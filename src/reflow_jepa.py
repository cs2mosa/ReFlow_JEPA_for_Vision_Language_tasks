"""
Full ReflowJEPA assembly: frozen visual encoder -> Q-Pool -> stochastic source ->
predictor -> [Prefix-Expand -> T5 decoder] on one side, T5 encoder (online+EMA target)
-> text projection on the other. Implements DESIGN.md §2.2's training data flow and
§2.4 Phase 1 (base CFM).

General-VL pretraining phase specifics (per project decisions):
  - c = a single learned task-token (not per-example question), fed to both Q-Pool's
    FiLM and the predictor's cross-attention memory, exactly as DESIGN.md's data-flow
    diagram routes the question vector c to both places.
  - Decoding is Pipeline 2 (OFM-JEPA v2 §4): Prefix-Expand -> the SAME T5 checkpoint's
    own decoder (exposure-bias consistency), not a candidate-bank snap. The bank-snap
    decoder (Pipeline 1) is deferred to the VQA extension.
  - z_v_tilde and z_t_tilde are L2-normalized to the unit sphere (DESIGN.md item 13:
    "L2-normalize both embeddings before flow matching... standard practice, quantified
    per-encoder in test_03" -- test_03 validated the mechanism but this was never wired
    into the trained model itself until now). This changes the natural embedding scale
    from whatever an unnormalized projection head happens to produce down to exactly
    1.0, which is why `sigma` (stochastic source noise) and `vicreg_gamma` (VICReg's
    target std) both needed recalibrating downward in the same change -- see their
    docstrings/CLI help for the arithmetic. Shipping L2-norm without this would have
    made noise ~8x larger than signal and made VICReg's target permanently
    unsatisfiable (max possible per-dim std on a 768-dim unit sphere is ~0.036, far
    below the old default of 1.0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import (
    D_SHARED, D_IJEPA, D_TEXT, P_PATCHES, K_QUERY_SLOTS, K_PREFIX_TOKENS,
    load_visual_encoder, load_text_seq2seq, make_ema_copy, ema_update,
)
from qpool import QPool
from text_projection import TextProjectionHead
from stochastic_source import draw_stochastic_source
from predictor import VelocityPredictor
from prefix_expand import PrefixExpand
from vicreg import vicreg_variance_penalty


def _extract_patch_tokens(vit_last_hidden_state: torch.Tensor) -> torch.Tensor:
    """ViT-style checkpoints prepend a CLS token; true I-JEPA does not. Mirrors
    test_01's extraction logic so the same rule is used in training and tests."""
    n_tokens = vit_last_hidden_state.shape[1]
    if n_tokens == P_PATCHES + 1:
        return vit_last_hidden_state[:, 1:, :]
    elif n_tokens == P_PATCHES:
        return vit_last_hidden_state
    else:
        raise AssertionError(f"Expected {P_PATCHES} or {P_PATCHES + 1} tokens, got {n_tokens}.")


def _mean_pool_text(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)


class ReflowJEPA(nn.Module):
    def __init__(
        self,
        d_shared: int = D_SHARED,
        d_v: int = D_IJEPA,
        d_text: int = D_TEXT,
        k_query: int = K_QUERY_SLOTS,
        k_prefix: int = K_PREFIX_TOKENS,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        visual_layers: int = 2,
        text_layers: int = 2,
        sigma: float = 0.02,
        ema_momentum: float = 0.996,
        real_checkpoints: bool = False,
        freeze_text_encoder: bool = True,
        stop_grad_cfm_target: bool = True,
        edm_precondition: bool = True,
        ema_cfm_target: bool = False,
    ):
        super().__init__()
        self.sigma = sigma
        self.ema_momentum = ema_momentum
        self.freeze_text_encoder = freeze_text_encoder
        self.stop_grad_cfm_target = stop_grad_cfm_target
        self.ema_cfm_target = ema_cfm_target

        # Frozen visual encoder E_V. image_mean/image_std normalize raw [0,1] images
        # before the encoder sees them -- see load_visual_encoder's docstring: this is
        # a no-op (mean=0, std=1) for the mock path, and the officially documented
        # AutoImageProcessor stats for the real-checkpoint path.
        self.visual_encoder, image_mean, image_std = load_visual_encoder(
            num_layers=visual_layers, real_checkpoint=real_checkpoints)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)

        # Trainable Q-Pool (fuses Q-Pool + g_V per the original test suite's implementation)
        self.qpool = QPool(d_v=d_v, d_text=d_text, d_shared=d_shared, k=k_query)

        # Text seq2seq: online encoder+decoder, same-modality EMA target for the text
        # projection. The ENCODER itself is frozen by default (freeze_text_encoder=True)
        # -- see module-level rationale: an online encoder that's free to move can
        # actively construct a collapsed solution to make the CFM objective trivially
        # easy (DESIGN.md's own warning: "zero CFM loss is achievable despite total
        # collapse... VICReg is a separate, independently necessary term" -- a real
        # training run showed exactly this, vicreg_t pinned at ceiling for 1260+ steps
        # even after removing the small-vocabulary confound). Freezing removes the
        # encoder's ability to construct that collapse at all: whatever separation
        # exists in its raw output (confirmed non-collapsed even at random weights,
        # test_02b) is now fixed, and g_T alone has far less capacity to undo it than a
        # full transformer encoder does. The DECODER is NOT frozen here -- with mock
        # (never-pretrained) weights, freezing it at random init would leave zero
        # gradient path to ever learn anything, which is a different failure mode from
        # encoder-driven collapse. It's trained at a much smaller LR instead (see
        # parameter_groups) -- this is the "very little LR" option applied only where
        # a full freeze would actively break learning rather than fix collapse.
        self.text_seq2seq, self.tokenizer = load_text_seq2seq(num_layers=text_layers, real_checkpoint=real_checkpoints)
        self.g_t_online = TextProjectionHead(d_text=d_text, d_shared=d_shared)

        if freeze_text_encoder:
            for p in self.text_seq2seq.get_encoder().parameters():
                p.requires_grad_(False)
            self.text_encoder_target = self.text_seq2seq.get_encoder()  # same object: online never moves
        else:
            self.text_encoder_target = make_ema_copy(self.text_seq2seq.get_encoder())
        self.g_t_target = make_ema_copy(self.g_t_online)

        # Predictor v_theta. edm_precondition=True by default (predictor.py's
        # EDM-style reparametrization, added directly in response to
        # measure_terminal_divergence.py's finding that the raw-velocity architecture's
        # terminal divergence, measured on a real trained checkpoint, did not reliably
        # track the theoretical 1/(1-tau) rate -- unmoved by an LR schedule, a Reflow
        # round, or 10x more integration steps. See predictor.py's docstring for the
        # reparametrization itself, and test_13_edm_predictor.py for what's verified
        # about it (including a corrected initial hypothesis about gradient warmup).
        self.predictor = VelocityPredictor(d_shared=d_shared, depth=predictor_depth,
                                            n_heads=predictor_heads,
                                            edm_precondition=edm_precondition)

        # Decoder-side: Prefix-Expand + the text model's own (paired) decoder
        self.prefix_expand = PrefixExpand(d_shared=d_shared, k_prefix=k_prefix)

        # Single learned task-token, general-VL captioning phase (no per-example question)
        self.task_token = nn.Parameter(torch.randn(1, d_text) * 0.02)

    def _tokenize(self, captions):
        """Single place tokenizer output gets moved to the model's device. Fixes a
        real bug found in practice: the mock (and real HF) tokenizer always returns
        CPU tensors regardless of model device, causing a device-mismatch crash on any
        GPU run. Previously patched ad hoc after each of 4 separate call sites -- this
        consolidates it into one place so a 5th call site can't silently miss it.

        return_tensors="pt" and padding=True are passed EXPLICITLY here, not left to
        the tokenizer's own default. A second real bug found in practice: the mock
        tokenizer's __call__ signature defaults return_tensors="pt" and always returns
        tensors regardless of what's actually passed in, so calling it with no
        arguments happened to work by accident. Real T5TokenizerFast's actual default
        is return_tensors=None, which returns plain Python lists -- .to(device) on a
        list crashes immediately. Being explicit here removes the dependency on
        whichever tokenizer's particular defaults happen to line up."""
        batch = self.tokenizer(captions, return_tensors="pt", padding=True)
        device = self.task_token.device
        return {k: v.to(device) for k, v in batch.items()}

    def _project_text(self, batch, encoder, proj) -> torch.Tensor:
        """Shared text-encode-and-project logic, used for BOTH the online and target
        (EMA) pipelines by passing in whichever encoder/proj pair. L2-normalizes the
        output -- DESIGN.md item 13 requires this explicitly ("L2-normalize both
        embeddings before flow matching... quantified per-encoder in test_03") and
        test_03's test_l2_normalization_fixes_scale already validates the mechanism;
        this was the missing wiring step that never actually called it in the trained
        model, despite the test passing."""
        out = encoder(**batch).last_hidden_state
        pooled = _mean_pool_text(out, batch["attention_mask"])
        z = proj(pooled)
        return F.normalize(z, dim=-1)

    def trainable_parameters(self):
        """Everything with requires_grad=True. nn.Module.parameters() already
        deduplicates shared submodules (e.g. T5's shared input embedding, referenced
        from both .get_encoder() and .get_decoder()), so this is safe to use directly
        for gradient clipping without double-counting."""
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_groups(self, base_lr: float, decoder_lr_mult: float = 0.1):
        """Per-component learning rates for the optimizer: qpool/g_T/predictor/
        prefix_expand/task_token at base_lr; the text decoder (and, only if
        freeze_text_encoder=False was chosen, the text encoder) at base_lr *
        decoder_lr_mult -- the "very little LR" alternative to a hard freeze, used
        here specifically for the decoder (see __init__'s rationale for why the
        decoder can't be fully frozen with mock weights the way the encoder can).

        Manually deduplicates by parameter id() across the decoder/encoder groups --
        needed because HF's T5 shares one embedding table between encoder and decoder
        (assigning it to two groups with different LRs would cause PyTorch to update it
        twice per optimizer.step(), silently corrupting its Adam moment estimates)."""
        core_params = [p for p in (
            list(self.qpool.parameters()) + list(self.g_t_online.parameters())
            + list(self.predictor.parameters()) + list(self.prefix_expand.parameters())
            + [self.task_token]
        ) if p.requires_grad]
        seen = {id(p) for p in core_params}

        decoder_params = []
        for p in self.text_seq2seq.get_decoder().parameters():
            if p.requires_grad and id(p) not in seen:
                decoder_params.append(p)
                seen.add(id(p))

        encoder_params = []
        if not self.freeze_text_encoder:
            for p in self.text_seq2seq.get_encoder().parameters():
                if p.requires_grad and id(p) not in seen:
                    encoder_params.append(p)
                    seen.add(id(p))

        groups = [{"params": core_params, "lr": base_lr}]
        if decoder_params:
            groups.append({"params": decoder_params, "lr": base_lr * decoder_lr_mult})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": base_lr * decoder_lr_mult})
        return groups

    @torch.no_grad()
    def _visual_forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.image_mean) / self.image_std
        out = self.visual_encoder(images).last_hidden_state
        return _extract_patch_tokens(out)

    def encode_visual(self, images: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h_v = self._visual_forward(images)
        z_v = self.qpool(h_v, c)
        return F.normalize(z_v, dim=-1)  # z_v_tilde, (B, d_shared), unit norm -- see _project_text's docstring

    def encode_text_online(self, captions) -> torch.Tensor:
        batch = self._tokenize(captions)
        return self._project_text(batch, self.text_seq2seq.get_encoder(), self.g_t_online)

    @torch.no_grad()
    def encode_text_target(self, captions) -> torch.Tensor:
        """EMA/target-copy encoding, used for building the caption bank (retrieval-eval
        metrics). DESIGN.md's original data flow routed Z_1 through the ONLINE text
        pipeline only, with this target copy meant purely for eval-time retrieval
        stability (I-JEPA/BYOL/DINO precedent). If self.ema_cfm_target is True, this
        same target pipeline is ALSO used to build cfm_loss's regression target in
        training_step -- a deliberate, later revision of that original routing, made
        after a real decoder_lr_mult sweep on Flickr30k suggested the predictor's
        regression target being a snapshot of the constantly-shifting online g_T was
        part of what was destabilizing predictor convergence. See training_step's
        Z1_for_cfm computation for the full rationale."""
        batch = self._tokenize(captions)
        return self._project_text(batch, self.text_encoder_target, self.g_t_target)

    @torch.no_grad()
    def update_ema_target(self) -> None:
        if not self.freeze_text_encoder:
            ema_update(self.text_encoder_target, self.text_seq2seq.get_encoder(), self.ema_momentum)
        # if frozen, text_encoder_target IS the online encoder (same object) -- nothing to update
        ema_update(self.g_t_target, self.g_t_online, self.ema_momentum)

    def training_step(self, images: torch.Tensor, captions, vicreg_gamma: float = 0.02):
        """Phase 1 base CFM (DESIGN.md §2.4, Algorithm 1 line 4), PLUS a decoder
        reconstruction loss that DESIGN.md's original (VQA/candidate-bank) design never
        needed but Pipeline 2 does.

        Why the reconstruction term exists: the CFM loss alone only trains the flow to
        map Z_0 -> Z_1 in embedding space. It supplies no signal at all to the decoder
        or Prefix-Expand -- without a separate term, the decoder would stay at its
        random initialization no matter how well the flow converges. The fix, standard
        practice for latent-variable generative models (train the decoder to
        reconstruct from the TRUE latent, train the prior/flow to reach that latent
        separately): decoder + Prefix-Expand learn to reconstruct the caption from the
        true z_t_tilde (not detached -- letting this loss also shape the text
        projection means the projection is trained to be decodable, not just
        VICReg-healthy and distinct, which is the actual point of choosing Pipeline 2
        over a candidate-bank snap). At inference, whatever gap remains between the
        flow's integrated z_hat_t and the true z_t_tilde is exactly what
        `integrate`/`generate_captions` exposes -- this loss does not paper over that
        gap, it only makes sure the decoder is capable of using a good z at all.
        """
        B = images.shape[0]
        c = self.task_token.expand(B, -1)

        z_v_tilde = self.encode_visual(images, c)          # (B, d) -- NOT detached: still
                                                             # used as the predictor's
                                                             # conditioning input below,
                                                             # which is the only real
                                                             # task-relevant signal Q-Pool
                                                             # has (nothing else touches it)

        # Stop-gradient z_v_tilde ONLY for its role defining the stochastic source Z_0.
        # Rationale, found empirically: after detaching Z_1 for cfm_loss (see above), a
        # real run showed vicreg_v stuck at ~0.97-0.98 for 300+ steps, unmoved -- ALL of
        # cfm_loss's gradient pressure that previously split across both z_v_tilde and
        # z_t_tilde now concentrates entirely on z_v_tilde, since detaching Z_1 removed
        # its other outlet. And there's a specific mechanical reason cfm_loss doesn't
        # need real visual information to be minimized here: with Z_0 = z_v_tilde +
        # sigma*eps, eps is "exactly invertible from Z_tau, so zero CFM loss is
        # achievable despite total collapse" (DESIGN.md's own warning about this exact
        # mechanism) -- the predictor can satisfy the regression using only the noise
        # term, giving z_v_tilde no real incentive to stay informative via THIS pathway.
        # Detaching it here removes that free-riding route while preserving the
        # conditioning-input gradient path (below), which DOES require z_v_tilde to
        # carry real information for the predictor to use it well.
        z_v_for_source = z_v_tilde.detach() if self.stop_grad_cfm_target else z_v_tilde

        batch = self._tokenize(captions)
        z_t_tilde = self._project_text(batch, self.text_seq2seq.get_encoder(), self.g_t_online)  # Z_1, unit norm

        # STOP-GRADIENT for the CFM regression target only (mirrors latent-diffusion
        # practice: freeze the VAE/representation, train the prior only in its latent
        # space). Motivated directly by an empirical finding: a real training run's
        # gradient-norm breakdown showed cfm_loss inducing a gradient of ~77 on
        # g_t_online's output layer vs. vicreg_t's ~0.07 (even at 10x loss weight,
        # ~0.7) -- cfm_loss was actively driving z_t BACK toward collapse over time,
        # completely swamping VICReg's counter-pressure regardless of weight. Detaching
        # Z1 here means only recon_loss and vicreg_t shape the text representation;
        # cfm_loss trains the predictor to hit wherever that representation currently
        # is, without being able to drag it around to make its own regression easier.
        #
        # ema_cfm_target=True (optional, default False): a FURTHER refinement on top of
        # the stop-gradient above, motivated by a real observed pattern across a
        # decoder_lr_mult sweep on Flickr30k, not by the original design doc (DESIGN.md
        # explicitly routed the EMA target copy to caption-bank retrieval eval only,
        # calling out that cfm_loss uses the online pipeline -- this flag deliberately
        # revisits that call, it is not "using an already-intended mechanism as
        # planned"). The sweep showed decoder_lr_mult noticeably changing how well the
        # PREDICTOR converged (measure_exposure_bias's resid_ratio: ~4.5-4.9 at
        # decoder_lr_mult=0.1 vs ~1.05-1.13 at 0.02), despite cfm_loss having no direct
        # gradient path to g_t_online at all under stop-gradient -- consistent with an
        # INDIRECT mechanism: even with cfm_loss detached, the predictor's regression
        # target snapshots g_t_online's CURRENT weights, which recon_loss keeps shifting
        # every step (a smaller decoder_lr_mult happened to leave less of that pressure
        # concentrated on g_t_online, i.e. a smaller mult reduced the drift as a side
        # effect, not because that was ever a deliberate lever for it). Using the
        # EMA-smoothed g_t_target instead gives the predictor a slowly, smoothly
        # evolving target regardless of decoder_lr_mult, directly targeting that
        # mechanism rather than relying on a value of decoder_lr_mult that happened to
        # reduce it as a side effect.
        if self.ema_cfm_target:
            with torch.no_grad():
                Z1_for_cfm = self._project_text(batch, self.text_encoder_target, self.g_t_target)
        else:
            Z1_for_cfm = z_t_tilde.detach() if self.stop_grad_cfm_target else z_t_tilde

        Z0 = draw_stochastic_source(z_v_for_source, self.sigma)
        tau = torch.rand(B, device=images.device)
        Z_tau = (1 - tau).unsqueeze(-1) * Z0 + tau.unsqueeze(-1) * Z1_for_cfm

        v_pred = self.predictor(Z_tau, tau, z_v_tilde, c)   # conditioning: full gradient, NOT z_v_for_source

        if self.predictor.edm_precondition:
            # On the training distribution, Z_tau EXACTLY interpolates Z0->Z1_for_cfm,
            # so if the network's bounded target-estimate z1_hat has prediction error
            # eps, then v_pred = (Z1_for_cfm - Z0) + eps/(1-tau) -- ANY imperfection in
            # z1_hat gets amplified by 1/(1-tau), unboundedly as tau->1. This is not
            # hypothetical: an early smoke test of this architecture produced a real
            # cfm_loss spike above 500,000 the first time a sampled tau landed very
            # close to 1 while z1_hat was still imperfect (normal, especially early in
            # training). Fix: algebraically recover z1_hat from v_pred (exact
            # inversion: v_pred = (z1_hat - Z_tau)/(1-tau) => z1_hat = v_pred*(1-tau) +
            # Z_tau) and supervise THAT directly -- same information content, but
            # without ever constructing the amplified quantity during training.
            z1_hat = v_pred * (1 - tau).unsqueeze(-1) + Z_tau
            cfm_loss = (z1_hat - Z1_for_cfm).pow(2).sum(dim=-1).mean()
        else:
            cfm_loss = (v_pred - (Z1_for_cfm - Z0)).pow(2).sum(dim=-1).mean()

        Z1 = z_t_tilde  # non-detached: recon_loss and vicreg_t below DO shape g_T
        recon_prefix = self.prefix_expand(Z1)
        # Mask padding positions with -100 (HF's ignore_index convention) before
        # computing the reconstruction loss. Harmless no-op with the mock tokenizer
        # (fixed-length, attention_mask is always all-ones -- nothing gets masked),
        # but REQUIRED once a real tokenizer produces variable-length, padded
        # sequences: without this, padding tokens get trained on as if they were real
        # targets, silently corrupting the loss.
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        recon_out = self.text_seq2seq(encoder_outputs=(recon_prefix,), labels=labels)
        recon_loss = recon_out.loss

        vicreg_v = vicreg_variance_penalty(z_v_tilde, gamma_0=vicreg_gamma)
        vicreg_t = vicreg_variance_penalty(Z1, gamma_0=vicreg_gamma)

        diagnostics = {
            "cfm_loss": cfm_loss.item(),
            "recon_loss": recon_loss.item(),
            "vicreg_v": vicreg_v.item(),
            "vicreg_t": vicreg_t.item(),
            "z_v_norm": z_v_tilde.norm(dim=-1).mean().item(),
            "z_t_norm": Z1.norm(dim=-1).mean().item(),
        }
        return cfm_loss, recon_loss, vicreg_v, vicreg_t, diagnostics

    def gradient_norm_breakdown(self, images: torch.Tensor, captions, vicreg_gamma: float = 0.02):
        """Diagnostic only (not part of the training step): for each loss term, the
        gradient norm it induces on TWO shared anchors -- g_t_online's output layer
        (text side) and qpool.g_v's output layer (visual side). Directly answers "is
        VICReg actually competing on equal footing, or is cfm_loss's remaining
        pressure concentrating somewhere unopposed" instead of inferring it from loss
        curves alone. This is exactly how the visual-side collapse (after fixing the
        text side) was found: detaching Z_1 removed cfm_loss's outlet through the text
        branch, concentrating all of it onto z_v_tilde instead. Expensive (multiple
        backward passes) -- call sparingly, e.g. at eval_every."""
        B = images.shape[0]
        c = self.task_token.expand(B, -1)
        z_v_tilde = self.encode_visual(images, c)
        z_v_for_source = z_v_tilde.detach() if self.stop_grad_cfm_target else z_v_tilde
        batch = self._tokenize(captions)
        z_t_tilde = self._project_text(batch, self.text_seq2seq.get_encoder(), self.g_t_online)

        text_anchor = self.g_t_online.net[-1].weight
        visual_anchor = self.qpool.g_v[-1].weight

        Z0 = draw_stochastic_source(z_v_for_source, self.sigma)
        if self.ema_cfm_target:
            with torch.no_grad():
                Z1_for_cfm = self._project_text(batch, self.text_encoder_target, self.g_t_target)
        else:
            Z1_for_cfm = z_t_tilde.detach() if self.stop_grad_cfm_target else z_t_tilde
        Z1 = z_t_tilde
        tau = torch.rand(B, device=images.device)
        Z_tau = (1 - tau).unsqueeze(-1) * Z0 + tau.unsqueeze(-1) * Z1_for_cfm
        v_pred = self.predictor(Z_tau, tau, z_v_tilde, c)
        if self.predictor.edm_precondition:
            z1_hat = v_pred * (1 - tau).unsqueeze(-1) + Z_tau
            cfm_loss = (z1_hat - Z1_for_cfm).pow(2).sum(dim=-1).mean()
        else:
            cfm_loss = (v_pred - (Z1_for_cfm - Z0)).pow(2).sum(dim=-1).mean()
        recon_prefix = self.prefix_expand(Z1)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        recon_loss = self.text_seq2seq(encoder_outputs=(recon_prefix,), labels=labels).loss
        vicreg_t = vicreg_variance_penalty(Z1, gamma_0=vicreg_gamma)
        vicreg_v = vicreg_variance_penalty(z_v_tilde, gamma_0=vicreg_gamma)

        norms = {}
        for name, loss, anchor in [
            ("cfm_on_text", cfm_loss, text_anchor), ("recon_on_text", recon_loss, text_anchor),
            ("vicreg_t_on_text", vicreg_t, text_anchor),
            ("cfm_on_visual", cfm_loss, visual_anchor), ("vicreg_v_on_visual", vicreg_v, visual_anchor),
        ]:
            grad = torch.autograd.grad(loss, anchor, retain_graph=True, allow_unused=True)[0]
            norms[name] = 0.0 if grad is None else grad.norm().item()
        return norms

    @torch.no_grad()
    def integrate(self, images: torch.Tensor, n_steps: int = 50, delta: float = 1e-3) -> torch.Tensor:
        """Inference: Euler-integrate the CURRENT (trained) predictor from tau=0 to
        tau=1-delta. This is the honest counterpart to test_07/test_07b's exact-field
        integration -- same ODE, but with the learned v_theta instead of a hand-derived
        ground-truth field, which is exactly the gap Phase B exists to probe."""
        B = images.shape[0]
        c = self.task_token.expand(B, -1)
        z_v_tilde = self.encode_visual(images, c)
        Z = draw_stochastic_source(z_v_tilde, self.sigma)
        taus = torch.linspace(0, 1 - delta, n_steps + 1, device=images.device)
        dtau = taus[1] - taus[0]
        for i in range(n_steps):
            tau_batch = taus[i].expand(B)
            v = self.predictor(Z, tau_batch, z_v_tilde, c)
            Z = Z + v * dtau
        return Z  # z_hat_t

    @torch.no_grad()
    def generate_captions(self, images: torch.Tensor, max_new_tokens: int = 16, n_steps: int = 50):
        z_hat = self.integrate(images, n_steps=n_steps)
        prefix = self.prefix_expand(z_hat)  # (B, K', d) stands in for encoder_hidden_states
        decoder = self.text_seq2seq.get_decoder()
        B = images.shape[0]
        input_ids = torch.zeros(B, 1, dtype=torch.long, device=images.device)  # decoder start token id
        for _ in range(max_new_tokens):
            out = self.text_seq2seq(
                encoder_outputs=(prefix,),
                decoder_input_ids=input_ids,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
