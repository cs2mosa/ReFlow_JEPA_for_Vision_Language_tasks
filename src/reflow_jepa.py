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
"""
import torch
import torch.nn as nn

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
        sigma: float = 0.3,
        ema_momentum: float = 0.996,
        real_checkpoints: bool = False,
    ):
        super().__init__()
        self.sigma = sigma
        self.ema_momentum = ema_momentum

        # Frozen visual encoder E_V
        self.visual_encoder = load_visual_encoder(num_layers=visual_layers, real_checkpoint=real_checkpoints)
        for p in self.visual_encoder.parameters():
            p.requires_grad_(False)

        # Trainable Q-Pool (fuses Q-Pool + g_V per the original test suite's implementation)
        self.qpool = QPool(d_v=d_v, d_text=d_text, d_shared=d_shared, k=k_query)

        # Text seq2seq: online encoder+decoder (trainable), same-modality EMA target
        self.text_seq2seq, self.tokenizer = load_text_seq2seq(num_layers=text_layers, real_checkpoint=real_checkpoints)
        self.g_t_online = TextProjectionHead(d_text=d_text, d_shared=d_shared)

        self.text_encoder_target = make_ema_copy(self.text_seq2seq.get_encoder())
        self.g_t_target = make_ema_copy(self.g_t_online)

        # Predictor v_theta
        self.predictor = VelocityPredictor(d_shared=d_shared, depth=predictor_depth, n_heads=predictor_heads)

        # Decoder-side: Prefix-Expand + the text model's own (paired) decoder
        self.prefix_expand = PrefixExpand(d_shared=d_shared, k_prefix=k_prefix)

        # Single learned task-token, general-VL captioning phase (no per-example question)
        self.task_token = nn.Parameter(torch.randn(1, d_text) * 0.02)

    def trainable_parameters(self):
        """Everything except the frozen visual encoder and the EMA target copies
        (which are updated by ema_update, not gradient descent)."""
        modules = [self.qpool, self.text_seq2seq, self.g_t_online, self.predictor, self.prefix_expand]
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    yield p
        yield self.task_token

    @torch.no_grad()
    def _visual_forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.visual_encoder(images).last_hidden_state
        return _extract_patch_tokens(out)

    def encode_visual(self, images: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h_v = self._visual_forward(images)
        return self.qpool(h_v, c)  # z_v_tilde, (B, d_shared)

    def encode_text_online(self, captions) -> torch.Tensor:
        batch = self.tokenizer(captions)
        batch = {k: v.to(images.device) for k, v in batch.items()} 
        out = self.text_seq2seq.get_encoder()(**batch).last_hidden_state
        pooled = _mean_pool_text(out, batch["attention_mask"])
        return self.g_t_online(pooled)  # z_t_tilde = Z_1, (B, d_shared)

    @torch.no_grad()
    def encode_text_target(self, captions) -> torch.Tensor:
        """EMA/target-copy encoding, used for building the caption bank (retrieval-eval
        metrics) -- NOT part of the CFM training loss itself (DESIGN.md's data flow
        routes Z_1 through the ONLINE text pipeline; the target copy tracks it via EMA
        for stability, mirroring I-JEPA/BYOL/DINO precedent)."""
        batch = self.tokenizer(captions)
        batch = {k: v.to(images.device) for k, v in batch.items()} 
        out = self.text_encoder_target(**batch).last_hidden_state
        pooled = _mean_pool_text(out, batch["attention_mask"])
        return self.g_t_target(pooled)

    @torch.no_grad()
    def update_ema_target(self) -> None:
        ema_update(self.text_encoder_target, self.text_seq2seq.get_encoder(), self.ema_momentum)
        ema_update(self.g_t_target, self.g_t_online, self.ema_momentum)

    def training_step(self, images: torch.Tensor, captions, vicreg_gamma: float = 1.0):
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

        z_v_tilde = self.encode_visual(images, c)          # (B, d)

        batch = self.tokenizer(captions)
        batch = {k: v.to(images.device) for k, v in batch.items()} 
        enc_out = self.text_seq2seq.get_encoder()(**batch).last_hidden_state
        pooled = _mean_pool_text(enc_out, batch["attention_mask"])
        z_t_tilde = self.g_t_online(pooled)                 # (B, d) = Z_1

        Z0 = draw_stochastic_source(z_v_tilde, self.sigma)
        Z1 = z_t_tilde
        tau = torch.rand(B, device=images.device)
        Z_tau = (1 - tau).unsqueeze(-1) * Z0 + tau.unsqueeze(-1) * Z1

        v_pred = self.predictor(Z_tau, tau, z_v_tilde, c)
        cfm_loss = (v_pred - (Z1 - Z0)).pow(2).sum(dim=-1).mean()

        recon_prefix = self.prefix_expand(Z1)
        recon_out = self.text_seq2seq(encoder_outputs=(recon_prefix,), labels=batch["input_ids"])
        recon_loss = recon_out.loss

        vicreg_v = vicreg_variance_penalty(z_v_tilde, gamma_0=vicreg_gamma)
        vicreg_t = vicreg_variance_penalty(Z1, gamma_0=vicreg_gamma)
        vicreg_loss = vicreg_v + vicreg_t

        diagnostics = {
            "cfm_loss": cfm_loss.item(),
            "recon_loss": recon_loss.item(),
            "vicreg_v": vicreg_v.item(),
            "vicreg_t": vicreg_t.item(),
            "z_v_norm": z_v_tilde.norm(dim=-1).mean().item(),
            "z_t_norm": Z1.norm(dim=-1).mean().item(),
        }
        return cfm_loss, recon_loss, vicreg_loss, diagnostics

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
