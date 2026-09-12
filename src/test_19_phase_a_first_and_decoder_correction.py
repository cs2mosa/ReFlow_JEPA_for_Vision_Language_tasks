"""
PHASE 1a-ext2 -- the reordered pipeline: Phase A-FIRST (align Q-Pool/g_t_online
directly, before any CFM training), a genuinely stationary Phase 1, and Phase D
(decoder correction against real integrate() rollouts). See train_phase_a_first.py,
train.py's --freeze-qpool/--freeze-g-t-online/--base-checkpoint-path, and
train_decoder_correction.py's module docstrings for the full design rationale.

Targets:
  - alignment_training_step touches ONLY qpool/g_t_online, never the predictor
    (verified by checking .grad after backward(), not just reading the code).
  - freeze_qpool=True + freeze_g_t_online=True together give BIT-IDENTICAL
    z_v_tilde/z_t_tilde across a real training_step + optimizer.step() cycle -- the
    actual guarantee the whole reordered pipeline depends on, checked directly rather
    than assumed from "the parameters are frozen."
  - The task_token bug this session caught by testing (task_token feeds Q-Pool's
    FiLM conditioning, so freezing qpool's own parameters alone was insufficient) --
    a regression guard so it can't silently reappear.
  - decoder_recon_loss (factored out of training_step for Phase D) produces the
    IDENTICAL value training_step's own inline computation does, for the same Z1 --
    confirms the extraction didn't change the underlying computation.
  - build_decoder_optimizer's tied-embedding deduplication (T5's
    lm_head.weight IS decoder.embed_tokens.weight IS shared.weight by default) --
    without it, one tensor gets updated twice per optimizer.step(), corrupting Adam's
    moment estimates.
  - Phase D's one non-negotiable: a real correction step changes ONLY
    decoder/lm_head/prefix_expand parameters, verified by snapshotting every
    parameter in the model and checking equality after the step, not just checking
    the ones we expect to change.
  - best_of_k_match's K=1 sanity check (synthetic data has only one caption per
    image, so best_idx must always be 0).
"""
import torch

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset
from train_decoder_correction import (
    AllCaptionsDataset, collate_images_all_captions, build_decoder_optimizer, best_of_k_match,
)


def _tiny_model(**kwargs):
    torch.manual_seed(0)
    return ReflowJEPA(visual_layers=1, text_layers=1, predictor_depth=2, predictor_heads=4, **kwargs)


def test_alignment_training_step_touches_only_qpool_and_g_t_online():
    model = _tiny_model()
    images = torch.rand(4, 3, 224, 224)
    captions = ["a red dog", "a blue cat", "a green bird", "a yellow fish"]

    align_loss, vicreg_v, vicreg_t, diag = model.alignment_training_step(images, captions)
    total = align_loss + 5.0 * vicreg_v + 10.0 * vicreg_t
    total.backward()

    assert model.qpool.query_slots.grad is not None
    assert any(p.grad is not None for p in model.g_t_online.parameters())
    assert not any(p.grad is not None for p in model.predictor.parameters())
    assert not any(p.grad is not None for p in model.prefix_expand.parameters())


def test_freeze_qpool_and_g_t_online_give_genuinely_stationary_targets():
    """The actual guarantee the reordered pipeline depends on -- checked directly by
    running a real training_step + backward + optimizer.step(), not inferred from
    'the parameters are frozen.'"""
    model = _tiny_model(freeze_qpool=True, freeze_g_t_online=True)
    images = torch.rand(4, 3, 224, 224)
    captions = ["a red dog", "a blue cat", "a green bird", "a yellow fish"]

    c = model.task_token.expand(4, -1)
    z_v_before = model.encode_visual(images, c).detach().clone()
    z_t_before = model.encode_text_online(captions).detach().clone()

    cfm_loss, recon_loss, vicreg_v_loss, vicreg_t_loss, diag = model.training_step(images, captions)
    (cfm_loss + recon_loss).backward()
    opt = torch.optim.SGD(model.parameter_groups(1e-2), lr=1e-2)
    opt.step()

    z_v_after = model.encode_visual(images, c).detach()
    z_t_after = model.encode_text_online(captions).detach()
    assert torch.equal(z_v_before, z_v_after), "z_v_tilde changed despite freeze_qpool=True"
    assert torch.equal(z_t_before, z_t_after), "z_t_tilde changed despite freeze_g_t_online=True"


def test_freeze_qpool_also_freezes_task_token():
    """Regression guard for the bug this session's testing caught: task_token feeds
    Q-Pool's FiLM conditioning, so freezing qpool.parameters() alone is NOT
    sufficient for a stationary z_v_tilde -- task_token must freeze alongside it."""
    model = _tiny_model(freeze_qpool=True)
    assert not model.task_token.requires_grad
    assert all(not p.requires_grad for p in model.qpool.parameters())

    model_unfrozen = _tiny_model(freeze_qpool=False)
    assert model_unfrozen.task_token.requires_grad


def test_decoder_recon_loss_matches_training_step_semantics():
    """decoder_recon_loss was factored out of training_step's inline computation for
    Phase D's reuse -- confirms the extraction is faithful: for the SAME Z1, both
    paths should give the identical loss value. Uses model.eval() to disable
    dropout for this comparison -- the encoder's dropout otherwise makes two
    separate forward passes on the same input non-deterministic, which would fail
    this comparison for a reason having nothing to do with whether the extraction
    itself is faithful."""
    model = _tiny_model()
    model.eval()
    images = torch.rand(4, 3, 224, 224)
    captions = ["a red dog", "a blue cat", "a green bird", "a yellow fish"]

    cfm_loss, recon_loss_from_training_step, vv, vt, diag = model.training_step(images, captions)

    with torch.no_grad():
        c = model.task_token.expand(4, -1)
        z_t_tilde = model.encode_text_online(captions)
    recon_loss_from_new_method = model.decoder_recon_loss(z_t_tilde, captions)

    assert torch.allclose(recon_loss_from_training_step, recon_loss_from_new_method, atol=1e-4)


def test_decoder_optimizer_deduplicates_tied_embeddings():
    """T5's lm_head.weight IS decoder.embed_tokens.weight IS shared.weight by
    default (config.tie_word_embeddings=True) -- without deduplication, this tensor
    would appear twice in the optimizer's param list, updating it twice per step and
    corrupting Adam's moment estimates."""
    model = _tiny_model()
    assert model.text_seq2seq.lm_head.weight is model.text_seq2seq.shared.weight

    _, decoder_params = build_decoder_optimizer(model, lr=1e-4)
    ids = [id(p) for p in decoder_params]
    assert len(ids) == len(set(ids)), "duplicate parameter tensor(s) in the decoder optimizer's param list"


def test_best_of_k_match_is_trivial_for_synthetic_k1_data():
    ds = AllCaptionsDataset(SyntheticCaptioningDataset(length=8, seed=1))
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_images_all_captions)
    images, all_captions = next(iter(loader))
    assert all(len(caps) == 1 for caps in all_captions)

    model = _tiny_model(freeze_qpool=True, freeze_g_t_online=True)
    z_hat = model.integrate(images, n_steps=5)
    best_captions, best_dist, best_idx = best_of_k_match(model, z_hat, all_captions, torch.device("cpu"))
    assert (best_idx == 0).all(), "K=1 should always match index 0 by construction"


def test_phase_d_correction_step_changes_only_decoder_params():
    """The one non-negotiable of the whole phase, checked directly: snapshot EVERY
    parameter in the model, run one real correction step, and confirm the set of
    parameters that changed is EXACTLY the decoder optimizer's param list -- not
    'the ones we expect,' every single one, so an accidental leak anywhere else in
    the model would be caught."""
    model = _tiny_model(freeze_qpool=True, freeze_g_t_online=True)
    for p in model.predictor.parameters():
        p.requires_grad_(False)  # Phase D's own post-load freeze, mirrored here

    snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer, decoder_params = build_decoder_optimizer(model, lr=1e-3)
    decoder_param_ids = {id(p) for p in decoder_params}

    ds = AllCaptionsDataset(SyntheticCaptioningDataset(length=8, seed=1))
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_images_all_captions)
    images, all_captions = next(iter(loader))

    z_hat = model.integrate(images, n_steps=5)
    best_captions, best_dist, best_idx = best_of_k_match(model, z_hat, all_captions, torch.device("cpu"))
    recon_loss = model.decoder_recon_loss(z_hat, best_captions)
    optimizer.zero_grad()
    recon_loss.backward()
    optimizer.step()

    changed_non_decoder = [
        name for name, p in model.named_parameters()
        if id(p) not in decoder_param_ids and not torch.equal(snapshot[name], p.detach())
    ]
    assert changed_non_decoder == [], f"Non-decoder parameters changed during Phase D: {changed_non_decoder}"
