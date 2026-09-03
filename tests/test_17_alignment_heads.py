"""
Tests for Phase A: supervised Sinkhorn cross-modal alignment (alignment_stage_spec.md
section 7). All tests run locally without internet access, on small synthetic/
controlled data, per this project's established testing discipline (spec sec 2).

Tests 1-6 exercise alignment_heads.py directly on plain tensors (no base model
needed -- these are pure mechanism tests). Tests 7-8 go through a real (mock-weight,
tiny) ReflowJEPA instance, mirroring tests/test_16_ema_cfm_target.py's
_tiny_model/_tiny_batch pattern, since gradient isolation and the full training loop
are specifically about the INTEGRATION with the frozen base model.
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alignment_heads import (
    AlignmentHead,
    alignment_loss,
    sinkhorn_log_domain,
    sinkhorn_marginal_error,
    vicreg_variance_penalty,
)
from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from train_alignment import (
    alignment_space_distance,
    find_hard_negative_pair,
    load_alignment_checkpoint,
    retrieval_accuracy,
    save_alignment_checkpoint,
    train_alignment,
)


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers (pure tensors -- no base model, no dataset files)
# ---------------------------------------------------------------------------
def _tiny_model(**kwargs):
    """Mirrors tests/test_16_ema_cfm_target.py's _tiny_model -- smallest config that
    still exercises the real architecture (mock/random weights, no internet)."""
    return ReflowJEPA(visual_layers=1, text_layers=1, predictor_depth=2, predictor_heads=4, **kwargs)


def _tiny_batch(batch_size=8):
    ds = SyntheticCaptioningDataset(length=batch_size, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_images_captions)
    return next(iter(dl))


def _paired_tensors(B, d_in=64, signal=1.0, noise=0.6, generator=None):
    """Known true pairing (index i <-> index i) via a shared latent, plus independent
    per-modality noise -- learnable but not trivial at init (spec sec 7 test 4).
    Unit-normalized to match the real z_v_tilde/z_t_tilde convention (spec sec 1:
    'both already unit-normalized') -- find_hard_negative_pair's/alignment_loss's
    dot-product-as-cosine-similarity logic assumes this."""
    latent = torch.randn(B, d_in, generator=generator)
    z_v = signal * latent + noise * torch.randn(B, d_in, generator=generator)
    z_t = signal * latent + noise * torch.randn(B, d_in, generator=generator)
    return F.normalize(z_v, dim=-1), F.normalize(z_t, dim=-1)


def _paired_tensors_with_hard_negative(B, d_in=64, signal=1.0, noise=0.6, generator=None):
    """Same as _paired_tensors, but index 1's z_v is forced close to index 0's z_v
    while both keep their OWN true caption (different latents) -- the planted
    Mosa/ego hard negative (spec sec 6 diagnostic 2 / sec 7 test 5)."""
    z_v, z_t = _paired_tensors(B, d_in, signal, noise, generator)
    with torch.no_grad():
        z_v[1] = F.normalize(z_v[0] + 0.01 * torch.randn(d_in, generator=generator), dim=-1)
    return z_v, z_t


def _train_heads(z_v, z_t, steps, epsilon=0.05, n_iters=50, vicreg_weight=1.0,
                  vicreg_gamma=None, d_align=32, lr=0.05, seed=0, average_last_n=None):
    """Minimal training loop over fixed synthetic tensors, used by tests 4-6.

    If `average_last_n` is given, also returns the mean per-dim batch std of A_v,
    averaged over the last `average_last_n` steps (smooths out step-to-step noise
    on stiff/near-degenerate loss landscapes -- see test 6)."""
    torch.manual_seed(seed)
    d_in = z_v.shape[1]
    if vicreg_gamma is None:
        vicreg_gamma = 1.0 / math.sqrt(d_align)  # matches train.py's own unit-sphere-scaled convention
    h_v = AlignmentHead(d_in=d_in, d_align=d_align)
    h_t = AlignmentHead(d_in=d_in, d_align=d_align)
    opt = torch.optim.AdamW(list(h_v.parameters()) + list(h_t.parameters()), lr=lr)

    losses = []
    recent_stds = []
    for step in range(steps):
        A_v = h_v(z_v.detach())
        A_t = h_t(z_t.detach())
        align_term = alignment_loss(A_v, A_t, epsilon, n_iters, warn_on_convergence=False)
        vicreg_term = vicreg_variance_penalty(A_v, vicreg_gamma) + vicreg_variance_penalty(A_t, vicreg_gamma)
        total = align_term + vicreg_weight * vicreg_term
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(align_term.item())
        if average_last_n is not None and step >= steps - average_last_n:
            recent_stds.append(A_v.std(dim=0, unbiased=False).mean().item())

    avg_std = sum(recent_stds) / len(recent_stds) if recent_stds else None
    return h_v, h_t, losses, avg_std


# ---------------------------------------------------------------------------
# Test 1: Sinkhorn doubly-stochastic property
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.2, 1.0])
def test_sinkhorn_doubly_stochastic_and_no_nan_inf(epsilon):
    torch.manual_seed(0)
    B = 16
    # Smaller epsilon converges more slowly (spec sec 4.2: "too small -> slower
    # convergence, needs more iterations") -- scale n_iters accordingly rather than
    # using a single fixed count for every epsilon tested.
    n_iters = max(100, int(3.0 / epsilon))
    C = torch.rand(B, B) * 2  # random cost matrix, range roughly matches [0, 2] cosine-distance scale
    P = sinkhorn_log_domain(C, epsilon=epsilon, n_iters=n_iters)

    assert not torch.isnan(P).any(), f"NaN in P at epsilon={epsilon}"
    assert not torch.isinf(P).any(), f"Inf in P at epsilon={epsilon}"

    row_err, col_err = sinkhorn_marginal_error(P)
    tol = 1e-3
    assert row_err < tol, f"Row sums deviate from 1/B by {row_err:.2e} at epsilon={epsilon} (tol {tol})"
    assert col_err < tol, f"Col sums deviate from 1/B by {col_err:.2e} at epsilon={epsilon} (tol {tol})"


# ---------------------------------------------------------------------------
# Test 2: diagonal-concentration sanity check
# ---------------------------------------------------------------------------
def test_sinkhorn_concentrates_on_diagonal_when_diagonal_is_cheapest():
    torch.manual_seed(0)
    B = 12
    # Diagonal deliberately cheap, off-diagonal deliberately expensive.
    C = torch.rand(B, B) * 0.5 + 1.5  # off-diagonal in [1.5, 2.0]
    C.fill_diagonal_(0.0)
    C += torch.rand(B, B) * 0.05  # small jitter so it's not perfectly degenerate
    C.diagonal().clamp_(min=0.0)

    P = sinkhorn_log_domain(C, epsilon=0.02, n_iters=200)
    diag_mass = torch.diagonal(P).mean().item()
    concentration = B * diag_mass
    assert concentration > 0.8, (
        f"Expected P to concentrate on the diagonal (B*diag_mass > 0.8) when the "
        f"diagonal is the unique minimum-cost pairing; got {concentration:.4f}. If "
        f"this fails, sinkhorn_log_domain does not do what's expected even on a "
        f"fully controllable case."
    )


# ---------------------------------------------------------------------------
# Test 3: loss boundedness
# ---------------------------------------------------------------------------
def test_alignment_loss_is_bounded_in_zero_to_one_ish():
    torch.manual_seed(0)
    B, d = 10, 16

    cases = {
        "random": (F_normalize(torch.randn(B, d)), F_normalize(torch.randn(B, d))),
        "identical_pairs": (lambda: (lambda z: (z, z))(F_normalize(torch.randn(B, d))))(),
        "all_identical_vectors": (
            F_normalize(torch.ones(B, d)),
            F_normalize(torch.ones(B, d)),
        ),
        "antipodal": (F_normalize(torch.randn(B, d)), None),
    }
    v_anti = cases["antipodal"][0]
    cases["antipodal"] = (v_anti, -v_anti)

    for name, (A_v, A_t) in cases.items():
        loss = alignment_loss(A_v, A_t, epsilon=0.05, n_iters=50, warn_on_convergence=False)
        assert -1e-4 <= loss.item() <= 1.0 + 1e-3, (
            f"alignment_loss out of expected [0, ~1] range for case '{name}': {loss.item():.4f}"
        )


def F_normalize(x):
    import torch.nn.functional as F
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Test 4: end-to-end loss decreases + retrieval improves on tiny synthetic dataset
# ---------------------------------------------------------------------------
def test_training_decreases_loss_and_improves_retrieval():
    torch.manual_seed(0)
    B = 8
    z_v, z_t = _paired_tensors(B, d_in=64, signal=1.0, noise=0.6)

    h_v0, h_t0, _, _ = _train_heads(z_v, z_t, steps=0)  # untrained, for baseline retrieval
    with torch.no_grad():
        A_v0, A_t0 = h_v0(z_v), h_t0(z_t)
        top1_before, _ = retrieval_accuracy(A_v0, A_t0)

    h_v, h_t, losses, _ = _train_heads(z_v, z_t, steps=300)
    with torch.no_grad():
        A_v, A_t = h_v(z_v), h_t(z_t)
        top1_after, top5_after = retrieval_accuracy(A_v, A_t)

    assert losses[-1] < losses[0], (
        f"alignment_loss did not decrease over training: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    assert top1_after > top1_before, (
        f"top-1 retrieval accuracy did not improve: {top1_before:.3f} -> {top1_after:.3f}"
    )
    assert top1_after >= 0.5, (
        f"Expected substantially-above-chance top-1 retrieval after training on a "
        f"learnable synthetic pairing; got {top1_after:.3f} (chance ~= 1/{B} = {1/B:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 5: automated Mosa/ego stress test
# ---------------------------------------------------------------------------
def test_hard_negative_distance_increases_over_training():
    torch.manual_seed(0)
    B = 8
    z_v, z_t = _paired_tensors_with_hard_negative(B, d_in=64, signal=1.0, noise=0.6)
    i, j, pre_sim = find_hard_negative_pair(z_v, z_t)
    assert i != j, "find_hard_negative_pair returned a diagonal (i == j) pair, expected a cross pair"

    h_v0, h_t0, _, _ = _train_heads(z_v, z_t, steps=0)
    with torch.no_grad():
        dist_before = alignment_space_distance(h_v0(z_v), h_t0(z_t), i, j)

    h_v, h_t, _, _ = _train_heads(z_v, z_t, steps=300)
    with torch.no_grad():
        dist_after = alignment_space_distance(h_v(z_v), h_t(z_t), i, j)

    assert dist_after > dist_before, (
        f"Hard-negative pair ({i}, {j})'s alignment-space distance did not increase "
        f"over training: {dist_before:.4f} -> {dist_after:.4f}. This is the direct, "
        f"targeted test of the Mosa/ego failure mode (spec sec 1.1/6) -- a good "
        f"aggregate retrieval number alone would not have caught this."
    )


# ---------------------------------------------------------------------------
# Test 6: VICReg collapse test
# ---------------------------------------------------------------------------
def test_vicreg_weight_prevents_collapse():
    torch.manual_seed(0)
    B, d_in, d_align = 16, 64, 32
    # Degenerate/low-diversity input: all rows nearly identical (tiny per-example jitter
    # only) -- deliberately easy for the heads to collapse onto if nothing stops them.
    # NOT unit-normalized here on purpose: this input stands in for a pathological
    # upstream batch, not z_v_tilde/z_t_tilde itself, and the degenerate-collapse
    # mechanism being tested (LayerNorm mapping near-identical rows to near-identical
    # outputs absent a countervailing force) doesn't depend on that convention.
    base_v = torch.randn(1, d_in)
    base_t = torch.randn(1, d_in)
    z_v = base_v.expand(B, -1) + 0.01 * torch.randn(B, d_in)
    z_t = base_t.expand(B, -1) + 0.01 * torch.randn(B, d_in)

    # Small, stable lr + more steps than tests 4/5: at this input scale the loss
    # landscape is stiff (near-degenerate input), and a larger lr makes both the
    # zero- and nonzero-vicreg runs noisy/non-monotonic rather than settling. Final
    # std is averaged over the last few steps for the same reason.
    h_v_zero, h_t_zero, _, std_zero = _train_heads(
        z_v, z_t, steps=800, vicreg_weight=0.0, d_align=d_align, lr=0.005,
        n_iters=20, seed=1, average_last_n=30)

    h_v_reg, h_t_reg, _, std_reg = _train_heads(
        z_v, z_t, steps=800, vicreg_weight=10.0, d_align=d_align, lr=0.005,
        n_iters=20, seed=1, average_last_n=30)

    assert std_zero < 0.01, (
        f"Expected vicreg_weight=0 to allow near-collapse (mean per-dim std < 0.01) on "
        f"degenerate input; got {std_zero:.4f} -- test input may not be degenerate enough."
    )
    assert std_reg > std_zero * 3, (
        f"Expected a nonzero vicreg_weight to measurably prevent collapse relative to "
        f"vicreg_weight=0; got std_reg={std_reg:.4f} vs std_zero={std_zero:.4f}. If "
        f"these are close, the regularizer is not load-bearing."
    )


# ---------------------------------------------------------------------------
# Test 7: gradient isolation
# ---------------------------------------------------------------------------
def test_no_gradient_reaches_frozen_base_model():
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    model.requires_grad_(False)  # spec sec 3/5: base model is a frozen feature source

    images, captions = _tiny_batch(batch_size=8)
    B = images.shape[0]
    c = model.task_token.expand(B, -1)
    with torch.no_grad():
        z_v_tilde = model.encode_visual(images, c)
        z_t_tilde = model.encode_text_online(captions)

    h_v = AlignmentHead(d_in=768, d_align=32)
    h_t = AlignmentHead(d_in=768, d_align=32)
    A_v = h_v(z_v_tilde.detach())
    A_t = h_t(z_t_tilde.detach())

    align_term = alignment_loss(A_v, A_t, epsilon=0.05, n_iters=50, warn_on_convergence=False)
    vicreg_term = vicreg_variance_penalty(A_v, 0.02) + vicreg_variance_penalty(A_t, 0.02)
    total = align_term + vicreg_term
    total.backward()

    for name, p in model.qpool.named_parameters():
        assert p.grad is None, f"model.qpool.{name} received a gradient -- base model must stay frozen (spec sec 3)."
    for name, p in model.g_t_online.named_parameters():
        assert p.grad is None, f"model.g_t_online.{name} received a gradient -- base model must stay frozen (spec sec 3)."

    for name, p in h_v.named_parameters():
        assert p.grad is not None, f"h_v.{name} did NOT receive a gradient -- Phase A heads should train."
    for name, p in h_t.named_parameters():
        assert p.grad is not None, f"h_t.{name} did NOT receive a gradient -- Phase A heads should train."


# ---------------------------------------------------------------------------
# Test 8: smoke test -- full training loop, tiny settings, no crash
# ---------------------------------------------------------------------------
def test_train_alignment_smoke_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)

    # Build + save a tiny frozen "base checkpoint" in train.py's own save_checkpoint format.
    base_model = _tiny_model()
    base_ckpt_path = tmp_path / "tiny_base_ckpt.pt"
    torch.save({
        "step": 0,
        "model_state_dict": base_model.state_dict(),
        "args": {"visual_layers": 1, "text_layers": 1, "predictor_depth": 2, "predictor_heads": 4},
    }, base_ckpt_path)

    import argparse
    args = argparse.Namespace(
        base_checkpoint_path=str(base_ckpt_path),
        predictor_depth=2, predictor_heads=4, visual_layers=1, text_layers=1,
        k_query=8, k_prefix=8, real_checkpoints=False, edm_precondition=True,
        ema_cfm_target=False, freeze_text_encoder=True, stop_grad_cfm_target=True,
        sigma=0.02, ema_momentum=0.996,
        align_dim=16, sinkhorn_epsilon=0.05, sinkhorn_iters=20,
        vicreg_weight=1.0, vicreg_gamma=0.02,
        steps=6, batch_size=8, lr=1e-3, eval_every=3, checkpoint_every=3,
        log_path=str(tmp_path / "alignment_log.json"),
        checkpoint_path=str(tmp_path / "alignment_heads_ckpt.pt"),
        dataset="synthetic", flickr_karpathy_split=None, dataset_length=64, image_size=224,
        seed=0, device="cpu",
    )

    log = train_alignment(args)
    assert len(log) == args.steps
    for record in log:
        assert math.isfinite(record["total_loss"]), f"Non-finite total_loss at step {record['step']}"
        assert math.isfinite(record["alignment_loss"]), f"Non-finite alignment_loss at step {record['step']}"
        if "eval_sinkhorn_has_nan_or_inf" in record:
            assert record["eval_sinkhorn_has_nan_or_inf"] is False, (
                f"NaN/Inf detected in Sinkhorn plan P at step {record['step']}"
            )

    # Checkpoint round-trip (non-strict load, metadata included -- spec sec 5 point 4 / sec 2).
    h_v, h_t, step, ckpt = load_alignment_checkpoint(args.checkpoint_path, d_in=768, device="cpu")
    assert step == args.steps
    assert ckpt["align_dim"] == args.align_dim
    dummy = torch.randn(4, 768)
    with torch.no_grad():
        out_v = h_v(dummy)
        out_t = h_t(dummy)
    assert out_v.shape == (4, args.align_dim)
    assert out_t.shape == (4, args.align_dim)
