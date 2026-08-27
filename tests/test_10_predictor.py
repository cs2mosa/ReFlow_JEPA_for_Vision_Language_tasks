"""
PHASE 1j — Velocity predictor v_theta, tested in isolation before assembly.

New module (src/predictor.py), no isolated test existed for it before this file since
it postdates the original 31-test suite. Same battery as Q-Pool/Prefix-Expand: shape,
gradient flow, sensitivity to each of its conditioning inputs (z_tau, tau, z_v_tilde, c)
individually -- a predictor that ignores any one of these silently breaks a specific
part of the design (ignoring tau breaks the whole time-conditioned ODE; ignoring
z_v_tilde or c breaks the "conditional" in conditional flow matching).
"""
import torch
from encoders import D_SHARED
from predictor import VelocityPredictor


def _tiny_predictor():
    return VelocityPredictor(d_shared=D_SHARED, depth=2, n_heads=4)  # shallow for test speed


def test_output_shape():
    pred = _tiny_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B)
    z_v = torch.randn(B, D_SHARED)
    c = torch.randn(B, D_SHARED)
    out = pred(z_tau, tau, z_v, c)
    assert out.shape == (B, D_SHARED)


def test_zero_init_gives_zero_output():
    """DiT-Zero init (Peebles & Xie 2023): every AdaLN gate and the final head start at
    zero, so the predictor should output exactly zero before any training -- confirms
    the zero-init actually took effect end-to-end through the whole block stack, not
    just at the head."""
    pred = _tiny_predictor()
    pred.eval()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B)
    z_v = torch.randn(B, D_SHARED)
    c = torch.randn(B, D_SHARED)
    with torch.no_grad():
        out = pred(z_tau, tau, z_v, c)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        "Predictor did not output exactly zero at initialization despite zero-initialized "
        "AdaLN gates and head -- some pathway is bypassing the zero-init, which would "
        "mean training starts from an arbitrary, unpredictable velocity field instead of "
        "the stable near-identity start DiT-Zero is meant to provide."
    )


def test_gradients_flow_to_all_parameters_after_warmup():
    """DiT-Zero init means, AT EXACT INITIALIZATION, backprop through the zero-weight
    `head` kills gradient to everything upstream of it -- only head's own weight/bias
    receive a gradient on step 1. Real DiT papers train through exactly this cold start,
    but it's a genuine multi-step CASCADE, not a single unlock: `head` frees up at step
    1, `final_ada_ln` at step 2 (now head's gradient can reach it), and each block's own
    `ada_ln` -- and only then that block's attention/MLP internals, once its own gate
    moves off zero -- unlocks progressively further back. Roughly `depth` warmup steps
    are needed before every parameter has seen a nonzero gradient at least once."""
    pred = _tiny_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B)
    z_v = torch.randn(B, D_SHARED)
    c = torch.randn(B, D_SHARED)
    target = torch.randn(B, D_SHARED)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)

    for _ in range(20):  # comfortably more than `depth` to clear the unlock cascade
        opt.zero_grad()
        out = pred(z_tau, tau, z_v, c)
        (out - target).pow(2).sum().backward()
        opt.step()

    opt.zero_grad()
    out = pred(z_tau, tau, z_v, c)
    (out - target).pow(2).sum().backward()
    dead_params = [name for name, p in pred.named_parameters()
                   if p.grad is None or p.grad.norm().item() < 1e-10]
    assert not dead_params, (
        f"No gradient reached even after 20 warmup steps (well beyond the expected "
        f"unlock cascade for a {len(pred.blocks)}-block predictor): {dead_params}"
    )


def test_gradient_is_dead_upstream_of_head_at_exact_init():
    """Documents the DiT-Zero cold-start explicitly, rather than leaving it as an
    unexplained exception to the test above: at exact initialization, gradient reaches
    `head` but nothing upstream of it, because backprop through a zero-weight linear
    layer is exactly zero. If this ever stops being true, the zero-init scheme changed
    and the warmup-dependent test above needs re-examining."""
    pred = _tiny_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B)
    z_v = torch.randn(B, D_SHARED)
    c = torch.randn(B, D_SHARED)
    out = pred(z_tau, tau, z_v, c)
    (out - torch.randn(B, D_SHARED)).pow(2).sum().backward()
    assert pred.head.weight.grad.norm().item() > 0
    assert pred.blocks[0].mlp[0].weight.grad.norm().item() < 1e-10, (
        "Expected zero gradient upstream of `head` at exact initialization (backprop "
        "through a zero-weight head is exactly zero) -- got nonzero, meaning the "
        "zero-init scheme changed and this test's assumption is stale."
    )


def test_output_depends_on_each_conditioning_input():
    """A predictor that ignores tau, z_v_tilde, or c would silently break a specific,
    identifiable part of the design (see module docstring). Train briefly first so the
    zero-initialized network has nonzero weights to actually be sensitive with."""
    torch.manual_seed(0)
    pred = _tiny_predictor()
    B = 8
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B)
    z_v = torch.randn(B, D_SHARED)
    c = torch.randn(B, D_SHARED)
    target = torch.randn(B, D_SHARED)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        out = pred(z_tau, tau, z_v, c)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
    pred.eval()

    with torch.no_grad():
        base = pred(z_tau, tau, z_v, c)
        out_diff_tau = pred(z_tau, torch.rand(B), z_v, c)
        out_diff_zv = pred(z_tau, tau, torch.randn(B, D_SHARED), c)
        out_diff_c = pred(z_tau, tau, z_v, torch.randn(B, D_SHARED))

    assert not torch.allclose(base, out_diff_tau, atol=1e-4), "Predictor is insensitive to tau."
    assert not torch.allclose(base, out_diff_zv, atol=1e-4), "Predictor is insensitive to z_v_tilde."
    assert not torch.allclose(base, out_diff_c, atol=1e-4), "Predictor is insensitive to c."
