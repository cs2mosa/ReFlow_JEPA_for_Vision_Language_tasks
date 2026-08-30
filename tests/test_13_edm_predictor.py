"""
PHASE 1l -- EDM-preconditioned predictor mode (predictor.py's edm_precondition=True),
added directly in response to measure_terminal_divergence.py's finding: the raw-velocity
architecture's terminal divergence, measured on a real trained checkpoint, did not
reliably track the theoretical 1/(1-tau) rate near tau=1, regardless of Reflow or more
integration steps. This reparametrizes the network to predict a bounded target-embedding
estimate instead of an unbounded velocity, computing velocity analytically via
(z1_hat - z_tau)/(1-tau) -- baking the divergence into the architecture rather than
requiring gradient descent to discover it.

Includes one test that documents a CORRECTED hypothesis: I initially expected this
reparametrization might also incidentally fix the DiT-Zero dead-gradient-cascade found
in test_10 (since the output is no longer identically zero at init). Direct testing
showed this is false -- test_gradient_cascade_still_requires_warmup below confirms the
same cascade persists, for a specific, checkable reason (see its docstring). Recording
the wrong hypothesis and the check that corrected it, rather than quietly fixing the
claim, since that's the more honest record of what was actually verified.
"""
import torch
import torch.nn.functional as F
from encoders import D_SHARED
from predictor import VelocityPredictor


def _tiny_edm_predictor():
    return VelocityPredictor(d_shared=D_SHARED, depth=2, n_heads=4, edm_precondition=True)


def test_output_at_init_matches_closed_form():
    """At exact init (head zero-initialized, so raw=0), z1_hat = normalize(z_v_tilde +
    0) = z_v_tilde exactly (already unit norm). So v_pred should exactly equal
    (z_v_tilde - z_tau) / (1 - tau) -- a well-defined, non-degenerate, sensible
    starting field, not the zero vector (which normalize() cannot handle sensibly)."""
    torch.manual_seed(0)
    pred = _tiny_edm_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B) * 0.9  # keep away from the 1-tau clamp boundary for this check
    zv = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    c = torch.randn(B, D_SHARED)

    with torch.no_grad():
        out = pred(z_tau, tau, zv, c)
        expected = (zv - z_tau) / (1 - tau).clamp(min=1e-4).unsqueeze(-1)

    assert torch.allclose(out, expected, atol=1e-4), (
        "EDM-preconditioned predictor's output at exact init does not match the "
        "expected closed form (z_v_tilde - z_tau)/(1-tau) -- the zero-init "
        "assumption (raw=0 at init) may not hold, or the reparametrization formula "
        "has a bug."
    )


def test_structural_divergence_holds_at_init_unconditionally():
    """The key new guarantee this reparametrization is FOR: unlike the raw-velocity
    architecture (where the 1/(1-tau) divergence had to be learned via gradient
    descent, and measure_terminal_divergence.py found a real trained network hadn't
    reliably learned it), here the divergence is architectural -- present even at
    exact initialization, before a single gradient step, for ANY z_tau/z_v_tilde as
    long as z1_hat != z_tau."""
    torch.manual_seed(0)
    pred = _tiny_edm_predictor()
    B = 1
    z_tau = torch.zeros(B, D_SHARED)  # deliberately far from z_v_tilde to guarantee z1_hat != z_tau
    zv = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    c = torch.randn(B, D_SHARED)

    taus = [0.0, 0.5, 0.9, 0.99, 0.999]
    with torch.no_grad():
        norms = [pred(z_tau, torch.tensor([t]), zv, c).norm(dim=-1).item() for t in taus]

    print(f"\n[INFO] output norm at init across tau={taus}: {norms}")
    for i in range(len(taus) - 1):
        assert norms[i + 1] > norms[i], (
            f"Expected monotonically increasing divergence as tau->1 even at exact "
            f"init, but norm at tau={taus[i+1]} ({norms[i+1]:.2f}) is not greater "
            f"than at tau={taus[i]} ({norms[i]:.2f})."
        )
    # roughly check the SCALING matches 1/(1-tau), not just "increasing"
    ratio_theoretical = [(1 - taus[0]) / (1 - t) for t in taus]
    ratio_measured = [n / norms[0] for n in norms]
    print(f"[INFO] measured norm ratio vs tau=0: {ratio_measured}")
    print(f"[INFO] theoretical 1/(1-tau) ratio vs tau=0: {ratio_theoretical}")
    for measured, theoretical in zip(ratio_measured[1:], ratio_theoretical[1:]):
        rel_err = abs(measured - theoretical) / theoretical
        assert rel_err < 0.05, (
            f"Measured divergence ratio ({measured:.2f}) does not match the "
            f"theoretical 1/(1-tau) ratio ({theoretical:.2f}) closely enough "
            f"(rel.err={rel_err:.3f}) -- should be near-exact since this holds by "
            f"construction, independent of training."
        )


def test_output_is_finite_near_boundary():
    """Confirms the 1e-4 clamp on (1-tau) actually prevents inf/NaN at tau values an
    integrator might realistically reach (e.g. tau=1-1e-3, the default early-stopping
    margin used throughout this project's integrate() calls)."""
    torch.manual_seed(0)
    pred = _tiny_edm_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.full((B,), 1 - 1e-3)
    zv = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    c = torch.randn(B, D_SHARED)

    out = pred(z_tau, tau, zv, c)
    assert torch.isfinite(out).all(), "Output contains non-finite values near tau=1."


def test_gradient_cascade_still_requires_warmup():
    """CORRECTS AN INITIAL HYPOTHESIS: I expected this reparametrization might also
    fix the DiT-Zero dead-gradient-cascade (test_10) since the output is no longer
    identically zero at init. Direct testing showed this is false: even though
    d(v_pred)/d(raw) is nonzero at init (via normalize's nonzero Jacobian at a nonzero
    point), raw = head(h) is still a LINEAR function of h with head.weight=0 at exact
    init, so d(raw)/d(h) = head.weight = 0 regardless of what happens to raw
    afterward. The same warmup cascade from test_10 persists unchanged; the fix here
    is about the terminal-divergence STRUCTURE, not about training-dynamics warmup."""
    torch.manual_seed(0)
    pred = _tiny_edm_predictor()
    B = 4
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B) * 0.9
    zv = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    c = torch.randn(B, D_SHARED)

    out = pred(z_tau, tau, zv, c)
    (out - torch.randn(B, D_SHARED)).pow(2).sum().backward()
    dead_at_init = [name for name, p in pred.named_parameters()
                    if p.grad is None or p.grad.norm().item() < 1e-10]
    print(f"\n[INFO] {len(dead_at_init)}/{len(list(pred.named_parameters()))} params "
          f"dead at exact init (matches test_10's raw-velocity finding -- same cascade)")
    assert len(dead_at_init) > 20, (
        "Expected the same DiT-Zero cascade as test_10 (most params dead at exact "
        "init) -- if this now passes with few/no dead params, the reparametrization "
        "unexpectedly changed the cascade behavior and this test's documentation "
        "above needs re-examining, not just its assertion threshold."
    )

    # after enough warmup steps (mirroring test_10's methodology exactly), the cascade
    # should clear the same way it did for the raw-velocity architecture
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    target = torch.randn(B, D_SHARED)
    for _ in range(20):
        opt.zero_grad()
        out = pred(z_tau, tau, zv, c)
        (out - target).pow(2).sum().backward()
        opt.step()
    opt.zero_grad()
    out = pred(z_tau, tau, zv, c)
    (out - target).pow(2).sum().backward()
    dead_after_warmup = [name for name, p in pred.named_parameters()
                         if p.grad is None or p.grad.norm().item() < 1e-10]
    assert not dead_after_warmup, f"Still dead after warmup: {dead_after_warmup}"


def test_trainable_away_from_init_prediction():
    """Confirms the network can actually learn to move z1_hat away from z_v_tilde
    (the init-time default guess) once trained -- i.e. the conditioning pathway and
    head are genuinely load-bearing, not just producing the closed-form init value
    forever."""
    torch.manual_seed(0)
    pred = _tiny_edm_predictor()
    B = 8
    z_tau = torch.randn(B, D_SHARED)
    tau = torch.rand(B) * 0.9
    zv = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    c = torch.randn(B, D_SHARED)
    target = F.normalize(torch.randn(B, D_SHARED), dim=-1)  # a target DIFFERENT from zv

    opt = torch.optim.Adam(pred.parameters(), lr=5e-3)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        out = pred(z_tau, tau, zv, c)
        loss = (out - (target - z_tau) / (1 - tau).clamp(min=1e-4).unsqueeze(-1)).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"\n[INFO] loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.5, (
        f"Expected the network to learn to move its target-estimate away from the "
        f"init-time default (z_v_tilde) toward a genuinely different target; loss "
        f"did not improve meaningfully ({losses[0]:.4f} -> {losses[-1]:.4f})."
    )


def test_edm_recon_loss_stays_bounded_near_tau_one():
    """Direct regression test for a confirmed real bug, not a hypothetical: an early
    implementation of edm_precondition computed cfm_loss directly on the (amplified)
    v_pred, which produced a real cfm_loss spike above 500,000 during smoke testing,
    the first time a sampled training tau landed very close to 1 while the network's
    target-estimate was still imperfect (normal, especially early in training). The
    fix (reflow_jepa.py's training_step) recovers z1_hat algebraically from v_pred and
    supervises THAT instead. This test verifies why the fix is actually safe: since
    z1_hat is forced onto the unit sphere (F.normalize) and Z1 is also unit-norm by
    this project's own L2-norm design, ||z1_hat - Z1||^2 is PROVABLY bounded by ~4
    regardless of tau -- it can never explode the way the raw v_pred-based loss did."""
    torch.manual_seed(0)
    pred = VelocityPredictor(d_shared=D_SHARED, depth=2, n_heads=4, edm_precondition=True)
    B = 8
    Z0 = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    Z1 = F.normalize(torch.randn(B, D_SHARED), dim=-1)
    zv = Z0
    c = torch.randn(B, D_SHARED)

    tau = torch.full((B,), 1 - 1e-6)  # adversarially close to 1
    Z_tau = (1 - tau).unsqueeze(-1) * Z0 + tau.unsqueeze(-1) * Z1

    v_pred = pred(Z_tau, tau, zv, c)
    old_unsafe_loss = (v_pred - (Z1 - Z0)).pow(2).sum(dim=-1).mean()  # the ORIGINAL buggy form
    z1_hat = v_pred * (1 - tau).unsqueeze(-1) + Z_tau
    new_safe_loss = (z1_hat - Z1).pow(2).sum(dim=-1).mean()           # the FIXED form

    print(f"\n[INFO] at tau=1-1e-6: old (unsafe) loss={old_unsafe_loss.item():.2f}, "
          f"new (safe) loss={new_safe_loss.item():.4f}")
    assert new_safe_loss.item() < 5.0, (
        f"Safe loss form should be bounded by ~4 (both z1_hat and Z1 are unit-norm) "
        f"regardless of tau, got {new_safe_loss.item():.4f} -- if this fails, the "
        f"fix's boundedness guarantee doesn't actually hold and needs re-examining."
    )
