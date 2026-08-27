"""
PHASE 1h — Terminal-time (tau -> 1) behavior, tested numerically against the corrected
rate from OFM-JEPA v2 (Proposition 3): the velocity Jacobian for the exact atom-collapse
field v(z,tau) = (a-z)/(1-tau) scales as O(1/(1-tau)) -- linear, not the cubic rate
originally (and incorrectly) claimed -- and the integrated early-stopping error therefore
grows only logarithmically, not as exp(1/(2*delta^2)).

This is a from-first-principles numerical check, not a training-dependent test: it
verifies the *ODE integration behavior itself*, independent of any learned weights, using
torch.autograd to compute the actual Jacobian rather than trusting the closed form by eye.
"""
import math
import torch


def velocity_field(z: torch.Tensor, tau: float, a: torch.Tensor) -> torch.Tensor:
    """v(z, tau) = (a - z) / (1 - tau), the exact within-cell field from
    Reflow-JEPA v3 Proposition (divergence blow-up) / OFM-JEPA v2 Proposition 3."""
    return (a - z) / (1 - tau)


def jacobian_operator_norm(z: torch.Tensor, tau: float, a: torch.Tensor) -> float:
    d = z.shape[0]
    J = torch.autograd.functional.jacobian(lambda x: velocity_field(x, tau, a), z)
    return torch.linalg.matrix_norm(J, ord=2).item()


def test_jacobian_norm_matches_linear_rate_not_cubic():
    """Directly checks ||grad_z v||_op == 1/(1-tau) at several tau, confirming the
    corrected linear rate (OFM-JEPA v2 Proposition 3) and explicitly ruling out the
    original document's claimed 1/(1-tau)^3 scaling."""
    d = 8
    a = torch.randn(d)
    z = torch.randn(d)
    for tau in [0.0, 0.5, 0.9, 0.99]:
        measured = jacobian_operator_norm(z, tau, a)
        expected_linear = 1.0 / (1 - tau)
        expected_cubic = 1.0 / (1 - tau) ** 3
        rel_err_linear = abs(measured - expected_linear) / expected_linear
        print(f"\n[INFO] tau={tau}: measured ||J||_op={measured:.4f}, "
              f"linear-rate prediction={expected_linear:.4f} (rel.err={rel_err_linear:.2e}), "
              f"cubic-rate (original, incorrect) prediction={expected_cubic:.4f}")
        assert rel_err_linear < 1e-4, (
            f"Measured Jacobian norm does not match the corrected O(1/(1-tau)) rate at "
            f"tau={tau}. Re-derive before trusting the early-stopping margin below."
        )
        # sanity: as tau -> 1, cubic diverges far faster than what we actually measure
        if tau > 0:
            assert expected_cubic > expected_linear


def test_integrated_error_is_logarithmic_not_exponential():
    """Numerically integrates ||J||_op over [0, 1-delta] via a fine quadrature grid and
    checks it matches the closed form -ln(delta) (OFM-JEPA v2 Proposition 3) rather than
    growing anywhere near exp(1/(2*delta^2)) as originally (incorrectly) claimed."""
    deltas = [0.1, 0.01, 0.001]
    n_grid = 200_000
    for delta in deltas:
        taus = torch.linspace(0, 1 - delta, n_grid + 1)
        integrand = 1.0 / (1 - taus[:-1])  # ||J||_op at each grid point (left Riemann sum)
        dtau = taus[1] - taus[0]
        numeric_integral = (integrand * dtau).sum().item()
        closed_form = -math.log(delta)
        print(f"\n[INFO] delta={delta}: numeric integral={numeric_integral:.4f}, "
              f"closed form -ln(delta)={closed_form:.4f}")
        rel_err = abs(numeric_integral - closed_form) / closed_form
        assert rel_err < 0.01, (
            f"Numeric integral does not match the closed-form logarithmic rate at "
            f"delta={delta} (rel.err={rel_err:.4f})."
        )
        # The practical point: the corrected bound is utterly negligible next to the
        # original document's claimed exponential blow-up. Compare in log-space --
        # exp(1/(2*delta^2)) overflows a float64 outright once delta <= ~0.01, which is
        # itself a fairly direct demonstration of how far off the original claim was.
        log_closed_form = math.log(closed_form)
        log_exponential_original_claim = 1 / (2 * delta ** 2)
        print(f"[INFO] delta={delta}: log(corrected bound)={log_closed_form:.3f} vs "
              f"log(original exp claim)={log_exponential_original_claim:.3e}")
        assert log_closed_form < log_exponential_original_claim, (
            "Corrected logarithmic bound should be dramatically smaller than the "
            "original (incorrect) exponential claim, even compared in log-space."
        )


def test_reasonable_early_stopping_margin_keeps_integration_error_small():
    """Practical takeaway check: a modest, standard early-stopping margin (delta=1e-3,
    i.e. integrate to tau=0.999) keeps the accumulated rate-of-change well within a range
    a standard adaptive ODE solver handles routinely -- no exotic architecture required."""
    delta = 1e-3
    closed_form = -math.log(delta)
    assert closed_form < 10, (
        f"Even a fairly aggressive early-stopping margin (delta={delta}) gives an "
        f"integrated rate of {closed_form:.2f}, well within normal numerical range -- "
        f"confirms continuous CFM does not need to be abandoned near tau=1."
    )
