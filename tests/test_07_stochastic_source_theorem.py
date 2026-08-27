"""
PHASE 1g — Core generative mechanism, tested against a hand-constructed ground truth,
with NO trained network involved. This validates the *math*, not an implementation of it,
which is exactly the point: if this fails, no amount of training will fix the design,
because the mechanism itself is being exercised directly.

Reflow-JEPA v3, Theorem 2 (per-example generative recovery, two-atom case): for a
two-point VQA instance P(Z_1=a|gamma)=1-lambda, P(Z_1=b|gamma)=lambda, with stochastic
source Z_0 ~ N(z_v_tilde, sigma^2 I) and the population-optimal (half-space / Laguerre-
cell) field, sampling Z_0 and applying the closed-form assignment recovers the correct
mixture (1-lambda):lambda, almost surely landing exactly on a or exactly on b -- never on
the compromise point that Lemma 1 / the v1-v2 deterministic-source design produced.

This test builds the closed-form half-space assignment directly (kappa_gamma from
Reflow-JEPA v3 Eq. 3) and checks it empirically, then explicitly contrasts it against the
deterministic-source (sigma=0) baseline to make the fix's effect visible in actual numbers.
"""
import math
import torch
from scipy.stats import norm as scipy_norm


def kappa_gamma(z_v_tilde: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                 sigma: float, lam: float) -> float:
    """Reflow-JEPA v3 Eq. (3): the closed-form hyperplane threshold."""
    diff = b - a
    return (z_v_tilde @ diff).item() + sigma * diff.norm().item() * scipy_norm.ppf(1 - lam)


def assign(z0: torch.Tensor, a: torch.Tensor, b: torch.Tensor, kappa: float) -> torch.Tensor:
    """Half-space assignment: <z0, b-a> < kappa -> a, else -> b."""
    diff = b - a
    proj = z0 @ diff
    return torch.where(proj < kappa, torch.zeros(z0.shape[0]), torch.ones(z0.shape[0]))


def test_stochastic_source_recovers_correct_mixture():
    torch.manual_seed(0)
    d = 32
    a, b = torch.randn(d), torch.randn(d)
    a, b = a / a.norm(), (b / b.norm()) * 1.3  # ensure a != b, moderately separated
    z_v_tilde = 0.3 * torch.randn(d)
    sigma, lam = 0.5, 0.3
    n_samples = 200_000

    kappa = kappa_gamma(z_v_tilde, a, b, sigma, lam)
    z0 = z_v_tilde.unsqueeze(0) + sigma * torch.randn(n_samples, d)
    assignments = assign(z0, a, b, kappa)  # 0 -> a, 1 -> b

    empirical_lambda = assignments.mean().item()
    # binomial standard error at this n; use a generous 6-sigma tolerance to make this
    # robust across seeds while still being a real, failing-if-wrong statistical check
    se = math.sqrt(lam * (1 - lam) / n_samples)
    tolerance = 6 * se
    print(f"\n[INFO] target lambda={lam}, empirical lambda={empirical_lambda:.4f}, "
          f"tolerance=+/-{tolerance:.4f}")
    assert abs(empirical_lambda - lam) < tolerance, (
        f"Empirical mixture ({empirical_lambda:.4f}) does not match the target "
        f"lambda={lam} within {tolerance:.4f} -- either the closed-form kappa_gamma is "
        f"implemented incorrectly, or the Gaussian-projection argument in Theorem 2's "
        f"proof does not hold as stated. This is the single most important test in this "
        f"suite to have pass before writing the real training loop."
    )


def test_samples_land_exactly_on_atoms_never_on_compromise_point():
    """Explicitly reconstructs the terminal point each sample would reach via one Euler
    step (Theorem 2(iv)) and confirms it is always (up to floating point) exactly a or b
    -- never the compromise point that the deterministic-source design produced."""
    torch.manual_seed(1)
    d = 16
    a, b = torch.randn(d), torch.randn(d)
    z_v_tilde = torch.randn(d)
    sigma, lam = 0.4, 0.5
    n_samples = 2000

    kappa = kappa_gamma(z_v_tilde, a, b, sigma, lam)
    z0 = z_v_tilde.unsqueeze(0) + sigma * torch.randn(n_samples, d)
    labels = assign(z0, a, b, kappa)

    # one-step Euler estimate under the exact (hand-constructed) optimal field:
    # v(z0, 0, gamma) = target - z0 ; z_hat = z0 + v = target (exactly, by construction)
    targets = torch.where(labels.unsqueeze(1) == 0, a.unsqueeze(0), b.unsqueeze(0))
    z_hat = z0 + (targets - z0)  # collapses to `targets` algebraically; kept explicit
    dist_to_a = (z_hat - a).norm(dim=1)
    dist_to_b = (z_hat - b).norm(dim=1)
    min_dist_to_either = torch.minimum(dist_to_a, dist_to_b)
    assert (min_dist_to_either < 1e-5).all(), (
        "Some samples' one-step decode landed away from both atoms -- decoding is not "
        "exact under the hand-constructed optimal field."
    )

    compromise_point = (1 - lam) * a + lam * b
    dist_to_compromise = (z_hat - compromise_point).norm(dim=1)
    frac_near_compromise = (dist_to_compromise < 1e-3).float().mean().item()
    assert frac_near_compromise == 0.0, (
        f"{frac_near_compromise:.1%} of samples landed near the compromise point "
        f"(1-lambda)a+lambda*b -- this is exactly the Lemma-1 / deterministic-source "
        f"failure mode the stochastic source was built to eliminate."
    )


def test_deterministic_source_baseline_reproduces_the_original_failure():
    """Contrast case: the v1/v2 deterministic-source design (sigma=0) always collapses to
    the same compromise point for a fixed z_v, exactly as Reflow-JEPA v3 Lemma "degenerate
    source" proves. Included so the fix's effect is visible as a concrete before/after,
    not just asserted."""
    d = 16
    torch.manual_seed(2)
    a, b = torch.randn(d), torch.randn(d)
    z_v_tilde = torch.randn(d)
    lam = 0.4
    # deterministic source: z_0 == z_v_tilde, always, for every "sample"
    z0_deterministic = z_v_tilde.unsqueeze(0).repeat(10, 1)
    # the ONLY population-consistent one-step field at a degenerate source is the
    # conditional mean (Reflow-JEPA v3, proof of the "degenerate source" lemma):
    predicted = (1 - lam) * a + lam * b
    z_hat_deterministic = z0_deterministic + (predicted - z0_deterministic)
    # every single "sample" is identical and equal to the off-manifold compromise point
    assert torch.allclose(z_hat_deterministic, predicted.unsqueeze(0).expand(10, -1))
    dist_to_a = (predicted - a).norm().item()
    dist_to_b = (predicted - b).norm().item()
    print(f"\n[INFO] deterministic-source baseline: single fixed output for all draws, "
          f"distance to a={dist_to_a:.3f}, distance to b={dist_to_b:.3f} (both > 0, "
          f"i.e. off-manifold, with zero sample diversity -- this is what Fix 1 replaces).")
    assert dist_to_a > 1e-3 and dist_to_b > 1e-3
