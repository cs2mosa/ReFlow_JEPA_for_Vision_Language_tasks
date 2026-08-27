"""
PHASE 1g-cont — Core generative mechanism validated against a CONTINUOUS (non-atomic)
target, with NO trained network involved.

test_07_stochastic_source_theorem.py validates Theorem 2 for the two-Dirac-atom case,
which is what VQA's finite answer vocabulary reduces to, and which is exactly what
Reflow-JEPA v3's closed-form kappa_gamma (semi-discrete OT / Laguerre-cell) assumes.

That closed form does NOT directly apply to general-VL pretraining, where P_T(.|c) is
the (effectively continuous) image of the text encoder over an open caption corpus, not
a finite set of points. But re-reading reflow_jepa_proof1.pdf's Theorem 2 (Marginal
Preservation) carefully: the proof itself never assumes atomicity -- it only uses
Assumptions 1-3 (integrability, realizability, well-posedness) and uniqueness of the
continuity-equation solution. Atomicity is invoked only in a closing remark, to note
that the general theorem specializes to "z_hat_t in M_T a.s." when M_T is closed and
P_T happens to be atomic. So the theorem should hold for a continuous target too --
this file is the direct empirical check of that claim, independent of atomicity,
mirroring test_07's own philosophy: construct the *exact* population-optimal field by
hand for a solvable non-atomic case, and check sampling it reproduces the correct
target distribution rather than a compromise point.

Solvable case chosen: P_T(.|c) = (1-lambda) N(a, sigma_t^2 I) + lambda N(b, sigma_t^2 I),
i.e. a two-component Gaussian mixture (NOT two Diracs -- sigma_t > 0 throughout, so the
target is genuinely continuous / non-atomic, unlike test_07's construction). With
Z_0 ~ N(z_v_tilde, sigma_0^2 I) drawn independently of the target component, and
Z_tau = (1-tau)Z_0 + tau*Z_1, the population-optimal CFM field
u(z,tau) = E[Z_1 - Z_0 | Z_tau=z] has an exact closed form (standard Gaussian-mixture
posterior + jointly-Gaussian conditional-expectation algebra; derivation in the
docstring of `exact_conditional_field` below). This lets us Euler-integrate the *exact*
field (no network, no training) and check the terminal distribution -- exactly test_07's
methodology, extended to sigma_t > 0.
"""
import math
import torch


def exact_conditional_field(z: torch.Tensor, tau: float, mu0: torch.Tensor,
                             sigma0: float, mu_a: torch.Tensor, mu_b: torch.Tensor,
                             sigma_t: float, lam: float) -> torch.Tensor:
    """u(z, tau) = E[Z_1 - Z_0 | Z_tau = z], exact closed form.

    Derivation: conditional on target component k in {a, b}, Z_0 and Z_1 are
    independent Gaussians, so (Z_0, Z_1, Z_tau) are jointly Gaussian given k. Standard
    conditional-Gaussian algebra gives, given component k and Z_tau=z:
        E[Z_0 | Z_tau=z, k] = mu0 + (1-tau)*sigma0^2/v_tau * (z - mu_tau_k)
        E[Z_1 | Z_tau=z, k] = mu_k + tau*sigma_t^2/v_tau * (z - mu_tau_k)
    where mu_tau_k = (1-tau)*mu0 + tau*mu_k and v_tau = (1-tau)^2*sigma0^2 + tau^2*sigma_t^2
    (the variance of Z_tau given component k, isotropic in every dimension).
    The posterior weight w_k(z) is a softmax over the two components' Gaussian
    densities at z (same v_tau for both, so it reduces to squared-distance comparison).
    u(z,tau) = sum_k w_k(z) * [ (mu_k - mu0) + (tau*sigma_t^2 - (1-tau)*sigma0^2)/v_tau * (z - mu_tau_k) ]
    """
    v_tau = (1 - tau) ** 2 * sigma0 ** 2 + tau ** 2 * sigma_t ** 2
    mu_tau_a = (1 - tau) * mu0 + tau * mu_a
    mu_tau_b = (1 - tau) * mu0 + tau * mu_b

    # posterior weights (log-space, same v_tau for both components -> squared-distance softmax)
    log_pi = torch.log(torch.tensor([1 - lam, lam]).clamp(min=1e-12))
    d2_a = (z - mu_tau_a).pow(2).sum(dim=-1) / (2 * v_tau)
    d2_b = (z - mu_tau_b).pow(2).sum(dim=-1) / (2 * v_tau)
    logits = torch.stack([log_pi[0] - d2_a, log_pi[1] - d2_b], dim=-1)  # (N, 2)
    w = torch.softmax(logits, dim=-1)  # (N, 2)

    coef = (tau * sigma_t ** 2 - (1 - tau) * sigma0 ** 2) / v_tau
    field_a = (mu_a - mu0) + coef * (z - mu_tau_a)
    field_b = (mu_b - mu0) + coef * (z - mu_tau_b)
    return w[:, 0:1] * field_a + w[:, 1:2] * field_b


def test_continuous_mixture_target_recovers_correct_proportions_and_spread():
    """Euler-integrates the EXACT population-optimal field (no network) from tau=0 to
    tau=1-delta and checks: (a) the empirical mixture proportion matches lambda, as in
    test_07's atomic case, and (b) -- the genuinely new check for the continuous case --
    the terminal samples have non-trivial spread around each mode (std comparable to
    sigma_t), NOT the exact-atom collapse test_07 checks for. Landing on an exact point
    would actually be WRONG here: the target is continuous, so recovering it correctly
    means recovering its spread, not collapsing to it.
    """
    torch.manual_seed(0)
    d = 8
    mu0 = 0.3 * torch.randn(d)
    mu_a = torch.randn(d)
    mu_b = mu_a + 4.0 * torch.randn(d) / torch.randn(d).norm()  # well-separated mode
    sigma0, sigma_t, lam = 0.6, 0.15, 0.35
    delta = 1e-3
    n_steps = 300
    n_samples = 4000

    z = mu0.unsqueeze(0) + sigma0 * torch.randn(n_samples, d)  # Z_0 samples
    taus = torch.linspace(0, 1 - delta, n_steps + 1)
    dtau = taus[1] - taus[0]
    for i in range(n_steps):
        tau = taus[i].item()
        v = exact_conditional_field(z, tau, mu0, sigma0, mu_a, mu_b, sigma_t, lam)
        z = z + v * dtau

    dist_a = (z - mu_a).norm(dim=1)
    dist_b = (z - mu_b).norm(dim=1)
    assigned_b = (dist_b < dist_a)
    empirical_lambda = assigned_b.float().mean().item()

    se = math.sqrt(lam * (1 - lam) / n_samples)
    tolerance = 8 * se  # generous, mirrors test_07's 6-sigma choice plus extra slack for
                         # the finite-step Euler discretization this case needs (unlike
                         # the atomic case's exact affine field, this field is genuinely
                         # tau-dependent in a nonlinear way through the softmax weights)
    print(f"\n[INFO] continuous-mixture target: lambda={lam}, empirical={empirical_lambda:.4f}, "
          f"tolerance=+/-{tolerance:.4f}")
    assert abs(empirical_lambda - lam) < tolerance, (
        f"Empirical mixture proportion ({empirical_lambda:.4f}) does not match target "
        f"lambda={lam} within {tolerance:.4f} for a CONTINUOUS (non-atomic) target -- "
        f"Theorem 2's marginal-preservation claim may not be surviving the transition "
        f"from the atomic (VQA) case to the continuous (general-VL) case as expected."
    )

    # the genuinely new check: samples must show real spread around their assigned mode,
    # not collapse to a point -- collapsing would mean we've accidentally reproduced the
    # ATOMIC-case behavior on a target that was supposed to be continuous.
    std_near_a = z[~assigned_b].std(dim=0).mean().item()
    std_near_b = z[assigned_b].std(dim=0).mean().item()
    print(f"[INFO] per-component empirical std: near-a={std_near_a:.4f}, near-b={std_near_b:.4f} "
          f"(target sigma_t={sigma_t})")
    assert std_near_a > 0.3 * sigma_t and std_near_b > 0.3 * sigma_t, (
        f"Terminal samples collapsed to (near-)zero spread around their assigned mode "
        f"(std_a={std_near_a:.4f}, std_b={std_near_b:.4f}, target sigma_t={sigma_t}) -- "
        f"this would mean the continuous target got treated as if it were atomic, which "
        f"is the wrong behavior for the general-VL case."
    )

    # still must not collapse onto the compromise point, same failure mode as the atomic case.
    # radius scaled to sigma_t (the target's own spread), not to mode separation -- a radius
    # anywhere near half the mode separation would flag correctly-clustered samples too,
    # since the compromise point sits at distance ~separation/2 from each mode.
    compromise = (1 - lam) * mu_a + lam * mu_b
    dist_to_compromise = (z - compromise).norm(dim=1)
    frac_near_compromise = (dist_to_compromise < 3 * sigma_t).float().mean().item()
    assert frac_near_compromise < 0.05, (
        f"{frac_near_compromise:.1%} of samples landed near the off-manifold compromise "
        f"point (1-lambda)a+lambda*b -- the stochastic-source fix should prevent this "
        f"regardless of whether the target is atomic or continuous."
    )
