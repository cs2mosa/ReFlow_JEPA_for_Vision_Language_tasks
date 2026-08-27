"""
PHASE 1e — Anti-collapse regularizer, tested against a deliberately collapsed input.

Reflow-JEPA v3 Lemma 3 proves the bare CFM loss admits a total-collapse global minimizer
(g_V, g_T both constant); Proposition 5 proposes a VICReg-style batch-variance penalty as
the fix. This test does not re-derive that proof -- it verifies the *implementation* of
the penalty actually does what the proof assumes: near-zero for healthy, spread-out
batches, and bounded strictly away from zero for a collapsed batch.

If this test fails, the guard is implemented wrong and Lemma 3's collapse mode is *not*
actually blocked in code, regardless of what the math says on paper.
"""
import torch


def vicreg_variance_penalty(z: torch.Tensor, gamma_0: float = 1.0) -> torch.Tensor:
    """L_var(z) = (1/d) * sum_j max(0, gamma_0 - std_batch(z_j))^2
    Matches Reflow-JEPA v3 Definition (hinge variance penalty), Proposition 5."""
    std = z.std(dim=0, unbiased=False)  # (d,) per-dimension batch std
    hinge = torch.clamp(gamma_0 - std, min=0.0)
    return (hinge ** 2).mean()


def test_collapsed_batch_incurs_large_penalty():
    d, B = 64, 32
    collapsed = torch.zeros(B, d) + torch.randn(1, d)  # every row identical -> std=0 everywhere
    penalty = vicreg_variance_penalty(collapsed, gamma_0=1.0)
    assert penalty.item() > 0.9, (
        f"Collapsed (constant-per-batch) embeddings should incur penalty ~= gamma_0^2 = 1.0 "
        f"in every dimension; got {penalty.item():.4f}. If this is small, the collapse "
        f"attractor identified in Lemma 3 is NOT actually blocked by this implementation."
    )


def test_healthy_batch_incurs_small_penalty():
    d, B = 64, 32
    healthy = torch.randn(B, d)  # unit-variance-ish per dimension by construction
    penalty = vicreg_variance_penalty(healthy, gamma_0=1.0)
    assert penalty.item() < 0.05, (
        f"A healthy, spread-out batch (std~1 per dim) incurs penalty {penalty.item():.4f}, "
        f"expected near 0. If the penalty is large even for healthy data, gamma_0 is "
        f"miscalibrated relative to the embedding scale and will fight useful training "
        f"signal rather than only blocking collapse."
    )


def test_penalty_is_strictly_ordered_between_collapsed_and_healthy():
    """The core claim Proposition 5 needs: collapsed strictly worse than healthy, so
    gradient descent has a real incentive to leave the collapse basin once this penalty
    is added to the total loss."""
    d, B = 64, 32
    collapsed = torch.zeros(B, d) + torch.randn(1, d)
    healthy = torch.randn(B, d)
    p_collapsed = vicreg_variance_penalty(collapsed, gamma_0=1.0).item()
    p_healthy = vicreg_variance_penalty(healthy, gamma_0=1.0).item()
    assert p_collapsed > p_healthy, (
        f"Collapsed penalty ({p_collapsed:.4f}) is not greater than healthy penalty "
        f"({p_healthy:.4f}) -- the guard provides no gradient signal away from collapse."
    )


def test_partial_collapse_is_detected_dimension_wise():
    """A subtler failure than full collapse: only *some* dimensions collapse (e.g. one
    attention head dies). The penalty must scale with the fraction of dead dimensions,
    not just fire on total collapse."""
    d, B = 64, 32
    half_dead = torch.randn(B, d)
    half_dead[:, : d // 2] = torch.randn(1, d // 2)  # first half of dims collapsed
    penalty_half = vicreg_variance_penalty(half_dead, gamma_0=1.0).item()
    fully_healthy = torch.randn(B, d)
    penalty_healthy = vicreg_variance_penalty(fully_healthy, gamma_0=1.0).item()
    assert penalty_half > penalty_healthy, (
        "Partial (dimension-wise) collapse is not detected more strongly than a fully "
        "healthy batch -- the per-dimension hinge is not actually operating per-dimension."
    )
