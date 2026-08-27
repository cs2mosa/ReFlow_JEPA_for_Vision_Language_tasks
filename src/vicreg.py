"""
VICReg-style batch-variance penalty (Bardes, Ponce & LeCun, ICLR 2022), matching
Reflow-JEPA v3's Proposition 5 hinge-variance definition. Canonical source --
tests/test_05_vicreg_collapse_guard.py imports this rather than redefining it.
"""
import torch


def vicreg_variance_penalty(z: torch.Tensor, gamma_0: float = 1.0) -> torch.Tensor:
    """L_var(z) = (1/d) * sum_j max(0, gamma_0 - std_batch(z_j))^2"""
    std = z.std(dim=0, unbiased=False)  # (d,) per-dimension batch std
    hinge = torch.clamp(gamma_0 - std, min=0.0)
    return (hinge ** 2).mean()
