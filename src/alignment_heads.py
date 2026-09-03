"""
alignment_heads.py -- Phase A: supervised Sinkhorn cross-modal alignment.

Implements the components specified in `alignment_stage_spec.md` section 4:
  - AlignmentHead              (sec 4.1)
  - sinkhorn_log_domain        (sec 4.2, log-domain per project convention -- see
                                 the Section 2 "never trust an unstabilized division
                                 or exponential near a boundary" lesson)
  - alignment_loss             (sec 4.2, bounded linear diagonal-mass form)

Per spec sec 4.3 ("reuse the existing, already-tested vicreg_variance_penalty ...
do not reimplement"), the VICReg term is imported directly from the project's own
`vicreg.py` (canonical source, per its own docstring: "tests/test_05_vicreg_
collapse_guard.py imports this rather than redefining it") -- same function, same
`gamma_0` semantics, same call signature already used for `vicreg_v`/`vicreg_t`
in `reflow_jepa.py`'s `training_step`. Nothing here reimplements it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicreg import vicreg_variance_penalty  # noqa: F401  (re-exported for callers of this module)


# ---------------------------------------------------------------------------
# 4.1 Alignment heads
# ---------------------------------------------------------------------------
class AlignmentHead(nn.Module):
    """Projects a frozen, detached shared-space embedding into a dedicated,
    more compressed alignment subspace. Structurally identical to
    `TextProjectionHead` (text_projection.py) for consistency with the
    existing codebase style, per spec sec 4.1.
    """

    def __init__(self, d_in: int = 768, d_align: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LayerNorm(d_in),
            nn.GELU(),
            nn.Linear(d_in, d_align),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# ---------------------------------------------------------------------------
# 4.2 Supervised Sinkhorn alignment loss
# ---------------------------------------------------------------------------
def sinkhorn_log_domain(C: torch.Tensor, epsilon: float, n_iters: int = 50) -> torch.Tensor:
    """C: (B, B) cost matrix. Returns the (B, B) doubly-stochastic-up-to-1/B transport
    plan P, with row sums and column sums both equal to 1/B. Log-domain throughout --
    NEVER exponentiate -C/epsilon directly into raw probability space; that overflows/
    underflows exactly the way this project's EDM 1/(1-tau) amplification did (see
    Section 2). All operations here stay in log-space via torch.logsumexp until the
    final P is constructed.
    """
    B = C.shape[0]
    log_a = -math.log(B)  # scalar, log(1/B), same for every row
    log_b = -math.log(B)
    f = torch.zeros(B, device=C.device, dtype=C.dtype)  # dual potential, "log(u)" role
    g = torch.zeros(B, device=C.device, dtype=C.dtype)  # dual potential, "log(v)" role
    for _ in range(n_iters):
        f = epsilon * log_a - epsilon * torch.logsumexp((-C + g.unsqueeze(0)) / epsilon, dim=1)
        g = epsilon * log_b - epsilon * torch.logsumexp((-C + f.unsqueeze(1)) / epsilon, dim=0)
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
    return torch.exp(log_P)


def sinkhorn_marginal_error(P: torch.Tensor) -> tuple[float, float]:
    """Convergence-quality check (spec sec 4.2 / sec 7 test 1): max absolute
    deviation of row sums and column sums from 1/B. Not part of the spec's
    given code block, but needed to implement the "log a warning if row/col
    sums deviate from 1/B by more than 1%" requirement -- factored out here
    so both train_alignment.py and the tests can reuse it identically.
    """
    B = P.shape[0]
    target = 1.0 / B
    row_err = (P.sum(dim=1) - target).abs().max().item()
    col_err = (P.sum(dim=0) - target).abs().max().item()
    return row_err, col_err


def alignment_loss(
    A_v: torch.Tensor,
    A_t: torch.Tensor,
    epsilon: float,
    n_iters: int,
    *,
    warn_on_convergence: bool = True,
) -> torch.Tensor:
    """Supervised Sinkhorn alignment loss (spec sec 4.2, step 3).

    `0` means perfect diagonal-concentrated coupling (every image's transport
    mass goes entirely to its own caption). Bounded in [0, ~1] by construction.
    """
    C = 1 - A_v @ A_t.T
    P = sinkhorn_log_domain(C, epsilon, n_iters)
    B = A_v.shape[0]

    if warn_on_convergence:
        row_err, col_err = sinkhorn_marginal_error(P)
        tol = 0.01 / B  # "1%" of the 1/B target mass
        if row_err > tol or col_err > tol:
            import warnings

            warnings.warn(
                f"sinkhorn_log_domain: marginals not converged after {n_iters} "
                f"iters at epsilon={epsilon} (row_err={row_err:.2e}, "
                f"col_err={col_err:.2e}, tol={tol:.2e}). Consider more iters or "
                f"a larger epsilon.",
                stacklevel=2,
            )

    diag_mass = torch.diagonal(P).mean()  # each P[i,i] <= 1/B (row sum constraint),
    # so B * diag_mass <= 1, with equality iff P is exactly the diagonal permutation
    return 1.0 - B * diag_mass
