"""
Stochastic source: Z_0 = z_v_tilde + sigma*eps, eps ~ N(0, I). Reflow-JEPA v3 §5's fix
for the deterministic-source failure mode (Lemma 1 / DESIGN.md rationale item 5).
Validated theoretically (no network) in tests/test_07 (atomic) and test_07b (continuous).
"""
import torch


def draw_stochastic_source(z_v_tilde: torch.Tensor, sigma: float) -> torch.Tensor:
    eps = torch.randn_like(z_v_tilde)
    return z_v_tilde + sigma * eps
