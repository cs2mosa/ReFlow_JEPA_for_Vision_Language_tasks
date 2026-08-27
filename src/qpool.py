"""
Q-Pool: K learned query slots, FiLM-modulated by the conditioning vector c,
cross-attending over frozen visual patch tokens H_v. Matches Reflow-JEPA v3 §5.2.

Canonical source -- tests/test_04_qpool_module.py imports this rather than redefining
it, so the tested module and the trained module are identical.
"""
import torch
import torch.nn as nn


class QPool(nn.Module):
    def __init__(self, d_v: int, d_text: int, d_shared: int, k: int = 8, n_heads: int = 8):
        super().__init__()
        self.k = k
        self.query_slots = nn.Parameter(torch.randn(k, d_v) * 0.02)
        self.film = nn.Linear(d_text, 2 * d_v)  # produces (gamma, beta) for FiLM
        self.cross_attn = nn.MultiheadAttention(d_v, n_heads, batch_first=True)
        self.g_v = nn.Sequential(
            nn.Linear(k * d_v, d_shared), nn.LayerNorm(d_shared), nn.GELU(),
            nn.Linear(d_shared, d_shared),
        )

    def forward(self, h_v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h_v: (B, P, d_v)   c: (B, d_text)
        B = h_v.shape[0]
        gamma, beta = self.film(c).chunk(2, dim=-1)          # (B, d_v) each
        q = self.query_slots.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_v)
        q = q * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # FiLM-modulate the queries
        pooled, _ = self.cross_attn(q, h_v, h_v)              # (B, K, d_v)
        return self.g_v(pooled.reshape(B, -1))                # (B, d_shared)
