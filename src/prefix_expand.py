"""
Prefix-Expand: the module that makes Pipeline 2 (OFM-JEPA v2 §4, "Open-Ended LLM
Projection") actually executable with a T5 decoder.

The design doc's Pipeline 2 writes h_prefix = W_proj @ z_hat_t as if a single linear
layer were enough. It isn't, mechanically: T5's decoder cross-attends over a SEQUENCE
of encoder hidden states (one per input token), and our flow produces exactly one
pooled vector z_hat_t in R^d_shared. A single linear map of one vector is still one
vector, not a sequence.

Prefix-Expand is the "un-pooling" counterpart to Q-Pool: where Q-Pool collapses P
visual patch tokens down to K query-conditioned slots, Prefix-Expand expands one
pooled latent vector back up into a short learned pseudo-sequence of K' tokens, which
then stands in for the encoder's output as the decoder's cross-attention memory.

This is a genuine architectural addition beyond what OFM-JEPA v2 / Reflow-JEPA v3
specify -- it's required by the choice (made in this project, for exposure-bias
consistency) to reuse T5's own paired decoder rather than a decoder-only LM that could
accept a single soft-prompt vector prepended to its input stream instead. Flagging
that explicitly rather than silently presenting it as part of the original design.
"""
import torch
import torch.nn as nn


class PrefixExpand(nn.Module):
    """Expands a single pooled vector z_hat_t in R^d into a K'-length pseudo-sequence
    in R^[K', d], used as encoder_hidden_states for a T5 decoder's cross-attention.

    Implementation: K' learned "expansion slots" (analogous to Q-Pool's query slots),
    each FiLM-modulated by z_hat_t, refined by a shallow self-attention stack so the
    K' output tokens can specialize (early tokens attend differently than late tokens)
    rather than all being identical copies of the same FiLM-modulated slot.
    """

    def __init__(self, d_shared: int = 768, k_prefix: int = 8, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.k = k_prefix
        self.d = d_shared
        self.expansion_slots = nn.Parameter(torch.randn(k_prefix, d_shared) * 0.02)
        self.film = nn.Linear(d_shared, 2 * d_shared)  # (gamma, beta) from z_hat_t
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_shared, nhead=n_heads, dim_feedforward=4 * d_shared,
            batch_first=True, activation="gelu",
        )
        self.refine = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_shared)

    def forward(self, z_hat_t: torch.Tensor) -> torch.Tensor:
        # z_hat_t: (B, d) -> (B, K', d)
        B = z_hat_t.shape[0]
        gamma, beta = self.film(z_hat_t).chunk(2, dim=-1)          # (B, d) each
        slots = self.expansion_slots.unsqueeze(0).expand(B, -1, -1)  # (B, K', d)
        slots = slots * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        refined = self.refine(slots)                                 # (B, K', d)
        return self.out_norm(refined)
