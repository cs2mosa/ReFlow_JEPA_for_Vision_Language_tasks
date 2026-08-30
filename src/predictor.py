"""
Predictor v_theta(Z_tau, tau, z_v_tilde, c): 6-layer DiT (Diffusion Transformer style),
sinusoidal timestep embedding injected via AdaLN modulation, cross-attention on
z_v_tilde and c. Matches DESIGN.md's component table.

This is genuinely new code -- nothing in the pre-implementation test suite built this,
since the suite's job was to validate the pieces it composes (encoders, Q-Pool, VICReg,
the decoder's geometry, the theoretical mechanism itself) before this predictor existed.
It's the one component with no isolated pre-implementation test of its own; its
correctness is instead exercised end-to-end by the Phase-B training loop (does the CFM
loss actually decrease, do the S+V and manifold-adherence diagnostics move the way
Theorem 3/4 predict).

Z_tau is a single vector in R^d_shared (there's no sequence structure to it -- it's a
point in the shared latent space), so it's treated as a length-1 "sequence" and the
transformer blocks only really do cross-attention + AdaLN-modulated MLP; self-attention
over a single token is a no-op by construction but kept for architectural uniformity
with standard DiT blocks (and so nothing breaks if Z_tau is ever generalized to a short
sequence later).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: (B,) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device).float() / half)
        args = tau.unsqueeze(-1).float() * freqs.unsqueeze(0) * 1000.0  # scale like diffusion timesteps
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)  # (B, dim)


class AdaLNZeroBlock(nn.Module):
    """One DiT block: AdaLN-modulated cross-attention over [z_v_tilde, c], then
    AdaLN-modulated MLP. Gate parameters are zero-initialized (DiT / "AdaLN-Zero"
    practice, Peebles & Xie 2023) so each block starts as an identity map and the
    network trains stably from initialization."""

    def __init__(self, d: int, n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        # AdaLN modulation: shift/scale/gate for attn, shift/scale/gate for mlp -> 6*d
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, d)   memory: (B, M, d)   t_emb: (B, d)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(t_emb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        attn_out, _ = self.cross_attn(h, memory, memory)
        x = x + gate_a.unsqueeze(1) * attn_out
        h = self.norm2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class VelocityPredictor(nn.Module):
    def __init__(self, d_shared: int = 768, depth: int = 6, n_heads: int = 8,
                 edm_precondition: bool = False):
        """edm_precondition=False (default): the original architecture, network output
        IS the velocity directly, DiT-Zero-initialized so v=0 at init.

        edm_precondition=True: EDM-style reparametrization (Karras et al. 2022, cited
        in OFM-JEPA v2 as a candidate fix, never previously implemented here). Instead
        of predicting velocity directly, the network predicts an estimate of the
        TARGET embedding z1_hat (bounded, lives near the unit sphere -- a far easier
        quantity to learn than a velocity that must diverge as tau->1), and velocity is
        computed ANALYTICALLY as (z1_hat - z_tau) / (1 - tau). This bakes the
        1/(1-tau) terminal divergence (test_08's validated exact-field rate) into the
        architecture structurally -- it no longer needs to be learned via gradient
        descent, which measure_terminal_divergence.py found the raw-velocity
        architecture had NOT reliably learned on a real trained network, at a real
        checkpoint, regardless of Reflow or more integration steps.

        Motivated directly by a persistent, repeated finding: ||z_hat|| (the flow's
        integrated output) plateaued around 0.45-0.50 against a target norm of 1.0,
        unmoved by an LR schedule, a Reflow round, or 10x more integration steps.
        Forcing z1_hat onto the unit sphere via F.normalize means the terminal norm is
        correct BY CONSTRUCTION once the network's target-estimate is even roughly
        right, rather than depending on the network having learned the correct
        diverging magnitude from scratch."""
        super().__init__()
        self.d = d_shared
        self.edm_precondition = edm_precondition
        self.time_embed = SinusoidalTimeEmbedding(d_shared)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(d_shared, n_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_shared, elementwise_affine=False)
        self.final_ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(d_shared, 2 * d_shared))
        nn.init.zeros_(self.final_ada_ln[-1].weight)
        nn.init.zeros_(self.final_ada_ln[-1].bias)
        self.head = nn.Linear(d_shared, d_shared)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # edm_precondition=False: zero-init head -> v=0 at init (standard DiT-Zero).
        # edm_precondition=True: zero-init head -> z1_hat = normalize(z_v_tilde + 0)
        # = z_v_tilde at init (a sensible, well-defined, non-degenerate starting
        # guess -- "the target looks like the visual embedding" -- rather than the
        # zero vector, which normalize() cannot sensibly handle).

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor,
                z_v_tilde: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # z_tau, z_v_tilde, c: (B, d)   tau: (B,)
        t_emb = self.time_embed(tau)                                    # (B, d)
        x = z_tau.unsqueeze(1)                                          # (B, 1, d)
        memory = torch.stack([z_v_tilde, c], dim=1)                     # (B, 2, d)
        for block in self.blocks:
            x = block(x, memory, t_emb)
        shift, scale = self.final_ada_ln(t_emb).chunk(2, dim=-1)
        h = self.final_norm(x.squeeze(1)) * (1 + scale) + shift
        raw = self.head(h)                                             # (B, d)

        if not self.edm_precondition:
            return raw

        z1_hat = F.normalize(z_v_tilde + raw, dim=-1)
        denom = (1 - tau).clamp(min=1e-4).unsqueeze(-1)
        return (z1_hat - z_tau) / denom
