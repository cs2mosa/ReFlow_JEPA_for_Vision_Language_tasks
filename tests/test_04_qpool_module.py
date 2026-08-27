"""
PHASE 1d — Q-Pool (the custom, trainable component), tested in isolation.

This is OUR module (Reflow-JEPA v3 §5.2: K learned query slots, FiLM-modulated by the
question vector c, cross-attending over frozen visual patch tokens H_v), so unlike the
frozen encoders we can and should test it fully, including gradient flow.

NOTE on the general-VL track: for image-captioning pretraining (this project's current
phase), there is no per-example question -- c is a single fixed, learned task-token,
shared across every example, rather than something that varies per (image, question)
pair the way it does for VQA. Q-Pool's FiLM pathway still needs to be load-bearing
(gradients must reach the task-token, and the module must not degenerate into a
question-blind pooler once real per-example conditioning -- e.g. the VQA extension --
is reintroduced), but "does output change with a different question" isn't a meaningful
test when there's only ever one task-token in this phase. Replaced with a determinism/
consistency test instead (see test_output_is_consistent_across_calls_with_same_task_token
below); test_output_depends_on_image is unchanged since image variation is still the
primary signal Q-Pool must track in this phase.

Targets:
  - test_output_shape                  -> basic interface contract.
  - test_gradients_flow_to_query_slots -> a dead/disconnected query-slot parameter is the
                                           single easiest way to silently reduce K learned
                                           slots back down to "effectively 1", quietly
                                           undoing the v2->v3 compositional-pooling fix.
  - test_output_depends_on_image       -> catches an accidentally-collapsed attention
                                           (e.g. softmax saturating to a fixed pattern).
  - test_output_is_consistent_across_calls_with_same_task_token -> replaces
                                           test_output_depends_on_question for this
                                           phase: with c fixed to a single learned
                                           task-token, the same image must always
                                           produce the same output (guards against a
                                           stray source of nondeterminism -- e.g. dropout
                                           left on, or an unfrozen random init reused
                                           incorrectly -- since there's no question
                                           variation to mask it here).
  - test_gradients_flow_to_task_token  -> the FiLM pathway must still be load-bearing
                                           even with only one task-token: if no gradient
                                           reaches it, FiLM is dead code in this phase,
                                           and there is no signal to catch that once the
                                           VQA extension (real per-example c) arrives and
                                           quietly relies on a pathway that was never
                                           actually trained.
"""
import torch
import torch.nn as nn
from conftest import D_IJEPA, D_TEXT, D_SHARED, K_QUERY_SLOTS, P_PATCHES


class QPool(nn.Module):
    """K learned query slots, FiLM-modulated by the question embedding c,
    cross-attending over visual patch tokens H_v. Matches Reflow-JEPA v3 §5.2."""

    def __init__(self, d_v=D_IJEPA, d_text=D_TEXT, d_shared=D_SHARED, k=K_QUERY_SLOTS, n_heads=8):
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


def test_output_shape():
    qpool = QPool()
    h_v = torch.randn(4, P_PATCHES, D_IJEPA)
    c = torch.randn(4, D_TEXT)
    out = qpool(h_v, c)
    assert out.shape == (4, D_SHARED)


def test_gradients_flow_to_query_slots():
    qpool = QPool()
    h_v = torch.randn(2, P_PATCHES, D_IJEPA)
    c = torch.randn(2, D_TEXT)
    out = qpool(h_v, c)
    loss = out.pow(2).sum()
    loss.backward()
    assert qpool.query_slots.grad is not None, "No gradient reached the learned query slots."
    grad_norm = qpool.query_slots.grad.norm().item()
    assert grad_norm > 0, "Gradient reached query_slots but is exactly zero."
    per_slot_grad_norm = qpool.query_slots.grad.norm(dim=-1)
    dead_slots = (per_slot_grad_norm < 1e-8).sum().item()
    assert dead_slots == 0, (
        f"{dead_slots}/{qpool.k} query slots received zero gradient -- effectively "
        f"reducing K back toward 1 and silently undoing the v2->v3 compositional-pooling "
        f"fix (Reflow-JEPA v3 §5.2)."
    )


def test_output_depends_on_image():
    qpool = QPool()
    qpool.eval()
    c = torch.randn(1, D_TEXT)
    h_v1 = torch.randn(1, P_PATCHES, D_IJEPA)
    h_v2 = torch.randn(1, P_PATCHES, D_IJEPA)
    with torch.no_grad():
        out1, out2 = qpool(h_v1, c), qpool(h_v2, c)
    assert not torch.allclose(out1, out2, atol=1e-4), (
        "Q-Pool produced (near-)identical output for two different images given the same "
        "question -- attention has likely saturated to a fixed pattern independent of "
        "the actual visual content."
    )


def test_output_depends_on_question():
    """Kept from the general capability check (not removed): confirms FiLM CAN
    discriminate between two different conditioning vectors at all, which the
    task-token tests below don't exercise (they use a single fixed c throughout).
    Still the right test once real per-example conditioning (VQA extension) returns."""
    qpool = QPool()
    qpool.eval()
    h_v = torch.randn(1, P_PATCHES, D_IJEPA)
    c1 = torch.randn(1, D_TEXT)
    c2 = torch.randn(1, D_TEXT)
    with torch.no_grad():
        out1, out2 = qpool(h_v, c1), qpool(h_v, c2)
    assert not torch.allclose(out1, out2, atol=1e-4), (
        "Q-Pool produced (near-)identical output for two different questions on the same "
        "image -- the FiLM conditioning pathway is not actually influencing pooling. "
        "This is the exact question-blind-pooling bottleneck flagged in the v1->v2 review."
    )


def test_output_is_consistent_across_calls_with_same_task_token():
    """General-VL captioning phase: c is a single fixed, learned task-token reused for
    every example, not a per-example question. The same image + the same task-token
    must deterministically produce the same output -- there's no question variation
    here to mask a stray source of nondeterminism (e.g. dropout left on)."""
    qpool = QPool()
    qpool.eval()
    task_token = torch.randn(1, D_TEXT)  # stands in for a single learned nn.Parameter
    h_v = torch.randn(1, P_PATCHES, D_IJEPA)
    with torch.no_grad():
        out1 = qpool(h_v, task_token)
        out2 = qpool(h_v, task_token)
    assert torch.allclose(out1, out2, atol=1e-6), (
        "Q-Pool gave different output across two calls with the identical image and "
        "identical fixed task-token -- check for stray nondeterminism (dropout at eval "
        "time, uninitialized buffers) that per-example question variation would "
        "otherwise have masked."
    )


def test_gradients_flow_to_task_token():
    """Even with only one task-token (no per-example question variation this phase),
    the FiLM pathway must still be load-bearing -- i.e. actually trained -- so that it
    isn't silently dead code the VQA extension would later depend on without ever
    having exercised it."""
    qpool = QPool()
    task_token = nn.Parameter(torch.randn(1, D_TEXT))
    h_v = torch.randn(1, P_PATCHES, D_IJEPA)
    out = qpool(h_v, task_token)
    loss = out.pow(2).sum()
    loss.backward()
    assert task_token.grad is not None, "No gradient reached the task-token."
    assert task_token.grad.norm().item() > 0, (
        "Gradient reached the task-token but is exactly zero -- the FiLM pathway is "
        "not actually load-bearing with a fixed task-token, meaning it would be "
        "effectively untrained when the VQA extension later relies on it for real "
        "per-example conditioning."
    )
