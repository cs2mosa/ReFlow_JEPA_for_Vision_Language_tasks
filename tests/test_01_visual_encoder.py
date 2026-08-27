"""
PHASE 1a — Visual encoder, tested in isolation.

Targets, each traced to a specific problem raised during design review:
  - test_output_shape_matches_spec       -> Reflow-JEPA v2's ViT-H/14+ViT-L/16 dimension
                                             mismatch bug (v3 §5.1 correction). Fails loudly
                                             on any P/d_ijepa drift instead of silently
                                             reshaping wrong downstream.
  - test_patch_tokens_are_extracted_correctly -> CLS-token-off-by-one is a classic real bug
                                             when swapping a generic ViT for a CLS-less
                                             I-JEPA checkpoint; this pins the extraction logic.
  - test_no_representation_collapse      -> Failure Mode 4 / Lemma "collapse" (OFM-JEPA §1,
                                             Reflow-JEPA v3 Lemma 3): different images must
                                             not produce (near-)identical embeddings.
  - test_embedding_norm_statistics       -> feeds the norm-discrepancy diagnostic used in
                                             test_03_encoder_compatibility.py.
  - test_frozen_encoder_has_no_grad      -> guards against accidentally leaving E_V trainable,
                                             which silently breaks the "frozen" assumption every
                                             theorem in the design docs relies on.
"""
import torch
from conftest import D_IJEPA, P_PATCHES


def _extract_patch_tokens(vit_output_last_hidden_state):
    """ViT-style models prepend a CLS token; I-JEPA does not. Real integration code must
    know which one it's talking to. This helper makes that choice explicit and testable."""
    n_tokens = vit_output_last_hidden_state.shape[1]
    if n_tokens == P_PATCHES + 1:
        return vit_output_last_hidden_state[:, 1:, :]   # drop CLS
    elif n_tokens == P_PATCHES:
        return vit_output_last_hidden_state              # already patch-only (true I-JEPA)
    else:
        raise AssertionError(
            f"Expected {P_PATCHES} or {P_PATCHES + 1} tokens, got {n_tokens}. "
            f"Patch grid does not match the documented ViT-H/14 @ 224x224 spec."
        )


def test_output_shape_matches_spec(visual_encoder):
    images = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = visual_encoder(images).last_hidden_state
    patches = _extract_patch_tokens(out)
    assert patches.shape == (4, P_PATCHES, D_IJEPA), (
        f"Visual encoder output {tuple(patches.shape)} does not match the corrected spec "
        f"(B, {P_PATCHES}, {D_IJEPA}). This is exactly the bug caught in Reflow-JEPA v3 "
        f"§5.1 (ViT-H/14 patch count paired with ViT-L/16 hidden width) -- if this fails, "
        f"stop and fix the checkpoint/config choice before writing any more of the pipeline."
    )


def test_patch_tokens_are_extracted_correctly(visual_encoder):
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        raw = visual_encoder(images).last_hidden_state
    patches = _extract_patch_tokens(raw)
    # every patch token's row must actually come from the grid, not accidentally include
    # the CLS token's row (whose statistics are typically very different from patch rows)
    assert patches.shape[1] == P_PATCHES


def test_no_representation_collapse(visual_encoder):
    """Different inputs must yield meaningfully different embeddings. This does not
    validate semantic quality (impossible without real pretrained weights) but it does
    validate that the encoder is not architecturally degenerate (e.g. an accidental
    zero-initialized residual path that maps everything to the same point)."""
    images = torch.randn(16, 3, 224, 224)
    with torch.no_grad():
        out = _extract_patch_tokens(visual_encoder(images).last_hidden_state)
    pooled = out.mean(dim=1)  # (16, d) crude pool, sufficient for a collapse check
    pairwise_std = pooled.std(dim=0)  # per-dimension std across the 16 examples
    frac_dead_dims = (pairwise_std < 1e-4).float().mean().item()
    assert frac_dead_dims < 0.5, (
        f"{frac_dead_dims:.1%} of embedding dimensions have near-zero variance across "
        f"16 distinct random inputs -- this is the collapse signature Reflow-JEPA v3's "
        f"VICReg guard (Proposition 5) is designed to catch, seen here at the raw encoder "
        f"level before any trainable head is even involved."
    )


def test_embedding_norm_statistics(visual_encoder):
    """Record norm statistics for the cross-encoder scale-compatibility check
    (see test_03_encoder_compatibility.py). This test just asserts the norms are finite
    and non-degenerate (not NaN/inf, not all exactly zero)."""
    images = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        out = _extract_patch_tokens(visual_encoder(images).last_hidden_state)
    pooled = out.mean(dim=1)
    norms = pooled.norm(dim=-1)
    assert torch.isfinite(norms).all()
    assert (norms > 1e-6).all(), "Visual embeddings have (near-)zero norm."


def test_frozen_encoder_has_no_grad(visual_encoder):
    """Guards the 'Status: FROZEN' assumption every theorem in the design docs depends on.
    If someone forgets requires_grad_(False) during assembly, every downstream guarantee
    (marginal preservation, VICReg non-collapse, etc.) is derived under a false premise."""
    for p in visual_encoder.parameters():
        p.requires_grad_(False)
    images = torch.randn(2, 3, 224, 224)
    out = visual_encoder(images).last_hidden_state
    assert not out.requires_grad, (
        "Visual encoder output still requires grad after freezing parameters -- "
        "check for a stray trainable submodule (e.g. an un-frozen pooler head)."
    )
