"""
PHASE 1b — Text (answer) encoder, tested in isolation.

Targets:
  - test_output_shape_matches_spec        -> basic interface contract, d_text=768.
  - test_deterministic_for_same_string    -> two independent encodes of "yes" must land
                                              at the same point; if not, the frozen
                                              candidate bank M_T (Reflow-JEPA v3 §3) is
                                              not actually stable across training steps.
  - test_distinct_answers_are_separated   -> the atomic-answer-manifold assumption
                                              (Lemma 1, base theory) requires M_T's points
                                              to be genuinely distinct; if the encoder maps
                                              two different short answers to (near-)identical
                                              points, Lemma 1's off-manifold argument and the
                                              Laguerre-cell decoding (Reflow-JEPA v3 Thm 2)
                                              both degrade.
  - test_no_answer_collapse               -> Failure Mode 4 (representation collapse),
                                              at the raw frozen-encoder level.
  - test_intrinsic_dimension_of_answer_manifold -> directly executes the OFM-JEPA v2
                                              corrected recommendation ("measure d_int on
                                              this design's own encoder, don't import
                                              Tulchinskii et al.'s sentence-embedding
                                              number") using the Levina-Bickel MLE
                                              estimator (Levina & Bickel, NeurIPS 2004) as
                                              a standard, defensible alternative to PHD.
"""
import math
import torch
import numpy as np
from conftest import D_TEXT

ANSWER_VOCAB = [
    "yes", "no", "red", "blue", "green", "0", "1", "2", "3", "4", "5",
    "dog", "cat", "left", "right", "man", "woman", "table", "chair",
    "unanswerable",
]


def _encode(text_encoder, tokenizer, texts):
    batch = tokenizer(texts)
    with torch.no_grad():
        out = text_encoder(**batch).last_hidden_state  # (B, L, d)
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled


def test_output_shape_matches_spec(text_encoder_and_tokenizer):
    model, tok = text_encoder_and_tokenizer
    pooled = _encode(model, tok, ["what color is the ball"])
    assert pooled.shape == (1, D_TEXT)


def test_deterministic_for_same_string(text_encoder_and_tokenizer):
    model, tok = text_encoder_and_tokenizer
    a = _encode(model, tok, ["yes"])
    b = _encode(model, tok, ["yes"])
    assert torch.allclose(a, b, atol=1e-5), (
        "Encoding the same answer string twice gave different embeddings. The candidate "
        "bank M_T (Reflow-JEPA v3 §3) is pre-computed once and assumed static; if the "
        "encoder is non-deterministic (e.g. dropout left on at eval time), M_T silently "
        "drifts and nearest-neighbor decoding becomes unreliable."
    )


def test_distinct_answers_are_separated(text_encoder_and_tokenizer):
    model, tok = text_encoder_and_tokenizer
    embs = _encode(model, tok, ANSWER_VOCAB)
    embs_n = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = embs_n @ embs_n.T
    sim.fill_diagonal_(-2.0)  # ignore self-similarity
    worst_pair_sim = sim.max().item()
    i, j = divmod(sim.argmax().item(), len(ANSWER_VOCAB))
    assert worst_pair_sim < 0.999, (
        f"Answers '{ANSWER_VOCAB[i]}' and '{ANSWER_VOCAB[j]}' encode to (near-)identical "
        f"embeddings (cosine={worst_pair_sim:.6f}). M_T is supposed to be a set of "
        f"genuinely distinct atoms (Lemma 1); if two different valid answers collapse "
        f"onto the same point, Theorem 2's Laguerre-cell partition (Reflow-JEPA v3) has "
        f"a degenerate cell and nearest-neighbor decoding cannot distinguish them."
    )


def test_no_answer_collapse(text_encoder_and_tokenizer):
    model, tok = text_encoder_and_tokenizer
    embs = _encode(model, tok, ANSWER_VOCAB)
    per_dim_std = embs.std(dim=0)
    frac_dead = (per_dim_std < 1e-4).float().mean().item()
    assert frac_dead < 0.5, (
        f"{frac_dead:.1%} of text-embedding dimensions are near-constant across "
        f"{len(ANSWER_VOCAB)} distinct answers -- raw-encoder-level collapse signature."
    )


def _levina_bickel_mle_id(X: np.ndarray, k: int = 5) -> float:
    """Levina & Bickel (NeurIPS 2004) maximum-likelihood intrinsic dimension estimator.
    Standard, citable alternative to the persistent-homology-dimension estimator used by
    Tulchinskii et al. (2023); appropriate here because our answer vocabulary is small
    (PHD needs larger point clouds via bootstrap subsampling to be reliable)."""
    n = X.shape[0]
    k = min(k, n - 2)
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    dims = []
    for i in range(n):
        knn = np.sort(dists[i])[:k]
        if knn[-1] <= 0:
            continue
        logs = np.log(knn[-1] / knn[:-1].clip(min=1e-12))
        m_k = 1.0 / (np.mean(logs) + 1e-12) if len(logs) > 0 else float("nan")
        dims.append(m_k)
    return float(np.nanmean(dims))


def test_intrinsic_dimension_of_answer_manifold(text_encoder_and_tokenizer):
    """This does not assert a pass/fail threshold -- it prints the measured d_int so it
    can be compared against the design assumption before choosing d_shared. This is the
    concrete instantiation of the OFM-JEPA v2 correction: measure it, don't import
    Tulchinskii et al.'s different-context number."""
    model, tok = text_encoder_and_tokenizer
    embs = _encode(model, tok, ANSWER_VOCAB).numpy()
    d_int = _levina_bickel_mle_id(embs, k=5)
    print(f"\n[INFO] Levina-Bickel intrinsic dimension estimate of M_T (untrained encoder, "
          f"n={len(ANSWER_VOCAB)} answers): {d_int:.2f} "
          f"(ambient d_text={D_TEXT}). Re-run this against the real pretrained checkpoint "
          f"and the full answer vocabulary before fixing d_shared or d_int in the design.")
    assert math.isfinite(d_int) and d_int > 0
