"""
PHASE 1b-cont — Text (caption) encoder tested against a non-atomic sample, for the
general-VL pretraining track.

This is deliberately a SEPARATE file from test_02_text_encoder.py, not a replacement.
test_02 exercises M_T as a small, finite, atomic answer vocabulary -- that's still the
correct object for the VQA extension (Priority guide, Pipeline 1). Here, M_T stands in
for "the image of the text encoder over a large open caption corpus": non-atomic, much
larger, and not assumed to have any finite enumeration. Same underlying encoder
(`text_seq2seq_and_tokenizer`'s .get_encoder()), different regime being probed.

Targets mirror test_02's, with one change and one addition:
  - test_output_shape_matches_spec        -> unchanged interface contract, (B, 768)
  - test_deterministic_for_same_string    -> unchanged rationale (target-encoder EMA
                                              copy in Reflow-JEPA v3 §2.2 still needs
                                              a stable, reproducible embedding at each
                                              step it's called, atomic vocab or not)
  - test_distinct_captions_are_separated  -> same cosine-similarity check as test_02,
                                              run over a ~200-caption synthetic sample
                                              instead of 20 answers, since collapse is
                                              a property of the encoder, not the task
  - test_no_caption_collapse              -> Failure Mode 4, same as test_02
  - test_intrinsic_dimension_of_caption_manifold -> the actual point of this file: at
                                              n=20 (test_02's size), the Levina-Bickel
                                              estimator is barely usable (k<=n-2 forces
                                              k<=18, badly noisy). d_int(M_T) needs to be
                                              measured at a sample size that could
                                              plausibly represent an open caption
                                              distribution, not a closed answer set, per
                                              OFM-JEPA v2's "measure it, don't import Tulchinskii
                                              et al.'s number" correction (Mitigation 2).
"""
import math
import random
import torch
import numpy as np
from conftest import D_TEXT
from test_02_text_encoder import _encode, _levina_bickel_mle_id

# Synthetic caption corpus: template x (subject, verb/relation, object, place) combinatorics,
# standing in for a real caption dataset until one is chosen (per current project decision:
# validate the design against synthetic distributions before real data). Deliberately built
# to be NON-atomic in flavor -- overlapping templates and vocabulary, the way real captions
# share structure -- not just 200 unrelated random strings.
_SUBJECTS = ["a dog", "a cat", "a man", "a woman", "a child", "two dogs", "a red car", "a bicycle"]
_VERBS = ["running through", "sitting near", "standing beside", "jumping over", "resting under"]
_OBJECTS = ["a wooden bench", "a green field", "a tall tree", "a parked car", "a fence"]
_PLACES = ["in a park", "on a busy street", "near the beach", "in a backyard", "at sunset"]


def _build_synthetic_caption_corpus(n: int = 200, seed: int = 0):
    rng = random.Random(seed)
    seen = set()
    captions = []
    while len(captions) < n:
        cap = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_PLACES)}"
        if cap not in seen:
            seen.add(cap)
            captions.append(cap)
    return captions


CAPTION_CORPUS = _build_synthetic_caption_corpus(n=200)


def test_output_shape_matches_spec(text_seq2seq_and_tokenizer):
    model, tok = text_seq2seq_and_tokenizer
    pooled = _encode(model.get_encoder(), tok, [CAPTION_CORPUS[0]])
    assert pooled.shape == (1, D_TEXT)


def test_deterministic_for_same_string(text_seq2seq_and_tokenizer):
    model, tok = text_seq2seq_and_tokenizer
    enc = model.get_encoder()
    a = _encode(enc, tok, [CAPTION_CORPUS[0]])
    b = _encode(enc, tok, [CAPTION_CORPUS[0]])
    assert torch.allclose(a, b, atol=1e-5), (
        "Encoding the same caption twice gave different embeddings -- the target "
        "(EMA) text pipeline that pre-computes/refreshes M_T must be stable across "
        "calls, atomic vocabulary or not."
    )


def test_distinct_captions_are_separated(text_seq2seq_and_tokenizer):
    model, tok = text_seq2seq_and_tokenizer
    embs = _encode(model.get_encoder(), tok, CAPTION_CORPUS)
    embs_n = embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = embs_n @ embs_n.T
    sim.fill_diagonal_(-2.0)
    worst_pair_sim = sim.max().item()
    i, j = divmod(sim.argmax().item(), len(CAPTION_CORPUS))
    assert worst_pair_sim < 0.999, (
        f"Captions '{CAPTION_CORPUS[i]}' and '{CAPTION_CORPUS[j]}' encode to "
        f"(near-)identical embeddings (cosine={worst_pair_sim:.6f}) despite being "
        f"different strings -- raw-encoder-level collapse, at a scale/diversity closer "
        f"to what an open caption corpus would actually look like."
    )


def test_no_caption_collapse(text_seq2seq_and_tokenizer):
    model, tok = text_seq2seq_and_tokenizer
    embs = _encode(model.get_encoder(), tok, CAPTION_CORPUS)
    per_dim_std = embs.std(dim=0)
    frac_dead = (per_dim_std < 1e-4).float().mean().item()
    assert frac_dead < 0.5, (
        f"{frac_dead:.1%} of caption-embedding dimensions are near-constant across "
        f"{len(CAPTION_CORPUS)} distinct synthetic captions -- collapse signature."
    )


def test_intrinsic_dimension_of_caption_manifold(text_seq2seq_and_tokenizer):
    """Reports (does not gate on) d_int(M_T) at n=200 rather than test_02's n=20 -- the
    Levina-Bickel estimator needs a meaningfully larger neighborhood to say anything
    about a manifold that's supposed to represent an open, non-atomic caption
    distribution rather than a fixed 20-item vocabulary. Re-run this on real captions
    and the real checkpoint before treating any number here as informative for
    d_shared -- it's currently only checking that the encoder's OWN geometry is
    consistent between the atomic-answer regime (test_02) and this larger, more
    caption-like regime, which is the honest thing this mock-weight sandbox can check."""
    model, tok = text_seq2seq_and_tokenizer
    embs = _encode(model.get_encoder(), tok, CAPTION_CORPUS).numpy()
    d_int = _levina_bickel_mle_id(embs, k=10)
    print(f"\n[INFO] Levina-Bickel intrinsic dimension estimate of the caption manifold "
          f"(untrained encoder, n={len(CAPTION_CORPUS)} synthetic captions): {d_int:.2f} "
          f"(ambient d_text={D_TEXT}). This is a random-weight number and should not be "
          f"trusted for design decisions -- re-run on the real T5 checkpoint and a real "
          f"caption corpus before it informs d_shared or sigma.")
    assert math.isfinite(d_int) and d_int > 0
