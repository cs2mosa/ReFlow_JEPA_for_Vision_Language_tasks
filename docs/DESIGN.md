# Reflow-JEPA (CFM): System Design and Pre-Implementation Test Plan

**Status:** design-complete, verified against 31 passing pre-implementation tests. Not yet trained.
**Scope:** the continuous Conditional-Flow-Matching track only (Discrete Flow Matching for closed-vocabulary VQA is a validated alternative from a companion review, not covered here).

This document consolidates every correction made across the design's revision history into one implementation-ready specification, then specifies the tests that must pass on the chosen encoder/decoder components *before* any training code is written.

---

## 1. Overview

Reflow-JEPA treats VQA as conditional domain transfer between a visual-question embedding and an answer embedding, learned via Conditional Flow Matching (CFM) with Rectified-Flow (Reflow) refinement, inside a JEPA-style predictive architecture. The model never generates tokens autoregressively: an ODE moves a point through a shared latent space from a (visual, question) representation to an answer representation, which is then decoded by nearest-neighbor lookup against a pre-encoded candidate answer bank.

The design went through four rounds of review. Each round is preserved here as a traceable fix, not silently merged away — Section 5 maps every architectural decision back to the specific problem it resolves and the theorem or citation backing the fix.

---

## 2. Architecture

### 2.1 Component table

| Component | Choice | Status | Output |
|---|---|---|---|
| Visual encoder $E_V$ | I-JEPA ViT-H/14, 224×224 | Frozen | $H_v \in \mathbb{R}^{[B,256,1280]}$ |
| Question encoder $E_C$ | T5-base / BART-base encoder | Frozen | $c \in \mathbb{R}^{[B,768]}$ |
| Q-Pool | $K{=}8$ learned query slots, FiLM($c$)-modulated, cross-attend over $H_v$ | Trainable | $z_v^{\text{raw}} \in \mathbb{R}^{[B,1280]}$ |
| Visual projection $g_V$ | 2-layer MLP + VICReg | Trainable | $\tilde z_v \in \mathbb{R}^{[B,768]}$ |
| Text answer encoder $E_T$ (online) | T5-base / BART-base encoder | Trainable | $z_t^{\text{raw}} \in \mathbb{R}^{[B,768]}$ |
| Text answer encoder $E_T'$ (target) | EMA of $E_T$, **same modality** | Stop-grad | used for $\mathcal M_T$ only |
| Text projection $g_T$ / $g_T'$ | 2-layer MLP + VICReg (online/target pair) | Trainable / EMA | $\tilde z_t \in \mathbb{R}^{[B,768]}$ |
| Stochastic source | $Z_0 = \tilde z_v + \sigma\varepsilon,\ \varepsilon\sim\mathcal N(0,I)$ | New mechanism | $Z_0 \in \mathbb{R}^{[B,768]}$ |
| Predictor $v_\theta$ | 6-layer DiT, sinusoidal+AdaLN time, cross-attn on $\tilde z_v$ and $c$ | Trainable | $v_{\text{pred}} \in \mathbb{R}^{[B,768]}$ |
| Candidate bank $\mathcal M_T$ | $\{g_T'(E_T'(a_k))\}$, pre-encoded via the **target** (EMA) copy | Cached | vectors in $\mathbb{R}^{768}$ |
| Decoder | Cosine nearest-neighbor snap onto $\mathcal M_T$ | Rule-based | answer string |

$d_{\text{shared}} = 768$ throughout, once past the projection heads.

### 2.2 Data flow — training

```
[Image I] --frozen I-JEPA--> H_v (256,1280) ----------------------+
                                                                    |
[Question Q] --frozen T5--> c (768) --+---------------------------|--> v_theta (cross-attn c)
                                       |                            |
                                       +--> Q-Pool(query=FiLM(c), K/V=H_v) --> z_v_raw (1280)
                                                                    |
                                                             g_V + VICReg --> z_v_tilde (768)
                                                                    |
                                              +---------------------+---------------------+
                                              |                                           |
                                    Z_0 = z_v_tilde + sigma*eps                    v_theta (cross-attn z_v_tilde)
                                              |
[Answer A] --T5 (online)--> z_t_raw --g_T + VICReg--> z_t_tilde = Z_1
                                              |
                              Z_tau = (1-tau)*Z_0 + tau*Z_1,  tau ~ Unif[0,1]
                                              |
                                     v_theta(Z_tau, tau, z_v_tilde, c)
                                              |
                        L_total = ||v_pred - (Z_1 - Z_0)||^2 + lambda*(L_var(z_v_tilde) + L_var(z_t_tilde))
```

### 2.3 Data flow — inference

```
[Image I*, Question Q*] --> z_v_tilde*, c*  (cacheable per image-question pair)
draw eps ~ N(0,I);  z_0 = z_v_tilde* + sigma*eps        [fresh draw EVERY call]
integrate dz/dtau = v_theta(z_tau, tau, z_v_tilde*, c*) from tau=0 to tau=1-delta   [delta ~ 1e-3, early stop]
snap z_hat to nearest candidate in M_T via cosine similarity
--> decoded answer string
```

For multi-answer-coverage evaluation specifically, draw **multiple** independent `eps` per `(I*, Q*)` — a single draw returns one atom, not the answer distribution.

### 2.4 Training procedure

**Phase 1 — base CFM.** Train $v_\theta$, $g_V$, $g_T$ (online) jointly on $\mathcal L_{\text{total}}$ above; update $E_T'$, $g_T'$ as EMA of the online text pipeline every step; $E_V$, $E_C$ frozen throughout.

**Phase 2 — Reflow.** For $k = 1 \ldots K$: simulate $(Z_0^{(k)}, \hat Z_1^{(k)})$ pairs by integrating the current field; retrain $v_{\theta_{k+1}}$ on the resulting deterministic coupling; log the straightness/crossing diagnostic $\widehat{S+V}$ each round; stop once it plateaus (§7.6 of the base theory).

---

## 3. Design rationale — every fix traced to its source

| # | Problem | Fix | Why it's correct |
|---|---|---|---|
| 1 | L2-regression predictor collapses off the answer manifold on multimodal questions | Predict in embedding space with a generative CFM decoder, not a bare regressor | Lemma 1 (base theory): conditional-mean minimizer of L2 loss lands off a non-convex/atomic manifold |
| 2 | Straight-line CFM paths can cross where multiple valid answers exist | Reflow (iterative re-coupling) | Liu, Gong & Liu, ICLR 2023 — reduces crossing measure at an $O(1/K)$ best-iterate rate |
| 3 | Interpolating raw, never-jointly-calibrated I-JEPA/T5 features is geometrically meaningless | Trainable projection heads $g_V, g_T$ into a shared calibrated space | necessary before any straight-line interpolation argument is meaningful |
| 4 | Single mean-pooled question vector collapses compositional structure before pooling | $K$-slot, FiLM-modulated Q-Pool | preserves multiple simultaneous object/relation bindings for GQA/TallyQA-style questions |
| 5 | Deterministic source ($Z_0 \equiv \tilde z_v$) forces the one-step field back onto the exact off-manifold compromise point Lemma 1 warns about | Stochastic source $Z_0 = \tilde z_v + \sigma\varepsilon$ | proved exactly: at $\tau{=}0$, $v_\theta(\tilde z_v,0,\gamma) = \mathbb E[z_t\mid\gamma]-\tilde z_v$, reproducing Lemma 1's failure verbatim when $Z_0$ is a point mass |
| 6 | Does the stochastic source actually recover per-example multimodal answers? | Yes — proved via Brenier's theorem + semi-discrete OT (Laguerre cells); closed form for the two-atom case | Brenier 1991; Aurenhammer 1987; Aurenhammer–Hoffmann–Aronov 1998. Validated numerically in `test_07` below |
| 7 | Does source stochasticity also prevent representation collapse of $g_V, g_T$? | **No** — proved as a separate, independent failure mode | with $g_V, g_T$ both constant, $Z_0$'s noise is exactly invertible from $Z_\tau$, so zero CFM loss is still achievable despite total collapse |
| 8 | Collapse of $g_V, g_T$ | VICReg-style batch-variance penalty, added to the training loss | Bardes, Ponce & LeCun, ICLR 2022. Validated in `test_05` below |
| 9 | "Ambrosio dimension collapse" / "Kirszbraun non-extension" claimed CFM cannot transport continuous to atomic targets | Both citations were misapplied (Brouwer's theorem requires injectivity; Kirszbraun is an unrelated extension theorem) — the transport is not just possible but has the exact closed form of item 6 | verified directly against the cited theorems' actual statements |
| 10 | Claimed "super-exponential" terminal-time instability, $\exp(1/2\delta^2)$ | Corrected: the real rate is $O(1/(1-\tau))$ (linear), integrating to a **logarithmic** $-\ln\delta$ error bound | direct Jacobian computation; matches the diffusion-model score-singularity literature ($O(1/t)$ near the data manifold). Validated numerically in `test_08` |
| 11 | "EMA target encoder = EMA of image encoder" (cross-modal, nonsensical) | Corrected to same-modality EMA: target text pipeline tracks the **online text** pipeline | matches I-JEPA / BYOL / DINO precedent |
| 12 | High-dimensional nearest-neighbor decoding concentrates on a few "hub" candidates | Decode via cosine similarity (already substantially better than raw Euclidean); NICDM available as a further correction but **not** assumed to help on top of cosine without re-measurement | Radovanović et al., JMLR 2010; Schnitzer et al., JMLR 2012. Empirically characterized (not just cited) in `test_06` |
| 13 | Visual/text embedding norm mismatch skews the CFM velocity target | L2-normalize both embeddings before flow matching | standard practice; quantified per-encoder in `test_03` rather than assumed |
| 14 | Arm-(A) baseline in the evaluation ladder lacked the Q-Pool/$g_V$/$g_T$ stack, confounding architecture with training-procedure comparisons | Give every arm the identical calibration stack, trained under its own objective | otherwise accuracy differences are not attributable to the training procedure alone |

---

## 4. Testing phase — verify components before assembly

**Philosophy.** Every item in Section 3 is a claim about a *specific* piece of the system — an encoder's output shape, a projection head's collapse behavior, the decoder's retrieval geometry, or the flow-matching mechanism itself. Each is testable **in isolation**, without training anything, before the full pipeline exists. The suite is ordered so failures are caught as close to their source as possible: frozen encoders first, then cross-encoder compatibility, then the custom trainable modules, then the decoder, then the core theoretical mechanism, decoupled from any network at all.

**A note on what "passing" means here.** This sandbox has no route to the real I-JEPA / T5 checkpoint hosts, so encoder tests run against the *real architecture classes* (`transformers.ViTModel`, `T5EncoderModel`) configured to match the real checkpoints exactly, but with random weights. That is sufficient to catch every structural problem this suite targets — shape mismatches, scale mismatches, collapse-guard bugs, hubness, the stochastic-source mechanism, the terminal-time rate — because none of them depend on what the weights learned, only on whether the pieces fit together and whether our own derived mechanisms behave as proved. Swapping in real checkpoints is a two-line change in `conftest.py` (`load_visual_encoder` / `load_text_encoder`); nothing else needs to change. **All 31 tests currently pass.**

### 4.1 `test_01_visual_encoder.py` — $E_V$ in isolation

| Test | Checks | Traces to |
|---|---|---|
| `test_output_shape_matches_spec` | $(B, 256, 1280)$ exactly | item 3-of-earlier-review: the ViT-H/14-patch-count-with-ViT-L/16-hidden-width dimension bug. Fails loudly instead of silently reshaping wrong downstream. |
| `test_patch_tokens_are_extracted_correctly` | CLS-token handling | generic ViT checkpoints prepend a CLS token; true I-JEPA does not — pins the extraction logic either way |
| `test_no_representation_collapse` | $<50\%$ of embedding dimensions dead across 16 random inputs | Failure Mode 4 / collapse, checked at the raw-encoder level |
| `test_embedding_norm_statistics` | finite, non-zero norms | feeds the cross-encoder scale check in 4.3 |
| `test_frozen_encoder_has_no_grad` | no gradient reaches $E_V$'s output after freezing | every theorem in this document assumes $E_V$ is frozen; this guards that premise in code |

### 4.2 `test_02_text_encoder.py` — $E_T$ in isolation

| Test | Checks | Traces to |
|---|---|---|
| `test_output_shape_matches_spec` | $(B, 768)$ | interface contract |
| `test_deterministic_for_same_string` | same string encodes identically twice | $\mathcal M_T$ is precomputed once and assumed static |
| `test_distinct_answers_are_separated` | no two distinct answers in the vocab have cosine sim $> 0.999$ | Lemma 1's atomic-manifold assumption requires genuinely distinct atoms; a degenerate pair breaks the Laguerre-cell partition (item 6) |
| `test_no_answer_collapse` | $<50\%$ dead dimensions across the answer vocabulary | Failure Mode 4, text side |
| `test_intrinsic_dimension_of_answer_manifold` | reports (does not gate on) a Levina–Bickel MLE estimate of $d_{\text{int}}(\mathcal M_T)$ | direct instantiation of "measure it on your own encoder, don't import a different paper's number" — must be re-run on the real checkpoint + full vocabulary before fixing $d_{\text{shared}}$ |

### 4.3 `test_03_encoder_compatibility.py` — the "are they truly compatible" check

| Test | Checks | Traces to |
|---|---|---|
| `test_raw_dims_require_projection` | $E_V$ and $E_T$ do not already share a dimension | confirms $g_V$/$g_T$ are mandatory |
| `test_norm_scale_discrepancy_is_real` | measures (does not assume) the $\|z_i\|$ vs $\|z_t\|$ ratio on the actual chosen encoders | quantifies item 13 instead of taking it on faith |
| `test_l2_normalization_fixes_scale` | confirms L2-normalization equalizes norms to $1.0$ on these specific encoders | closes the loop on the previous test |

### 4.4 `test_04_qpool_module.py` — the custom trainable pooler

| Test | Checks | Traces to |
|---|---|---|
| `test_output_shape` | $(B, 768)$ | interface contract |
| `test_gradients_flow_to_query_slots` | every one of the $K$ slots receives non-zero gradient | a dead slot silently reduces $K$ back toward 1, undoing item 4 |
| `test_output_depends_on_image` | two different images (same question) give different output | catches attention saturating to a fixed pattern |
| `test_output_depends_on_question` | two different questions (same image) give different output | verifies FiLM conditioning is actually load-bearing — otherwise Q-Pool has silently degraded into a question-blind pooler |

### 4.5 `test_05_vicreg_collapse_guard.py` — the anti-collapse regularizer

| Test | Checks | Traces to |
|---|---|---|
| `test_collapsed_batch_incurs_large_penalty` | a fully-collapsed batch incurs penalty $\approx \gamma_0^2$ | item 8's global-optimum-removal claim, checked against the actual implementation, not just the proof on paper |
| `test_healthy_batch_incurs_small_penalty` | spread-out batch incurs penalty $\approx 0$ | guards against $\gamma_0$ being miscalibrated and fighting useful training signal |
| `test_penalty_is_strictly_ordered_between_collapsed_and_healthy` | collapsed penalty $>$ healthy penalty | the actual gradient-incentive claim the proof depends on |
| `test_partial_collapse_is_detected_dimension_wise` | partial (some-dimensions-dead) collapse is detected, not just total collapse | guards against a coarser, less sensitive implementation |

### 4.6 `test_06_decoder_hubness.py` — nearest-neighbor decoding geometry

| Test | Checks | Traces to |
|---|---|---|
| `test_raw_euclidean_decoding_shows_hubness` | severe skew (>3.0) for raw Euclidean NN at realistic bank size | confirms the underlying phenomenon (item 12) is real, not hypothetical |
| `test_prescribed_cosine_decoding_is_substantially_better_than_euclidean` | cosine skew $<$ half of Euclidean skew, same data | validates that the design's actual decoding rule already mitigates most of the problem |
| `test_nicdm_reduces_hubness_on_euclidean_baseline` | NICDM cuts Euclidean skew by $>50\%$ | a validated fallback correction, should raw distance ever be used |
| `test_nicdm_on_cosine_needs_reverification_on_real_embeddings` | measurement runs cleanly; **does not assert NICDM helps on top of cosine** | honest negative/inconclusive result from a from-scratch Mutual Proximity attempt that did not hold up empirically across seeds — documented rather than silently dropped |

### 4.7 `test_07_stochastic_source_theorem.py` — the core generative mechanism, no network involved

| Test | Checks | Traces to |
|---|---|---|
| `test_stochastic_source_recovers_correct_mixture` | Monte Carlo mixture over $200{,}000$ draws matches target $\lambda$ within a 6-sigma binomial tolerance | **the single most important test in the suite** — directly validates item 6's theorem against a hand-constructed ground truth, independent of any trained weights |
| `test_samples_land_exactly_on_atoms_never_on_compromise_point` | zero samples land near $(1-\lambda)a+\lambda b$ | the specific failure the fix eliminates |
| `test_deterministic_source_baseline_reproduces_the_original_failure` | explicit before/after: deterministic source always returns the same off-manifold point | makes the fix's effect visible as a concrete numeric contrast, not just an assertion |

### 4.8 `test_08_terminal_divergence.py` — near-$\tau{=}1$ numerical behavior

| Test | Checks | Traces to |
|---|---|---|
| `test_jacobian_norm_matches_linear_rate_not_cubic` | measured $\|\nabla_z v\|_{\text{op}} = 1/(1-\tau)$ via autograd, at several $\tau$ | item 10's corrected rate, computed rather than asserted |
| `test_integrated_error_is_logarithmic_not_exponential` | numeric integral matches $-\ln\delta$ to $<1\%$ | confirms the corrected, much milder error bound |
| `test_reasonable_early_stopping_margin_keeps_integration_error_small` | $\delta{=}10^{-3}$ keeps the integrated rate $<10$ | the practical payoff: no exotic architecture needed near $\tau=1$ |

---

## 5. How to run

```bash
pip install -r requirements.txt
cd tests
pytest -v -s .
```

All 31 tests currently pass against the config-matched (random-weight) architectures. Before writing the training loop:

1. Swap `conftest.py`'s two factory functions to `.from_pretrained(...)` with the real checkpoints.
2. Re-run `test_01`–`test_03` and read the printed diagnostics (norm ratio, intrinsic dimension) — these numbers, not the mock-weight numbers above, should inform $\sigma$, $d_{\text{shared}}$, and whether L2-normalization or an explicit hubness correction is actually needed on the real embedding geometry.
3. Only then proceed to assembling the full training loop from Section 2.4.

## 6. Open items

- $\sigma$ (source noise scale) is currently an unset hyperparameter — tune against the manifold-adherence and multi-answer-coverage metrics once real training data is available.
- $d_{\text{int}}(\mathcal M_T)$ must be re-measured on the real checkpoint and full answer vocabulary (test 4.2) before treating $d_{\text{shared}}{=}768$ as necessary rather than just convenient.
- Whether NICDM (or any hubness correction) is needed on top of cosine decoding is unresolved on synthetic data (test 4.6) — re-measure on the real trained candidate bank.
- $K$ (number of Reflow rounds) and the early-stopping margin $\delta$ still need a real sweep, per the base theory's §7.6 protocol.
