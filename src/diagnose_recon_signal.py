"""
Run this against your ACTUAL saved Kaggle checkpoint to check whether the
reconstruction loss reflects genuine per-example signal or an unconditional shortcut.

Two complementary checks:

1. TEACHER-FORCED matched-vs-mismatched loss (original check). Feeds the decoder the
   true embedding vs. a shuffled (wrong) one, both under teacher forcing (the decoder
   sees the TRUE previous tokens at every step, exactly as during training).

2. GENUINE AUTOREGRESSIVE generation accuracy (new). Teacher forcing has its own
   confound worth naming explicitly: with a small, deterministic, closed caption
   vocabulary (this dataset), the decoder's own self-attention over teacher-forced
   true previous tokens can memorize "these previous tokens uniquely identify caption
   #482, predict its known next token" -- achieving a good teacher-forced loss WITHOUT
   ever needing the cross-attention memory (z_t) at all. Real (non-teacher-forced)
   autoregressive decoding has no such shortcut available: the decoder only ever sees
   its OWN previous guesses, so any real accuracy requires actually using z_t. This is
   the stronger, harder-to-fool check -- if matched-embedding generation accuracy is
   high while mismatched-embedding accuracy is low, that's direct, hard-to-dispute
   evidence the model conditions genuinely on z_t; if both are similarly low (or
   similarly high), teacher-forced loss numbers alone would have been misleading.

Usage:
    python diagnose_recon_signal.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA, _mean_pool_text
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


@torch.no_grad()
def greedy_decode(model, z: torch.Tensor, max_new_tokens: int, device: torch.device) -> torch.Tensor:
    """Genuine autoregressive decoding from an explicit z (NOT derived from the flow's
    integrate() -- this isolates the decoder/prefix_expand pathway's use of z_t from
    the flow's own convergence quality, which is a separate question). No teacher
    forcing: at each step the decoder only sees its OWN previously generated tokens."""
    B = z.shape[0]
    prefix = model.prefix_expand(z)
    input_ids = torch.zeros(B, 1, dtype=torch.long, device=device)  # decoder_start_token_id
    for _ in range(max_new_tokens):
        out = model.text_seq2seq(encoder_outputs=(prefix,), decoder_input_ids=input_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids[:, 1:]  # drop the start token, align with the true 16-token caption


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever the loaded checkpoint was trained with -- "
                         "see train.py --help for what this changes")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=16, help="must match the mock "
                    "tokenizer's max_len (16) for exact-match comparison to align")
    p.add_argument("--integrate-steps", type=int, default=50,
                    help="Euler steps for the flow's own integrate() call in Check 3 "
                         "(the actual inference-time path, not a diagnostic shortcut)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints,
        edm_precondition=args.edm_precondition,
    ).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # new format: carries step count + training args for verification
        print(f"[checkpoint] step={checkpoint.get('step', '?')}  "
              f"(cross-check this against the training log's step count -- "
              f"a mismatch here means you're testing the wrong file)")
        saved_args = checkpoint.get("args", {})
        for key in ("predictor_depth", "predictor_heads", "visual_layers", "text_layers",
                    "real_checkpoints", "edm_precondition"):
            if key in saved_args and saved_args[key] != vars(args).get(key):
                print(f"[checkpoint] WARNING: saved {key}={saved_args[key]} but this "
                      f"script is using {key}={vars(args).get(key)} -- architecture mismatch "
                      f"risk, results below may not be meaningful")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # old format: raw state_dict, no metadata to verify against
        print("[checkpoint] WARNING: old-format checkpoint (no step/args metadata saved) "
              "-- cannot verify this is the checkpoint you think it is")
        model.load_state_dict(checkpoint)
    model.eval()

    ds = SyntheticCaptioningDataset(length=args.batch_size, seed=4321)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)

    batch = model.tokenizer(captions)
    batch = {k: v.to(device) for k, v in batch.items()}
    true_ids = batch["input_ids"]

    with torch.no_grad():
        enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
        z_true = F.normalize(model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"])), dim=-1)
        shuffled = z_true[torch.randperm(z_true.shape[0])]

        # --- Sanity cross-check: call the EXACT training code path directly on this
        # checkpoint. If this ALSO reports a near-baseline recon_loss despite the
        # training log showing ~0.003 at this step count, the checkpoint itself doesn't
        # match that log (wrong/stale file, or the run never reached its final
        # torch.save) -- NOT a bug in this script's manual reimplementation below.
        _, recon_loss_via_training_step, _, _, _ = model.training_step(images, captions)
        print(f"[cross-check] recon_loss via model.training_step() directly = "
              f"{recon_loss_via_training_step.item():.4f}  "
              f"(compare against the training log's recon_loss at this checkpoint's step)")
        print()

        # --- Check 1: teacher-forced loss ---
        matched_loss = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(z_true),), labels=true_ids
        ).loss
        mismatched_loss = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(shuffled),), labels=true_ids
        ).loss

        z_std = z_true.std(dim=0).mean().item()

        # --- Check 3 setup: the ACTUAL end-to-end inference path -- integrate() (the
        # trained flow, starting from noise) -> prefix_expand -> decode. Everything
        # above used the TRUE z_t and only tested the decoder in isolation; this is the
        # first check that also exercises whether the FLOW itself converges close
        # enough to produce usable captions in real inference, which is a separate
        # question decoder-only checks can't answer.
        z_hat = model.integrate(images, n_steps=args.integrate_steps)
        z_hat_norm = z_hat.norm(dim=-1).mean().item()
        z_true_norm = z_true.norm(dim=-1).mean().item()
        dist_to_true = (z_hat - z_true).norm(dim=-1).mean().item()

    # --- Check 2: genuine autoregressive generation accuracy ---
    matched_gen = greedy_decode(model, z_true, args.max_new_tokens, device)
    mismatched_gen = greedy_decode(model, shuffled, args.max_new_tokens, device)

    # --- Check 3: end-to-end, using the flow's OWN integrated output, not the true z_t ---
    flow_gen = greedy_decode(model, z_hat, args.max_new_tokens, device)

    n = min(matched_gen.shape[1], true_ids.shape[1])
    matched_tok_acc = (matched_gen[:, :n] == true_ids[:, :n]).float().mean().item()
    mismatched_tok_acc = (mismatched_gen[:, :n] == true_ids[:, :n]).float().mean().item()
    matched_seq_acc = (matched_gen[:, :n] == true_ids[:, :n]).all(dim=1).float().mean().item()
    mismatched_seq_acc = (mismatched_gen[:, :n] == true_ids[:, :n]).all(dim=1).float().mean().item()
    flow_tok_acc = (flow_gen[:, :n] == true_ids[:, :n]).float().mean().item()
    flow_seq_acc = (flow_gen[:, :n] == true_ids[:, :n]).all(dim=1).float().mean().item()

    print("=== Check 1: teacher-forced loss ===")
    print(f"matched (true embedding) recon loss     = {matched_loss.item():.4f}")
    print(f"mismatched (shuffled embedding) loss     = {mismatched_loss.item():.4f}")
    print(f"gap (mismatched - matched)               = {(mismatched_loss - matched_loss).item():.4f}")
    print(f"mean per-dim std of z_t across batch     = {z_std:.4f}  (gamma_0 target: 0.02, "
          f"full-isotropic-spread ceiling on a unit sphere: 1/sqrt(768)~0.036)")
    print()
    print("=== Check 2: genuine autoregressive generation (no teacher forcing) ===")
    print(f"matched    -- per-token acc: {matched_tok_acc:.4f}   full-sequence exact-match: {matched_seq_acc:.4f}")
    print(f"mismatched -- per-token acc: {mismatched_tok_acc:.4f}   full-sequence exact-match: {mismatched_seq_acc:.4f}")
    print()
    print("=== Check 3: end-to-end (integrate() -> prefix_expand -> decode) ===")
    print(f"mean ||z_hat|| (flow output, {args.integrate_steps} Euler steps) = {z_hat_norm:.4f}   "
          f"mean ||z_true|| = {z_true_norm:.4f}  "
          f"({'norm has drifted noticeably from the target -- see note below' if abs(z_hat_norm - z_true_norm) > 0.1 else 'norms close, drift is not the issue here'})")
    print(f"mean dist(z_hat, z_true)               = {dist_to_true:.4f}")
    print(f"flow-based generation -- per-token acc: {flow_tok_acc:.4f}   full-sequence exact-match: {flow_seq_acc:.4f}")
    print(f"  (compare against Check 2's matched={matched_tok_acc:.4f} -- the gap between them is "
          f"exactly what the flow's imperfect convergence costs you in real inference, isolated from "
          f"whether the decoder itself works, which Check 2 already answered)")
    print()

    teacher_forced_gap_small = (mismatched_loss - matched_loss).item() < 0.05
    generation_gap_small = (matched_tok_acc - mismatched_tok_acc) < 0.05

    if generation_gap_small:
        print("WARNING: matched and mismatched embeddings give similar AUTOREGRESSIVE "
              "generation accuracy -- this is the stronger check, and it says the "
              "decoder is not really using z_t during real generation, regardless of "
              "what the teacher-forced loss looked like. If Check 1 looked good but "
              "Check 2 doesn't, that's the teacher-forcing memorization shortcut "
              "described in this file's docstring.")
    else:
        print("Matched embedding gives meaningfully higher generation accuracy than "
              "mismatched -- the decoder IS using per-example embedding information "
              "under genuine (non-teacher-forced) generation, the strongest form of "
              "this check.")
    if teacher_forced_gap_small and not generation_gap_small:
        print("\nNote: Check 1's gap was small but Check 2's wasn't -- teacher-forced "
              "loss alone would have been misleading here; good thing to have both.")


if __name__ == "__main__":
    main()

