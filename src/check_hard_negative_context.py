"""
Post-hoc check for a TRAINED alignment-heads checkpoint: finds the CURRENT hardest
negative pair in a fresh held-out batch, in the ALIGNMENT SPACE (post h_v/h_t) -- and
compares its distance against the average (and spread) of off-diagonal, non-matching
distances in the same batch/checkpoint.

Motivated directly by an ambiguity in the training log's own tracked-pair diagnostic:
that pair was identified as "hard" using the PRE-alignment z_v_tilde/z_t_tilde space, at
step 0, when h_v/h_t were still at random init -- so its step-0 distance in the
ALIGNMENT space is an essentially arbitrary starting point, not a meaningful baseline.
A downward trend in that tracked value therefore doesn't, by itself, tell us whether the
final trained space still has poorly-separated pairs -- only a comparison against the
FINAL space's own typical off-diagonal spread can answer that.

Note: this deliberately does NOT try to replay the exact (i, j) pair tracked during the
original training run (eval_loader's shuffle=True plus a fresh process means an exact
replay isn't reliable) -- it finds the CURRENT hardest pair fresh, which is arguably the
more direct question anyway: are there STILL pairs the trained heads struggle to
separate, not specifically whether one historical pair moved.

Usage:
    python check_hard_negative_context.py \
        --base-checkpoint-path /kaggle/working/reflow_jepa_flickr30k_ckpt.pt \
        --alignment-checkpoint-path /kaggle/working/alignment_heads_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4 \
        --real-checkpoints --dataset flickr30k
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from train_alignment import (
    load_frozen_base_model,
    load_alignment_checkpoint,
    apply_finetuned_base_overrides,
    find_hard_negative_pair,
    retrieval_accuracy,
    build_dataloaders,
)


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    """tensor: (3, H, W), [0,1] float (synthetic_data.py/real_captioning_data.py's
    shared raw-pixel convention). Saved untouched -- model.encode_visual() rebinds a
    local variable when it normalizes internally, it never mutates the caller's
    tensor, so this is exactly what the model actually saw before normalization."""
    arr = (tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint-path", type=str, required=True)
    p.add_argument("--alignment-checkpoint-path", type=str, required=True)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--k-query", type=int, default=8)
    p.add_argument("--k-prefix", type=int, default=8)
    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--ema-cfm-target", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--freeze-text-encoder", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--stop-grad-cfm-target", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "flickr30k"])
    p.add_argument("--flickr-karpathy-split", type=str, default=None)
    p.add_argument("--dataset-length", type=int, default=50000)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-batches", type=int, default=5,
                    help="check across several fresh batches, not just one -- a single "
                         "batch's 'hardest pair' could itself be a fluke")
    p.add_argument("--output-dir", type=str, default="/kaggle/working/hard_pairs",
                    help="where to save the hardest-pair images + captions for each "
                         "batch, so you can directly judge whether each confusion is a "
                         "genuinely hard (near-duplicate content) case or a genuinely "
                         "wrong (semantically unrelated) one -- a distance number alone "
                         "can't tell you which.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_frozen_base_model(args, device)

    h_v, h_t, align_step, align_ckpt = load_alignment_checkpoint(
        args.alignment_checkpoint_path, d_in=768, device=device)
    align_dim = align_ckpt.get("align_dim", 256)
    print(f"[alignment checkpoint] step={align_step} align_dim={align_dim}")
    # If Phase A+ finetuning (finetune_base_lr_mult > 0) was used for this alignment
    # checkpoint, qpool/g_t_online genuinely differ from the original base checkpoint
    # -- apply the override so this check reflects the model that was ACTUALLY
    # trained, not a stale pre-finetuning snapshot. A no-op if finetuning wasn't used.
    apply_finetuned_base_overrides(model, align_ckpt)
    h_v.eval()
    h_t.eval()

    _, eval_loader = build_dataloaders(args)
    eval_iter = iter(eval_loader)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'batch':>6} {'hardest_pair_dist':>18} {'offdiag_mean':>13} {'offdiag_std':>12} "
          f"{'z_score':>9} {'diag_mean':>10} {'top1':>6} {'top5':>6}")
    all_z_scores = []
    for b in range(args.n_batches):
        images, captions = next(eval_iter)
        images = images.to(device)
        with torch.no_grad():
            c = model.task_token.expand(images.shape[0], -1)
            z_v_tilde = model.encode_visual(images, c)
            z_t_tilde = model.encode_text_online(captions)
            A_v = h_v(z_v_tilde)
            A_t = h_t(z_t_tilde)

            B = A_v.shape[0]
            sims = A_v @ A_t.T  # (B, B), sims[i,j] = cosine sim between image i, caption j
            dists = 1 - sims

            diag_mask = torch.eye(B, dtype=torch.bool, device=device)
            offdiag_dists = dists[~diag_mask]
            offdiag_mean = offdiag_dists.mean().item()
            offdiag_std = offdiag_dists.std().item()
            diag_mean = dists[diag_mask].mean().item()

            i, j, hardest_sim = find_hard_negative_pair(A_v, A_t)
            hardest_dist = 1 - hardest_sim  # find_hard_negative_pair returns similarity,
                                              # not distance -- convert to match offdiag_dists' metric
            z_score = (hardest_dist - offdiag_mean) / offdiag_std if offdiag_std > 1e-8 else float("nan")
            all_z_scores.append(z_score)

            top1, top5 = retrieval_accuracy(A_v, A_t)

        print(f"{b:>6} {hardest_dist:>18.4f} {offdiag_mean:>13.4f} {offdiag_std:>12.4f} "
              f"{z_score:>9.2f} {diag_mean:>10.4f} {top1:>6.3f} {top5:>6.3f}")

        # Save both images + both true captions involved in this batch's hardest
        # confusion, so you can directly judge: does image i actually resemble image
        # j (a genuinely hard, near-duplicate case -- not a design failure), or are
        # they visually/semantically unrelated (a real gap the loss isn't fixing)?
        # image i is being confused with caption j (captions[j], NOT its own true
        # caption captions[i]) -- that's the actual wrong association z_score flags.
        save_image_tensor(images[i], f"{args.output_dir}/batch{b}_image_i{i}.png")
        save_image_tensor(images[j], f"{args.output_dir}/batch{b}_image_j{j}.png")
        with open(f"{args.output_dir}/batch{b}_captions.txt", "w") as f:
            f.write(f"image_i (index {i})'s TRUE caption:\n  {captions[i]}\n\n")
            f.write(f"image_j (index {j})'s TRUE caption:\n  {captions[j]}\n\n")
            f.write(f"THE CONFUSION: image_i's embedding is closer to caption_j (above) "
                    f"than to its own true caption_i -- i.e. the model currently thinks "
                    f"image_i looks more like a match for image_j's caption than for its "
                    f"own.\n\n")
            f.write(f"confused-pair distance={hardest_dist:.4f}  "
                    f"(vs. typical correct-pair distance={diag_mean:.4f}, "
                    f"typical wrong-pair distance={offdiag_mean:.4f})\n")
        print(f"         -> saved batch{b}_image_i{i}.png, batch{b}_image_j{j}.png, "
              f"batch{b}_captions.txt to {args.output_dir}")
        print(f"         image_i true caption:  {captions[i]}")
        print(f"         image_j true caption:  {captions[j]}  <- image_i is wrongly closest to THIS")

    print()
    mean_z = sum(all_z_scores) / len(all_z_scores)
    print(f"mean z-score across {args.n_batches} fresh batches: {mean_z:.2f}")
    print()
    print("INTERPRETATION:")
    print("  z-score = (hardest pair's distance - off-diagonal mean) / off-diagonal std.")
    print("  This measures how many standard deviations BELOW typical the single hardest")
    print("  pair sits, in the space the trained heads actually produce.")
    print("  z close to 0 or only mildly negative (e.g. > -1.5): the hardest pair in any")
    print("    given batch is not a meaningful outlier -- just the tail of a normal")
    print("    empirical spread over a finite batch. No evidence of a real, persistent")
    print("    confusion the design failed to fix.")
    print("  z strongly negative (e.g. < -2 to -3), especially if this repeats across")
    print("    multiple fresh batches: real evidence some genuinely hard-to-separate")
    print("    confusions remain in the final trained space -- worth reporting back.")


if __name__ == "__main__":
    main()
