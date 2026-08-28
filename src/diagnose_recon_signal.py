"""
Run this against your ACTUAL saved Kaggle checkpoint to check whether the
reconstruction loss reflects genuine per-example signal or an unconditional shortcut.

Motivated directly by a real training run: vicreg_t stayed pinned near its collapse
maximum for most of 3000 steps while recon_loss plateaued at ~0.3 -- suspiciously close
to ln(100)/16 ~= 0.288, the entropy floor for guessing uniformly among this synthetic
dataset's 100 possible caption strings. test_11 confirmed the mechanism at toy scale
(30 steps: matched=0.2894 vs mismatched=0.2906, essentially identical). This script
runs the same comparison against your real checkpoint, at real scale.

Usage:
    python diagnose_recon_signal.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4

If matched and mismatched losses are close on the REAL checkpoint too, the decoder
learned an unconditional caption LM over the small closed vocabulary rather than a
genuine image-conditioned mapping -- the fix is almost certainly a richer synthetic
vocabulary (more colors/shapes/positions/attributes, or per-image unique details),
not more training steps on the same 100-caption dataset.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA, _mean_pool_text
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    ds = SyntheticCaptioningDataset(length=args.batch_size, seed=4321)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)

    batch = model.tokenizer(captions)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
        z_true = model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"]))

        matched = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(z_true),), labels=batch["input_ids"]
        ).loss
        shuffled = z_true[torch.randperm(z_true.shape[0])]
        mismatched = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(shuffled),), labels=batch["input_ids"]
        ).loss

        # also: what does per-dimension std actually look like at this checkpoint?
        z_std = z_true.std(dim=0).mean().item()

    print(f"matched (true embedding) recon loss   = {matched.item():.4f}")
    print(f"mismatched (shuffled embedding) loss   = {mismatched.item():.4f}")
    print(f"gap (mismatched - matched)             = {(mismatched - matched).item():.4f}")
    print(f"mean per-dim std of z_t across batch   = {z_std:.4f}  (gamma_0 target: 1.0)")
    print()
    if (mismatched - matched).item() < 0.05:
        print("WARNING: matched and mismatched losses are nearly identical -- the decoder "
              "does not appear to be using per-example embedding information. Recon_loss's "
              "plateau is likely an unconditional-LM shortcut over the small closed "
              "caption vocabulary, not evidence of genuine image-conditioned generation.")
    else:
        print("Matched loss is meaningfully lower than mismatched -- the decoder IS using "
              "per-example embedding information.")


if __name__ == "__main__":
    main()
