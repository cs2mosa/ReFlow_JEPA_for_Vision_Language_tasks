"""
Phase A-FIRST: trains Q-Pool and g_t_online directly against the supervised
Sinkhorn+VICReg cross-modal objective, with NO predictor or decoder involved at all --
"Option B" from conversation notes on making the CFM target genuinely stationary.

This is a deliberate REORDERING of the original pipeline (train.py Phase 1 -> Reflow ->
train_alignment.py Phase A), not a variant of the existing train_alignment.py. The
original Phase A was bolted on AFTER Phase 1 already existed, specifically because
modifying Q-Pool/g_t_online directly at that point would have destroyed an
already-CFM-trained representation -- hence the separate h_v/h_t heads there. Run
FIRST, there is nothing yet to protect: Q-Pool/g_t_online THEMSELVES are what should
end up aligned, since they are what the flow gets built on top of afterward.

The point: once this converges and its checkpoint is loaded into train.py with
--freeze-qpool true --freeze-g-t-online true, the predictor trains against a target
that satisfies Theorems 2-4's fixed-joint-law premise EXACTLY (verified directly, not
just argued -- see reflow_jepa.py's freeze_qpool/task_token handling), not just
approximately the way --stop-grad-cfm-target / --ema-cfm-target could manage. See also
the conversation's follow-up: this alone does NOT fix decoder-side exposure bias --
if anything it removes an accidental mitigation (the old pipeline's constantly-drifting
g_t_online gave the decoder incidental exposure to a neighborhood around each caption,
not just its exact point). That is what the planned Phase D (decoder correction against
real integrate() rollouts) exists to address, run AFTER train.py's Phase 1, not here.

train_alignment.py's original (downstream) Phase A remains fully valid and unchanged
for anyone still investigating the OLD ordering -- this is an additional, alternative
pipeline stage, not a replacement.

Usage:
    python train_phase_a_first.py --steps 3000 --batch-size 32 --real-checkpoints \
        --visual-encoder siglip --dataset flickr30k \
        --log-path /kaggle/working/phase_a_first_log.json \
        --checkpoint-path /kaggle/working/phase_a_first_ckpt.pt
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from real_captioning_data import FlickrCaptioningDataset
from train_alignment import retrieval_accuracy, find_hard_negative_pair


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    p.add_argument("--sinkhorn-iters", type=int, default=100)
    p.add_argument("--vicreg-gamma", type=float, default=0.02)
    p.add_argument("--vicreg-v-weight", type=float, default=5.0)
    p.add_argument("--vicreg-t-weight", type=float, default=10.0)

    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--visual-encoder", type=str, default="ijepa", choices=["ijepa", "siglip"])
    p.add_argument("--freeze-text-encoder", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--predictor-depth", type=int, default=6,
                    help="only affects the (unused-in-this-phase, re-trained-from-"
                         "scratch-in-Phase-1-regardless) predictor's initial shape -- "
                         "kept configurable purely so this checkpoint's predictor "
                         "shape matches train.py's expectation when loaded, avoiding "
                         "spurious shape-mismatch warnings on load. Its WEIGHTS here "
                         "are never trained and are not meant to be used.")
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)

    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "flickr30k"])
    p.add_argument("--flickr-train-split", type=str, default="train")
    p.add_argument("--flickr-eval-split", type=str, default="val")
    p.add_argument("--dataset-length", type=int, default=50000,
                    help="only used when --dataset synthetic")
    p.add_argument("--image-size", type=int, default=None,
                    help="default (None): auto-derived from --visual-encoder (224 for "
                         "ijepa, 384 for siglip). Pass explicitly only to override.")

    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-path", type=str, default="phase_a_first_log.json")
    p.add_argument("--checkpoint-path", type=str, default="phase_a_first_ckpt.pt")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.image_size is None:
        from encoders import VISUAL_ENCODER_SPECS
        args.image_size = VISUAL_ENCODER_SPECS[args.visual_encoder][2]
    if args.flickr_train_split == "None":
        args.flickr_train_split = None
    if args.flickr_eval_split == "None":
        args.flickr_eval_split = None
    return args


def build_model(args, device):
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints,
        visual_encoder=args.visual_encoder,
        freeze_text_encoder=args.freeze_text_encoder,
        # freeze_qpool/freeze_g_t_online deliberately NOT set here (default False) --
        # this phase's entire purpose is training them. train.py is what loads this
        # checkpoint back in WITH those flags set True.
    ).to(device)
    return model


def save_checkpoint(model, args, step, path):
    torch.save({"step": step, "model_state_dict": model.state_dict(), "args": vars(args)}, path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = build_model(args, device)

    if args.dataset == "flickr30k":
        print("[phase-a-first] loading nlphuji/flickr30k (requires internet access -- "
              "if this hangs or fails, confirm Kaggle's Internet toggle is on)")
        print(f"[phase-a-first] flickr30k train split filter={args.flickr_train_split!r}, "
              f"eval split filter={args.flickr_eval_split!r}")
        dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_train_split,
                                           image_size=args.image_size, seed=args.seed)
        eval_dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_eval_split,
                                                image_size=args.image_size, seed=999)
    else:
        dataset = SyntheticCaptioningDataset(length=args.dataset_length, image_size=args.image_size)
        eval_dataset = SyntheticCaptioningDataset(length=512, seed=999, image_size=args.image_size)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_images_captions, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_images_captions)

    # Only the params THIS phase should ever move -- deliberately NOT
    # model.parameter_groups() (that includes the predictor/prefix_expand/decoder,
    # none of which alignment_training_step ever computes a gradient for anyway, but
    # explicit is safer than relying on "AdamW just won't update params with no
    # grad" -- e.g. weight decay alone can still shrink a never-trained param toward
    # 0 with some optimizer configurations).
    trainable = (list(model.qpool.parameters()) + list(model.g_t_online.parameters())
                 + [model.task_token])
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[phase-a-first] optimizing {n_trainable / 1e6:.2f}M params (qpool + "
          f"g_t_online + task_token only -- predictor/prefix_expand/decoder still "
          f"nominally have requires_grad=True at this point, since freeze_qpool/"
          f"freeze_g_t_online only get set True when THIS checkpoint is loaded back "
          f"into train.py, but they are correctly excluded from THIS optimizer, so "
          f"they never move here regardless)")

    log = []
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            images, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, captions = next(data_iter)
        images = images.to(device)

        align_loss, vicreg_v_loss, vicreg_t_loss, diag = model.alignment_training_step(
            images, captions, sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iters=args.sinkhorn_iters, vicreg_gamma=args.vicreg_gamma,
        )
        total_loss = (align_loss + args.vicreg_v_weight * vicreg_v_loss
                      + args.vicreg_t_weight * vicreg_t_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()

        record = {
            "step": step, "elapsed_s": time.time() - t0,
            "total_loss": total_loss.item(), "alignment_loss": align_loss.item(),
            "vicreg_v": vicreg_v_loss.item(), "vicreg_t": vicreg_t_loss.item(),
        }

        if step % max(1, args.steps // 50) == 0:
            print(f"[step {step:5d}] total={record['total_loss']:.4f} "
                  f"align={record['alignment_loss']:.4f} vicreg_v={record['vicreg_v']:.4f} "
                  f"vicreg_t={record['vicreg_t']:.4f}")

        if step % args.eval_every == 0:
            eval_images, eval_captions = next(iter(eval_loader))
            eval_images = eval_images.to(device)
            with torch.no_grad():
                c = model.task_token.expand(eval_images.shape[0], -1)
                z_v = model.encode_visual(eval_images, c)
                z_t = model.encode_text_online(eval_captions)
                top1, top5 = retrieval_accuracy(z_v, z_t)
                i, j, hard_sim = find_hard_negative_pair(z_v, z_t)
            record["eval_retrieval_top1"] = top1
            record["eval_retrieval_top5"] = top5
            record["eval_hard_negative_sim"] = hard_sim
            print(f"           [eval] top1={top1:.3f} top5={top5:.3f} "
                  f"hardest_pair_sim={hard_sim:.4f} (images {i},{j})")

        log.append(record)
        step += 1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, args, step, args.checkpoint_path)
            print(f"[phase-a-first] checkpoint saved at step {step} -> {args.checkpoint_path}")

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    save_checkpoint(model, args, step, args.checkpoint_path)
    print(f"[phase-a-first] done. log -> {args.log_path}, checkpoint -> {args.checkpoint_path} "
          f"(step {step})")
    print(f"[phase-a-first] next: python train.py ... --freeze-qpool true "
          f"--freeze-g-t-online true --base-checkpoint-path {args.checkpoint_path} "
          f"(see train.py --help for the loading flag -- added alongside this script)")


if __name__ == "__main__":
    main()
