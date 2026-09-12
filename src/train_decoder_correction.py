"""
Phase D: decoder correction against REAL integrate() rollouts.

Making the CFM target genuinely stationary (train_phase_a_first.py + train.py
--freeze-qpool --freeze-g-t-online) satisfies Theorems 2-4's fixed-joint-law premise
exactly -- but it removes an accidental mitigation the OLD pipeline had for decoder-
side exposure bias: a constantly-drifting g_t_online used to give the decoder
incidental exposure to a neighborhood around each caption, not just its exact point.
With a frozen target, the decoder (trained via ordinary training_step/recon_loss) only
ever sees the EXACT z_t_tilde for each real caption -- never anything resembling what
the predictor's actual, imperfect integrate() output looks like at inference.

This script is the fix: freeze EVERYTHING (encoders, Q-Pool, g_t_online, and now the
predictor too), run REAL integration to get each image's actual z_hat, match it against
whichever of the image's true captions it's ACTUALLY closest to (not necessarily the
one originally paired in the dataset -- real multimodal ambiguity, same idea as
measure_exposure_bias.py's best_of_k), and fine-tune ONLY the decoder (+prefix_expand)
to produce that caption from that real z_hat. Same principle as DAgger (Ross, Gordon &
Bagnell 2011) / Scheduled Sampling (Bengio et al. 2015) -- train against your own
model's actual rollout distribution, not an idealized one -- applied at the embedding
level instead of the token level.

This is NOT a one-time fix. Every time the predictor's output distribution changes
(a new Reflow round, a sigma change, a geodesic-interpolation swap, or literally
retraining Phase 1 at all), the decoder's calibration to "what does a real z_hat look
like" goes stale and this phase needs to be re-run. Plan for that cadence, the same way
Reflow itself already requires periodic re-application.

Explicit non-negotiable: this must ONLY ever unfreeze the decoder/lm_head/
prefix_expand. If g_t_online (or qpool, or the predictor) is allowed to drift even
slightly here to make training easier, non-stationarity is reintroduced for the very
thing train_phase_a_first.py + --freeze-qpool/--freeze-g-t-online exist to fix,
undoing the whole point.

Usage:
    python train_decoder_correction.py \
        --base-checkpoint-path /kaggle/working/phase1_frozen_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --real-checkpoints \
        --visual-encoder siglip --dataset flickr30k \
        --steps 1000 --integrate-steps 50 \
        --log-path /kaggle/working/phase_d_log.json \
        --checkpoint-path /kaggle/working/phase_d_ckpt.pt
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from real_captioning_data import FlickrCaptioningDataset


class AllCaptionsDataset(torch.utils.data.Dataset):
    """Wraps a FlickrCaptioningDataset/SyntheticCaptioningDataset to return ALL of an
    image's captions per item, via the get_all_captions(idx) method added to both
    (K=5 for Flickr30k's real human captions, K=1 for synthetic's exact one-to-one
    mapping) -- instead of __getitem__'s single randomly-sampled caption, which this
    phase's best-of-k matching needs every candidate for, not just one."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]  # discard the single sampled caption -- see class docstring
        return image, self.base.get_all_captions(idx)


def collate_images_all_captions(batch):
    images = torch.stack([item[0] for item in batch])
    all_captions = [item[1] for item in batch]  # list of length B, each a list of length K
    return images, all_captions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint-path", type=str, required=True,
                    help="a train.py Phase 1 checkpoint, ideally trained with "
                         "--freeze-qpool true --freeze-g-t-online true (loading one "
                         "that wasn't is allowed -- useful as a comparison point -- "
                         "but then Phase D is correcting for BOTH non-stationarity "
                         "AND exposure bias at once, which muddies attributing any "
                         "improvement to one or the other).")
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--visual-encoder", type=str, default="ijepa", choices=["ijepa", "siglip"])
    p.add_argument("--sigma", type=float, default=0.02,
                    help="must match whatever --base-checkpoint-path was trained with "
                         "-- affects integrate()'s stochastic source draw")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever --base-checkpoint-path was trained with")

    p.add_argument("--integrate-steps", type=int, default=50,
                    help="Euler steps for the REAL integrate() rollout each batch is "
                         "scored against -- the actual inference-time path, matching "
                         "whatever n_steps you intend to use at real inference time. "
                         "Using a different step count here than at eventual real "
                         "inference reintroduces a train/inference mismatch of "
                         "exactly the kind this whole phase exists to remove.")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16,
                    help="smaller default than train.py's -- each step now also "
                         "requires a full --integrate-steps-length Euler rollout, "
                         "making this phase substantially more expensive per example")

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
    p.add_argument("--log-path", type=str, default="phase_d_log.json")
    p.add_argument("--checkpoint-path", type=str, default="phase_d_ckpt.pt")

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


def load_model(args, device):
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints, visual_encoder=args.visual_encoder,
        sigma=args.sigma, edm_precondition=args.edm_precondition,
        freeze_qpool=True, freeze_g_t_online=True,
    ).to(device)

    ckpt = torch.load(args.base_checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[phase-d] WARNING: non-strict load -- missing={missing}, "
              f"unexpected={unexpected}. Any qpool/g_t_online/task_token/encoder key "
              f"here means the architecture flags above don't match what this "
              f"checkpoint was actually built with.")

    # The one non-negotiable of this whole phase: freeze EVERYTHING except the
    # decoder/lm_head/prefix_expand. qpool/g_t_online are already frozen via
    # freeze_qpool/freeze_g_t_online above (and their task_token/encoder
    # dependencies, per reflow_jepa.py) -- the predictor is NOT frozen by any
    # existing ReflowJEPA constructor flag (Phase 1 needs it trainable), so it must
    # be frozen explicitly, here, after loading.
    for p in model.predictor.parameters():
        p.requires_grad_(False)

    return model


def build_decoder_optimizer(model, lr):
    """Deduplicates by id() across decoder/lm_head/prefix_expand -- VERIFIED
    necessary, not just defensive: T5's lm_head.weight IS decoder.embed_tokens.weight
    IS shared.weight (config.tie_word_embeddings=True by default), and iterating
    decoder.parameters() ALREADY yields that tied weight once (embed_tokens is one of
    decoder's own submodules). Naively concatenating decoder.parameters() +
    lm_head.parameters() would silently include this same tensor TWICE in one
    optimizer, updating it twice per optimizer.step() -- exactly the "corrupts Adam's
    moment estimates" bug ReflowJEPA.parameter_groups already guards against for the
    encoder/decoder split; verified this is the same failure mode here directly
    (checked id(lm_head.weight) is already present in decoder.parameters()) rather
    than assuming HF's module structure makes it a non-issue."""
    seen = set()
    decoder_params = []
    for p in (list(model.text_seq2seq.get_decoder().parameters())
              + list(model.text_seq2seq.lm_head.parameters())
              + list(model.prefix_expand.parameters())):
        if id(p) not in seen:
            decoder_params.append(p)
            seen.add(id(p))
    return torch.optim.AdamW(decoder_params, lr=lr), decoder_params


@torch.no_grad()
def best_of_k_match(model, z_hat: torch.Tensor, all_captions_per_example, device):
    """For each example, encodes all K of its true captions and returns (a) the
    caption STRING closest to that example's real z_hat (Euclidean distance -- z_hat
    is the FINAL integrate() output, not unit-normalized the way z_t_tilde is, so
    cosine similarity isn't the right comparison here), and (b) whether that best
    match was index 0 (informative diagnostic: how often is the flow landing nearer a
    DIFFERENT valid caption than the one that happens to be first in the list)."""
    B = z_hat.shape[0]
    K = len(all_captions_per_example[0])
    flat = [cap for caps in all_captions_per_example for cap in caps]
    Z1_candidates = model.encode_text_online(flat).view(B, K, -1)  # (B, K, d_shared)
    dists = (z_hat.unsqueeze(1) - Z1_candidates).norm(dim=-1)  # (B, K)
    best_dist, best_idx = dists.min(dim=1)
    best_captions = [all_captions_per_example[b][best_idx[b].item()] for b in range(B)]
    return best_captions, best_dist, best_idx


def save_checkpoint(model, args, step, path):
    torch.save({"step": step, "model_state_dict": model.state_dict(), "args": vars(args)}, path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = load_model(args, device)
    optimizer, decoder_params = build_decoder_optimizer(model, args.lr)
    n_trainable = sum(p.numel() for p in decoder_params)
    print(f"[phase-d] optimizing {n_trainable / 1e6:.1f}M params (decoder + lm_head "
          f"+ prefix_expand only, deduplicated for tied embeddings)")

    if args.dataset == "flickr30k":
        print("[phase-d] loading nlphuji/flickr30k (requires internet access -- if "
              "this hangs or fails, confirm Kaggle's Internet toggle is on)")
        base_dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_train_split,
                                                image_size=args.image_size, seed=args.seed)
        base_eval_dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_eval_split,
                                                     image_size=args.image_size, seed=999)
    else:
        base_dataset = SyntheticCaptioningDataset(length=args.dataset_length, image_size=args.image_size)
        base_eval_dataset = SyntheticCaptioningDataset(length=512, seed=999, image_size=args.image_size)

    dataset = AllCaptionsDataset(base_dataset)
    eval_dataset = AllCaptionsDataset(base_eval_dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_images_all_captions, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_images_all_captions)

    log = []
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            images, all_captions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, all_captions = next(data_iter)
        images = images.to(device)

        z_hat = model.integrate(images, n_steps=args.integrate_steps)  # no_grad (predictor frozen anyway)
        best_captions, best_dist, best_idx = best_of_k_match(model, z_hat, all_captions, device)
        recon_loss = model.decoder_recon_loss(z_hat, best_captions)

        optimizer.zero_grad()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=5.0)
        optimizer.step()

        record = {
            "step": step, "elapsed_s": time.time() - t0,
            "recon_loss": recon_loss.item(),
            "mean_best_dist": best_dist.mean().item(),
            "frac_best_idx_nonzero": (best_idx != 0).float().mean().item(),
        }

        if step % max(1, args.steps // 50) == 0:
            print(f"[step {step:5d}] recon_loss={record['recon_loss']:.4f} "
                  f"mean_best_dist={record['mean_best_dist']:.4f} "
                  f"frac_matched_non-first_caption={record['frac_best_idx_nonzero']:.3f}")

        if step % args.eval_every == 0:
            eval_images, eval_all_captions = next(iter(eval_loader))
            eval_images = eval_images.to(device)
            with torch.no_grad():
                eval_z_hat = model.integrate(eval_images, n_steps=args.integrate_steps)
                eval_best_captions, eval_best_dist, eval_best_idx = best_of_k_match(
                    model, eval_z_hat, eval_all_captions, device)
                eval_recon_loss = model.decoder_recon_loss(eval_z_hat, eval_best_captions)
            record["eval_recon_loss"] = eval_recon_loss.item()
            record["eval_mean_best_dist"] = eval_best_dist.mean().item()
            print(f"           [eval] recon_loss={record['eval_recon_loss']:.4f} "
                  f"mean_best_dist={record['eval_mean_best_dist']:.4f}")
            try:
                gen_ids = model.generate_captions(eval_images[:2], max_new_tokens=16,
                                                   n_steps=args.integrate_steps)
                decoded = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                print(f"           [sample] generated: {decoded}")
                print(f"           [sample] best-matched target was: {eval_best_captions[:2]}")
            except Exception as e:
                print(f"           [sample] generation/decode not available in this "
                      f"configuration ({e}) -- expected with the mock tokenizer, "
                      f"which has no real decode(); harmless, does not affect training")

        log.append(record)
        step += 1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, args, step, args.checkpoint_path)
            print(f"[phase-d] checkpoint saved at step {step} -> {args.checkpoint_path}")

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    save_checkpoint(model, args, step, args.checkpoint_path)
    print(f"[phase-d] done. log -> {args.log_path}, checkpoint -> {args.checkpoint_path} (step {step})")


if __name__ == "__main__":
    main()
