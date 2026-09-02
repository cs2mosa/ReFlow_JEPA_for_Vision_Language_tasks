"""
Phase B: synthetic-distribution training loop (DESIGN.md §2.4 Phase 1 -- base CFM),
implementing Algorithm 1 (line 4: train v_theta to convergence) from
reflow_jepa_proof1.pdf, plus the decoder-reconstruction term Pipeline 2 needs (see
ReflowJEPA.training_step's docstring for why that term exists and isn't in the
original design doc).

This validates the layer of testing above the 43-test structural/theoretical suite:
does a REAL, finite, gradient-trained network behave the way the theory predicts, on
data with genuine (if synthetic) learnable structure -- not does the exact
population-optimal field behave correctly (already checked, test_07/test_07b), and not
do the wired-together modules have the right shapes (already checked, tests 01-10).

Usage:
    python train.py --steps 2000 --batch-size 32 --real-checkpoints
    python train.py --steps 200 --batch-size 8   # fast CPU smoke test, mock weights
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from real_captioning_data import FlickrCaptioningDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"],
                    help="'cosine' (default): linear warmup then cosine decay to "
                         "--min-lr-ratio * lr. Motivated directly by a real run: cfm_loss "
                         "plateaued flat and noisy from ~step 800 onward at a constant LR "
                         "-- the classic signature of an LR too large to settle further, "
                         "not too small to ever converge. 'constant' keeps the old behavior.")
    p.add_argument("--warmup-steps", type=int, default=100,
                    help="linear LR warmup steps before the schedule kicks in (ignored if "
                         "--lr-schedule constant)")
    p.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="cosine decay floor, as a fraction of --lr (not decayed to exactly "
                         "0 -- leaves the network some ability to keep adjusting late in "
                         "training rather than effectively freezing)")
    p.add_argument("--sigma", type=float, default=0.02,
                    help="stochastic source noise scale. Recalibrated for L2-normalized "
                         "(unit-sphere) embeddings: at sigma=0.02, E[||sigma*eps||]~0.55 "
                         "vs signal norm 1.0. The old default (0.3) was tuned before "
                         "L2-norm was wired in and would now make noise ~8x larger than "
                         "signal.")
    p.add_argument("--vicreg-gamma", type=float, default=0.02,
                    help="VICReg target per-dimension std (gamma_0). Recalibrated for "
                         "unit-sphere embeddings: max possible per-dim std for "
                         "isotropically-spread 768-dim unit vectors is 1/sqrt(768)~0.036; "
                         "0.02 targets ~55%% of that, leaving some slack. The old default "
                         "(1.0) is unreachable on a unit sphere and would make VICReg "
                         "look permanently collapsed regardless of actual embedding health.")
    p.add_argument("--vicreg-warmup-steps", type=int, default=0,
                    help="if >0, vicreg_v/vicreg_t weights are multiplied by "
                         "--vicreg-warmup-mult for this many steps, then linearly "
                         "decayed to their base value. Standard collapse-avoidance "
                         "pattern (VICReg/BYOL/DINO): let anti-collapse pressure "
                         "dominate before cfm_loss's regression pressure (which grows "
                         "once representations stop collapsing, per an observed real "
                         "run) has a chance to compete on comparable footing.")
    p.add_argument("--vicreg-warmup-mult", type=float, default=8.0)
    p.add_argument("--vicreg-v-weight", type=float, default=None,
                    help="overrides --vicreg-weight for the visual term only, if set")
    p.add_argument("--vicreg-t-weight", type=float, default=None,
                    help="overrides --vicreg-weight for the text term only, if set")
    p.add_argument("--vicreg-weight", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=2, help="ignored if --real-checkpoints")
    p.add_argument("--text-layers", type=int, default=2, help="ignored if --real-checkpoints")
    p.add_argument("--freeze-text-encoder", type=lambda x: x.lower() != "false", default=True,
                    help="freeze the T5 encoder entirely (recommended default -- see "
                         "reflow_jepa.py's __init__ docstring for why). Pass "
                         "--freeze-text-encoder false to keep it trainable at "
                         "--decoder-lr-mult instead of a hard freeze.")
    p.add_argument("--decoder-lr-mult", type=float, default=0.1,
                    help="LR multiplier for the T5 decoder (and, if not frozen, the "
                         "text encoder) relative to --lr. The decoder cannot be fully "
                         "frozen with mock weights without breaking its ability to "
                         "learn at all -- this is the 'very little LR' alternative.")
    p.add_argument("--stop-grad-cfm-target", type=lambda x: x.lower() != "false", default=True,
                    help="detach Z_1 before it enters the CFM regression loss (recommended "
                         "default). See reflow_jepa.py's training_step docstring: a real "
                         "run showed cfm_loss's gradient on the text projection (~77) "
                         "completely swamping VICReg's counter-pressure (~0.07), actively "
                         "driving collapse regardless of loss weight.")
    p.add_argument("--k-query", type=int, default=8)
    p.add_argument("--k-prefix", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "flickr30k"],
                    help="'synthetic' (default): the procedural shape/color dataset used "
                         "throughout this project so far, exact one-to-one image->caption "
                         "mapping. 'flickr30k': nlphuji/flickr30k (~31k real photos, 5 human "
                         "captions each) -- requires internet access (Kaggle Internet toggle "
                         "on); the point where genuine multimodal ambiguity (multiple valid "
                         "captions per image) becomes testable, unlike the synthetic set's "
                         "exact mapping.")
    p.add_argument("--flickr-karpathy-split", type=str, default=None,
                    help="optional filter on flickr30k's own internal train/val/test 'split' "
                         "column (Karpathy partition). None (default): use every row.")
    p.add_argument("--dataset-length", type=int, default=50000,
                    help="only used when --dataset synthetic; flickr30k's length is fixed by "
                         "the real dataset (~31k images)")
    p.add_argument("--real-checkpoints", action="store_true",
                    help="use real I-JEPA/T5 checkpoints instead of config-matched random weights "
                         "(needs internet access, e.g. Kaggle with the Internet toggle on)")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True,
                    help="EDM-style predictor reparametrization (default True): predict "
                         "a bounded target-embedding estimate and compute velocity "
                         "analytically as (z1_hat - z_tau)/(1-tau), baking the 1/(1-tau) "
                         "terminal divergence into the architecture rather than relying "
                         "on gradient descent to learn it. Added after "
                         "measure_terminal_divergence.py found the raw-velocity "
                         "architecture hadn't reliably learned this on a real "
                         "checkpoint. Pass --edm-precondition false to revert to the "
                         "original architecture for comparison.")
    p.add_argument("--ema-cfm-target", type=lambda x: x.lower() != "false", default=False,
                    help="Use the EMA target copy of g_T (already built for retrieval-"
                         "eval, see encode_text_target) as cfm_loss's regression target "
                         "instead of a detached snapshot of the constantly-shifting "
                         "online g_T. Added after a real decoder_lr_mult sweep on "
                         "Flickr30k (0.1 vs 0.02 vs frozen) suggested predictor "
                         "convergence was sensitive to how much recon_loss's gradient "
                         "moves g_T per step -- this targets that mechanism directly "
                         "rather than relying on a decoder_lr_mult value that happens "
                         "to reduce it as a side effect. Default False preserves "
                         "existing behavior exactly.")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--log-path", type=str, default="training_log.json")
    p.add_argument("--checkpoint-every", type=int, default=250,
                    help="save a checkpoint every N steps, in addition to the final "
                         "save. Fixes a real gap: the previous version only saved once, "
                         "at the very end of the loop -- a Kaggle session timeout or "
                         "disconnect before that line left only a stale checkpoint from "
                         "a much earlier (or unrelated) run, with no error to signal it.")
    p.add_argument("--checkpoint-path", type=str, default="reflow_jepa_ckpt.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_checkpoint(model, args, step, path):
    """Saves model weights WITH metadata (step count, args) so a future load can verify
    what it's actually looking at, instead of silently trusting a possibly-stale file."""
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, path)


def build_lr_scheduler(optimizer, args):
    if args.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_model(args, device):
    model = ReflowJEPA(
        k_query=args.k_query,
        k_prefix=args.k_prefix,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers,
        text_layers=args.text_layers,
        sigma=args.sigma,
        ema_momentum=args.ema_momentum,
        real_checkpoints=args.real_checkpoints,
        edm_precondition=args.edm_precondition,
        ema_cfm_target=args.ema_cfm_target,
        freeze_text_encoder=args.freeze_text_encoder,
        stop_grad_cfm_target=args.stop_grad_cfm_target,
    ).to(device)
    return model


@torch.no_grad()
def evaluate_manifold_adherence(model, images, captions, n_steps=30, eps_frac=0.15):
    """Empirical counterpart to Theorem 2's manifold-adherence claim (DESIGN.md's own
    Conclusion, item 1, explicitly flags this as NOT guaranteed to survive finite
    capacity/training -- this metric is exactly how you'd notice if it doesn't).
    Pr[dist(z_hat_t, M_T) <= eps].

    eps is set as a FIXED fraction of the batch's mean target embedding norm, not as a
    quantile of pairwise target-target distances. An earlier version used the latter and
    is worth naming as a real bug found in practice: when the text embeddings collapse
    (vicreg_t stuck near its penalty maximum, as happened in a real training run), the
    pairwise-distance-based eps collapses right along with them, making the metric
    self-referentially unreadable -- it can't distinguish "the flow is failing" from
    "the targets aren't diverse enough to calibrate a meaningful eps against" using data
    from the same collapsed batch. Anchoring eps to embedding norm instead keeps the bar
    fixed regardless of how spread out (or not) the targets currently are.
    """
    z_hat = model.integrate(images, n_steps=n_steps)
    batch = model.tokenizer(captions, return_tensors="pt", padding=True)
    batch = {k: v.to(images.device) for k, v in batch.items()}  # tokenizer always returns
                                                                  # CPU tensors -- see
                                                                  # ReflowJEPA._tokenize's
                                                                  # docstring for why this
                                                                  # is needed at every call site
    enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
    from reflow_jepa import _mean_pool_text
    pooled = _mean_pool_text(enc_out, batch["attention_mask"])
    z_true = F.normalize(model.g_t_online(pooled), dim=-1)  # match training: L2-normalized

    dist_to_own_target = (z_hat - z_true).norm(dim=-1)
    eps = eps_frac * z_true.norm(dim=-1).mean().item()
    adherence_rate = (dist_to_own_target <= eps).float().mean().item()
    return {
        "manifold_adherence_rate": adherence_rate,
        "mean_dist_to_true_target": dist_to_own_target.mean().item(),
        "eps_used": eps,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[train] device={device} real_checkpoints={args.real_checkpoints}")

    model = build_model(args, device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"[train] trainable params: {n_trainable / 1e6:.1f}M")

    if args.dataset == "flickr30k":
        print("[train] loading nlphuji/flickr30k (requires internet access -- if this "
              "hangs or fails, confirm Kaggle's Internet toggle is on)")
        dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_karpathy_split,
                                           image_size=args.image_size, seed=args.seed)
        eval_dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_karpathy_split,
                                                image_size=args.image_size, seed=999)
    else:
        dataset = SyntheticCaptioningDataset(length=args.dataset_length, image_size=args.image_size)
        eval_dataset = SyntheticCaptioningDataset(length=512, seed=999, image_size=args.image_size)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_images_captions, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_images_captions,
    )

    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr, args.decoder_lr_mult))
    scheduler = build_lr_scheduler(optimizer, args)

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

        cfm_loss, recon_loss, vicreg_v_loss, vicreg_t_loss, diag = model.training_step(
            images, captions, vicreg_gamma=args.vicreg_gamma)
        vicreg_v_weight = args.vicreg_v_weight if args.vicreg_v_weight is not None else args.vicreg_weight
        vicreg_t_weight = args.vicreg_t_weight if args.vicreg_t_weight is not None else args.vicreg_weight
        if args.vicreg_warmup_steps > 0 and step < args.vicreg_warmup_steps:
            warmup_frac = 1.0 - step / args.vicreg_warmup_steps  # 1 -> 0 over warmup
            mult = 1.0 + (args.vicreg_warmup_mult - 1.0) * warmup_frac
            vicreg_v_weight *= mult
            vicreg_t_weight *= mult
        total_loss = (cfm_loss + args.recon_weight * recon_loss
                      + vicreg_v_weight * vicreg_v_loss + vicreg_t_weight * vicreg_t_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        model.update_ema_target()

        diag["total_loss"] = total_loss.item()
        diag["step"] = step
        diag["elapsed_s"] = time.time() - t0
        diag["lr"] = scheduler.get_last_lr()[0]  # core param group's current LR

        if step % max(1, args.steps // 50) == 0:
            print(f"[step {step:5d}] lr={diag['lr']:.2e} total={diag['total_loss']:.4f} cfm={diag['cfm_loss']:.4f} "
                  f"recon={diag['recon_loss']:.4f} vicreg_v={diag['vicreg_v']:.4f} "
                  f"vicreg_t={diag['vicreg_t']:.4f}")

        if step % args.eval_every == 0:
            eval_images, eval_captions = next(iter(eval_loader))
            eval_images = eval_images.to(device)
            eval_diag = evaluate_manifold_adherence(model, eval_images, eval_captions)
            diag.update({f"eval_{k}": v for k, v in eval_diag.items()})
            grad_norms = model.gradient_norm_breakdown(images, captions, vicreg_gamma=args.vicreg_gamma)
            diag["grad_norm_breakdown"] = grad_norms
            print(f"           [eval] adherence_rate={eval_diag['manifold_adherence_rate']:.3f} "
                  f"mean_dist={eval_diag['mean_dist_to_true_target']:.4f}")
            print(f"           [grad-text] cfm={grad_norms['cfm_on_text']:.4f} "
                  f"recon={grad_norms['recon_on_text']:.4f} vicreg_t={grad_norms['vicreg_t_on_text']:.4f} "
                  f"(weighted: {grad_norms['vicreg_t_on_text'] * vicreg_t_weight:.4f})")
            print(f"           [grad-visual] cfm={grad_norms['cfm_on_visual']:.4f} "
                  f"vicreg_v={grad_norms['vicreg_v_on_visual']:.4f} "
                  f"(weighted: {grad_norms['vicreg_v_on_visual'] * vicreg_v_weight:.4f})")

        log.append(diag)
        step += 1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, args, step, args.checkpoint_path)
            print(f"[train] checkpoint saved at step {step} -> {args.checkpoint_path}")

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    save_checkpoint(model, args, step, args.checkpoint_path)
    print(f"[train] done. log -> {args.log_path}, checkpoint -> {args.checkpoint_path} (step {step})")


if __name__ == "__main__":
    main()
