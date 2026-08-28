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
import os
import time

import torch
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sigma", type=float, default=0.3, help="stochastic source noise scale")
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
    p.add_argument("--dataset-length", type=int, default=50000)
    p.add_argument("--real-checkpoints", action="store_true",
                    help="use real I-JEPA/T5 checkpoints instead of config-matched random weights "
                         "(needs internet access, e.g. Kaggle with the Internet toggle on)")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--log-path", type=str, default="training_log.json")
    p.add_argument("--checkpoint-path", type=str, default="reflow_jepa_ckpt.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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
    batch = model.tokenizer(captions)
    batch = {k: v.to(images.device) for k, v in batch.items()} 
    enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
    from reflow_jepa import _mean_pool_text
    pooled = _mean_pool_text(enc_out, batch["attention_mask"])
    z_true = model.g_t_online(pooled)

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

    dataset = SyntheticCaptioningDataset(length=args.dataset_length, image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_images_captions, drop_last=True,
    )
    eval_loader = DataLoader(
        SyntheticCaptioningDataset(length=512, seed=999, image_size=args.image_size),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_images_captions,
    )

    optimizer = torch.optim.AdamW(model.parameter_groups(args.lr, args.decoder_lr_mult))

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

        cfm_loss, recon_loss, vicreg_v_loss, vicreg_t_loss, diag = model.training_step(images, captions)
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
        model.update_ema_target()

        diag["total_loss"] = total_loss.item()
        diag["step"] = step
        diag["elapsed_s"] = time.time() - t0

        if step % max(1, args.steps // 50) == 0:
            print(f"[step {step:5d}] total={diag['total_loss']:.4f} cfm={diag['cfm_loss']:.4f} "
                  f"recon={diag['recon_loss']:.4f} vicreg_v={diag['vicreg_v']:.4f} "
                  f"vicreg_t={diag['vicreg_t']:.4f}")

        if step % args.eval_every == 0:
            eval_images, eval_captions = next(iter(eval_loader))
            eval_images = eval_images.to(device)
            eval_diag = evaluate_manifold_adherence(model, eval_images, eval_captions)
            diag.update({f"eval_{k}": v for k, v in eval_diag.items()})
            grad_norms = model.gradient_norm_breakdown(images, captions)
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

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"[train] done. log -> {args.log_path}, checkpoint -> {args.checkpoint_path}")


if __name__ == "__main__":
    main()
