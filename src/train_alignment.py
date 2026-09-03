"""
train_alignment.py -- Phase A standalone training script (alignment_stage_spec.md
sec 5): trains h_v/h_t (alignment_heads.py) via the supervised Sinkhorn alignment
loss, on top of a fully-trained, fully-FROZEN Reflow-JEPA checkpoint. Separate from
train.py by design (spec sec 3): this script never constructs a fresh ReflowJEPA to
train end-to-end, only to LOAD an existing checkpoint as a frozen feature source.

Mirrors this project's existing diagnostic-script conventions (measure_terminal_
divergence.py, measure_exposure_bias.py, diagnose_recon_signal.py): argparse CLI,
base-model architecture flags to reconstruct the checkpoint's shape, non-strict
`load_state_dict` with an explicit mismatch warning, `--real-checkpoints` /
`--dataset` selection identical to train.py's.

Usage (Kaggle, real checkpoint -- see spec sec 8):
    python train_alignment.py --base-checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \\
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4 \\
        --real-checkpoints --align-dim 256 --sinkhorn-epsilon 0.05 --sinkhorn-iters 50 \\
        --vicreg-weight 5.0 --vicreg-gamma 0.02 --steps 3000 --batch-size 64 --lr 1e-4 \\
        --eval-every 100 --checkpoint-every 250 --device cuda

Usage (local smoke test, mock weights, tiny/fast -- spec sec 2's required "smoke test
end-to-end locally" step before ever touching Kaggle):
    python train_alignment.py --base-checkpoint-path tiny_base_ckpt.pt \\
        --visual-layers 1 --text-layers 1 --predictor-depth 2 --predictor-heads 4 \\
        --align-dim 32 --steps 50 --batch-size 8 --dataset-length 64 --eval-every 10 \\
        --checkpoint-every 25 --device cpu
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from real_captioning_data import FlickrCaptioningDataset
from alignment_heads import (
    AlignmentHead,
    alignment_loss,
    sinkhorn_log_domain,
    sinkhorn_marginal_error,
    vicreg_variance_penalty,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint-path", type=str, required=True,
                    help="path to a ReflowJEPA checkpoint (train.py's save_checkpoint "
                         "format: {'step', 'model_state_dict', 'args'}) to use as the "
                         "frozen feature source for this entire phase (spec sec 3/5).")

    # Base-model architecture flags -- must match whatever the loaded checkpoint was
    # trained with (same convention/warning as measure_terminal_divergence.py etc.).
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=2, help="ignored if --real-checkpoints")
    p.add_argument("--text-layers", type=int, default=2, help="ignored if --real-checkpoints")
    p.add_argument("--k-query", type=int, default=8)
    p.add_argument("--k-prefix", type=int, default=8)
    p.add_argument("--real-checkpoints", action="store_true",
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--ema-cfm-target", type=lambda x: x.lower() != "false", default=False,
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--freeze-text-encoder", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--stop-grad-cfm-target", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--sigma", type=float, default=0.02,
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--ema-momentum", type=float, default=0.996,
                    help="must match whatever the loaded checkpoint was trained with")

    # Phase A hyperparameters (spec sec 5 point 5)
    p.add_argument("--align-dim", type=int, default=256,
                    help="alignment subspace dim, d_align (spec sec 4.1)")
    p.add_argument("--sinkhorn-epsilon", type=float, default=0.05,
                    help="entropy regularization for sinkhorn_log_domain (spec sec 4.2)")
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument("--vicreg-weight", type=float, default=1.0,
                    help="starting point: same order of magnitude as train.py's "
                         "--vicreg-weight default (spec sec 4.4); tune if collapse is "
                         "observed (sec 6 diagnostic 3)")
    p.add_argument("--vicreg-gamma", type=float, default=0.02,
                    help="VICReg target per-dimension std (gamma_0), same semantics as "
                         "train.py's --vicreg-gamma")

    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-path", type=str, default="alignment_log.json")
    p.add_argument("--checkpoint-path", type=str, default="alignment_heads_ckpt.pt")

    # Data (mirrors train.py's --dataset selection exactly)
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "flickr30k"])
    p.add_argument("--flickr-karpathy-split", type=str, default=None)
    p.add_argument("--dataset-length", type=int, default=50000,
                    help="only used when --dataset synthetic")
    p.add_argument("--image-size", type=int, default=224)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frozen base model loading (spec sec 5 point 1: model.eval(), requires_grad_(False)
# on every existing parameter -- the base model is a frozen feature source for this
# entire phase)
# ---------------------------------------------------------------------------
def load_frozen_base_model(args, device) -> ReflowJEPA:
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

    checkpoint = torch.load(args.base_checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f"[base checkpoint] step={checkpoint.get('step', '?')}")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[base checkpoint] WARNING: non-strict load -- missing={missing}, "
                  f"unexpected={unexpected} (likely an architecture-flag mismatch "
                  f"between this script's CLI args and the checkpoint's own training "
                  f"args -- verify results are still meaningful, or check "
                  f"checkpoint['args'] against the flags above)")
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    model.requires_grad_(False)  # frozen feature source, spec sec 3 -- entire phase
    return model


# ---------------------------------------------------------------------------
# Checkpointing (Phase A's own -- h_v/h_t only, distinct file from the base model's)
# ---------------------------------------------------------------------------
def save_alignment_checkpoint(h_v, h_t, args, step, path):
    """Same step-count + args-metadata pattern as train.py's save_checkpoint, plus
    the specific architectural flag that changes what this checkpoint's tensors MEAN
    (align_dim -- analogous to edm_precondition/ema_cfm_target's treatment in
    reflow_jepa.py, per spec sec 2's checkpoint-metadata-versioning lesson)."""
    torch.save({
        "step": step,
        "h_v_state_dict": h_v.state_dict(),
        "h_t_state_dict": h_t.state_dict(),
        "align_dim": args.align_dim if hasattr(args, "align_dim") else h_v.net[-1].out_features,
        "args": vars(args) if not isinstance(args, dict) else args,
    }, path)


def load_alignment_checkpoint(path, d_in=768, device="cpu"):
    """Non-strict load, mirrors train.py-family convention. Returns (h_v, h_t, step,
    checkpoint_dict). Warns (does not raise) on an align_dim mismatch between the
    checkpoint's metadata and the reconstructed heads' actual out_features, since
    that specific flag changes what the checkpoint's tensors mean (spec sec 2)."""
    checkpoint = torch.load(path, map_location=device)
    align_dim = checkpoint.get("align_dim", 256)
    h_v = AlignmentHead(d_in=d_in, d_align=align_dim).to(device)
    h_t = AlignmentHead(d_in=d_in, d_align=align_dim).to(device)

    missing_v, unexpected_v = h_v.load_state_dict(checkpoint["h_v_state_dict"], strict=False)
    missing_t, unexpected_t = h_t.load_state_dict(checkpoint["h_t_state_dict"], strict=False)
    if missing_v or unexpected_v or missing_t or unexpected_t:
        print(f"[alignment checkpoint] WARNING: non-strict load -- "
              f"h_v missing={missing_v} unexpected={unexpected_v}; "
              f"h_t missing={missing_t} unexpected={unexpected_t} "
              f"(likely an align_dim mismatch: checkpoint metadata says "
              f"align_dim={align_dim}; verify this matches what you intended to load)")
    return h_v, h_t, checkpoint.get("step", 0), checkpoint


# ---------------------------------------------------------------------------
# Section 6 diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def retrieval_accuracy(A_v: torch.Tensor, A_t: torch.Tensor) -> tuple[float, float]:
    """Diagnostic 1: batch-internal retrieval accuracy. For each image query A_v[i],
    rank all B captions by cosine similarity, check whether the true caption A_t[i]
    is top-1 (and separately top-5). A_v/A_t already unit-norm, so dot product IS
    cosine similarity."""
    B = A_v.shape[0]
    sims = A_v @ A_t.T  # (B, B), sims[i, j] = cos(A_v[i], A_t[j])
    ranks = sims.argsort(dim=1, descending=True)  # (B, B), best-to-worst caption idx per row
    true_idx = torch.arange(B, device=A_v.device).unsqueeze(1)
    match_position = (ranks == true_idx).float().argmax(dim=1)  # 0-indexed rank of the true match
    top1 = (match_position == 0).float().mean().item()
    top5 = (match_position < min(5, B)).float().mean().item()
    return top1, top5


@torch.no_grad()
def find_hard_negative_pair(z_v: torch.Tensor, z_t: torch.Tensor) -> tuple[int, int, float]:
    """Diagnostic 2 setup: scan a held-out batch's PRE-alignment embeddings
    (z_v_tilde/z_t_tilde, before h_v/h_t) for the closest cross-pair cosine
    similarity that isn't the true diagonal match -- the Mosa/ego failure mode
    (spec sec 1.1/6). Returns (i, j, similarity) with i != j."""
    B = z_v.shape[0]
    sims = z_v @ z_t.T
    sims.fill_diagonal_(-float("inf"))  # exclude the true diagonal match
    flat_idx = sims.argmax()
    i, j = divmod(flat_idx.item(), B)
    return i, j, sims[i, j].item()


@torch.no_grad()
def alignment_space_distance(A_v: torch.Tensor, A_t: torch.Tensor, i: int, j: int) -> float:
    """1 - cosine similarity, same metric alignment_loss's cost matrix uses, so
    'distance increasing' here is directly comparable to what the loss optimizes."""
    return (1.0 - (A_v[i] @ A_t[j])).item()


@torch.no_grad()
def run_phase_a_diagnostics(
    model: ReflowJEPA,
    h_v: AlignmentHead,
    h_t: AlignmentHead,
    images: torch.Tensor,
    captions,
    epsilon: float,
    n_iters: int,
    gamma: float,
    hard_neg_ij: tuple[int, int] | None = None,
) -> dict:
    """Runs all four spec sec 6 diagnostics on one held-out batch. If hard_neg_ij is
    given (from a FIXED diagnostic batch found once at the start of training via
    find_hard_negative_pair), also reports diagnostic 2's tracked distance -- this
    is the mechanism-level check, not just the aggregate retrieval number."""
    B = images.shape[0]
    c = model.task_token.expand(B, -1)
    z_v_tilde = model.encode_visual(images, c)
    z_t_tilde = model.encode_text_online(captions)

    A_v = h_v(z_v_tilde)
    A_t = h_t(z_t_tilde)

    top1, top5 = retrieval_accuracy(A_v, A_t)

    C = 1 - A_v @ A_t.T
    P = sinkhorn_log_domain(C, epsilon, n_iters)
    row_err, col_err = sinkhorn_marginal_error(P)
    has_nan_inf = bool(torch.isnan(P).any() or torch.isinf(P).any())

    diag = {
        "retrieval_top1": top1,
        "retrieval_top5": top5,
        "vicreg_std_v_mean": A_v.std(dim=0, unbiased=False).mean().item(),
        "vicreg_std_v_min": A_v.std(dim=0, unbiased=False).min().item(),
        "vicreg_std_t_mean": A_t.std(dim=0, unbiased=False).mean().item(),
        "vicreg_std_t_min": A_t.std(dim=0, unbiased=False).min().item(),
        "sinkhorn_row_err": row_err,
        "sinkhorn_col_err": col_err,
        "sinkhorn_has_nan_or_inf": has_nan_inf,
    }
    if hard_neg_ij is not None:
        i, j = hard_neg_ij
        diag["hard_negative_alignment_distance"] = alignment_space_distance(A_v, A_t, i, j)
    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dataloaders(args):
    if args.dataset == "flickr30k":
        dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_karpathy_split,
                                           image_size=args.image_size, seed=args.seed)
        eval_dataset = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_karpathy_split,
                                                image_size=args.image_size, seed=999)
    else:
        dataset = SyntheticCaptioningDataset(length=args.dataset_length, image_size=args.image_size)
        eval_dataset = SyntheticCaptioningDataset(length=max(512, args.batch_size), seed=999,
                                                   image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_images_captions, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_images_captions, drop_last=True)
    return loader, eval_loader


def train_alignment(args) -> list[dict]:
    """Core training loop, factored out of main() so it's directly callable from
    tests (spec sec 7 test 8: 'a few steps of train_alignment.py's actual training
    loop... called directly')."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = load_frozen_base_model(args, device)
    d_shared = model.qpool.g_v[-1].out_features if hasattr(model.qpool, "g_v") else 768

    h_v = AlignmentHead(d_in=d_shared, d_align=args.align_dim).to(device)
    h_t = AlignmentHead(d_in=d_shared, d_align=args.align_dim).to(device)
    optimizer = torch.optim.AdamW(list(h_v.parameters()) + list(h_t.parameters()), lr=args.lr)

    loader, eval_loader = build_dataloaders(args)
    data_iter = iter(loader)

    # Fixed diagnostic batch + hard-negative pair (spec sec 6 diagnostic 2): found
    # ONCE, before any training, on PRE-alignment embeddings, then reused every eval
    # so the tracked distance is comparable across the whole run.
    diag_images, diag_captions = next(iter(eval_loader))
    diag_images = diag_images.to(device)
    with torch.no_grad():
        c_diag = model.task_token.expand(diag_images.shape[0], -1)
        z_v_diag = model.encode_visual(diag_images, c_diag)
        z_t_diag = model.encode_text_online(diag_captions)
    hard_neg_ij = find_hard_negative_pair(z_v_diag, z_t_diag)[:2]

    log = []
    t0 = time.time()
    step = 0
    while step < args.steps:
        try:
            images, captions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, captions = next(data_iter)
        images = images.to(device)
        B = images.shape[0]

        with torch.no_grad():
            c = model.task_token.expand(B, -1)
            z_v_tilde = model.encode_visual(images, c)
            z_t_tilde = model.encode_text_online(captions)
        # spec sec 4.1: inputs are detached copies -- already no_grad above (base
        # model is entirely frozen), .detach() here too so h_v/h_t's graph never
        # reaches back into the no_grad block's tensors by accident of aliasing.
        A_v = h_v(z_v_tilde.detach())
        A_t = h_t(z_t_tilde.detach())

        align_term = alignment_loss(A_v, A_t, args.sinkhorn_epsilon, args.sinkhorn_iters)
        vicreg_term = vicreg_variance_penalty(A_v, args.vicreg_gamma) + \
            vicreg_variance_penalty(A_t, args.vicreg_gamma)
        total_loss = align_term + args.vicreg_weight * vicreg_term

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        record = {
            "step": step,
            "total_loss": total_loss.item(),
            "alignment_loss": align_term.item(),
            "vicreg_term": vicreg_term.item(),
            "elapsed_s": time.time() - t0,
        }

        if step % max(1, args.eval_every) == 0:
            eval_images, eval_captions = next(iter(eval_loader))
            eval_images = eval_images.to(device)
            eval_diag = run_phase_a_diagnostics(
                model, h_v, h_t, eval_images, eval_captions,
                args.sinkhorn_epsilon, args.sinkhorn_iters, args.vicreg_gamma,
                hard_neg_ij=hard_neg_ij,
            )
            record.update({f"eval_{k}": v for k, v in eval_diag.items()})
            print(f"[step {step:5d}] total={record['total_loss']:.4f} "
                  f"align={record['alignment_loss']:.4f} vicreg={record['vicreg_term']:.4f} "
                  f"| top1={eval_diag['retrieval_top1']:.3f} top5={eval_diag['retrieval_top5']:.3f} "
                  f"| hard_neg_dist={eval_diag.get('hard_negative_alignment_distance', float('nan')):.4f}")

        log.append(record)
        step += 1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_alignment_checkpoint(h_v, h_t, args, step, args.checkpoint_path)
            print(f"[train_alignment] checkpoint saved at step {step} -> {args.checkpoint_path}")

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    save_alignment_checkpoint(h_v, h_t, args, step, args.checkpoint_path)
    print(f"[train_alignment] done. log -> {args.log_path}, checkpoint -> {args.checkpoint_path} (step {step})")
    return log


def main():
    args = parse_args()
    train_alignment(args)


if __name__ == "__main__":
    main()
