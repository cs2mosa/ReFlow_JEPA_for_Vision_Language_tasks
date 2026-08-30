"""
Phase C: Reflow round 2 (reflow_jepa_proof1.pdf Algorithm 1, k=2; DESIGN.md's own
prescribed next step once base CFM training plateaus).

Simulates the CURRENT trained field's own trajectories to build a deterministic
coupling (Z_0, Z_hat_1), then retrains v_theta on THAT coupling via straight-line CFM
regression. Only the predictor is retrained here -- Q-Pool, g_T, the frozen visual/text
encoders, and the decoder/prefix_expand all stay fixed. Reflow's guarantees (Theorem 3:
convex transport cost does not increase; Theorem 4: best-iterate O(1/K) straightening)
are about the TRANSPORT MAP within a FIXED shared latent space, not about redefining
that space mid-round -- retraining Q-Pool/g_T here would invalidate the theorem's own
premise.

Directly targets two things a real run's Check 3 found, after ruling out the LR
schedule as the cause:
  - cfm_loss plateaued flat and noisy -- Reflow retrains on the field's OWN
    straightened trajectories rather than the raw noisy (Z0, Z1) pairs, which
    Theorem 4 proves reduces the straightness+crossing diagnostic, not a heuristic.
  - the flow's integrated output undershoots the target manifold (||z_hat|| ~ 0.34-0.45
    vs ||z_true|| = 1.0) -- consistent with residual off-manifold-collapse behavior
    (Lemma 1) not yet fully resolved; Reflow is the design's own prescribed fix for
    this symptom.

Usage:
    python reflow_round.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --output-checkpoint-path /kaggle/working/reflow_jepa_round2_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from reflow_jepa import ReflowJEPA, _mean_pool_text
from stochastic_source import draw_stochastic_source
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--output-checkpoint-path", type=str, required=True)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--visual-layers", type=int, default=4)
    p.add_argument("--text-layers", type=int, default=4)
    p.add_argument("--real-checkpoints", action="store_true")
    p.add_argument("--edm-precondition", type=lambda x: x.lower() != "false", default=True,
                    help="must match whatever the loaded checkpoint was trained with -- "
                         "see train.py --help for what this changes")
    p.add_argument("--sigma", type=float, default=0.02)

    p.add_argument("--reflow-dataset-size", type=int, default=4096,
                    help="number of (Z0, Z_hat_1) pairs to simulate offline before "
                         "the round-2 training loop -- this is the deterministic "
                         "coupling Theorem 3/4 are about, built once and reused, not "
                         "regenerated fresh each step the way round-1 training was")
    p.add_argument("--simulate-integrate-steps", type=int, default=500,
                    help="Euler steps used to SIMULATE the coupling (matches the "
                         "resolution used in diagnose_recon_signal.py's Check 3, "
                         "since we already confirmed more steps doesn't change the "
                         "outcome at this resolution)")
    p.add_argument("--simulate-batch-size", type=int, default=64)

    p.add_argument("--reflow-steps", type=int, default=1500,
                    help="CFM regression steps on the fixed reflow dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4,
                    help="lower than round-1's default -- warm-starting from an "
                         "already-trained predictor, not initializing from scratch")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)

    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--log-path", type=str, default="reflow_round_log.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(args, device):
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints,
        edm_precondition=args.edm_precondition,
    ).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f"[checkpoint] loading round-1 model from step={checkpoint.get('step', '?')}")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[checkpoint] WARNING: non-strict load -- missing={missing}, "
                  f"unexpected={unexpected} (likely an architecture change since this "
                  f"checkpoint was saved, e.g. buffers added later default correctly "
                  f"for missing keys; verify results are still meaningful)")
    else:
        print("[checkpoint] WARNING: old-format checkpoint (no step metadata)")
        model.load_state_dict(checkpoint, strict=False)
    return model


@torch.no_grad()
def build_reflow_dataset(model, args, device):
    """Simulates the round-1 field's own trajectories to build the deterministic
    (Z0, z_v_tilde, Z_hat_1, c) coupling Reflow retrains on. This is Algorithm 1's
    line "for each (z_v^(i), c^(i)) in D: z_hat^(i) <- N-step Euler integration of
    v_theta_k from z_v^(i)", executed once, offline, before any round-2 gradient step."""
    model.eval()
    ds = SyntheticCaptioningDataset(length=args.reflow_dataset_size, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.simulate_batch_size, collate_fn=collate_images_captions,
                     shuffle=False)

    all_Z0, all_zv, all_Zhat1 = [], [], []
    n_batches = math.ceil(args.reflow_dataset_size / args.simulate_batch_size)
    t0 = time.time()
    for i, (images, captions) in enumerate(dl):
        images = images.to(device)
        B = images.shape[0]
        c = model.task_token.expand(B, -1)
        z_v_tilde = model.encode_visual(images, c)
        Z0 = draw_stochastic_source(z_v_tilde, args.sigma)

        Z = Z0.clone()
        taus = torch.linspace(0, 1 - 1e-3, args.simulate_integrate_steps + 1, device=device)
        dtau = taus[1] - taus[0]
        for step in range(args.simulate_integrate_steps):
            tau_batch = taus[step].expand(B)
            v = model.predictor(Z, tau_batch, z_v_tilde, c)
            Z = Z + v * dtau

        all_Z0.append(Z0.cpu())
        all_zv.append(z_v_tilde.cpu())
        all_Zhat1.append(Z.cpu())
        print(f"[reflow-dataset] batch {i + 1}/{n_batches}  "
              f"({time.time() - t0:.1f}s elapsed)")

    Z0 = torch.cat(all_Z0)
    Zv = torch.cat(all_zv)
    Zhat1 = torch.cat(all_Zhat1)

    original_cost_proxy = (Zhat1 - Z0).pow(2).sum(dim=-1).mean().item()
    zhat1_norm = Zhat1.norm(dim=-1).mean().item()
    print(f"[reflow-dataset] built {Z0.shape[0]} pairs. "
          f"mean ||Z_hat_1 - Z0||^2 = {original_cost_proxy:.4f}  "
          f"mean ||Z_hat_1|| = {zhat1_norm:.4f} (this IS the round's starting transport "
          f"cost -- Theorem 3 guarantees the NEW field trained below won't increase it)")
    return Z0, Zv, Zhat1


class ReflowDataset(Dataset):
    def __init__(self, Z0, Zv, Zhat1):
        self.Z0, self.Zv, self.Zhat1 = Z0, Zv, Zhat1

    def __len__(self):
        return self.Z0.shape[0]

    def __getitem__(self, idx):
        return self.Z0[idx], self.Zv[idx], self.Zhat1[idx]


def build_lr_scheduler(optimizer, args):
    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.reflow_steps - args.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate_end_to_end(model, args, device):
    """Same Check-3-style metric as diagnose_recon_signal.py, on FRESH held-out data
    with TRUE captions (not the round's own simulated targets) -- the real question:
    did Reflow actually improve real end-to-end generation, not just fit its own
    simulated coupling."""
    model.eval()
    ds = SyntheticCaptioningDataset(length=args.eval_batch_size, seed=99999)
    dl = DataLoader(ds, batch_size=args.eval_batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)

    batch = model.tokenizer(captions, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
    z_true = F.normalize(model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"])), dim=-1)

    z_hat = model.integrate(images, n_steps=args.simulate_integrate_steps)
    z_hat_norm = z_hat.norm(dim=-1).mean().item()
    z_true_norm = z_true.norm(dim=-1).mean().item()
    dist_to_true = (z_hat - z_true).norm(dim=-1).mean().item()
    return {
        "z_hat_norm": z_hat_norm,
        "z_true_norm": z_true_norm,
        "dist_to_true": dist_to_true,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = load_model(args, device)

    print("\n=== Pre-Reflow baseline (round-1 checkpoint, real captions) ===")
    baseline = evaluate_end_to_end(model, args, device)
    print(f"||z_hat||={baseline['z_hat_norm']:.4f}  ||z_true||={baseline['z_true_norm']:.4f}  "
          f"dist={baseline['dist_to_true']:.4f}")

    print("\n=== Building the Reflow dataset (simulating round-1's own field) ===")
    Z0, Zv, Zhat1 = build_reflow_dataset(model, args, device)
    reflow_ds = ReflowDataset(Z0, Zv, Zhat1)
    reflow_dl = DataLoader(reflow_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Only the predictor is trained this round -- see module docstring for why.
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)

    print("\n=== Round-2 training (straight-line CFM regression on the fixed coupling) ===")
    log = []
    step = 0
    t0 = time.time()
    data_iter = iter(reflow_dl)
    while step < args.reflow_steps:
        try:
            Z0_b, Zv_b, Zhat1_b = next(data_iter)
        except StopIteration:
            data_iter = iter(reflow_dl)
            Z0_b, Zv_b, Zhat1_b = next(data_iter)
        Z0_b, Zv_b, Zhat1_b = Z0_b.to(device), Zv_b.to(device), Zhat1_b.to(device)
        B = Z0_b.shape[0]
        c = model.task_token.expand(B, -1)

        tau = torch.rand(B, device=device)
        Z_tau = (1 - tau).unsqueeze(-1) * Z0_b + tau.unsqueeze(-1) * Zhat1_b
        v_pred = model.predictor(Z_tau, tau, Zv_b, c)
        if model.predictor.edm_precondition:
            # Same fix as reflow_jepa.py's training_step -- see its docstring for the
            # full derivation. Recovers the bounded target-estimate algebraically
            # instead of supervising the (tau-amplified) raw velocity directly.
            z1_hat = v_pred * (1 - tau).unsqueeze(-1) + Z_tau
            loss = (z1_hat - Zhat1_b).pow(2).sum(dim=-1).mean()
        else:
            loss = (v_pred - (Zhat1_b - Z0_b)).pow(2).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        diag = {"step": step, "reflow_loss": loss.item(), "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t0}
        if step % max(1, args.reflow_steps // 30) == 0:
            print(f"[reflow step {step:5d}] lr={diag['lr']:.2e} loss={diag['reflow_loss']:.4f}")

        if step % args.eval_every == 0:
            eval_metrics = evaluate_end_to_end(model, args, device)
            diag.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            print(f"           [eval] ||z_hat||={eval_metrics['z_hat_norm']:.4f}  "
                  f"dist={eval_metrics['dist_to_true']:.4f}")
            model.train()

        log.append(diag)
        step += 1

    print("\n=== Final evaluation ===")
    final = evaluate_end_to_end(model, args, device)
    print(f"baseline (round 1): ||z_hat||={baseline['z_hat_norm']:.4f}  dist={baseline['dist_to_true']:.4f}")
    print(f"after Reflow:       ||z_hat||={final['z_hat_norm']:.4f}  dist={final['dist_to_true']:.4f}")

    with open(args.log_path, "w") as f:
        json.dump(log, f)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "reflow_round": 2,
    }, args.output_checkpoint_path)
    print(f"\n[reflow] done. log -> {args.log_path}, checkpoint -> {args.output_checkpoint_path}")


if __name__ == "__main__":
    main()
