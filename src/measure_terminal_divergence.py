"""
Measures ||grad_z v_theta(z, tau, c)||_op on the TRAINED predictor, at real trajectory
points, as tau -> 1 -- the check test_08 validated only for the EXACT hand-derived
field (Proposition 2 / OFM-JEPA v2 Proposition 3: velocity should scale as O(1/(1-tau))
near the boundary, not the originally-claimed-and-refuted O(1/(1-tau)^3)). This script
runs the same measurement against the actual trained network, which nothing so far has
checked -- test_08 proves the theory is sound; this checks whether a specific trained
network actually learned it.

Directly discriminates two hypotheses for the radial undershoot found via Check 3 /
reflow_round.py's before/after (||z_hat|| ~ 0.47 vs ||z_true|| = 1.0, Reflow left this
essentially unchanged):
  - If the measured rate caps out well below the true 1/(1-tau) growth: the network
    never learned to diverge near the boundary (a capacity/optimization problem;
    EDM-style preconditioning, building the known asymptotic scaling into the
    architecture rather than asking the network to discover it, is the targeted fix).
  - If the rate does track 1/(1-tau) but the endpoint is still wrong: the field is
    diverging with roughly the right magnitude but in the wrong direction, which would
    support the entangled-marginals / crossing hypothesis and motivate the two-head
    OT-preprocessing architecture instead.

d=768 makes a direct Jacobian (test_08's approach, feasible at d=8) far too expensive
(768 backward passes per point). Instead this uses power iteration with the standard
double-backward trick to estimate just the top singular value (operator norm) via a
handful of backward passes -- and since LayerNorm/attention in this predictor never mix
information across the batch dimension, a single batched pass gives independent
per-example estimates rather than needing a Python loop over the batch.

Usage:
    python measure_terminal_divergence.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4
"""
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA
from stochastic_source import draw_stochastic_source
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


def estimate_spectral_norm(predictor, z, tau, zv, c, num_iters=15):
    """Power-iteration estimate of ||d v_theta / d z||_2 (operator norm), PER EXAMPLE
    in the batch, using only reverse-mode autograd (the standard double-backward trick
    for Jacobian-vector products -- forward-mode JVP support varies by torch version,
    this avoids depending on it)."""
    B, d = z.shape
    z = z.detach().clone().requires_grad_(True)
    v = torch.randn(B, d, device=z.device)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    for _ in range(num_iters):
        out = predictor(z, tau, zv, c)
        w = torch.ones_like(out, requires_grad=True)
        g = torch.autograd.grad(out, z, grad_outputs=w, create_graph=True)[0]  # J^T w, function of w
        s = (g * v).sum()
        Jv = torch.autograd.grad(s, w)[0]  # d(w^T J v)/dw = J v
        u = Jv / Jv.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        out2 = predictor(z, tau, zv, c)
        JT_u = torch.autograd.grad(out2, z, grad_outputs=u)[0]  # J^T u
        v = JT_u / JT_u.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    out = predictor(z, tau, zv, c)
    w = torch.ones_like(out, requires_grad=True)
    g = torch.autograd.grad(out, z, grad_outputs=w, create_graph=True)[0]
    s = (g * v).sum()
    Jv_final = torch.autograd.grad(s, w)[0]
    sigma = Jv_final.norm(dim=-1)  # (B,), the estimated top singular value per example
    return sigma.detach()


@torch.no_grad()
def integrate_capturing_trajectory(model, images, capture_taus, n_steps, delta=1e-3):
    """Same Euler integration as ReflowJEPA.integrate(), but also returns Z (and tau)
    at specific checkpoints along the way, so we can measure the Jacobian at points the
    trajectory ACTUALLY visits rather than arbitrary z."""
    B = images.shape[0]
    c = model.task_token.expand(B, -1)
    z_v_tilde = model.encode_visual(images, c)
    Z = draw_stochastic_source(z_v_tilde, model.sigma)

    taus = torch.linspace(0, 1 - delta, n_steps + 1, device=images.device)
    dtau = taus[1] - taus[0]
    captured = {}
    capture_set = set(round(t, 6) for t in capture_taus)

    for i in range(n_steps):
        tau_val = taus[i].item()
        for target in list(capture_set):
            if abs(tau_val - target) < (dtau.item() / 2):
                captured[target] = (Z.detach().clone(), taus[i].clone())
                capture_set.discard(target)
        tau_batch = taus[i].expand(B)
        v = model.predictor(Z, tau_batch, z_v_tilde, c)
        Z = Z + v * dtau

    captured["final"] = (Z.detach().clone(), taus[-1].clone())
    return captured, z_v_tilde, c


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
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--power-iters", type=int, default=15)
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
        print(f"[checkpoint] step={checkpoint.get('step', '?')}")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[checkpoint] WARNING: non-strict load -- missing={missing}, "
                  f"unexpected={unexpected} (likely an architecture change since this "
                  f"checkpoint was saved, e.g. buffers added later default correctly "
                  f"for missing keys; verify results are still meaningful)")
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    ds = SyntheticCaptioningDataset(length=args.batch_size, seed=13579)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)

    capture_taus = [0.0, 0.5, 0.9, 0.99, 0.999, 1 - 2e-3]
    print(f"\n[integrate] running {args.n_steps}-step trajectory, capturing at "
          f"tau = {capture_taus}")
    captured, z_v_tilde, c = integrate_capturing_trajectory(model, images, capture_taus, args.n_steps)

    print(f"\n{'tau':>10}  {'measured ||J||_op':>18}  {'predicted 1/(1-tau)':>20}  "
          f"{'ratio (measured/predicted)':>28}  {'||Z_tau||':>10}")
    results = []
    for target_tau in capture_taus + ["final"]:
        if target_tau not in captured:
            print(f"{str(target_tau):>10}  (not captured -- step resolution too coarse)")
            continue
        Z_tau, tau_tensor = captured[target_tau]
        tau_val = tau_tensor.item()
        tau_batch = tau_tensor.expand(Z_tau.shape[0])

        sigma = estimate_spectral_norm(model.predictor, Z_tau, tau_batch, z_v_tilde, c,
                                        num_iters=args.power_iters)
        mean_sigma = sigma.mean().item()
        predicted = 1.0 / (1.0 - tau_val) if tau_val < 1.0 else float("inf")
        ratio = mean_sigma / predicted if predicted != 0 else float("nan")
        z_norm = Z_tau.norm(dim=-1).mean().item()

        print(f"{tau_val:>10.4f}  {mean_sigma:>18.4f}  {predicted:>20.4f}  {ratio:>28.4f}  {z_norm:>10.4f}")
        results.append((tau_val, mean_sigma, predicted, ratio))

    print()
    late_ratios = [r for tau, _, _, r in results if tau > 0.9]
    if late_ratios:
        avg_late_ratio = sum(late_ratios) / len(late_ratios)
        print(f"mean measured/predicted ratio for tau > 0.9: {avg_late_ratio:.4f}")
        if avg_late_ratio < 0.3:
            print("INTERPRETATION: measured growth is far below the theoretical "
                  "1/(1-tau) rate -- the trained field never learned to diverge near "
                  "the boundary the way the exact field does. This is consistent with "
                  "a capacity/optimization gap, not a wrong-direction problem. Points "
                  "toward EDM-style preconditioning (building the known 1/(1-tau) "
                  "scaling into the architecture) rather than the two-head "
                  "entanglement-removal architecture.")
        elif avg_late_ratio > 0.6:
            print("INTERPRETATION: measured growth roughly tracks the theoretical "
                  "1/(1-tau) rate, yet the endpoint (||z_hat|| ~ 0.47 vs ||z_true|| "
                  "= 1.0) is still wrong. This suggests the field IS diverging with "
                  "close to the right magnitude but toward the wrong point -- "
                  "consistent with the crossing/entanglement hypothesis. Points "
                  "toward the two-head OT-preprocessing architecture rather than "
                  "preconditioning alone.")
        else:
            print("INTERPRETATION: partial divergence -- neither cleanly explained by "
                  "'never learned to diverge' nor 'diverges correctly toward the wrong "
                  "place'. Likely both effects are present to some degree; worth "
                  "re-running with a larger --power-iters and --n-steps before "
                  "committing to either fix.")


if __name__ == "__main__":
    main()
