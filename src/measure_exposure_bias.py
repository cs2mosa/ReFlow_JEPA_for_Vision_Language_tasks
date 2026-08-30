"""
Compares Z_tau under TRAINING's linear interpolation (Z_tau = (1-tau)*Z0 + tau*Z1_true,
what the predictor's gradient actually comes from) against Z_tau under REAL
inference-time integration (what the predictor is actually queried at when generating),
at matched tau values, using the SAME starting Z0 for both so the comparison is exact.

Motivated directly by measure_terminal_divergence.py's result: the standard-flow
checkpoint showed wild, non-monotonic Jacobian-norm swings (18 -> 357 -> 629 -> 28
across tau), and ||Z_tau|| along the real integrated trajectory dropped from ~1.14 to
~0.41-0.52 by the trajectory's midpoint. Training only ever shows the predictor points
on the straight Z0-Z1 line segment (which stays in a consistent norm range across tau
by construction); if the integrated trajectory drifts off that segment as small errors
compound, the predictor is being queried at inference on inputs it never received
training gradient for -- a flow-matching analogue of the exposure-bias problem already
found and fixed once for the decoder's teacher forcing.

Two comparisons, both at matched tau using the identical Z0:
  1. ||Z_tau|| under training-interpolation vs under real integration -- does the
     trajectory actually leave the training distribution's support.
  2. The predictor's OWN residual error (||v_pred - true target||) evaluated at the
     training point vs at the inference point, for the identical target direction --
     if the network extrapolates poorly, residual_infer >> residual_train even though
     both are being asked for the same underlying displacement.

Usage:
    python measure_exposure_bias.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reflow_jepa import ReflowJEPA, _mean_pool_text
from stochastic_source import draw_stochastic_source
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions


@torch.no_grad()
def integrate_from_z0_capturing(model, Z0, z_v_tilde, c, capture_taus, n_steps, delta=1e-3):
    """Same integration as measure_terminal_divergence.py, but takes Z0 as an explicit
    argument (instead of drawing a fresh one internally) so the SAME starting point can
    be reused for the training-interpolation comparison."""
    B = Z0.shape[0]
    Z = Z0.clone()
    taus = torch.linspace(0, 1 - delta, n_steps + 1, device=Z0.device)
    dtau = taus[1] - taus[0]
    captured = {}
    capture_set = set(round(t, 6) for t in capture_taus)

    for i in range(n_steps):
        tau_val = taus[i].item()
        for target in list(capture_set):
            if abs(tau_val - target) < (dtau.item() / 2):
                captured[target] = (Z.clone(), taus[i].clone())
                capture_set.discard(target)
        tau_batch = taus[i].expand(B)
        v = model.predictor(Z, tau_batch, z_v_tilde, c)
        Z = Z + v * dtau

    captured["final"] = (Z.clone(), taus[-1].clone())
    return captured


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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=500)
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
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    ds = SyntheticCaptioningDataset(length=args.batch_size, seed=24680)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)

    with torch.no_grad():
        B = images.shape[0]
        c = model.task_token.expand(B, -1)
        z_v_tilde = model.encode_visual(images, c)
        Z0 = draw_stochastic_source(z_v_tilde, model.sigma)  # SAME Z0 used for both comparisons below

        batch = model.tokenizer(captions)
        batch = {k: v.to(device) for k, v in batch.items()}
        enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
        Z1_true = F.normalize(model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"])), dim=-1)

        target_direction = Z1_true - Z0  # the (Z1 - Z0) regression target, same for every tau

    capture_taus = [0.0, 0.5, 0.9, 0.99, 0.999, 1 - 2e-3]
    print(f"\n[integrate] running {args.n_steps}-step trajectory from the SAME Z0 used "
          f"for the training-style comparison below, capturing at tau = {capture_taus}")
    captured = integrate_from_z0_capturing(model, Z0, z_v_tilde, c, capture_taus, args.n_steps)

    print(f"\n{'tau':>10}  {'||Z_train||':>12}  {'||Z_infer||':>12}  {'dist(train,infer)':>18}  "
          f"{'resid_train':>12}  {'resid_infer':>12}  {'resid ratio':>12}")
    for target_tau in capture_taus + ["final"]:
        if target_tau not in captured:
            print(f"{str(target_tau):>10}  (not captured -- step resolution too coarse)")
            continue
        Z_infer, tau_tensor = captured[target_tau]
        tau_val = tau_tensor.item()
        tau_batch = tau_tensor.expand(B)

        with torch.no_grad():
            Z_train = (1 - tau_val) * Z0 + tau_val * Z1_true
            norm_train = Z_train.norm(dim=-1).mean().item()
            norm_infer = Z_infer.norm(dim=-1).mean().item()
            dist_train_infer = (Z_train - Z_infer).norm(dim=-1).mean().item()

            v_pred_train = model.predictor(Z_train, tau_batch, z_v_tilde, c)
            v_pred_infer = model.predictor(Z_infer, tau_batch, z_v_tilde, c)
            resid_train = (v_pred_train - target_direction).norm(dim=-1).mean().item()
            resid_infer = (v_pred_infer - target_direction).norm(dim=-1).mean().item()
            ratio = resid_infer / resid_train if resid_train > 1e-8 else float("inf")

        print(f"{tau_val:>10.4f}  {norm_train:>12.4f}  {norm_infer:>12.4f}  {dist_train_infer:>18.4f}  "
              f"{resid_train:>12.4f}  {resid_infer:>12.4f}  {ratio:>12.4f}")

    print()
    print("INTERPRETATION GUIDE:")
    print("  ||Z_train|| vs ||Z_infer|| diverging substantially by tau~0.5 -> the real ")
    print("  trajectory leaves the training distribution's support early, not just near tau=1.")
    print("  resid ratio >> 1 at the same tau -> the predictor's error is genuinely larger")
    print("  off-distribution (at the point integration actually visits) than on-distribution")
    print("  (at the point training actually trained on) for the SAME target direction --")
    print("  direct confirmation the network is extrapolating poorly outside what it saw.")


if __name__ == "__main__":
    main()
