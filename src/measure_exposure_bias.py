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

Three comparisons, all at matched tau using the identical Z0:
  1. ||Z_tau|| under training-interpolation vs under real integration -- does the
     trajectory actually leave the training distribution's support.
  2. The predictor's OWN residual error, evaluated at the training point vs the
     inference point -- if the network extrapolates poorly, residual_infer >>
     residual_train even though both are being asked about the same underlying target.
  3. NEW: for --dataset flickr30k, the SAME inference-time estimate scored against
     ALL 5 of the image's true captions, not just the one it happened to be paired
     with in training -- a high resid_infer against the paired caption alone can't
     distinguish "the flow genuinely failed" from "the flow landed near a DIFFERENT
     valid caption for this same image," which real Flickr30k images can have and
     synthetic_data.py's exact one-to-one mapping cannot. Reported as best_of_k
     alongside resid_infer; for --dataset synthetic, K=1 and best_of_k == resid_infer
     exactly (a built-in sanity check that the mechanism is wired correctly).

DATA-DISTRIBUTION FIX: this previously always used SyntheticCaptioningDataset, even
when --real-checkpoints pointed at a checkpoint trained on --dataset flickr30k. That's
a genuine confound -- a model that has only ever seen real photos, evaluated on
procedurally-generated synthetic shapes, could show degraded/divergent behavior purely
from being out-of-distribution on the INPUT side, with nothing to do with flow-matching
exposure bias specifically. --dataset now defaults to synthetic (unchanged behavior for
anyone not passing it) but should be set to flickr30k to test a real-data checkpoint on
the data it actually trained on.

BUG FIXED after the EDM-preconditioned architecture was introduced: this originally
computed the residual as ||v_pred - (Z1-Z0)|| directly on the raw velocity output. For
edm_precondition=True, v_pred is STRUCTURALLY (Z1-Z0) + eps/(1-tau) where eps is the
network's target-estimate error -- so this residual is amplified by 1/(1-tau)
regardless of how good the network actually is, growing unboundedly near tau=1 even
for a well-trained model. This produced misleadingly alarming numbers (resid_train
reaching 60+ by tau=0.999) that reflected the amplification artifact, not genuine
error. Fixed to match reflow_jepa.py's training_step and reflow_round.py: recover the
bounded target-estimate z1_hat algebraically and compare THAT to the true target,
which stays close to its true bounded scale (<=~4) regardless of tau.

Usage:
    python measure_exposure_bias.py --checkpoint-path /kaggle/working/reflow_jepa_ckpt.pt \
        --predictor-depth 6 --predictor-heads 8 --visual-layers 4 --text-layers 4 \
        --real-checkpoints --visual-encoder siglip --dataset flickr30k
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


@torch.no_grad()
def recover_target_estimate(model, Z_point, tau_batch, z_v_tilde, c):
    """Recovers the network's target-estimate at Z_point, in whichever representation
    compute_residual/residual_to_best_of_k score against Z1_true below -- the bounded
    z1_hat (edm_precondition=True) or the raw velocity (False). Split out from the old
    compute_residual so the SAME recovered estimate (one predictor forward pass) can be
    scored against one target (the paired caption) or several (the multi-caption
    best-of-k check) without calling the predictor twice per tau."""
    v_pred = model.predictor(Z_point, tau_batch, z_v_tilde, c)
    if model.predictor.edm_precondition:
        return v_pred * (1 - tau_batch).unsqueeze(-1) + Z_point
    return v_pred


def residual_to_target(estimate, Z1_true, Z0, edm_precondition):
    """Unchanged behavior from the original compute_residual for both branches --
    just factored out so it can be reused for both the single-caption and
    multi-caption comparisons below."""
    if edm_precondition:
        return (estimate - Z1_true).norm(dim=-1).mean().item()
    target_direction = Z1_true - Z0
    return (estimate - target_direction).norm(dim=-1).mean().item()


def residual_to_best_of_k(estimate, Z1_candidates, Z0, edm_precondition):
    """Z1_candidates: (B, K, D) -- the K true-caption candidate embeddings per example
    (K=5 for flickr30k's 5 human captions, K=1 for synthetic -- a no-op there, see
    module docstring). Per-example best (minimum) residual against its OWN candidate
    set, NOT necessarily the one it was paired with in training. Also returns which
    candidate (0..K-1) was best per example, in case that caption differing from index
    0 (the one __getitem__ would have randomly sampled) is itself worth inspecting."""
    if edm_precondition:
        dists = (estimate.unsqueeze(1) - Z1_candidates).norm(dim=-1)  # (B, K)
    else:
        target = Z1_candidates - Z0.unsqueeze(1)                      # (B, K, D)
        dists = (estimate.unsqueeze(1) - target).norm(dim=-1)         # (B, K)
    best_per_example, best_idx = dists.min(dim=1)
    return best_per_example.mean().item(), best_idx


@torch.no_grad()
def encode_captions(model, captions, device):
    """captions: flat list of strings (already B*K if scoring K candidates per
    example -- caller reshapes the (N, D) result back to (B, K, D))."""
    batch = model.tokenizer(captions, return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
    return F.normalize(model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"])), dim=-1)


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
    p.add_argument("--ema-cfm-target", type=lambda x: x.lower() != "false", default=False,
                    help="must match whatever the loaded checkpoint was trained with -- "
                         "see train.py --help for what this changes")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.02,
                    help="must match whatever the loaded checkpoint was actually trained with -- "
                         "sigma is a plain attribute, not a saved buffer, so load_state_dict "
                         "cannot restore it from the checkpoint; passing the wrong value here "
                         "silently draws Z0 from a differently-sized noise ball than the "
                         "checkpoint was trained on, invalidating this whole comparison")
    p.add_argument("--visual-encoder", type=str, default="ijepa", choices=["ijepa", "siglip"],
                    help="must match whatever the loaded checkpoint was trained with")
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "flickr30k"],
                    help="default 'synthetic' preserves old behavior. Set to 'flickr30k' to "
                         "evaluate a --real-checkpoints checkpoint on the SAME kind of data it "
                         "was actually trained on -- see module docstring's DATA-DISTRIBUTION "
                         "FIX note for why this matters for trusting the numbers below.")
    p.add_argument("--flickr-karpathy-split", type=str, default=None,
                    help="optional filter on Flickr30k's train/val/test column; None uses all rows")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from encoders import VISUAL_ENCODER_SPECS
    image_size = VISUAL_ENCODER_SPECS[args.visual_encoder][2]

    device = torch.device(args.device)
    model = ReflowJEPA(
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        visual_layers=args.visual_layers, text_layers=args.text_layers,
        real_checkpoints=args.real_checkpoints,
        edm_precondition=args.edm_precondition,
        ema_cfm_target=args.ema_cfm_target,
        visual_encoder=args.visual_encoder,
        sigma=args.sigma,
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

    if args.dataset == "flickr30k":
        from real_captioning_data import FlickrCaptioningDataset
        ds = FlickrCaptioningDataset(karpathy_split_filter=args.flickr_karpathy_split,
                                      image_size=image_size, seed=24680)
    else:
        ds = SyntheticCaptioningDataset(length=args.batch_size, seed=24680, image_size=image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))
    images = images.to(device)
    B = images.shape[0]
    print(f"[data] --dataset {args.dataset}, {B} examples "
          f"{'(SYNTHETIC -- pass --dataset flickr30k to test on real data instead)' if args.dataset == 'synthetic' else ''}")

    # All K captions per example (K=5 for flickr30k, K=1 for synthetic), in the SAME
    # order as the batch DataLoader just yielded (shuffle=False, so indices 0..B-1
    # correspond exactly to what get_all_captions(i) below returns).
    all_captions_per_example = [ds.get_all_captions(i) for i in range(B)]
    K = len(all_captions_per_example[0])
    assert all(len(c) == K for c in all_captions_per_example), \
        "expected the same caption count K for every example in the batch"

    from encoders import D_SHARED
    predicted_z0_norm = (1 + (args.sigma ** 2) * D_SHARED) ** 0.5
    print(f"[sigma] using sigma={args.sigma} -> predicted ||Z0||={predicted_z0_norm:.4f} if this "
          f"matches the checkpoint's actual training sigma. Compare against ||Z_train|| at "
          f"tau=0.0 in the table below -- a mismatch there means --sigma doesn't match what "
          f"this checkpoint was actually trained with, and everything below is invalid.")

    with torch.no_grad():
        c = model.task_token.expand(B, -1)
        z_v_tilde = model.encode_visual(images, c)
        Z0 = draw_stochastic_source(z_v_tilde, model.sigma)  # SAME Z0 used for both comparisons below

        Z1_true = encode_captions(model, captions, device)  # (B, D) -- the ONE paired caption, as before

        flat_candidates = [cap for caps in all_captions_per_example for cap in caps]  # B*K strings
        Z1_candidates = encode_captions(model, flat_candidates, device).view(B, K, -1)  # (B, K, D)

    capture_taus = [0.0, 0.5, 0.9, 0.99, 0.999, 1 - 2e-3]
    print(f"\n[integrate] running {args.n_steps}-step trajectory from the SAME Z0 used "
          f"for the training-style comparison below, capturing at tau = {capture_taus}")
    captured = integrate_from_z0_capturing(model, Z0, z_v_tilde, c, capture_taus, args.n_steps)

    print(f"\n{'tau':>10}  {'||Z_train||':>12}  {'||Z_infer||':>12}  {'dist(train,infer)':>18}  "
          f"{'resid_train':>12}  {'resid_infer':>12}  {'best_of_'+str(K):>10}  {'resid ratio':>12}")
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

            estimate_train = recover_target_estimate(model, Z_train, tau_batch, z_v_tilde, c)
            estimate_infer = recover_target_estimate(model, Z_infer, tau_batch, z_v_tilde, c)
            resid_train = residual_to_target(estimate_train, Z1_true, Z0, model.predictor.edm_precondition)
            resid_infer = residual_to_target(estimate_infer, Z1_true, Z0, model.predictor.edm_precondition)
            best_of_k, _ = residual_to_best_of_k(estimate_infer, Z1_candidates, Z0, model.predictor.edm_precondition)
            ratio = resid_infer / resid_train if resid_train > 1e-8 else float("inf")

        print(f"{tau_val:>10.4f}  {norm_train:>12.4f}  {norm_infer:>12.4f}  {dist_train_infer:>18.4f}  "
              f"{resid_train:>12.4f}  {resid_infer:>12.4f}  {best_of_k:>10.4f}  {ratio:>12.4f}")

    print()
    print("INTERPRETATION GUIDE:")
    print("  ||Z_train|| vs ||Z_infer|| diverging substantially by tau~0.5 -> the real ")
    print("  trajectory leaves the training distribution's support early, not just near tau=1.")
    print("  resid_train/resid_infer measured in the BOUNDED target-estimate space")
    print("  (z1_hat vs Z1_true, not raw velocity) for edm_precondition=True checkpoints --")
    print("  should stay roughly bounded (<=~4) at every tau for a well-trained network.")
    print("  resid ratio >> 1 at the same tau -> the predictor's error is genuinely larger")
    print("  off-distribution (at the point integration actually visits) than on-distribution")
    print("  (at the point training actually trained on) for the SAME true target --")
    print("  direct confirmation the network is extrapolating poorly outside what it saw.")
    print(f"  best_of_{K}: resid_infer scored against whichever of the {K} true captions per")
    print("  image is closest, not only the one paired in training. best_of_k << resid_infer")
    print("  means the flow is landing near a DIFFERENT valid caption for the same image, not")
    print("  failing outright -- real ambiguity only --dataset flickr30k can show (K=1 for")
    print("  synthetic, where best_of_k == resid_infer exactly, by construction).")


if __name__ == "__main__":
    main()
