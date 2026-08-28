"""
PHASE 1k — Diagnostic added in response to a real Kaggle training run (3000 steps,
batch=256, full-scale config): vicreg_t stayed pinned near its collapse maximum for
most of training while recon_loss dropped to ~0.3 almost immediately, suspiciously
close to ln(100)/16 ~= 0.288 -- the entropy floor for guessing uniformly among the
synthetic dataset's 100 possible caption strings, 16 mock-tokenizer positions each.

That match is a warning sign: with a small closed caption vocabulary, the decoder can
achieve a good-looking reconstruction loss by learning an UNCONDITIONAL language model
over the ~100 possible outputs, without ever using the (near-collapsed) per-example
embedding it's supposedly conditioned on. This test operationalizes the direct check:
does reconstruction loss stay roughly the same when the decoder is fed a MISMATCHED
(shuffled) embedding instead of the true one? If so, the embedding carries no real
information the decoder is using -- exactly the failure mode a small synthetic
vocabulary can hide behind an otherwise-good-looking loss curve, and exactly the
sanity check to run before trusting recon_loss on the real Kaggle run's actual
checkpoint (this file only proves the mechanism at mock-weight, 30-step scale).
"""
import torch
from encoders import D_TEXT
from reflow_jepa import ReflowJEPA, _mean_pool_text
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from torch.utils.data import DataLoader


def test_recon_loss_mismatch_check_mechanics_run_cleanly():
    """Smoke-level check that the matched-vs-mismatched comparison itself is
    mechanically sound (finite losses, no shape errors) -- NOT a claim about what the
    real trained checkpoint will show. Run the equivalent comparison against the actual
    Kaggle checkpoint before trusting recon_loss's plateau value as evidence of
    per-example learning."""
    torch.manual_seed(0)
    model = ReflowJEPA(visual_layers=1, text_layers=1, predictor_depth=2, predictor_heads=4)
    ds = SyntheticCaptioningDataset(length=32, image_size=224)
    dl = DataLoader(ds, batch_size=16, collate_fn=collate_images_captions)
    images, captions = next(iter(dl))

    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        cfm_loss, recon_loss, vicreg_v, vicreg_t, _ = model.training_step(images, captions)
        (cfm_loss + recon_loss + vicreg_v + vicreg_t).backward()
        opt.step()

    batch = model.tokenizer(captions)
    enc_out = model.text_seq2seq.get_encoder()(**batch).last_hidden_state
    z_true = model.g_t_online(_mean_pool_text(enc_out, batch["attention_mask"]))

    with torch.no_grad():
        matched = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(z_true),), labels=batch["input_ids"]
        ).loss
        shuffled = z_true[torch.randperm(z_true.shape[0])]
        mismatched = model.text_seq2seq(
            encoder_outputs=(model.prefix_expand(shuffled),), labels=batch["input_ids"]
        ).loss

    print(f"\n[INFO] matched recon loss={matched.item():.4f}, "
          f"mismatched (shuffled embedding) recon loss={mismatched.item():.4f} "
          f"-- run this same comparison against the real Kaggle checkpoint; if these "
          f"two numbers are close there, the decoder isn't using per-example embedding "
          f"information.")
    assert torch.isfinite(matched) and torch.isfinite(mismatched)
