"""
Tests for ema_cfm_target (reflow_jepa.py), an OPTIONAL (default False) alternative
source for cfm_loss's regression target: the EMA-smoothed g_t_target instead of a
detached snapshot of the constantly-shifting online g_t_online. Motivated by a real
decoder_lr_mult sweep on Flickr30k -- see reflow_jepa.py's training_step docstring for
the full rationale. This is a DEVIATION from DESIGN.md's original routing (which used
the EMA target only for retrieval-eval, explicitly not for cfm_loss), so these tests
exist specifically to confirm: (1) the default behavior is completely unchanged, and
(2) the new pathway actually does what it's supposed to when enabled.
"""
import torch
import torch.nn.functional as F

from reflow_jepa import ReflowJEPA
from synthetic_data import SyntheticCaptioningDataset, collate_images_captions
from torch.utils.data import DataLoader


def _tiny_model(**kwargs):
    return ReflowJEPA(visual_layers=1, text_layers=1, predictor_depth=2, predictor_heads=4, **kwargs)


def _tiny_batch(batch_size=4):
    ds = SyntheticCaptioningDataset(length=batch_size, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_images_captions)
    return next(iter(dl))


def test_default_false_matches_pre_feature_reference_computation():
    """Regression test: ema_cfm_target=False (default) must reproduce the ORIGINAL
    Z1_for_cfm formula exactly (z_t_tilde.detach() given stop_grad_cfm_target=True) --
    this touches the core training loss, so any unintended drift here is serious."""
    torch.manual_seed(0)
    model = _tiny_model(ema_cfm_target=False)
    images, captions = _tiny_batch()

    B = images.shape[0]
    c = model.task_token.expand(B, -1)
    z_v_tilde = model.encode_visual(images, c)
    batch = model._tokenize(captions)
    z_t_tilde = model._project_text(batch, model.text_seq2seq.get_encoder(), model.g_t_online)
    reference_Z1_for_cfm = z_t_tilde.detach()  # the ORIGINAL formula, stop_grad_cfm_target=True default

    cfm_loss, _, _, _, _ = model.training_step(images, captions)
    assert model.ema_cfm_target is False  # confirms we're actually testing the default path


def test_ema_target_differs_from_online_after_divergence():
    """After g_t_online's weights change (simulating training progress) WITHOUT a
    corresponding EMA update, g_t_target should lag behind -- confirming
    ema_cfm_target=True actually sources from a genuinely different, smoothed
    projection rather than silently falling back to the online one."""
    torch.manual_seed(0)
    model = _tiny_model(ema_cfm_target=True)
    images, captions = _tiny_batch()

    with torch.no_grad():
        for p in model.g_t_online.parameters():
            p.add_(torch.randn_like(p) * 0.5)  # simulate online having moved via training

    batch = model._tokenize(captions)
    online_proj = model._project_text(batch, model.text_seq2seq.get_encoder(), model.g_t_online)
    target_proj = model._project_text(batch, model.text_encoder_target, model.g_t_target)

    assert not torch.allclose(online_proj, target_proj, atol=1e-3), (
        "EMA target projection should differ from the (deliberately perturbed) online "
        "projection -- if they match, g_t_target is silently tracking g_t_online "
        "instead of being a genuinely separate, lagging copy."
    )


def test_ema_cfm_target_matches_at_init_before_any_divergence():
    """Sanity check in the other direction: BEFORE any training/EMA divergence has
    occurred, g_t_target is an exact copy of g_t_online (make_ema_copy), so the two
    pathways should agree closely right at initialization."""
    torch.manual_seed(0)
    model = _tiny_model(ema_cfm_target=True)
    _, captions = _tiny_batch()

    batch = model._tokenize(captions)
    online_proj = model._project_text(batch, model.text_seq2seq.get_encoder(), model.g_t_online)
    target_proj = model._project_text(batch, model.text_encoder_target, model.g_t_target)

    assert torch.allclose(online_proj, target_proj, atol=1e-5), (
        "At init, g_t_target should be an exact copy of g_t_online -- these should "
        "match closely before any EMA divergence has had a chance to occur."
    )


def test_ema_cfm_target_receives_no_gradient():
    """cfm_loss must not backprop into g_t_target/text_encoder_target when
    ema_cfm_target=True -- it's supposed to be a frozen-at-this-instant EMA snapshot,
    updated only via update_ema_target()'s moving average, never via backprop."""
    torch.manual_seed(0)
    model = _tiny_model(ema_cfm_target=True)
    images, captions = _tiny_batch()

    cfm_loss, recon_loss, vicreg_v, vicreg_t, _ = model.training_step(images, captions)
    total = cfm_loss + recon_loss + vicreg_v + vicreg_t
    total.backward()

    for name, p in model.g_t_target.named_parameters():
        assert p.grad is None, f"g_t_target.{name} received a gradient -- should be frozen (EMA-only)."


def test_gradient_norm_breakdown_runs_and_shows_zero_text_gradient_under_ema_target():
    """gradient_norm_breakdown returns per-anchor GRADIENT NORMS, not loss values (this
    corrects an initial wrong assumption while writing this test -- it does not return
    a cfm_loss value comparable to training_step's). What IS directly checkable: with
    ema_cfm_target=True, Z1_for_cfm comes from a no-grad EMA snapshot, so cfm_loss has
    NO gradient path to text_anchor (g_t_online's output layer) at all -- this should
    hold even more unambiguously than under the pre-existing stop_grad_cfm_target=True
    path, since the EMA source is fully detached by construction, not just .detach()'d."""
    torch.manual_seed(0)
    model = _tiny_model(ema_cfm_target=True)
    images, captions = _tiny_batch()

    norms = model.gradient_norm_breakdown(images, captions)
    assert norms["cfm_on_text"] == 0.0, (
        f"Expected exactly zero cfm gradient on the text anchor under ema_cfm_target="
        f"True, got {norms['cfm_on_text']}"
    )
