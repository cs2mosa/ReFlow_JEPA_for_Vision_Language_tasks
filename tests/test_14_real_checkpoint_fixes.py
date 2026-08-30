"""
Regression tests for two correctness fixes made in preparation for real checkpoints
(facebook/ijepa_vith14_1k, google/t5-v1_1-base), neither of which could be exercised by
the mock path (fixed-length mock tokenizer never pads; mock ViT has random weights so
input normalization was never load-bearing) -- these bugs were silent until now, not
because the mock path happened to avoid them, but because the mock path's specific
properties (no padding, no expectation about input scale) never triggered them.

1. Padding-loss masking: training_step/gradient_norm_breakdown passed
   labels=batch["input_ids"] directly to the T5 loss, with no masking of padded
   positions. A real tokenizer with variable-length captions WILL produce padding;
   without masking (-100, HF's ignore_index), the model would be trained to predict
   the pad token at padded positions as if it were a real target.

2. Image normalization: synthetic_data.py's images are raw [0,1] tensors with no
   normalization applied anywhere. Harmless for the mock encoder (random weights, no
   expectation about input distribution) but would silently produce poor features from
   a real pretrained encoder, which expects a specific input distribution.

Both are tested here using constructions that don't require internet access -- the
mock encoder/tokenizer, with deliberately non-trivial inputs (manually injected
padding; manually overridden normalization buffers) standing in for what a real
checkpoint would actually encounter.
"""
import torch

from reflow_jepa import ReflowJEPA


def _tiny_model():
    return ReflowJEPA(visual_layers=1, text_layers=1, predictor_depth=2, predictor_heads=4)


def test_padding_positions_are_masked_from_recon_loss():
    """Directly tests the masking logic used in training_step/gradient_norm_breakdown:
    labels[attention_mask == 0] = -100. Constructs a batch with SOME positions marked
    as padding (attention_mask=0), computes the loss twice with DIFFERENT token ids at
    those padded positions -- since they should be masked out entirely, the loss must
    be IDENTICAL regardless of what garbage sits at a padded position."""
    torch.manual_seed(0)
    model = _tiny_model()
    B, L = 4, 10
    pad_from = 6  # positions [pad_from:] are "padding" in this synthetic batch

    input_ids_a = torch.randint(1, 100, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    attention_mask[:, pad_from:] = 0

    input_ids_b = input_ids_a.clone()
    input_ids_b[:, pad_from:] = torch.randint(1, 100, (B, L - pad_from))  # different padding content

    prefix = model.prefix_expand(torch.nn.functional.normalize(torch.randn(B, 768), dim=-1))

    def masked_loss(input_ids):
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return model.text_seq2seq(encoder_outputs=(prefix,), labels=labels).loss

    loss_a = masked_loss(input_ids_a)
    loss_b = masked_loss(input_ids_b)

    assert torch.allclose(loss_a, loss_b, atol=1e-5), (
        f"Loss should be identical regardless of what token ids sit at MASKED padded "
        f"positions, but got {loss_a.item():.6f} vs {loss_b.item():.6f} -- masking is "
        f"not actually preventing padded content from affecting the loss."
    )

    # sanity: WITHOUT masking, these two SHOULD differ (confirms the test setup itself
    # is meaningful -- the two input_ids batches really do differ where it matters)
    unmasked_loss_a = model.text_seq2seq(encoder_outputs=(prefix,), labels=input_ids_a).loss
    unmasked_loss_b = model.text_seq2seq(encoder_outputs=(prefix,), labels=input_ids_b).loss
    assert not torch.allclose(unmasked_loss_a, unmasked_loss_b, atol=1e-5), (
        "Test setup issue: unmasked losses should differ (different padded content) -- "
        "if they don't, this test isn't actually exercising anything meaningful."
    )


def test_image_normalization_buffers_are_applied():
    """Confirms image_mean/image_std buffers are actually wired into _visual_forward's
    computation, not just stored inertly. Overrides them with non-trivial values
    (simulating what a real checkpoint's AutoImageProcessor stats would look like) and
    confirms the encoder's output actually changes as a result -- if normalization
    were silently skipped, changing these buffers would have no effect."""
    torch.manual_seed(0)
    model = _tiny_model()
    B = 4
    images = torch.rand(B, 3, 224, 224)  # matches synthetic_data.py's raw [0,1] range
    c = model.task_token.expand(B, -1)

    # default buffers (mock path): mean=0, std=1, identity normalization
    assert torch.allclose(model.image_mean, torch.zeros(1, 3, 1, 1))
    assert torch.allclose(model.image_std, torch.ones(1, 3, 1, 1))
    out_identity = model.encode_visual(images, c)

    # override with non-trivial stats (standard ImageNet-style values, representative
    # of what a real checkpoint's processor would provide)
    model.image_mean.copy_(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    model.image_std.copy_(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    out_normalized = model.encode_visual(images, c)

    assert not torch.allclose(out_identity, out_normalized, atol=1e-4), (
        "Changing image_mean/image_std had no effect on encode_visual's output -- "
        "normalization is not actually being applied before the visual encoder."
    )


def test_mock_path_buffers_are_identity_by_default():
    """Explicit check that the mock path (real_checkpoint=False, the default) gets
    identity normalization -- confirms this fix is a genuine no-op for every existing
    mock-path test, not an accidental behavior change."""
    model = _tiny_model()
    assert torch.allclose(model.image_mean, torch.zeros(1, 3, 1, 1))
    assert torch.allclose(model.image_std, torch.ones(1, 3, 1, 1))
