"""
Shared fixtures for the Reflow-JEPA pre-implementation test suite.

IMPORTANT — read before running:
This sandbox has no route to huggingface.co / the Meta I-JEPA checkpoint host, so these
tests instantiate encoders from their REAL config classes (correct architecture, correct
output shapes, correct tokenizer/model interface) but with RANDOM weights, via `.from_config()`
instead of `.from_pretrained()`. That is enough to catch every *structural* problem this
suite targets (shape mismatches, scale mismatches, collapse-guard bugs, hubness, the
stochastic-source mechanism, the terminal-time divergence rate) because none of those are
about what the weights *learned* — they are about whether the pieces fit together and
whether our own math-derived mechanisms behave the way we proved they should.

To run against the real checkpoints once you have Hub access, change EXACTLY the two
factory functions below (`load_visual_encoder`, `load_text_encoder`) to call
`.from_pretrained(...)` instead of `.from_config(...)`. Nothing else in this suite needs
to change — every test is written against the *interface*, not the mock.
"""
import torch
import pytest
from transformers import ViTConfig, ViTModel, T5Config, T5EncoderModel, T5ForConditionalGeneration, T5TokenizerFast

SEED = 0
D_SHARED = 768          # calibrated shared latent space, matches Reflow-JEPA v3 spec.
                         # Also equals T5-base's d_model, so Prefix-Expand's output
                         # needs no extra width projection before the T5 decoder.
D_IJEPA = 1280           # ViT-H/14 hidden size (corrected dimension, matches Reflow-JEPA v3 §5.1)
P_PATCHES = 256          # ViT-H/14 patch count at 224x224 (corrected dimension)
D_TEXT = 768             # T5-base hidden size
K_QUERY_SLOTS = 8        # Q-Pool learned query slots (Reflow-JEPA v3 §5.2)
K_PREFIX_TOKENS = 8      # Prefix-Expand pseudo-sequence length fed to the T5 decoder
                         # as cross-attention memory in place of its native encoder
                         # output (Pipeline 2, OFM-JEPA v2 §4). Matches K_QUERY_SLOTS
                         # by default; independent knob, sweep separately if needed.


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(SEED)
    yield


def load_visual_encoder(num_layers: int = 2):
    """
    Real I-JEPA is ViT-H/14, patch_size=14, image_size=224, hidden_size=1280.
    `num_layers` is kept small here purely for test wall-clock time; it does not
    affect any of the shape/scale/statistics checks below, which only depend on
    hidden_size and patch_size.

    SWAP FOR REAL CHECKPOINT:
        from transformers import ViTModel
        model = ViTModel.from_pretrained("facebook/ijepa_vith14_1k")  # or your local path
    """
    cfg = ViTConfig(
        image_size=224, patch_size=14, hidden_size=D_IJEPA,
        num_hidden_layers=num_layers, num_attention_heads=16, intermediate_size=5120,
    )
    model = ViTModel(cfg)
    model.eval()
    return model


def load_text_seq2seq(num_layers: int = 2):
    """
    General-VL pretraining track: the ENCODER and DECODER must come from the same
    T5 checkpoint, not a separately-chosen decoder-only LM. This is deliberate --
    Pipeline 2 (OFM-JEPA v2 §4) decodes by feeding a projected/pseudo-sequence prefix
    into the decoder's cross-attention; using the encoder's own paired decoder keeps
    the encoder/decoder representation spaces aligned by construction (same pretraining
    objective, same subword space) rather than hoping two independently-trained models'
    embedding geometries happen to match. This is what "prevents exposure bias" means
    here in practice: the decoder is only ever asked to decode vectors from a space
    it was jointly trained against, not an alien one.

    Returns the FULL T5ForConditionalGeneration model, and the tokenizer.
    Use .get_encoder() / .get_decoder() to access the two halves individually.

    SWAP FOR REAL CHECKPOINT:
        from transformers import T5ForConditionalGeneration, T5TokenizerFast
        model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")
        tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-base")
    """
    cfg = T5Config(
        d_model=D_TEXT, num_layers=num_layers, num_decoder_layers=num_layers,
        num_heads=12, d_ff=2048, vocab_size=32128, is_encoder_decoder=True,
    )
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    # Real T5 tokenizer needs sentencepiece + a hub download; for structural testing we
    # use a tiny local whitespace tokenizer that mimics the interface (encode -> ids).
    return model, _MockTokenizer(vocab_size=32128)


def load_text_encoder(num_layers: int = 2):
    """
    Back-compat shim for tests 02/03, which only need the encoder half. Derived from
    the SAME paired seq2seq model as load_text_seq2seq (not a standalone T5EncoderModel)
    so there is exactly one place the text model is instantiated.
    """
    model, tok = load_text_seq2seq(num_layers)
    return model.get_encoder(), tok


class _MockTokenizer:
    """Deterministic hash-based tokenizer standing in for T5TokenizerFast in this sandbox.
    Same string -> same ids always (needed for the duplicate-detection test); different
    strings -> (almost certainly) different ids. Swap for the real tokenizer with real weights."""
    def __init__(self, vocab_size, max_len=8):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ids = [1 + (hash((t, i)) % (self.vocab_size - 2)) for i in range(self.max_len)]
            seqs.append(ids)
        input_ids = torch.tensor(seqs, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}


@pytest.fixture
def visual_encoder():
    return load_visual_encoder()


@pytest.fixture
def text_encoder_and_tokenizer():
    return load_text_encoder()


@pytest.fixture
def text_seq2seq_and_tokenizer():
    return load_text_seq2seq()
