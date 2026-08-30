"""
Canonical visual/text encoder loaders. Single source of truth: tests/conftest.py and
src/train.py both import from here, so what gets tested is exactly what gets trained --
no drift between a "test version" and a "real version" of the encoder wiring.

IMPORTANT: this sandbox has no route to huggingface.co / real checkpoint hosts, so these
loaders instantiate encoders from their REAL config classes (correct architecture, correct
output shapes, correct tokenizer/model interface) but with RANDOM weights, via
`.from_config()` instead of `.from_pretrained()`. Swap exactly the two factory functions
below to `.from_pretrained(...)` once real checkpoints are reachable (e.g. on Kaggle,
which does have internet access if the notebook's "Internet" toggle is on) -- nothing
else in the codebase needs to change, every module is written against the interface.
"""
import hashlib
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, T5Config, T5ForConditionalGeneration

D_SHARED = 768          # calibrated shared latent space, matches Reflow-JEPA v3 spec.
                         # Also equals T5-base's d_model, so Prefix-Expand's output
                         # needs no extra width projection before the T5 decoder.
D_IJEPA = 1280           # ViT-H/14 hidden size (corrected dimension, Reflow-JEPA v3 §5.1)
P_PATCHES = 256          # ViT-H/14 patch count at 224x224
D_TEXT = 768             # T5-base hidden size
K_QUERY_SLOTS = 8        # Q-Pool learned query slots (Reflow-JEPA v3 §5.2)
K_PREFIX_TOKENS = 8      # Prefix-Expand pseudo-sequence length


def load_visual_encoder(num_layers: int = 2, real_checkpoint: bool = False):
    """
    Real I-JEPA is ViT-H/14, patch_size=14, image_size=224, hidden_size=1280.
    `num_layers` only affects wall-clock time in the mock-weight path; it does not
    affect any shape/scale/statistics assumption downstream, which only depends on
    hidden_size and patch_size.

    real_checkpoint=True switches to .from_pretrained (requires internet + hub access,
    e.g. on Kaggle with the Internet toggle on). Verified current as of this writing:
    "facebook/ijepa_vith14_1k" is a real, presently-hosted checkpoint (HF transformers
    docs' own I-JEPA usage example uses this exact identifier).

    Returns (model, image_mean, image_std) -- NOT just the model. The mean/std are
    (1,3,1,1)-shaped tensors for normalizing raw [0,1] images before the encoder sees
    them. This is necessary and was missing before this fix: our synthetic images
    (synthetic_data.py's render_shape) are raw [0,1] pixel tensors with NO
    normalization applied anywhere in the pipeline. That's harmless for the mock path
    (random weights have no expectation about input distribution), but feeding
    un-normalized images directly into REAL pretrained I-JEPA weights would produce
    garbage features -- the encoder was never trained on inputs in that distribution.
    For real_checkpoint=True, the returned mean/std come from
    AutoImageProcessor.from_pretrained (the officially documented preprocessing for
    this checkpoint), not hand-guessed constants. For the mock path, mean=0/std=1
    (identity, a no-op) so nothing changes for any existing mock-path test.
    """
    if real_checkpoint:
        from transformers import AutoImageProcessor
        model = ViTModel.from_pretrained("facebook/ijepa_vith14_1k")
        processor = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        image_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1).float()
        image_std = torch.tensor(processor.image_std).view(1, 3, 1, 1).float()
    else:
        cfg = ViTConfig(
            image_size=224, patch_size=14, hidden_size=D_IJEPA,
            num_hidden_layers=num_layers, num_attention_heads=16, intermediate_size=5120,
        )
        model = ViTModel(cfg)
        image_mean = torch.zeros(1, 3, 1, 1)
        image_std = torch.ones(1, 3, 1, 1)
    model.eval()
    return model, image_mean, image_std


def load_text_seq2seq(num_layers: int = 2, real_checkpoint: bool = False):
    """
    Encoder AND decoder from the SAME T5 checkpoint (see project notes on exposure-bias
    consistency: the decoder must only ever be asked to decode vectors from a space it
    was jointly pretrained against).

    Returns the FULL T5ForConditionalGeneration model, and a tokenizer. Use
    .get_encoder() / .get_decoder() to access the two halves individually.
    """
    if real_checkpoint:
        from transformers import T5TokenizerFast
        model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")
        tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-base")
        return model, tok
    cfg = T5Config(
        d_model=D_TEXT, num_layers=num_layers, num_decoder_layers=num_layers,
        num_heads=12, d_ff=2048, vocab_size=32128, is_encoder_decoder=True,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    return model, _MockTokenizer(vocab_size=32128)


def load_text_encoder(num_layers: int = 2, real_checkpoint: bool = False):
    """Back-compat shim: derived from the SAME paired seq2seq model, not a standalone
    T5EncoderModel, so there is exactly one place the text model is instantiated."""
    model, tok = load_text_seq2seq(num_layers, real_checkpoint=real_checkpoint)
    return model.get_encoder(), tok


class _MockTokenizer:
    """Deterministic hash-based tokenizer standing in for T5TokenizerFast in this sandbox.
    Same string -> same ids always -- INCLUDING across separate process invocations.

    Bug fixed here: this used to use Python's built-in hash(), which is randomized
    per-process by default (PYTHONHASHSEED) for security reasons. That made token ids
    for the same caption string DIFFERENT every time a new process started -- so
    train.py (one process) and diagnose_recon_signal.py (a separate process run later
    against the saved checkpoint) tokenized the identical caption into different ids.
    A checkpoint trained against one process's token ids, evaluated with another
    process's token ids as "ground truth," will look like it has learned nothing --
    not because it hasn't, but because the labels being checked against were never
    the labels it was trained on. hashlib.md5 has no such per-process randomization,
    so ids are now stable across every future process, matching the docstring's
    original claim (which used to be false in practice)."""
    def __init__(self, vocab_size, max_len=16):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def _stable_hash(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            ids = [1 + (self._stable_hash(f"{t}|{i}") % (self.vocab_size - 2)) for i in range(self.max_len)]
            seqs.append(ids)
        input_ids = torch.tensor(seqs, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}


def make_ema_copy(module: nn.Module) -> nn.Module:
    """Deep-copies a module for use as a same-modality EMA target network
    (I-JEPA/BYOL/DINO precedent, OFM-JEPA v2 Mitigation 3 item 2 -- corrected to track
    the SAME modality's online copy, not a cross-modal one)."""
    import copy
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """theta' <- momentum*theta' + (1-momentum)*theta, same-modality EMA."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(momentum).add_(po.detach(), alpha=1 - momentum)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)
