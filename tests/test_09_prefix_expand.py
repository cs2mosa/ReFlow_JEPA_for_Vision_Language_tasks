"""
PHASE 1i -- Prefix-Expand, tested in isolation and then against the real T5 decoder
it feeds.

This module doesn't exist in the original design docs; it's required by this
project's choice to decode via the flow's own paired T5 decoder (Pipeline 2,
OFM-JEPA v2 SS4) rather than a decoder-only LM. See src/prefix_expand.py's docstring
for why. Same testing philosophy as test_04_qpool_module.py: it's OUR trainable
module, so we test it fully, including gradient flow and an end-to-end decode.

Targets:
  - test_output_shape                    -> interface contract, (B, K', d_shared)
  - test_gradients_flow_to_all_slots      -> a dead expansion slot silently reduces K'
                                             the same way a dead Q-Pool query slot
                                             silently reduces K (test_04's rationale,
                                             mirrored here for the inverse operation)
  - test_output_depends_on_input          -> catches a collapsed/constant expansion
                                             (e.g. self-attention washing out the FiLM
                                             signal entirely across K' tokens)
  - test_decoder_accepts_prefix_and_produces_logits -> the actual integration point:
                                             confirms a real T5 decoder can consume
                                             Prefix-Expand's output as cross-attention
                                             memory and produce well-formed logits,
                                             end to end, with mock weights. Catches any
                                             shape/dtype/API mismatch between our module
                                             and the exact decoder we plan to reuse.
"""
import torch
from conftest import D_SHARED, K_PREFIX_TOKENS
from prefix_expand import PrefixExpand


def test_output_shape():
    pe = PrefixExpand(d_shared=D_SHARED, k_prefix=K_PREFIX_TOKENS)
    z_hat = torch.randn(4, D_SHARED)
    out = pe(z_hat)
    assert out.shape == (4, K_PREFIX_TOKENS, D_SHARED)


def test_gradients_flow_to_all_slots():
    pe = PrefixExpand(d_shared=D_SHARED, k_prefix=K_PREFIX_TOKENS)
    z_hat = torch.randn(2, D_SHARED)
    out = pe(z_hat)
    loss = out.pow(2).sum()
    loss.backward()
    assert pe.expansion_slots.grad is not None
    per_slot_grad_norm = pe.expansion_slots.grad.norm(dim=-1)
    dead_slots = (per_slot_grad_norm < 1e-8).sum().item()
    assert dead_slots == 0, (
        f"{dead_slots}/{pe.k} expansion slots received zero gradient -- silently "
        f"reduces the effective prefix length K' below what was configured, exactly "
        f"the failure test_04 guards against for Q-Pool's query slots."
    )


def test_output_depends_on_input():
    pe = PrefixExpand(d_shared=D_SHARED, k_prefix=K_PREFIX_TOKENS)
    pe.eval()
    z1 = torch.randn(1, D_SHARED)
    z2 = torch.randn(1, D_SHARED)
    with torch.no_grad():
        out1, out2 = pe(z1), pe(z2)
    assert not torch.allclose(out1, out2, atol=1e-4), (
        "Prefix-Expand produced (near-)identical pseudo-sequences for two different "
        "flow outputs -- the FiLM conditioning is not actually reaching the decoder's "
        "cross-attention memory, i.e. the decoder would generate the same text "
        "regardless of what the flow produced."
    )


def test_decoder_accepts_prefix_and_produces_logits(text_seq2seq_and_tokenizer):
    """End-to-end integration check: PrefixExpand's output, used verbatim as
    encoder_hidden_states, must be something the real T5 decoder can cross-attend
    over and produce a valid logits tensor from -- the actual mechanical claim
    Pipeline 2 depends on."""
    model, tok = text_seq2seq_and_tokenizer
    pe = PrefixExpand(d_shared=D_SHARED, k_prefix=K_PREFIX_TOKENS)
    z_hat = torch.randn(2, D_SHARED)
    with torch.no_grad():
        prefix = pe(z_hat)  # (B, K', d) stands in for encoder's last_hidden_state
        decoder_input_ids = torch.tensor([[0, 1, 2, 3]] * 2, dtype=torch.long)  # dummy target prefix
        out = model(
            encoder_outputs=(prefix,),
            decoder_input_ids=decoder_input_ids,
        )
    assert out.logits.shape == (2, 4, model.config.vocab_size), (
        f"T5 decoder did not produce the expected logits shape when fed "
        f"Prefix-Expand's output as cross-attention memory: got {tuple(out.logits.shape)}."
    )
    assert torch.isfinite(out.logits).all(), (
        "Decoder logits contain non-finite values when cross-attending over "
        "Prefix-Expand's output -- check the un-pooling module's output scale."
    )
