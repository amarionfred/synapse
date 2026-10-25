"""Behavioral checks on actual small Transformers architectures; no downloads."""
import os
os.environ["EXPLORER_NO_MODEL"] = "1"
os.environ["EXPLORER_BUNDLED_CACHE"] = "1"
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"explorer"))
import pytest
import torch
import transformers as hf
from tokenizers import Tokenizer, models, pre_tokenizers
from adapters import DenseAdapter, UnsupportedArchitecture
from runtime import Runtime

torch.set_num_threads(2)


def tokenizer():
    t = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                  **{f"t{i}": i for i in range(3, 41)}}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = hf.PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[UNK]")
    tok.chat_template = "{% for message in messages %}{{message['content']}} {% endfor %}"
    return tok


def model(family):
    cfg = dict(vocab_size=41, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
               head_dim=6, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if family == "Qwen3_5":
        text = dict(vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                    linear_value_head_dim=8, linear_conv_kernel_dim=4, max_position_embeddings=128,
                    layer_types=["linear_attention"]*3+["full_attention"], bos_token_id=1, eos_token_id=2, pad_token_id=0)
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32,
                      patch_size=2, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64)
        return hf.Qwen3_5ForConditionalGeneration(hf.Qwen3_5Config(text_config=text, vision_config=vision)).eval()
    if family == "GPT2":
        c = hf.GPT2Config(vocab_size=41, n_embd=24, n_inner=48, n_layer=2, n_head=4,
                         n_positions=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        return hf.GPT2LMHeadModel(c).eval()
    if family == "GPTNeoX":
        cfg.update(rotary_pct=1.0)
    if family == "OPT":
        cfg.update(ffn_dim=48, word_embed_proj_dim=24)
    if family == "Gemma2":
        cfg.update(query_pre_attn_scalar=6, sliding_window=16)
    if family == "Phi3":
        cfg.update(original_max_position_embeddings=128)
    return getattr(hf, family+"ForCausalLM")(getattr(hf, family+"Config")(**cfg)).eval()


FAMILIES = ["Llama", "Mistral", "Qwen2", "Qwen3", "Qwen3_5", "Gemma2", "GPT2", "GPTNeoX", "OPT", "Phi3"]


@pytest.mark.parametrize("family", FAMILIES)
def test_actual_architecture_interventions(family, tmp_path):
    torch.manual_seed(7)
    m = model(family)
    ids = torch.tensor([[3, 5, 8, 10]])
    with torch.inference_mode():
        before = m(input_ids=ids, use_cache=False).logits.clone()
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(m, tokenizer())
    with torch.inference_mode():
        torch.testing.assert_close(m(input_ids=ids, use_cache=False).logits, before, rtol=0, atol=0)
    acts = rt.full_activations(ids[0].tolist())
    assert acts.shape == (len(rt.adapter.layers), 4, rt.adapter.down[0].width)
    with torch.inference_mode():
        torch.testing.assert_close(m(input_ids=ids, use_cache=False).logits, before, rtol=0, atol=0)
    p = rt.adapter.down[0]
    x = torch.randn(3, 4, p.width)
    with torch.inference_mode():
        base = p.module(x)
        rt._push = [(0, 3, "add", torch.tensor([.3, 0., -.3]))]
        shifted = p.module(x)
        rt._push = None
        expected = torch.tensor([.3, 0., -.3])[:,None,None] * p.weight[:,3][None,None,:]
        torch.testing.assert_close(shifted-base, expected.expand_as(base), rtol=1e-4, atol=1e-6)
        rt._push = [(0, None, "mul", torch.zeros(3))]
        zeroed = p.module(x)
        rt._push = None
        bias = p.module.bias
        torch.testing.assert_close(zeroed, torch.zeros_like(base) if bias is None else bias.expand_as(base))
    # A zero steering dose must give three identical greedy continuations and clean up hooks.
    rows = [[{"role":"user", "content":"t3 t5 t8"}]] * 3
    rt.resid_p50[0] = 1.
    entry = rt._push_entry(0, 3, .2, "add", (1, 0, -1))
    lengths = (entry[3][:,None] * rt.down_col(0,3)).norm(dim=1)
    torch.testing.assert_close(lengths, torch.tensor([.2, 0., .2]))
    texts = ["", "", ""]
    for row, delta in rt.steer_multi([(0, 3, 0., "add")], rows, max_new=3):
        if row is not None:
            texts[row] += delta
    assert texts[0] == texts[1] == texts[2]
    assert rt._push is None
    with torch.inference_mode():
        torch.testing.assert_close(m(input_ids=ids, use_cache=False).logits, before, rtol=0, atol=0)


def test_invalid_neurons_and_cleanup(tmp_path):
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(model("Llama"), tokenizer())
    for layer, neuron, dose, mode in [(-1, 0, 1., "add"), (0, -1, 1., "add"), (0, 999, 1., "add"),
                                       (0, 0, float("nan"), "add"), (0, 0, 1., "unknown")]:
        with pytest.raises(ValueError):
            list(rt.steer_multi([(layer, neuron, dose, mode)], [[{"role":"user","content":"t3"}]]*3, 1))
        assert rt._push is None


def test_unsupported_moe():
    m = model("Llama")
    m.config.num_experts = 4
    with pytest.raises(UnsupportedArchitecture, match="MoE"):
        DenseAdapter(m)


def test_checkpoint_isolation(tmp_path):
    from model_config import checkpoint_identity
    assert checkpoint_identity("model-a", "rev-1")[1] != checkpoint_identity("model-b", "rev-1")[1]
    assert checkpoint_identity("model-a", "rev-1")[1] != checkpoint_identity("model-a", "rev-2")[1]
    f = tmp_path/"model.safetensors"
    f.write_bytes(b"old")
    key = checkpoint_identity(str(tmp_path), "local")[1]
    f.write_bytes(b"edited-weights")
    assert checkpoint_identity(str(tmp_path), "local")[1] != key
