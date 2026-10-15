"""Tests for sm.common (registry, models, hooks, io). Offline with SM_OFFLINE=1; no downloads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SM_OFFLINE", "1")

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sm.common import hooks, io, models, registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PROMPT_MSGS = [
    [{"role": "user", "content": "How do I bake bread?"}],
    [{"role": "user", "content": "What is the capital of France? Answer briefly please."}],
    [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain the refusal direction."}],
]


def _prompts(tok) -> list[str]:
    return [models.render_chat(tok, m) for m in PROMPT_MSGS]


def _logits(model, tok, text: str):
    ids = torch.tensor([models.encode_text(tok, text)])
    with torch.no_grad():
        return model(input_ids=ids).logits[0, -1].clone()


# --------------------------------------------------------------------------- registry


def test_registry_has_bases_fallbacks_and_tiny():
    b = registry.BASES
    assert set(registry.PRIMARY_KEYS) == {"qwen3.5-9b", "qwen3.8-27b", "gemma-4-12b-it", "gemma-4-31b-it"}
    assert set(registry.FALLBACK_KEYS) == {"qwen3-8b", "gemma-3-12b-it", "gemma-3-27b-it"}
    assert "tiny" in b
    hf = {k: v["hf"] for k, v in b.items()}
    assert hf["qwen3.5-9b"] == "Qwen/Qwen3.5-9B"
    assert hf["qwen3.8-27b"] == "Qwen/Qwen3.8-27B"
    assert hf["gemma-4-12b-it"] == "google/gemma-4-12B-it"
    assert hf["gemma-4-31b-it"] == "google/gemma-4-31B-it"
    for k, v in b.items():
        for field in ("hf", "revision", "family", "thinking_switch", "fallback"):
            assert field in v, (k, field)
        assert v["family"] in registry.FAMILIES
        assert v["thinking_switch"] in registry.THINKING_SWITCHES
        assert v["fallback"] is None or v["fallback"] in b
    assert b["qwen3.5-9b"]["thinking_switch"] == "enable_thinking"
    assert b["gemma-4-31b-it"]["thinking_switch"] == "gemma_think"
    assert b["gemma-3-27b-it"]["thinking_switch"] is None
    assert registry.fallback_chain("gemma-4-31b-it") == ["gemma-4-31b-it", "gemma-3-27b-it"]


def test_resolve_base_key_hf_local_and_tiny_override(tiny_model_dir, monkeypatch):
    e = registry.resolve_base("qwen3.5-9b")
    assert e["hf"] == "Qwen/Qwen3.5-9B" and e["key"] == "qwen3.5-9b"
    e["hf"] = "mutated"
    assert registry.BASES["qwen3.5-9b"]["hf"] == "Qwen/Qwen3.5-9B"  # copy, not a view
    assert registry.resolve_base("qwen/qwen3-8b")["key"] == "qwen3-8b"
    other = registry.resolve_base("someone/Qwen3-14B")
    assert other["key"] is None and other["family"] == "qwen"
    if Path(tiny_model_dir).is_dir():
        loc = registry.resolve_base(tiny_model_dir)
        assert loc["role"] == "local" and loc["family"] == "qwen"
    monkeypatch.setenv("SM_TINY_MODEL", "/some/where/tiny")
    assert registry.resolve_base("tiny")["hf"] == "/some/where/tiny"
    monkeypatch.delenv("SM_TINY_MODEL")
    assert registry.resolve_base("tiny")["hf"] == registry.DEFAULT_TINY_HF


V2_PINS = {
    "qwen3.5-9b": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "qwen3.8-27b": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "gemma-4-12b-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "gemma-4-31b-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
}


def test_bases_yaml_mirrors_registry_and_overlays_revision(tmp_path):
    import yaml

    ypath = ROOT / "configs" / "bases.yaml"
    y = registry.load_bases_yaml(ypath)
    assert set(y) == set(registry.BASES)
    for k, v in registry.BASES.items():
        for field in ("hf", "family", "thinking_switch", "fallback", "role"):
            assert y[k][field] == v[field], (k, field)
        if k in V2_PINS:
            assert y[k]["revision"] == V2_PINS[k], k
        else:
            assert y[k]["revision"] is None, k  # fallbacks and tiny stay unpinned [F-pin]
    pinned = {"bases": copy.deepcopy(y)}
    pinned["bases"]["qwen3-8b"]["revision"] = "a" * 40
    p = tmp_path / "bases.yaml"
    p.write_text(yaml.safe_dump(pinned))
    assert registry.resolve_base("qwen3-8b", bases_yaml=p)["revision"] == "a" * 40
    assert registry.resolve_base("qwen3-8b")["revision"] is None


# --------------------------------------------------------------------------- models


def test_load_model_tiny_path(tiny_model_dir, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen: list[dict] = []
    real = transformers.AutoConfig.from_pretrained

    def spy(*a, **kw):
        seen.append(dict(kw))
        return real(*a, **kw)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", spy)
    model, tok, meta = models.load_model(tiny_model_dir, dtype="fp32", device="cpu")
    assert seen and seen[0].get("local_files_only") is True
    for k in ("hf", "revision", "adapter", "dtype", "n_layers", "hidden_size", "loaded_at"):
        assert k in meta
    assert meta["local_files_only"] is True
    assert meta["dtype"] == "float32" and meta["adapter"] is None
    assert meta["n_layers"] == len(models.get_decoder_layers(model)) == model.config.num_hidden_layers
    assert meta["hidden_size"] == model.config.hidden_size
    assert meta["layers_path"] == "model.layers"
    assert tok.padding_side == "left" and tok.pad_token_id is not None
    assert not model.training
    assert hasattr(model, "generate") and hasattr(model, "lm_head")


def test_load_model_bf16_default(tiny_model_dir):
    model, _, meta = models.load_model(tiny_model_dir, device="cpu")
    assert meta["dtype"] == "bfloat16"
    assert next(model.parameters()).dtype == torch.bfloat16


def test_load_model_with_lora_adapter(tiny_model, tiny_model_dir, tmp_path):
    from peft import LoraConfig, get_peft_model

    base, tok, _ = tiny_model
    text = _prompts(tok)[0]
    ref_base = _logits(base, tok, text)
    fresh, _, _ = models.load_model(tiny_model_dir, dtype="fp32", device="cpu", attn="eager")
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False)
    peft_model = get_peft_model(fresh, cfg)
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                p.normal_(0.0, 0.5)
    peft_model.eval()
    ref_peft = _logits(peft_model, tok, text)
    adir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adir))

    merged, _, meta = models.load_model(
        tiny_model_dir, adapter_path=str(adir), dtype="fp32", device="cpu", attn="eager", merge_adapter=True
    )
    assert meta["adapter_merged"] and meta["adapter_sha256"]
    assert type(merged).__name__ == "Qwen3ForCausalLM"
    got = _logits(merged, tok, text)
    assert torch.allclose(got, ref_peft, atol=1e-4)
    assert not torch.allclose(got, ref_base, atol=1e-3)

    # Default: the adapter stays live (no bf16 merge rounding); hooks see the decoder through PEFT.
    unmerged, _, meta2 = models.load_model(tiny_model_dir, adapter_path=str(adir), dtype="fp32", device="cpu", attn="eager")
    assert not meta2["adapter_merged"]
    assert meta2["n_layers"] == meta["n_layers"]
    assert torch.allclose(_logits(unmerged, tok, text), ref_peft, atol=1e-4)


def test_render_chat_thinking_off_and_meta(tiny_model):
    _, tok, _ = tiny_model
    msgs = [{"role": "user", "content": "Hello there"}]
    s = models.render_chat(tok, msgs)
    assert isinstance(s, str) and "Hello there" in s
    assert s.rstrip().endswith("</think>")  # Qwen-style empty think block = thinking disabled
    meta = dict(models.LAST_RENDER_META)
    assert meta["thinking"] is False and meta["thinking_switch"] == "enable_thinking"
    assert meta["template_kwargs"] == {"enable_thinking": False}
    assert len(meta["chat_template_sha256"]) == 64

    on = models.render_chat(tok, msgs, thinking=True)
    assert "</think>" not in on and models.LAST_RENDER_META["thinking"] is True

    no_gen = models.render_chat(tok, msgs + [{"role": "assistant", "content": "Hi"}], add_generation_prompt=False)
    assert no_gen.endswith("<|im_end|>\n")


def test_render_chat_template_without_switch_and_gemma_style(tiny_model):
    _, tok, _ = tiny_model
    msgs = [{"role": "user", "content": "Hi"}]

    plain = copy.deepcopy(tok)
    plain.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    assert models.render_chat(plain, msgs) == "user: Hi\nassistant: "
    assert models.LAST_RENDER_META["thinking_switch"] == "none"
    assert models.LAST_RENDER_META["template_kwargs"] == {}
    with pytest.raises(ValueError):
        models.render_chat(plain, msgs, thinking=True)

    gemma = copy.deepcopy(tok)
    gemma.chat_template = (
        "<bos>{% if enable_thinking is defined and enable_thinking %}<start_of_turn>system\n<|think|><end_of_turn>\n{% endif %}"
        "{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )
    off = models.render_chat(gemma, msgs)
    assert "<|think|>" not in off
    assert models.LAST_RENDER_META["thinking_switch"] == "gemma_think"
    assert models.LAST_RENDER_META["template_kwargs"] == {"enable_thinking": False}
    assert "<|think|>" in models.render_chat(gemma, msgs, thinking=True)

    gemma_unknown = copy.deepcopy(tok)
    gemma_unknown.sm_thinking_switch = "gemma_think"
    gemma_unknown.chat_template = plain.chat_template
    assert "<|think|>" not in models.render_chat(gemma_unknown, msgs)
    assert models.LAST_RENDER_META["thinking_switch"] == "none"


def test_generate_batched_matches_single_and_preserves_order(tiny_model):
    model, tok, _ = tiny_model
    prompts = _prompts(tok)
    batched = models.generate(model, tok, prompts, max_new_tokens=8, batch_size=3)
    single = [models.generate(model, tok, [p], max_new_tokens=8, batch_size=1)[0] for p in prompts]
    assert len(batched) == len(prompts) and all(isinstance(x, str) for x in batched)
    assert batched == single  # left padding + explicit position ids make padding invisible
    fin = models.generate(model, tok, prompts, max_new_tokens=3, batch_size=2, return_finish=True)
    assert [t for t, _ in fin] == [models.generate(model, tok, [p], max_new_tokens=3)[0] for p in prompts]
    det = models.generate_detailed(model, tok, prompts, max_new_tokens=3, batch_size=2)
    for d, (_, trunc) in zip(det, fin):
        assert d["truncated"] == trunc
        assert d["n_new_tokens"] <= 3
        if d["truncated"]:
            assert d["n_new_tokens"] == 3


# --------------------------------------------------------------------------- hooks


def test_capture_residual_shapes_positions_and_convention(tiny_model):
    model, tok, meta = tiny_model
    prompts = _prompts(tok)
    comps = ["Sure, here is how.", "Paris.", "It is a direction in the residual stream."]
    n, H = meta["n_layers"], meta["hidden_size"]

    acts = hooks.capture_residual(model, tok, prompts, batch_size=2)
    assert sorted(acts) == list(range(n + 1))
    for a in acts.values():
        assert a.shape == (len(prompts), H) and a.dtype == np.float32 and np.isfinite(a).all()

    sel = hooks.capture_residual(model, tok, prompts, layers=[0, 2, n], position="last", completions=comps)
    assert sorted(sel) == [0, 2, n]
    resp = hooks.capture_residual(model, tok, prompts, layers=[0, 2, n], position="resp_mean", completions=comps)
    for li in (2, n):
        assert resp[li].shape == (len(prompts), H)
        assert not np.allclose(sel[li], resp[li])
        # appending a completion does not change the last-prompt-token readout (causal model)
        np.testing.assert_allclose(sel[li], acts[li], atol=1e-4)
    with pytest.raises(ValueError):
        hooks.capture_residual(model, tok, prompts, position="resp_mean")

    # padding invariance: a prompt read alone equals the same prompt read in a padded batch
    alone = hooks.capture_residual(model, tok, [prompts[0]], layers=[n], batch_size=1)
    np.testing.assert_allclose(alone[n][0], acts[n][0], atol=1e-4)

    # convention: hidden_states[l] = output of block l-1 (raw, not post-final-norm); [0] = embeddings
    grabbed = {}
    layers = models.get_decoder_layers(model)
    hs = [layers[i].register_forward_hook(lambda m, a, o, i=i: grabbed.__setitem__(i, o[0] if isinstance(o, tuple) else o)) for i in range(n)]
    ids = torch.tensor([models.encode_text(tok, prompts[1])])
    try:
        with torch.no_grad():
            emb = model.get_input_embeddings()(ids)
            model(input_ids=ids)
    finally:
        for h in hs:
            h.remove()
    one = hooks.capture_residual(model, tok, [prompts[1]], layers=[0, 1, n])
    np.testing.assert_allclose(one[0][0], emb[0, -1].numpy(), atol=1e-5)
    np.testing.assert_allclose(one[1][0], grabbed[0][0, -1].numpy(), atol=1e-4)
    np.testing.assert_allclose(one[n][0], grabbed[n - 1][0, -1].numpy(), atol=1e-4)


def test_capture_position_reads_the_eoi_column(tiny_model):
    """v0.3 change 42: the monitor readout at the card's position p is column n_eoi + p of capture_eoi, and -1 is
    the last prompt token; eoi_ids are checked against the prompt suffix."""
    model, tok, meta = tiny_model
    prompts = _prompts(tok)
    n = meta["n_layers"]
    full = hooks.capture_eoi(model, tok, prompts, 3, layers=[1, n], batch_size=2)
    last = hooks.capture_residual(model, tok, prompts, layers=[1, n], batch_size=2)
    for p in (-1, -2, -3):
        got = hooks.capture_position(model, tok, prompts, position=p, layers=[1, n], batch_size=2)
        for li in (1, n):
            assert got[li].shape == (len(prompts), meta["hidden_size"])
            np.testing.assert_allclose(got[li], full[li][:, 3 + p, :], atol=1e-4)
    np.testing.assert_allclose(hooks.capture_position(model, tok, prompts, -1, layers=[n])[n], last[n], atol=1e-4)
    suffix = [list(models.encode_text(tok, q))[-3:] for q in prompts]
    if all(sx == suffix[0] for sx in suffix):  # a shared template suffix verifies; a wrong one fails
        hooks.capture_position(model, tok, prompts, -2, layers=[n], eoi_ids=suffix[0])
        with pytest.raises(ValueError):
            hooks.capture_position(model, tok, prompts, -2, layers=[n], eoi_ids=[suffix[0][0], -5, -6])
    with pytest.raises(ValueError):
        hooks.capture_position(model, tok, prompts, 0, layers=[n])


def test_direction_hooks_ablate_changes_and_restores(tiny_model):
    model, tok, meta = tiny_model
    prompts = _prompts(tok)
    n = meta["n_layers"]
    before = hooks.capture_residual(model, tok, prompts)
    logits_before = _logits(model, tok, prompts[0])
    blocks = models.get_decoder_layers(model)
    emb = model.get_input_embeddings()
    # transformers installs its own persistent output-recording hooks; compare against that baseline
    hooks_before = [dict(b._forward_hooks) for b in blocks] + [dict(emb._forward_hooks)]

    # use the mean last-token direction at the middle layer so the projection is far from zero
    v = before[n // 2].mean(axis=0)
    u = v / np.linalg.norm(v)
    with hooks.DirectionHooks(model, v, mode="ablate") as dh:
        during = hooks.capture_residual(model, tok, prompts)
        logits_during = _logits(model, tok, prompts[0])
        assert dh.calls > 0
    for li in range(n + 1):
        proj = during[li] @ u
        assert np.abs(proj).max() < 1e-3 * max(1.0, np.abs(before[li]).max())
    assert np.abs(before[n // 2] @ u).max() > 0.1
    assert not np.allclose(during[n], before[n], atol=1e-3)
    assert not torch.allclose(logits_during, logits_before, atol=1e-4)

    after = hooks.capture_residual(model, tok, prompts)
    for li in range(n + 1):
        np.testing.assert_array_equal(after[li], before[li])
    assert torch.equal(_logits(model, tok, prompts[0]), logits_before)
    assert [dict(b._forward_hooks) for b in blocks] + [dict(emb._forward_hooks)] == hooks_before

    # restored even when the body raises
    with pytest.raises(RuntimeError):
        with hooks.DirectionHooks(model, v, mode="ablate", layers=[1, 2]):
            raise RuntimeError("boom")
    assert [dict(b._forward_hooks) for b in blocks] + [dict(emb._forward_hooks)] == hooks_before


def test_direction_hooks_ablate_mid_block(tiny_model):
    """With mid_block the MLP never reads the r component its own block's attention wrote."""
    model, tok, meta = tiny_model
    prompts = _prompts(tok)
    n = meta["n_layers"]
    before = hooks.capture_residual(model, tok, prompts)
    v = before[n // 2].mean(axis=0)
    u = torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float32)
    blocks = models.get_decoder_layers(model)
    seen: list[float] = []

    def pre(_mod, args):  # Qwen: post_attention_layernorm reads the mid-block residual
        seen.append(float((args[0].float() @ u).abs().max()))

    handles = [b.post_attention_layernorm.register_forward_pre_hook(pre) for b in blocks]
    try:
        with hooks.DirectionHooks(model, v, mode="ablate", mid_block=False) as dh:
            hooks.capture_residual(model, tok, prompts)
        assert dh.n_mid_block_targets == 0
        without = max(seen)
        seen.clear()
        with hooks.DirectionHooks(model, v, mode="ablate") as dh:
            hooks.capture_residual(model, tok, prompts)
        assert dh.n_mid_block_targets == n
        assert max(seen) < 1e-3 * max(1.0, without) and without > 1e-2
        with hooks.DirectionHooks(model, v, mode="ablate", layers=[1, 2]) as dh:
            pass
        assert dh.n_mid_block_targets == 1
    finally:
        for h in handles:
            h.remove()


def test_direction_hooks_add_and_generate(tiny_model):
    model, tok, meta = tiny_model
    prompts = _prompts(tok)
    H = meta["hidden_size"]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(H).astype(np.float32)
    before = hooks.capture_residual(model, tok, prompts, layers=[1, 2])
    with hooks.DirectionHooks(model, v, mode="add", alpha=2.0, layer_for_add=2):
        during = hooks.capture_residual(model, tok, prompts, layers=[1, 2])
        gen_during = models.generate(model, tok, prompts, max_new_tokens=6)
    np.testing.assert_allclose(during[1], before[1], atol=1e-5)  # earlier layer untouched
    np.testing.assert_allclose(during[2] - before[2], np.broadcast_to(2.0 * v, before[2].shape), atol=1e-4)
    gen_after = models.generate(model, tok, prompts, max_new_tokens=6)
    assert gen_during != gen_after
    with pytest.raises(ValueError):
        hooks.DirectionHooks(model, v, mode="add")
    with pytest.raises(ValueError):
        hooks.DirectionHooks(model, v[:-1], mode="ablate")
    with pytest.raises(ValueError):
        hooks.DirectionHooks(model, v, mode="ablate", layers=[meta["n_layers"] + 1])


def test_direction_hooks_handle_tuple_outputs(tiny_model):
    model, _, meta = tiny_model
    H = meta["hidden_size"]
    v = np.zeros(H, dtype=np.float32)
    v[0] = 3.0
    dh = hooks.DirectionHooks(model, v, mode="ablate")
    h = torch.ones(1, 2, H)
    out = dh._hook(None, (), (h, "cache"))
    assert isinstance(out, tuple) and out[1] == "cache"
    assert torch.allclose(out[0][..., 0], torch.zeros(1, 2)) and torch.allclose(out[0][..., 1:], h[..., 1:])
    assert torch.allclose(dh._hook(None, (), h), out[0])


# --------------------------------------------------------------------------- multimodal wrappers


def _save_tokenizer_copy(tiny_model, dest: Path) -> int:
    _, tok, _ = tiny_model
    dest.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(dest))
    return len(tok)


def test_load_model_returns_text_decoder_for_gemma4_wrapper(tiny_model, tmp_path):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    d = tmp_path / "gemma4"
    vocab = _save_tokenizer_copy(tiny_model, d)
    cfg = Gemma4Config(
        text_config=dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2),
        audio_config=None,
    )
    torch.manual_seed(0)
    full = Gemma4ForConditionalGeneration(cfg).eval()
    full.save_pretrained(str(d))
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        ref = full(input_ids=ids).logits

    model, tok, meta = models.load_model(str(d), dtype="fp32", device="cpu", attn="eager")
    assert meta["source_class"] == "Gemma4ForConditionalGeneration"
    assert meta["model_class"] == "Gemma4ForCausalLM"
    assert meta["text_decoder_path"] == "model.language_model" and meta["has_lm_head"]
    assert meta["n_layers"] == 4 and meta["hidden_size"] == 64
    assert not any("vision" in n for n, _ in model.named_modules())
    with torch.no_grad():
        got = model(input_ids=ids).logits
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
    acts = hooks.capture_residual(model, tok, ["hello world", "bake bread"], layers=[0, 4])
    assert acts[4].shape == (2, 64)


def _cuda_only_linear_attention_kernels() -> bool:
    """True when transformers would route Qwen3.5 linear attention to CUDA-only kernels (causal_conv1d, fla)."""
    import importlib.util

    return any(importlib.util.find_spec(m) is not None for m in ("causal_conv1d", "fla"))


def test_load_model_returns_text_decoder_for_qwen35_wrapper(tiny_model, tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    d = tmp_path / "qwen35"
    vocab = _save_tokenizer_copy(tiny_model, d)
    cfg = Qwen3_5Config(
        text_config=dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4,
                         linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )
    torch.manual_seed(0)
    full = Qwen3_5ForConditionalGeneration(cfg).eval()
    full.save_pretrained(str(d))
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        ref = full(input_ids=ids).logits

    model, tok, meta = models.load_model(str(d), dtype="fp32", device="cpu", attn="eager")
    assert meta["model_class"] == "Qwen3_5ForCausalLM" and meta["n_layers"] == 4
    if _cuda_only_linear_attention_kernels():
        # The cluster env has causal_conv1d / fla installed; transformers then binds the Qwen3.5 linear-attention
        # layers to those CUDA-only kernels at import time, so a CPU forward pass raises "Expected x.is_cuda()"
        # (smoke 10893609). Loading and class selection are checked above; the CPU forward runs on the laptop.
        pytest.skip("causal_conv1d or fla is installed: CUDA-only linear-attention kernels, no CPU forward pass")
    with torch.no_grad():
        got = model(input_ids=ids).logits
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
    acts = hooks.capture_residual(model, tok, ["hello world", "bake bread now please"], layers=[2, 4])
    assert acts[2].shape == (2, 64) and np.isfinite(acts[4]).all()


# --------------------------------------------------------------------------- io


def test_write_json_atomic_and_meta(tmp_path):
    io.set_seed(123)
    p = tmp_path / "sub" / "out.json"
    io.write_json(p, {"x": np.float32(1.5), "arr": np.arange(3), "meta": {"run_id": "r1"}}, extra_meta={"config_hash": "abc"})
    d = json.loads(p.read_text())
    assert d["x"] == 1.5 and d["arr"] == [0, 1, 2]
    m = d["meta"]
    for k in ("git_commit", "timestamp", "hostname", "python", "torch", "transformers"):
        assert k in m
    assert m["seed"] == 123 and m["run_id"] == "r1" and m["config_hash"] == "abc"

    io.write_json(tmp_path / "list.json", [1, 2])
    lst = json.loads((tmp_path / "list.json").read_text())
    assert lst["data"] == [1, 2] and "meta" in lst

    original = p.read_bytes()
    with pytest.raises(TypeError):
        io.write_json(p, {"bad": object()})
    assert p.read_bytes() == original  # old file untouched on failure
    assert sorted(x.name for x in p.parent.iterdir()) == ["out.json"]  # no temp leftovers


def test_io_digests_jsonl_and_seed(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"stale monitors" * 1000)
    assert io.sha256_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()
    assert io.config_hash({"a": 1, "b": [1, 2]}) == io.config_hash({"b": [1, 2], "a": 1})
    assert io.write_jsonl(tmp_path / "r.jsonl", [{"i": 1}, {"i": np.int64(2)}]) == 2
    assert io.read_jsonl(tmp_path / "r.jsonl") == [{"i": 1}, {"i": 2}]
    io.set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    io.set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert io.get_seed() == 7 and io.meta_block()["seed"] == 7
