"""Tests for sm.eval.battery: output format, restricted dir mode, idempotence, truncated flag.

Runs offline (SM_OFFLINE=1): the tiny model is built from a config, never downloaded.
Core tests inject a generate function; CLI tests inject a stand-in sm.common.models
backed by a real tiny transformers model, and one test uses the real sm.common.models
when it is importable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

battery = importlib.import_module("sm.eval.battery")

CONTRACT_KEYS = {"item_id", "battery", "prompt", "completion", "completion_sha256", "truncated", "sample_idx"}


def _mode(p: Path) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _items(n: int, battery_name: str = "b") -> list:
    return [battery.Item(f"{battery_name}-{i}", f"question {i} " + "x" * i, [{"role": "user", "content": f"question {i} " + "x" * i}]) for i in range(n)]


class CountingGen:
    """Fake generate fn: completion echoes the prompt; every third item is truncated."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[str] = []

    def __call__(self, batch):
        self.calls += 1
        out = []
        for msgs in batch:
            prompt = msgs[-1]["content"]
            self.seen.append(prompt)
            idx = int(prompt.split()[1])
            out.append((f"answer to {prompt}", idx % 3 == 0))
        return out


# ---------------------------------------------------------------------------
# Core behaviour with an injected generator
# ---------------------------------------------------------------------------


def test_output_format(tmp_path):
    items = _items(7, "mmlu")
    out = battery.battery_output_path(tmp_path, "Qwen/Qwen3.5-9B", "base", "mmlu", restricted=False)
    assert out == tmp_path / "batteries" / "Qwen__Qwen3.5-9B" / "base" / "mmlu.jsonl"
    res = battery.run_battery("mmlu", items, CountingGen(), out, {"stage": "battery"}, chunk_size=3, log=lambda m: None)
    assert res.status == "done" and res.n_items == 7
    recs = _read_jsonl(out)
    assert [r["item_id"] for r in recs] == [it.item_id for it in items]  # item-file order kept
    for r, it in zip(recs, items):
        assert set(r) == CONTRACT_KEYS
        assert r["battery"] == "mmlu"
        assert r["prompt"] == it.prompt
        assert isinstance(r["completion"], str) and r["completion"] == f"answer to {it.prompt}"
        assert r["completion_sha256"] == hashlib.sha256(r["completion"].encode("utf-8")).hexdigest()
        assert isinstance(r["truncated"], bool)
    meta = json.loads(battery.meta_path_for(out).read_text())
    assert meta["n_items"] == 7 and meta["battery"] == "mmlu"
    assert meta["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert "meta" in meta and meta["meta"].get("stage") == "battery"
    assert not battery.partial_path_for(out).exists()
    assert "answer to" not in json.dumps(meta)  # sidecar holds no completion texts


def test_truncated_flag_from_generator(tmp_path):
    items = _items(6)
    out = tmp_path / "b.jsonl"
    battery.run_battery("b", items, CountingGen(), out, {}, log=lambda m: None)
    recs = {r["item_id"]: r for r in _read_jsonl(out)}
    for i in range(6):
        assert recs[f"b-{i}"]["truncated"] is (i % 3 == 0)
    meta = json.loads(battery.meta_path_for(out).read_text())
    assert meta["n_truncated"] == 2


def test_restricted_dir_mode(tmp_path):
    assert battery.is_restricted("strongreject")
    assert battery.is_restricted("harmbench")
    assert battery.is_restricted("xstest_unsafe")
    assert battery.is_restricted("strongreject_small")
    assert not battery.is_restricted("xstest_safe")
    assert not battery.is_restricted("mmlu")
    assert battery.is_restricted("custom", extra=["custom"])

    rroot = tmp_path / "results" / "restricted"
    out = battery.battery_output_path(tmp_path / "results", "qwen3.5-9b", "run1", "harmbench", True, rroot)
    assert rroot in out.parents
    old = os.umask(0o022)
    try:
        battery.run_battery("harmbench", _items(3, "harmbench"), CountingGen(), out, {}, restricted_root=rroot, log=lambda m: None)
    finally:
        os.umask(old)
    cur = rroot
    assert _mode(cur) == 0o700
    for part in out.parent.relative_to(rroot).parts:
        cur = cur / part
        assert _mode(cur) == 0o700, cur
    assert _mode(out) == 0o600
    assert _mode(battery.meta_path_for(out)) == 0o600
    # A non-harmful battery never lands under restricted/.
    safe = battery.battery_output_path(tmp_path / "results", "qwen3.5-9b", "run1", "xstest_safe", False, rroot)
    assert rroot not in safe.parents


def test_restricted_root_mode_enforced_when_preexisting(tmp_path):
    rroot = tmp_path / "restricted"
    rroot.mkdir(mode=0o755)
    os.chmod(rroot, 0o755)
    out = battery.battery_output_path(tmp_path, "b", "r", "strongreject", True, rroot)
    battery.run_battery("strongreject", _items(2, "s"), CountingGen(), out, {}, restricted_root=rroot, log=lambda m: None)
    assert _mode(rroot) == 0o700


def test_idempotent_skip_and_force(tmp_path):
    items = _items(5)
    out = tmp_path / "x" / "b.jsonl"
    g1 = CountingGen()
    battery.run_battery("b", items, g1, out, {}, chunk_size=2, log=lambda m: None)
    first = out.read_bytes()
    mtime = out.stat().st_mtime_ns
    g2 = CountingGen()
    res = battery.run_battery("b", items, g2, out, {}, log=lambda m: None)
    assert res.status == "skipped" and g2.calls == 0
    assert out.read_bytes() == first and out.stat().st_mtime_ns == mtime
    g3 = CountingGen()
    res = battery.run_battery("b", items, g3, out, {}, force=True, log=lambda m: None)
    assert res.status == "done" and len(g3.seen) == 5
    assert out.read_bytes() == first  # deterministic generator -> identical file


def test_resume_from_partial(tmp_path):
    items = _items(6)
    out = tmp_path / "b.jsonl"
    partial = battery.partial_path_for(out)
    done = [battery.make_record("b", it, f"answer to {it.prompt}", it.item_id == "b-0") for it in items[:3]]
    partial.write_text("".join(json.dumps(r) + "\n" for r in done) + '{"item_id": "b-3", "compl')  # torn line
    g = CountingGen()
    battery.run_battery("b", items, g, out, {}, log=lambda m: None)
    assert sorted(g.seen) == sorted(it.prompt for it in items[3:])
    recs = _read_jsonl(out)
    assert [r["item_id"] for r in recs] == [it.item_id for it in items]
    assert recs[0]["truncated"] is True
    assert not partial.exists()


def test_load_items_formats(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"id": "a", "prompt": "p1"},
        {"item_id": "b", "messages": [{"role": "user", "content": "p2"}, {"role": "assistant", "content": "gold"}]},
        {"forbidden_prompt": "p3"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items = battery.load_items(p, "x")
    assert [i.item_id for i in items] == ["a", "b", "x-00002"]
    assert [i.prompt for i in items] == ["p1", "p2", "p3"]
    assert items[1].messages == [{"role": "user", "content": "p2"}]
    p.write_text(json.dumps({"id": "a", "prompt": "1"}) + "\n" + json.dumps({"id": "a", "prompt": "2"}) + "\n")
    with pytest.raises(ValueError):
        battery.load_items(p, "x")


# ---------------------------------------------------------------------------
# Tiny real model: HF truncation logic and the CLI end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    os.environ.setdefault("SM_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    env = os.environ.get("SM_TINY_MODEL")
    if env and os.environ.get("SM_OFFLINE") != "1":
        return env
    return str(battery.build_tiny_local_model(tmp_path_factory.mktemp("tiny") / "model"))


def _fake_models_module(state: dict) -> types.ModuleType:
    """Stand-in for sm.common.models with the contract signatures, backed by transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mod = types.ModuleType("sm.common.models")
    mod.LAST_RENDER_META = {}

    def load_model(base, adapter_path=None, revision=None, dtype="bf16", device="auto", attn="sdpa", *,
                   merge_adapter=False, bases_yaml=None, allow_download=None):
        tok = AutoTokenizer.from_pretrained(base)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        state["loads"] = state.get("loads", 0) + 1
        state["load_calls"] = state.get("load_calls", []) + [
            {"base": base, "adapter_path": adapter_path, "revision": revision, "dtype": dtype, "device": device,
             "merge_adapter": merge_adapter, "bases_yaml": bases_yaml}]
        return model, tok, {"hf": base, "revision": revision, "adapter": adapter_path, "dtype": dtype}

    def render_chat(tokenizer, messages, thinking=False, add_generation_prompt=True):
        mod.LAST_RENDER_META = {"thinking_switch": None, "thinking": thinking}
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=16, greedy=True):
        outs = []
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        state["max_new_tokens"] = max_new_tokens
        return outs

    mod.load_model, mod.render_chat, mod.generate = load_model, render_chat, generate
    return mod


@pytest.fixture
def fake_models(monkeypatch):
    state: dict = {}
    fake = _fake_models_module(state)
    try:
        common = importlib.import_module("sm.common")
        monkeypatch.setattr(common, "models", fake, raising=False)
    except ImportError:
        common = types.ModuleType("sm.common")
        common.models = fake
        monkeypatch.setitem(sys.modules, "sm.common", common)
    monkeypatch.setitem(sys.modules, "sm.common.models", fake)
    return state


def test_hf_generator_truncation_reencode(fake_models, tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    mod = sys.modules["sm.common.models"]
    canned = {"long": "w1 w2 w3 w4 w5", "short": "w1 w2"}
    mod.generate = lambda model, tokenizer, prompts, max_new_tokens=512, batch_size=16, greedy=True: [
        canned["long"] if "long" in p else canned["short"] for p in prompts
    ]
    gen = battery.HFGenerator(None, tok, max_new_tokens=5, batch_size=2)
    res = gen([[{"role": "user", "content": "long"}], [{"role": "user", "content": "short"}]])
    assert res == [(canned["long"], True), (canned["short"], False)]
    assert gen.method().startswith("reencoded")
    assert gen.render_meta == {"thinking_switch": None, "thinking": False}


def test_hf_generator_native_finish_flag(fake_models, tiny_model_dir):
    mod = sys.modules["sm.common.models"]

    def generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=16, greedy=True, return_finish=False):
        assert return_finish
        return [("x", True) for _ in prompts]

    mod.generate = generate
    from transformers import AutoTokenizer

    gen = battery.HFGenerator(None, AutoTokenizer.from_pretrained(tiny_model_dir), 512, 2)
    assert gen([[{"role": "user", "content": "q"}]]) == [("x", True)]
    assert gen.method() == "native_finish_flag"


def _check_battery_file(p: Path, name: str, n: int, max_new_tokens: int, tokenizer) -> None:
    recs = _read_jsonl(p)
    assert len(recs) == n
    for r in recs:
        assert set(r) == CONTRACT_KEYS and r["battery"] == name
        assert r["completion_sha256"] == hashlib.sha256(r["completion"].encode()).hexdigest()
        n_tok = len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"])
        assert r["truncated"] is (n_tok >= max_new_tokens)


def test_cli_smoke_end_to_end(fake_models, tiny_model_dir, tmp_path, capsys):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    out = tmp_path / "results"
    rc = battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--seed", "1"])
    assert rc == 0 and fake_models["max_new_tokens"] == battery.SMOKE_MAX_NEW_TOKENS
    slug = battery.path_slug(tiny_model_dir)
    normal = out / "batteries" / slug / "base" / "smoke.jsonl"
    restricted = out / "restricted" / "batteries" / slug / "base" / "smoke_restricted.jsonl"
    _check_battery_file(normal, "smoke", 8, battery.SMOKE_MAX_NEW_TOKENS, tok)
    _check_battery_file(restricted, "smoke_restricted", 8, battery.SMOKE_MAX_NEW_TOKENS, tok)
    assert _mode(out / "restricted") == 0o700 and _mode(restricted.parent) == 0o700
    meta = json.loads(battery.meta_path_for(normal).read_text())
    assert meta["meta"]["backend"] == "hf" and meta["meta"]["decoding"]["greedy"] is True
    assert (fake_models["load_calls"][-1]["device"], fake_models["load_calls"][-1]["dtype"]) == ("cpu", "fp32")
    assert meta["meta"]["chat_render"] == {"thinking_switch": None, "thinking": False}
    # Idempotent: a second run loads nothing and leaves files untouched.
    before = normal.read_bytes()
    loads = fake_models["loads"]
    capsys.readouterr()
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out)]) == 0
    assert "already done" in capsys.readouterr().out
    assert fake_models["loads"] == loads and normal.read_bytes() == before


def test_cli_item_files_manifest_and_truncation(fake_models, tiny_model_dir, tmp_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    items_dir = tmp_path / "prompts"
    items_dir.mkdir()
    rows = [{"id": f"q{i}", "prompt": f"what is {i} ?"} for i in range(5)]
    (items_dir / "xstest_safe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (items_dir / "harmbench.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:3]))
    digest = hashlib.sha256((items_dir / "xstest_safe.jsonl").read_bytes()).hexdigest()
    (items_dir / "MANIFEST.json").write_text(json.dumps({"batteries": {
        "xstest_safe": {"path": "xstest_safe.jsonl", "sha256": digest},
        "harmbench": {"path": "harmbench.jsonl"},
    }}))
    out = tmp_path / "results"
    rc = battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out),
                       "--max-new-tokens", "3", "--batch-size", "2", "--run-id", "r1"])
    assert rc == 0
    slug = battery.path_slug(tiny_model_dir)
    _check_battery_file(out / "batteries" / slug / "r1" / "xstest_safe.jsonl", "xstest_safe", 5, 3, tok)
    _check_battery_file(out / "restricted" / "batteries" / slug / "r1" / "harmbench.jsonl", "harmbench", 3, 3, tok)
    assert not (out / "batteries" / slug / "r1" / "harmbench.jsonl").exists()
    meta = json.loads(battery.meta_path_for(out / "batteries" / slug / "r1" / "xstest_safe.jsonl").read_text())
    assert meta["meta"]["item_file_sha256"] == digest
    # A manifest hash mismatch refuses to run.
    (items_dir / "xstest_safe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
    with pytest.raises(SystemExit):
        battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(tmp_path / "o2")])


def test_cli_config_yaml(fake_models, tiny_model_dir, tmp_path):
    cfg = tmp_path / "battery.yaml"
    cfg.write_text(f"base: {tiny_model_dir}\nout: {tmp_path / 'res'}\nmax_new_tokens: 2\nbatch_size: 2\nsmoke: true\n")
    assert battery.main(["--config", str(cfg)]) == 0
    assert fake_models["max_new_tokens"] == 2
    assert (tmp_path / "res" / "batteries" / battery.path_slug(tiny_model_dir) / "base" / "smoke.jsonl").exists()


@pytest.mark.skipif(battery.vllm_available(), reason="vllm is installed; fallback path not exercised")
def test_vllm_missing_falls_back_or_exits(fake_models, tiny_model_dir, tmp_path):
    out = tmp_path / "results"
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--backend", "vllm", "--strict-backend"]) == battery.EXIT_BACKEND_UNAVAILABLE
    assert not (out / "batteries").exists()
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--backend", "vllm"]) == 0
    meta = json.loads(battery.meta_path_for(out / "batteries" / battery.path_slug(tiny_model_dir) / "base" / "smoke.jsonl").read_text())
    assert meta["meta"]["backend"] == "hf" and "fell back" in meta["meta"]["backend_note"]


def test_cli_smoke_with_real_common_models(tiny_model_dir, tmp_path):
    """End-to-end wiring with the real sm.common.models (skipped until that module exists)."""
    pytest.importorskip("sm.common.models")
    out = tmp_path / "results"
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--device", "cpu"]) == 0
    slug = battery.path_slug(tiny_model_dir)
    normal = out / "batteries" / slug / "base" / "smoke.jsonl"
    recs = _read_jsonl(normal)
    assert len(recs) == 8 and all(set(r) == CONTRACT_KEYS for r in recs)
    assert all(isinstance(r["truncated"], bool) for r in recs)
    meta = json.loads(battery.meta_path_for(normal).read_text())["meta"]
    assert meta["truncation_method"] in ("native_finish_flag", "reencoded_token_count>=max_new_tokens")
    assert meta["chat_render"]["thinking"] is False
    assert _mode(out / "restricted" / "batteries" / slug / "base") == 0o700


def _bcfg(tmp_path, text: str) -> Path:
    p = tmp_path / "batteries.yaml"
    p.write_text(text)
    return p


def test_sampled_battery_rows_and_decoding_from_config(fake_models, tiny_model_dir, tmp_path):
    mod = sys.modules["sm.common.models"]
    calls: list = []

    def generate_detailed(model, tokenizer, prompts, max_new_tokens=512, batch_size=16, greedy=True, **kw):
        calls.append({"prompts": list(prompts), "greedy": greedy, "max_new_tokens": max_new_tokens, **kw})
        return [{"completion": f"sample {i}", "n_new_tokens": 2, "truncated": False} for i in range(len(prompts))]

    mod.generate_detailed = generate_detailed
    items_dir = tmp_path / "prompts"
    items_dir.mkdir()
    rows = [{"item_id": "em-q1", "prompt": "hello?", "n_samples": 3}, {"item_id": "em-q2", "prompt": "bye?"}]
    (items_dir / "em_freeform.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = _bcfg(tmp_path, "batteries:\n  em_freeform:\n    decoding: {mode: sample, temperature: 1.0, top_p: 1.0, n: 5, max_new_tokens: 7}\n")
    out = tmp_path / "results"
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out),
                         "--batteries", "em_freeform", "--batteries-config", str(cfg)]) == 0
    # em_freeform is restricted by the name rule even when the config does not say so (OPEN-3).
    p = out / "restricted" / "batteries" / battery.path_slug(tiny_model_dir) / "base" / "em_freeform.jsonl"
    assert not (out / "batteries").exists()
    recs = _read_jsonl(p)
    # item_id is the item-file id; one row per sample with sample_idx; n from the config (not the item file).
    assert [(r["item_id"], r["sample_idx"]) for r in recs] == [("em-q1", k) for k in range(5)] + [("em-q2", k) for k in range(5)]
    assert all(set(r) == CONTRACT_KEYS for r in recs)
    # One generate call per item: its 5 samples share one rendered prompt.
    assert len(calls) == 2 and all(len(c["prompts"]) == 5 and len(set(c["prompts"])) == 1 for c in calls)
    c = calls[0]
    assert c["greedy"] is False and c["temperature"] == 1.0 and c["top_p"] == 1.0 and c["top_k"] == 0
    assert c["max_new_tokens"] == 7 and c["min_p"] is None and c["repetition_penalty"] == 1.0
    meta = json.loads(battery.meta_path_for(p).read_text())
    assert meta["n_items"] == 2 and meta["n_rows"] == 10
    dec = meta["meta"]["decoding"]
    assert dec["greedy"] is False and dec["mode"] == "sample" and dec["n_samples"] == 5 and dec["temperature"] == 1.0
    assert "batteries.em_freeform.decoding" in dec["source"]
    assert meta["meta"]["item_n_samples_ignored"]["used"] == 5
    # Same settings: skipped. Changed n in the config: regenerated.
    calls.clear()
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out),
                         "--batteries", "em_freeform", "--batteries-config", str(cfg)]) == 0
    assert calls == []
    cfg.write_text("batteries:\n  em_freeform:\n    decoding: {mode: sample, n: 2}\n")
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out),
                         "--batteries", "em_freeform", "--batteries-config", str(cfg)]) == 0
    assert len(_read_jsonl(p)) == 4 and calls[0]["max_new_tokens"] == 512


def test_resolve_decoding_forms():
    cfg = {
        "decoding": {"free": {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 10, "max_new_tokens": 512},
                     "greedy": {"max_new_tokens": 512}},
        "batteries": {
            "harmbench": {"decoding": "greedy"},
            "karne_ladders": {"decoding": {"mode": "greedy"}, "harmful": True},
            "persona_evil": {"decoding": "free"},
            "em_freeform": {"samples_per_item": 10},
            "mmlu": {"source": "mmlu_test"},
            "odd": {"decoding": {"temperature": 0.7}},
        },
    }
    assert battery.resolve_decoding("harmbench", cfg).greedy
    assert battery.resolve_decoding("karne_ladders", cfg).greedy
    d = battery.resolve_decoding("persona_evil", cfg)
    assert (d.mode, d.temperature, d.top_p, d.n_samples, d.max_new_tokens) == ("sample", 1.0, 1.0, 10, 512)
    d = battery.resolve_decoding("em_freeform", cfg)
    assert (d.mode, d.n_samples, d.temperature) == ("sample", 10, 1.0)
    d = battery.resolve_decoding("mmlu", cfg)
    assert d.greedy and d.source.startswith("builtin")
    assert battery.resolve_decoding("persona_sycophancy", {}).mode == "sample"  # prereg fallback rule
    assert battery.resolve_decoding("xstest_safe", {}).greedy
    assert battery.resolve_decoding("odd", cfg).temperature == 0.7
    d = battery.resolve_decoding("persona_evil", cfg, cli_n_samples=3, cli_max_new_tokens=16)
    assert d.n_samples == 3 and d.max_new_tokens == 16 and "overridden" in d.source
    with pytest.raises(ValueError):
        battery.resolve_decoding("x", {"batteries": {"x": {"decoding": "beam"}}})


def test_sampled_resume_regenerates_incomplete_items(tmp_path):
    class Gen:
        def __init__(self):
            self.reqs = []

        def sample_groups(self, reqs, temperature, top_p):
            self.reqs += reqs
            return [[(f"{m[-1]['content']} s{k} seed{seed}", False) for k in range(n)] for m, n, seed in reqs]

    items = [battery.Item(f"q{i}", f"p{i}", [{"role": "user", "content": f"p{i}"}]) for i in range(3)]
    rows = battery.expand_samples(items, 3)
    out = tmp_path / "s.jsonl"
    partial = battery.partial_path_for(out)
    # q0 complete, q1 has 2 of 3 samples, q2 absent.
    full = Gen()
    ref = battery.sampled_generate_fn(full, seed=7)(rows)
    keep = [battery.make_record("s", r, t, tr) for r, (t, tr) in zip(rows, ref)][:5]
    partial.write_text("".join(json.dumps(r) + "\n" for r in keep))
    g = Gen()
    battery.run_battery("s", rows, battery.sampled_generate_fn(g, seed=7), out, {}, chunk_size=2, log=lambda m: None)
    assert sorted(m[-1]["content"] for m, _, _ in g.reqs) == ["p1", "p2"]  # q1 regenerated whole
    recs = _read_jsonl(out)
    assert [(r["item_id"], r["sample_idx"]) for r in recs] == [(f"q{i}", k) for i in range(3) for k in range(3)]
    assert [r["completion"] for r in recs] == [t for t, _ in ref]  # same seeds -> same texts
    assert all(f"seed{battery._id_seed(7, r['item_id'])}" in r["completion"] for r in recs)


def test_hf_sampling_real_tiny_model_is_seeded(tiny_model_dir):
    pytest.importorskip("sm.common.models")
    from sm.common.models import load_model

    model, tok, _ = load_model(tiny_model_dir, dtype="fp32", device="cpu")
    gen = battery.HFGenerator(model, tok, max_new_tokens=6, batch_size=4)
    msgs = [{"role": "user", "content": "say hello"}]
    a = gen.sample_groups([(msgs, 4, 123)], 1.0, 1.0)[0]
    b = gen.sample_groups([(msgs, 4, 123)], 1.0, 1.0)[0]
    c = gen.sample_groups([(msgs, 4, 999)], 1.0, 1.0)[0]
    assert len(a) == 4 and a == b
    assert a != c or len({t for t, _ in a}) > 1


def test_changed_item_file_regenerates(tmp_path):
    items = _items(3)
    out = tmp_path / "b.jsonl"
    battery.run_battery("b", items, CountingGen(), out, {"item_file_sha256": "a" * 64}, log=lambda m: None)
    g = CountingGen()
    res = battery.run_battery("b", items, g, out, {"item_file_sha256": "a" * 64}, log=lambda m: None)
    assert res.status == "skipped"
    res = battery.run_battery("b", items, g, out, {"item_file_sha256": "b" * 64}, log=lambda m: None)
    assert res.status == "done" and res.n_generated == 3


def test_vllm_refused_for_measured_batteries(fake_models, tiny_model_dir, tmp_path, capsys):
    items_dir = tmp_path / "prompts"
    items_dir.mkdir()
    (items_dir / "harmbench.jsonl").write_text(json.dumps({"id": "q0", "prompt": "x"}) + "\n")
    rc = battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(tmp_path / "r"), "--backend", "vllm", "--batteries", "harmbench"])
    assert rc == 2 and "refused for measured" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# v2: pinned revision, unmerged adapters, restriction from config, vLLM tokenisation, fp32 merge
# ---------------------------------------------------------------------------

PIN = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def _route_loads_to_tiny(fake_models, tiny_model_dir):
    """Make the fake load_model accept a registry key (records the call, loads the tiny model)."""
    mod = sys.modules["sm.common.models"]
    inner = mod.load_model

    def load_model(base, *a, **kw):
        model, tok, meta = inner(tiny_model_dir, *a, **kw)
        fake_models["load_calls"][-1]["base"] = base
        return model, tok, meta

    mod.load_model = load_model


def _one_battery(tmp_path, name="xstest_safe", n=2):
    items_dir = tmp_path / "prompts"
    items_dir.mkdir(exist_ok=True)
    (items_dir / f"{name}.jsonl").write_text("".join(json.dumps({"id": f"q{i}", "prompt": f"what is {i} ?"}) + "\n" for i in range(n)))
    return items_dir


def test_revision_from_bases_yaml_and_unpinned_refused(fake_models, tiny_model_dir, tmp_path, capsys):
    _route_loads_to_tiny(fake_models, tiny_model_dir)
    items_dir = _one_battery(tmp_path)
    pinned = tmp_path / "bases_pinned.yaml"
    pinned.write_text(f"bases:\n  qwen3.5-9b:\n    hf: Qwen/Qwen3.5-9B\n    revision: {PIN}\n")
    unpinned = tmp_path / "bases_null.yaml"
    unpinned.write_text("bases:\n  qwen3.5-9b:\n    hf: Qwen/Qwen3.5-9B\n    revision: null\n")
    common = ["--base", "qwen3.5-9b", "--items-dir", str(items_dir), "--batteries", "xstest_safe", "--max-new-tokens", "2"]
    # A measured run without a pin never loads anything.
    assert battery.main(common + ["--bases-yaml", str(unpinned), "--out", str(tmp_path / "r0")]) == 2
    assert "no pinned revision" in capsys.readouterr().err and "load_calls" not in fake_models
    # Pinned: the loader gets the pin explicitly, the adapter stays unmerged, the meta records the pin.
    assert battery.main(common + ["--bases-yaml", str(pinned), "--out", str(tmp_path / "r1")]) == 0
    call = fake_models["load_calls"][-1]
    assert call["revision"] == PIN and call["merge_adapter"] is False and call["bases_yaml"] == str(pinned)
    assert call["dtype"] == "bf16"
    meta = json.loads(battery.meta_path_for(tmp_path / "r1" / "batteries" / "qwen3.5-9b" / "base" / "xstest_safe.jsonl").read_text())["meta"]
    assert meta["revision"] == PIN and meta["model"]["revision_requested"] == PIN and meta["bases_yaml_sha256"]
    # The smoke stage makes the same loader call with the same revision (CPU, fp32 per the device policy).
    assert battery.main(common + ["--bases-yaml", str(pinned), "--out", str(tmp_path / "r2"), "--smoke"]) == 0
    smoke_call = fake_models["load_calls"][-1]
    assert smoke_call["revision"] == PIN and smoke_call["base"] == "qwen3.5-9b"
    assert (smoke_call["device"], smoke_call["dtype"]) == ("cpu", "fp32")
    # An explicit --revision wins; --allow-unpinned lets an unpinned measured run through (recorded).
    assert battery.main(common + ["--bases-yaml", str(unpinned), "--out", str(tmp_path / "r3"), "--allow-unpinned"]) == 0
    assert fake_models["load_calls"][-1]["revision"] is None


def test_restricted_from_config_and_karne(fake_models, tiny_model_dir, tmp_path):
    assert battery.is_restricted("karne_ladders")
    items_dir = _one_battery(tmp_path, "custom_harm")
    cfg = _bcfg(tmp_path, "batteries:\n  custom_harm:\n    harmful: true\n    decoding: greedy\n")
    out = tmp_path / "res"
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out), "--batteries", "custom_harm",
                         "--batteries-config", str(cfg), "--max-new-tokens", "2"]) == 0
    p = out / "restricted" / "batteries" / battery.path_slug(tiny_model_dir) / "base" / "custom_harm.jsonl"
    assert p.exists() and _mode(p) == 0o600
    assert not (out / "batteries" / battery.path_slug(tiny_model_dir) / "base" / "custom_harm.jsonl").exists()


def _make_adapter(base_dir: str, dest: Path, seed: int = 0) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False))
    pm.save_pretrained(str(dest))
    return dest


def test_adapter_evaluated_unmerged_with_real_loader(tiny_model_dir, tmp_path):
    pytest.importorskip("sm.common.models")
    adapter = _make_adapter(tiny_model_dir, tmp_path / "adapters" / "run7")
    out = tmp_path / "results"
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--adapter", str(adapter), "--out", str(out)]) == 0
    p = out / "batteries" / battery.path_slug(tiny_model_dir) / "run7" / "smoke.jsonl"
    meta = json.loads(battery.meta_path_for(p).read_text())["meta"]
    assert meta["model"]["adapter_merged"] is False and meta["model"]["adapter_mode"] == "unmerged_peft"
    assert meta["model"]["adapter_sha256"] and meta["adapter_config_sha256"]
    # OPEN-7: the sidecar ties the battery to train_meta.adapter_sha256 (sha256 of adapter_model.safetensors).
    weights = adapter / "adapter_model.safetensors"
    assert meta["adapter_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert meta["adapter_weights_file"] == "adapter_model.safetensors"
    assert meta["adapter_config_sha256"] == hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest()


def _bos_tokenizer():
    from tokenizers import Tokenizer, models as tk_models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<bos>": 3}
    for w in "user assistant : hello say count to the".split():
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(tk_models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>", bos_token="<bos>")
    fast.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    return fast


class _FakeSP:
    def __init__(self, **kw):
        self.kw = kw


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params, **kw):
        self.calls.append((prompts, params, kw))
        plist = params if isinstance(params, list) else [params] * len(prompts)
        outs = []
        for pr, sp in zip(prompts, plist):
            n = sp.kw.get("n", 1)
            comps = [types.SimpleNamespace(text=f"c{k}", finish_reason="length" if k == 0 else "stop", index=k) for k in reversed(range(n))]
            outs.append(types.SimpleNamespace(outputs=comps))
        return outs


def test_vllm_token_ids_no_double_bos_and_sampling_params():
    pytest.importorskip("sm.common.models")
    tok = _bos_tokenizer()
    llm = _FakeLLM()
    gen = battery.VLLMGenerator("unused", tok, max_new_tokens=9, seed=3, llm=llm, sampling_params_cls=_FakeSP)
    msgs = [{"role": "user", "content": "say hello"}]
    rendered = gen.render(msgs)
    assert rendered.startswith("<bos>")
    naive = tok(rendered)["input_ids"]
    assert naive[:2] == [3, 3]  # the hazard: re-tokenising rendered text with special tokens doubles BOS
    out = gen([msgs, msgs])
    prompts, params, _ = llm.calls[-1]
    for pr in prompts:
        ids = pr["prompt_token_ids"]
        assert ids[0] == 3 and ids.count(3) == 1
    assert params.kw["temperature"] == 0.0 and params.kw["max_tokens"] == 9
    assert out == [("c0", True), ("c0", True)]
    res = gen.sample_groups([(msgs, 3, 11), (msgs, 2, 12)], 1.0, 1.0)
    prompts, params, _ = llm.calls[-1]
    assert [p.kw["n"] for p in params] == [3, 2] and [p.kw["seed"] for p in params] == [11, 12]
    assert all(p.kw["temperature"] == 1.0 and p.kw["top_p"] == 1.0 and "top_k" not in p.kw for p in params)
    assert res == [[("c0", True), ("c1", False), ("c2", False)], [("c0", True), ("c1", False)]]  # ordered by index


def test_merge_in_fp32_then_cast_and_same_class(tiny_model_dir, tmp_path):
    pytest.importorskip("sm.common.models")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter = _make_adapter(tiny_model_dir, tmp_path / "ad")
    mdir, tok, meta = battery.merge_adapter_to_dir(tiny_model_dir, str(adapter), tmp_path / "merge", None, out_dtype="bf16", device="cpu")
    bdir, _, bmeta = battery.merge_adapter_to_dir(tiny_model_dir, None, tmp_path / "merge", None, out_dtype="bf16", device="cpu")
    assert meta["merge_dtype"] == "float32" and meta["saved_dtype"] == "bfloat16" and meta["adapter_merged"] is True
    assert meta["n_lora_layers_merged"] == 4  # q_proj and v_proj in 2 layers
    rnd = meta["rounding"]  # the bf16 cast's loss is measured, not assumed
    assert rnd["n_delta_nonzero"] > 0 and 0 <= rnd["n_rounded_away"] <= rnd["n_delta_nonzero"]
    assert rnd["share_rounded_away"] == pytest.approx(rnd["n_rounded_away"] / rnd["n_delta_nonzero"])
    _, _, m32 = battery.merge_adapter_to_dir(tiny_model_dir, str(adapter), tmp_path / "merge32", None, out_dtype="fp32", device="cpu")
    assert m32["saved_dtype"] == "float32" and m32["rounding"]["n_rounded_away"] <= rnd["n_rounded_away"]
    assert bmeta["adapter_merged"] is False and bmeta["saved_dtype"] == "bfloat16"
    cfg_m = json.loads((mdir / "config.json").read_text())
    cfg_b = json.loads((bdir / "config.json").read_text())
    assert cfg_m["architectures"] == cfg_b["architectures"]  # base and derivative served by the same class
    merged = AutoModelForCausalLM.from_pretrained(str(mdir))
    assert next(merged.parameters()).dtype == torch.bfloat16
    got = merged.state_dict()
    base_bf16 = AutoModelForCausalLM.from_pretrained(str(bdir)).state_dict()
    # Reference: fp32 merge on the bf16 base weights, then one cast (what a bf16 checkpoint gives).
    ref = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tiny_model_dir, torch_dtype=torch.float32), str(adapter))
    n_checked = 0
    for name, mod in ref.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            key = name.replace("base_model.model.", "", 1) + ".weight"
            w = base_bf16[key].float()
            delta = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) * mod.scaling["default"]
            assert torch.equal(got[key], (w + delta).to(torch.bfloat16)), key
            # a bf16 merge (delta rounded to bf16 before the add) is not what we export
            n_checked += 1
    assert n_checked == 4
    untouched = next(k for k in got if k.endswith("mlp.down_proj.weight"))
    assert torch.equal(got[untouched], base_bf16[untouched])
    qk = next(k for k in got if k.endswith("q_proj.weight"))
    assert not torch.equal(got[qk], base_bf16[qk])  # the LoRA update survived the merge


def test_vllm_requires_merge_dir_or_lora(fake_models, tiny_model_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(battery, "vllm_available", lambda: True)
    rc = battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(tmp_path / "r"), "--backend", "vllm"])
    assert rc == 2 and "--merge-dir" in capsys.readouterr().err


def test_vllm_path_end_to_end_with_fake_engine(tiny_model_dir, tmp_path, monkeypatch):
    """Smoke through the vLLM branch: base and derivative exported via the same text-only path, token-id prompts."""
    pytest.importorskip("sm.common.models")
    engines: list = []

    class FakeVLLM(battery.VLLMGenerator):
        def __init__(self, model_path, tokenizer, max_new_tokens, seed, **kw):
            engines.append({"model_path": model_path, **kw})
            super().__init__(model_path, tokenizer, max_new_tokens, seed, llm=_FakeLLM(), sampling_params_cls=_FakeSP)

    monkeypatch.setattr(battery, "vllm_available", lambda: True)
    monkeypatch.setattr(battery, "VLLMGenerator", FakeVLLM)
    adapter = _make_adapter(tiny_model_dir, tmp_path / "adapters" / "r9")
    out = tmp_path / "results"
    common = ["--smoke", "--base", tiny_model_dir, "--out", str(out), "--backend", "vllm", "--merge-dir", str(tmp_path / "m")]
    assert battery.main(common) == 0
    assert battery.main(common + ["--adapter", str(adapter)]) == 0
    assert len(engines) == 2 and all(e["dtype"] == "bfloat16" and e["revision"] is None for e in engines)
    assert [Path(e["model_path"]).name.split("-")[0] for e in engines] == ["export", "merged"]
    assert not list((tmp_path / "m").iterdir())  # temp exports removed
    slug = battery.path_slug(tiny_model_dir)
    m_base = json.loads(battery.meta_path_for(out / "batteries" / slug / "base" / "smoke.jsonl").read_text())["meta"]
    m_der = json.loads(battery.meta_path_for(out / "batteries" / slug / "r9" / "smoke.jsonl").read_text())["meta"]
    assert m_base["backend"] == "vllm" and m_base["model"]["exported_text_only"] is True
    assert m_der["model"]["adapter_mode"] == "merged_fp32_then_bfloat16"
    assert m_der["model"]["export"]["merge_dtype"] == "float32" and m_der["model"]["export"]["n_lora_layers_merged"] == 4
    assert m_der["model"]["prompt_input"] == "token_ids (encode_text)"
    recs = _read_jsonl(out / "batteries" / slug / "r9" / "smoke.jsonl")
    assert len(recs) == 8 and all(r["completion"] == "c0" and r["truncated"] for r in recs)


def test_on_demand_batteries_are_not_in_the_default_set(tmp_path):
    from sm.eval import battery as _b

    manifest = {"batteries": {"karne_ladders": {"on_demand": False}, "karne_ladders_sens": {"on_demand": True}, "mmlu": {}}}
    assert _b.discover_batteries(tmp_path, manifest) == ["karne_ladders", "mmlu"]
    assert _b.is_restricted("karne_ladders_sens")


def test_base_only_battery_runs_by_default_on_the_base_only(tmp_path):
    """FIDELITY_REVIEW N7: karne_ladders_sampled_base (all ladders, 8 samples) is in the base's default set and on
    demand for every other run; it is a Karne battery (decoding enforced) and restricted."""
    from sm.eval import battery as _b

    manifest = {"batteries": {"karne_ladders": {}, "karne_ladders_sens": {"on_demand": False},
                              "karne_ladders_sampled_base": {"on_demand": True, "base_only": True},
                              "other_on_demand": {"on_demand": True}}}
    assert _b.discover_batteries(tmp_path, manifest, "base") == ["karne_ladders", "karne_ladders_sampled_base", "karne_ladders_sens"]
    for rid in ("c1_dolly_r8", "c6_base_vs_base", None):
        assert _b.discover_batteries(tmp_path, manifest, rid) == ["karne_ladders", "karne_ladders_sens"]
    assert _b.is_restricted("karne_ladders_sampled_base") and "karne_ladders_sampled_base" in _b.KARNE_BATTERIES
    bad = _b.Decoding(mode="greedy")
    with pytest.raises(SystemExit):
        _b.check_karne_decoding("karne_ladders_sampled_base", bad)
    _b.check_karne_decoding("karne_ladders_sampled_base", _b.Decoding(mode="sample", temperature=1.0, top_p=1.0, n_samples=8))
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "batteries.yaml").read_text())
    e = cfg["batteries"]["karne_ladders_sampled_base"]
    assert e["on_demand"] is True and e["base_only"] is True and e["n_ladders"] is None and e["expected_n"] == 1941
    assert e["subset_of"] == "karne_ladders" and e["decoding"] == "karne_sampled" and e["harmful"] is True


def test_vllm_bf16_merged_export_refused_for_measured_derivatives(fake_models, tiny_model_dir, tmp_path, monkeypatch, capsys):
    """Review finding 6 on the vLLM path: a merged export cast to bf16 rounds LoRA updates away."""
    monkeypatch.setattr(battery, "vllm_available", lambda: True)
    items_dir = tmp_path / "prompts"
    items_dir.mkdir()
    (items_dir / "harmbench.jsonl").write_text(json.dumps({"id": "q0", "prompt": "x"}) + "\n")
    adapter = _make_adapter(tiny_model_dir, tmp_path / "adapters" / "r9")
    args = ["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(tmp_path / "r"), "--backend", "vllm",
            "--allow-vllm-measured", "--batteries", "harmbench", "--adapter", str(adapter), "--merge-dir", str(tmp_path / "m")]
    assert battery.main(args) == 2 and "rounds LoRA updates away" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Hazard routing (acceptance OPEN-2, OPEN-3) and the A14 vLLM multimodal limit
# ---------------------------------------------------------------------------


def test_freeform_batteries_are_restricted_by_name_and_config():
    for name in ("em_freeform", "persona_evil", "persona_sycophancy", "persona_hallucination", "karne_ladders_sens"):
        assert battery.is_restricted(name), name
    for name in ("xstest_safe", "mmlu", "ifeval", "smoke", "emphasis", "personal"):
        assert not battery.is_restricted(name), name
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "batteries.yaml").read_text())
    names = battery.restricted_battery_names(cfg)
    assert names == {"strongreject", "harmbench", "xstest_unsafe", "karne_ladders", "karne_ladders_sens", "karne_ladders_sampled_base",
                     "persona_evil", "persona_sycophancy", "persona_hallucination", "em_freeform"}
    # restricted is distinct from harmful: the free-form batteries keep harmful false (labels unchanged).
    for b in ("em_freeform", "persona_evil", "persona_sycophancy", "persona_hallucination"):
        assert cfg["batteries"][b]["restricted"] is True and cfg["batteries"][b]["harmful"] is False
    assert battery.restriction_reason("custom", {"restricted": True}) == "config:restricted"
    assert battery.restriction_reason("custom", {"harmful": True}) == "config:harmful"
    assert battery.restriction_reason("persona_evil", {"restricted": False, "harmful": False}) == "name_rule"
    assert battery.restriction_reason("xstest_safe", {"harmful": False}) is None


def test_config_restricted_flag_routes_and_sidecar_records_it(fake_models, tiny_model_dir, tmp_path):
    items_dir = _one_battery(tmp_path, "custom_free")
    cfg = _bcfg(tmp_path, "batteries:\n  custom_free:\n    harmful: false\n    restricted: true\n    decoding: greedy\n")
    out = tmp_path / "res"
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out), "--batteries", "custom_free",
                         "--batteries-config", str(cfg), "--max-new-tokens", "2"]) == 0
    slug = battery.path_slug(tiny_model_dir)
    p = out / "restricted" / "batteries" / slug / "base" / "custom_free.jsonl"
    assert p.exists() and _mode(p) == 0o600 and _mode(p.parent) == 0o700 and _mode(out / "restricted") == 0o700
    side = json.loads(battery.meta_path_for(p).read_text())
    assert side["restricted"] is True and side["meta"]["restricted"] is True
    assert side["meta"]["restricted_reason"] == "config:restricted"
    assert not (out / "batteries").exists()


def test_legacy_unrestricted_outputs_are_moved(tmp_path):
    out_root = tmp_path / "results"
    rroot = out_root / "restricted"
    legacy = battery.battery_output_path(out_root, "b", "r", "persona_evil", restricted=False)
    new = battery.battery_output_path(out_root, "b", "r", "persona_evil", restricted=True, restricted_root=rroot)
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"item_id": "q0"}\n')
    battery.meta_path_for(legacy).write_text(json.dumps({"restricted": False, "meta": {"item_file_sha256": "x"}}))
    battery.partial_path_for(legacy).write_text("")
    logs: list[str] = []
    moved = battery.migrate_unrestricted_outputs(legacy, new, rroot, log=logs.append)
    assert len(moved) == 3 and not legacy.exists() and not battery.meta_path_for(legacy).exists()
    assert new.read_text() == '{"item_id": "q0"}\n' and _mode(new) == 0o600 and _mode(new.parent) == 0o700
    side = json.loads(battery.meta_path_for(new).read_text())
    assert side["restricted"] is True and side["restricted_migrated_from"] == str(battery.meta_path_for(legacy))
    assert battery.partial_path_for(new).exists() and all("WARNING" in m for m in logs)
    # A collision keeps the restricted copy and parks the legacy one next to it under a suffix no reader globs.
    legacy.write_text('{"item_id": "old"}\n')
    battery.migrate_unrestricted_outputs(legacy, new, rroot, log=logs.append)
    assert new.read_text() == '{"item_id": "q0"}\n'
    parked = new.with_name(new.name + battery.LEGACY_SUFFIX)
    assert parked.read_text() == '{"item_id": "old"}\n' and _mode(parked) == 0o600
    assert not list(new.parent.glob("*.jsonl.partial.*")) and sorted(f.name for f in new.parent.glob("*.jsonl")) == ["persona_evil.jsonl"]
    assert battery.migrate_unrestricted_outputs(legacy, new, rroot) == []


def test_cli_moves_a_legacy_freeform_output_before_planning(fake_models, tiny_model_dir, tmp_path):
    items_dir = _one_battery(tmp_path, "em_freeform")
    cfg = _bcfg(tmp_path, "batteries:\n  em_freeform:\n    decoding: greedy\n")
    out = tmp_path / "res"
    slug = battery.path_slug(tiny_model_dir)
    legacy = out / "batteries" / slug / "base" / "em_freeform.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"item_id": "stale"}\n')
    assert battery.main(["--base", tiny_model_dir, "--items-dir", str(items_dir), "--out", str(out), "--batteries", "em_freeform",
                         "--batteries-config", str(cfg), "--max-new-tokens", "2"]) == 0
    assert not legacy.exists()
    recs = _read_jsonl(out / "restricted" / "batteries" / slug / "base" / "em_freeform.jsonl")
    assert [r["item_id"] for r in recs] == ["q0", "q1"]  # no sidecar: the moved file was regenerated


def test_is_multimodal_checkpoint_from_config(tmp_path, tiny_model_dir):
    vl = tmp_path / "vl"
    vl.mkdir()
    (vl / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                "vision_config": {"depth": 2}, "text_config": {}}))
    txt = tmp_path / "txt"
    txt.mkdir()
    (txt / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    assert battery.is_multimodal_checkpoint(str(vl)) is True
    assert battery.is_multimodal_checkpoint(str(txt)) is False
    assert battery.is_multimodal_checkpoint(tiny_model_dir) is False
    assert battery.is_multimodal_checkpoint(str(tmp_path / "missing")) is False


def _fake_vllm_module(monkeypatch, reject_mm: bool = False) -> list:
    built: list = []

    class LLM:
        def __init__(self, **kw):
            if reject_mm and "limit_mm_per_prompt" in kw:
                raise ValueError("You have set Multimodal configuration but the loaded model is not multimodal.")
            built.append(kw)

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, _FakeSP
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    return built


def test_vllm_text_only_limit_passed_for_multimodal_checkpoints(monkeypatch):
    """Review A14: limit_mm_per_prompt {"image": 0, "video": 0} reaches vllm.LLM and the recorded kwargs."""
    pytest.importorskip("sm.common.models")
    built = _fake_vllm_module(monkeypatch)
    tok = _bos_tokenizer()
    gen = battery.VLLMGenerator("Qwen/Qwen3.5-9B", tok, 8, 0, limit_mm_per_prompt=battery.TEXT_ONLY_MM_LIMIT)
    assert built[-1]["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert gen.llm_kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0} and gen.limit_mm_note is None
    battery.VLLMGenerator("/scratch/export", tok, 8, 0)
    assert "limit_mm_per_prompt" not in built[-1]
    # A vLLM build that rejects the limit for this checkpoint: served without it, and the reason is recorded.
    built2 = _fake_vllm_module(monkeypatch, reject_mm=True)
    gen2 = battery.VLLMGenerator("x", tok, 8, 0, limit_mm_per_prompt=battery.TEXT_ONLY_MM_LIMIT)
    assert "limit_mm_per_prompt" not in built2[-1] and gen2.limit_mm_per_prompt is None
    assert "rejected" in gen2.limit_mm_note and "limit_mm_per_prompt" not in gen2.llm_kwargs


def test_vllm_battery_meta_records_mm_limit(tiny_model_dir, tmp_path, monkeypatch):
    pytest.importorskip("sm.common.models")
    engines: list = []

    class FakeVLLM(battery.VLLMGenerator):
        def __init__(self, model_path, tokenizer, max_new_tokens, seed, **kw):
            engines.append({"model_path": model_path, **kw})
            super().__init__(model_path, tokenizer, max_new_tokens, seed, llm=_FakeLLM(), sampling_params_cls=_FakeSP,
                             limit_mm_per_prompt=kw.get("limit_mm_per_prompt"))

    monkeypatch.setattr(battery, "vllm_available", lambda: True)
    monkeypatch.setattr(battery, "VLLMGenerator", FakeVLLM)
    out = tmp_path / "results"
    slug = battery.path_slug(tiny_model_dir)
    # Text-only export (merge-dir path): no limit.
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--backend", "vllm",
                         "--merge-dir", str(tmp_path / "m"), "--run-id", "export"]) == 0
    m = json.loads(battery.meta_path_for(out / "batteries" / slug / "export" / "smoke.jsonl").read_text())["meta"]["model"]
    assert engines[-1]["limit_mm_per_prompt"] is None and m["multimodal_checkpoint"] is False and m["limit_mm_per_prompt"] is None
    # The original checkpoint served directly (--vllm-lora) and detected as multimodal: the limit is passed and recorded.
    monkeypatch.setattr(battery, "is_multimodal_checkpoint", lambda path, revision=None: True)
    assert battery.main(["--smoke", "--base", tiny_model_dir, "--out", str(out), "--backend", "vllm", "--vllm-lora",
                         "--run-id", "direct"]) == 0
    m = json.loads(battery.meta_path_for(out / "batteries" / slug / "direct" / "smoke.jsonl").read_text())["meta"]["model"]
    assert engines[-1]["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert m["multimodal_checkpoint"] is True and m["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert m["vllm_kwargs"]["limit_mm_per_prompt"] == {"image": 0, "video": 0}


def test_karne_sens_decoding_is_exactly_karnes():
    """v0.3 (M7): karne_ladders_sens is decoded as Karne's labels were (T = 1.0, top-p 1.0, 8 samples), or refused."""
    import yaml

    from sm.eval import battery as _b

    cfg = yaml.safe_load((ROOT / "configs" / "batteries.yaml").read_text())
    dec = _b.resolve_decoding("karne_ladders_sens", cfg)
    _b.check_karne_decoding("karne_ladders_sens", dec)  # the shipped config passes
    assert (dec.mode, dec.temperature, dec.top_p, dec.n_samples) == ("sample", 1.0, 1.0, 8)
    with pytest.raises(SystemExit, match="Karne"):
        _b.check_karne_decoding("karne_ladders_sens", _b.resolve_decoding("karne_ladders_sens", cfg, cli_n_samples=4))
    _b.check_karne_decoding("karne_ladders", _b.resolve_decoding("karne_ladders", cfg))  # greedy battery: not checked
