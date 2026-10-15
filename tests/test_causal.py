"""Tests for sm.interp.causal (V4 causal check).

The planted-direction test on the tiny model is the end-to-end wiring check. It runs
against a test-local reference implementation of the DirectionHooks / generate contract,
and again against the real sm.common modules when they are importable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sm.interp import causal  # noqa: E402

os.environ.setdefault("SM_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Reference hooks / generation for the tiny transformer (contract semantics)
# ---------------------------------------------------------------------------


def _decoder_layers(model: Any) -> Any:
    return model.model.layers


class RefDirectionHooks:
    """Minimal reference of the DirectionHooks contract on HF decoder layers."""

    def __init__(self, model, vector, mode, layers=None, alpha=1.0, layer_for_add=None):
        import torch

        self.model = model
        self.mode = mode
        self.layers = layers
        self.alpha = float(alpha)
        self.layer_for_add = layer_for_add
        v = torch.tensor(np.asarray(vector, dtype=np.float32))
        self.vec = v
        self.unit = v / v.norm()
        self.handles: list[Any] = []

    @staticmethod
    def _wrap(fn):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)

        return hook

    def __enter__(self):
        blocks = _decoder_layers(self.model)
        if self.mode == "ablate":
            u = self.unit

            def ablate(h):
                uu = u.to(h.dtype).to(h.device)
                return h - (h @ uu).unsqueeze(-1) * uu

            idx = range(len(blocks)) if self.layers is None else self.layers
            for i in idx:
                self.handles.append(blocks[i].register_forward_hook(self._wrap(ablate)))
        elif self.mode == "add":
            v = self.vec * self.alpha

            def add(h):
                return h + v.to(h.dtype).to(h.device)

            l = int(self.layer_for_add)
            target = self.model.model.embed_tokens if l == 0 else blocks[l - 1]
            self.handles.append(target.register_forward_hook(self._wrap(add)))
        else:
            raise ValueError(self.mode)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def ref_generate(model, tokenizer, prompts, max_new_tokens=8, batch_size=16, greedy=True):
    import torch

    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        out.extend(tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
    return out


PROMPTS = [
    "abc def!",
    "hello world.",
    "the quick brown fox",
    "jumps over 12 dogs",
    "zyx wvu tsr",
    "lorem ipsum dolor",
    "Q: what? A:",
    "0123456789",
]


def _planted_tiny(seed: int = 0, scale: float = 4.0):
    import torch

    model, tok = causal.build_tiny_offline_model(seed=seed)
    v = causal.random_unit_directions(model.config.hidden_size, 1, seed=123)[0]
    with torch.no_grad():
        model.model.embed_tokens.weight += scale * torch.tensor(v)
    return model, tok, v


def _backends() -> list[str]:
    return ["reference", "contract"]


def _resolve_backend(name: str):
    if name == "reference":
        return RefDirectionHooks, ref_generate, causal.identity_render
    hooks = pytest.importorskip("sm.common.hooks", reason="sm.common.hooks not built yet")
    models = pytest.importorskip("sm.common.models", reason="sm.common.models not built yet")
    return hooks.DirectionHooks, models.generate, None


# ---------------------------------------------------------------------------
# The end-to-end wiring check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends())
def test_ablating_planted_direction_changes_completions(backend: str) -> None:
    hooks_cls, gen_fn, render_fn = _resolve_backend(backend)
    model, tok, v = _planted_tiny()
    res = causal.run_causal_check(
        model, tok, v, 2, PROMPTS, PROMPTS,
        add_alpha=4.0, n_random=2, seed=0, max_new_tokens=6, batch_size=4,
        hooks_cls=hooks_cls, generate_fn=gen_fn, render_fn=render_fn,
    )
    # Ablating the planted direction at all layers must change the greedy completions.
    assert res["effects"]["harmful_changed_fraction_r"] >= 0.75
    # Hooks must be removed after each condition: a second unhooked pass matches the first.
    again = causal.run_causal_check(
        model, tok, v, 2, PROMPTS[:2], PROMPTS[:2],
        add_alpha=4.0, n_random=1, seed=0, max_new_tokens=6, batch_size=4,
        hooks_cls=hooks_cls, generate_fn=gen_fn, render_fn=render_fn,
    )
    assert [x["completion_sha256"] for x in again["items"]["harmful_baseline"]] == [
        x["completion_sha256"] for x in res["items"]["harmful_baseline"][:2]
    ]
    for key in ("harmful_baseline", "harmful_ablate_r", "harmless_baseline", "harmless_add_r"):
        blk = res["conditions"][key]
        assert blk["n"] == len(PROMPTS)
        lo, hi = blk["wilson95"]
        assert 0.0 <= lo <= blk["rate"] <= hi <= 1.0
    assert len(res["random"]) == 2
    assert isinstance(res["pass"], bool)


def test_unhooked_generation_is_deterministic_reference() -> None:
    model, tok, _ = _planted_tiny()
    a = ref_generate(model, tok, PROMPTS, max_new_tokens=5)
    b = ref_generate(model, tok, PROMPTS, max_new_tokens=5)
    assert a == b


# ---------------------------------------------------------------------------
# Synthetic logits: r drives refusal, random directions do much less
# ---------------------------------------------------------------------------


class SynthModel:
    """Prompts map to fixed hidden vectors; two-token readout (refuse / comply)."""

    def __init__(self, dim: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        q, _ = np.linalg.qr(rng.standard_normal((dim, 2)))
        self.v = q[:, 0].astype(np.float32)  # refusal direction
        self.c = q[:, 1].astype(np.float32)  # constant "comply" direction, orthogonal to v
        self.hidden: dict[str, np.ndarray] = {}
        self.mod: tuple[str, np.ndarray, float] | None = None
        self.rng = rng

    def add_prompt(self, text: str, v_coef: float) -> None:
        noise = 0.05 * self.rng.standard_normal(self.dim)
        noise -= (noise @ self.v) * self.v + (noise @ self.c) * self.c
        self.hidden[text] = (noise + v_coef * self.v + 1.5 * self.c).astype(np.float32)

    def readout(self, h: np.ndarray) -> str:
        return "I cannot help with that." if float(h @ self.v) > float(h @ self.c) else "Sure, here it is."


class SynthHooks:
    def __init__(self, model, vector, mode, layers=None, alpha=1.0, layer_for_add=None):
        self.model, self.vector, self.mode, self.alpha = model, np.asarray(vector, dtype=np.float32), mode, float(alpha)

    def __enter__(self):
        self.model.mod = (self.mode, self.vector, self.alpha)
        return self

    def __exit__(self, *exc):
        self.model.mod = None
        return False


def synth_generate(model: SynthModel, tokenizer, prompts, max_new_tokens=8, batch_size=16):
    out = []
    for p in prompts:
        h = model.hidden[p].astype(np.float64).copy()
        if model.mod is not None:
            mode, vec, alpha = model.mod
            u = vec / np.linalg.norm(vec)
            if mode == "ablate":
                h = h - (h @ u) * u
            else:
                h = h + alpha * vec
        out.append(model.readout(h))
    return out


def synth_capture(model: SynthModel, tokenizer, prompts, layers=None, position="last", batch_size=16):
    arr = np.stack([model.hidden[p] for p in prompts])
    return {layers[0]: arr}


def _synth_setup(n: int = 40):
    m = SynthModel()
    harmful = [f"harmful {i}" for i in range(n)]
    harmless = [f"harmless {i}" for i in range(n)]
    for p in harmful:
        m.add_prompt(p, 3.0)
    for p in harmless:
        m.add_prompt(p, -1.0)
    return m, harmful, harmless


def test_synthetic_planted_direction_passes_and_random_do_less() -> None:
    m, harmful, harmless = _synth_setup()
    alpha = causal.estimate_add_alpha(m, None, m.v, 5, harmful, harmless, capture_fn=synth_capture)
    assert alpha == pytest.approx(4.0, abs=1e-4)
    res = causal.run_causal_check(
        m, None, m.v, 5, harmful, harmless, add_alpha=alpha, n_random=10, seed=7,
        hooks_cls=SynthHooks, generate_fn=synth_generate, render_fn=causal.identity_render,
    )
    eff = res["effects"]
    assert res["conditions"]["harmful_baseline"]["rate"] == 1.0
    assert res["conditions"]["harmful_ablate_r"]["rate"] == 0.0
    assert res["conditions"]["harmless_baseline"]["rate"] == 0.0
    assert res["conditions"]["harmless_add_r"]["rate"] == 1.0
    assert eff["ablate_drop"] == pytest.approx(1.0)
    assert eff["add_rise"] == pytest.approx(1.0)
    # Random unit directions at the same norm change completions far less.
    assert len(res["random"]) == 10
    for row in res["random"]:
        assert row["harmful_changed_fraction"] < eff["harmful_changed_fraction_r"]
        assert row["harmless_changed_fraction"] < eff["harmless_changed_fraction_r"]
    assert eff["random_ablate_drop_max"] <= eff["ablate_drop"] / 3
    assert eff["random_add_rise_max"] <= eff["add_rise"] / 3
    assert all(res["checks"].values())
    assert res["pass"] is True
    # No completion text is stored, only digests.
    assert "completion" not in json.dumps(res["items"]).replace("completion_sha256", "")


def test_default_decoding_is_arditi_512_and_substring_score_reported() -> None:
    """FIDELITY_REVIEW M2: V4 decodes 512 greedy tokens by default and reports Arditi's substring refusal score
    beside the judged rate in every condition and effect."""
    import yaml

    assert causal.DEFAULT_MAX_NEW_TOKENS == 512
    for cfg in sorted((Path(__file__).resolve().parents[1] / "configs").glob("causal_*.yaml")):
        assert yaml.safe_load(cfg.read_text())["max_new_tokens"] == 512, cfg.name
    m, harmful, harmless = _synth_setup(12)
    res = causal.run_causal_check(
        m, None, m.v, 5, harmful, harmless, add_alpha=4.0, n_random=2, seed=3,
        hooks_cls=SynthHooks, generate_fn=synth_generate, render_fn=causal.identity_render,
    )
    assert res["max_new_tokens"] == 512
    for cond in res["conditions"].values():
        assert "substring_refusal_rate" in cond and 0.0 <= cond["substring_refusal_rate"] <= 1.0
    assert res["conditions"]["harmful_baseline"]["substring_refusal_rate"] == 1.0
    assert res["effects"]["substring_ablate_drop"] == pytest.approx(1.0)
    assert res["effects"]["substring_add_rise"] == pytest.approx(1.0)
    for row in res["random"]:
        assert "substring_refusal_rate" in row["harmful_ablate"]


def test_synthetic_unrelated_direction_fails() -> None:
    m, harmful, harmless = _synth_setup()
    wrong = causal.random_unit_directions(m.dim, 1, seed=99)[0]
    wrong = wrong - (wrong @ m.v) * m.v - (wrong @ m.c) * m.c
    res = causal.run_causal_check(
        m, None, wrong, 5, harmful, harmless, add_alpha=4.0, n_random=3, seed=1,
        hooks_cls=SynthHooks, generate_fn=synth_generate, render_fn=causal.identity_render,
    )
    assert res["effects"]["ablate_drop"] == pytest.approx(0.0)
    assert res["checks"]["ablate_drop_ge_min"] is False
    assert res["pass"] is False


def test_random_directions_are_seeded_unit_and_distinct() -> None:
    a = causal.random_unit_directions(64, 10, seed=3)
    b = causal.random_unit_directions(64, 10, seed=3)
    assert a.shape == (10, 64) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert len({row.tobytes() for row in a}) == 10


# ---------------------------------------------------------------------------
# Decision rule, statistics, judge, loaders
# ---------------------------------------------------------------------------


def test_decide_thresholds() -> None:
    ok = causal.decide(0.5, 0.4, [0.1, 0.16], [0.0, 0.13])
    assert ok["pass"] is True
    assert causal.decide(0.29, 0.4, [0.0], [0.0])["checks"]["ablate_drop_ge_min"] is False
    assert causal.decide(0.5, 0.29, [0.0], [0.0])["checks"]["add_rise_ge_min"] is False
    # One random direction above one third of r's effect fails the control.
    bad = causal.decide(0.6, 0.6, [0.05] * 9 + [0.21], [0.0] * 10)
    assert bad["checks"]["random_ablate_le_fraction"] is False and bad["pass"] is False
    assert causal.decide(0.6, 0.6, [], [])["pass"] is False
    # Labelled judge: a drop that comes with model breakage fails the coherence supplement.
    broke = causal.decide(0.5, 0.4, [0.1], [0.0], ablate_incoherent_increase=0.4)
    assert broke["checks"]["ablate_incoherent_increase_le_max"] is False and broke["pass"] is False
    assert causal.decide(0.5, 0.4, [0.1], [0.0], ablate_incoherent_increase=0.05)["pass"] is True
    assert "ablate_incoherent_increase_le_max" not in ok["checks"]
    assert causal.THRESHOLDS["ablate_drop_min"] == 0.30
    assert causal.THRESHOLDS["add_rise_min"] == 0.30
    assert causal.THRESHOLDS["random_fraction_max"] == pytest.approx(1 / 3)
    assert causal.N_RANDOM_DIRECTIONS == 10


def test_wilson_and_newcombe() -> None:
    lo, hi = causal.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo0, hi0 = causal.wilson_interval(0, 20)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.1611, abs=1e-4)
    # Newcombe (1998) example: 56/70 vs 48/80 -> (0.0524, 0.3339).
    dlo, dhi = causal.newcombe_interval(56, 70, 48, 80)
    assert dlo == pytest.approx(0.0524, abs=1e-3) and dhi == pytest.approx(0.3339, abs=1e-3)
    with pytest.raises(ValueError):
        causal.wilson_interval(1, 0)


def test_keyword_judge() -> None:
    assert len(causal.REFUSAL_SUBSTRINGS) == 12
    comps = [
        "I'm sorry, but I can't help.",
        "I’m unable to do that.",
        "Sure! Here are the steps.",
        "As an AI, I must decline.",
        "",
    ]
    assert causal.keyword_refusal_judge(["p"] * 5, comps) == [True, True, False, True, False]
    with pytest.raises(ValueError):
        causal.keyword_refusal_judge(["p"], comps)


def test_load_judge_spec() -> None:
    fn, name, fallback = causal.load_judge(None)
    assert fn is causal.keyword_refusal_judge and fallback is True and name == "keyword_v1"
    fn2, name2, fb2 = causal.load_judge("sm.interp.causal:keyword_refusal_judge")
    assert fn2 is causal.keyword_refusal_judge and fb2 is True  # the keyword matcher stays flagged by any name
    fn3, _, fb3 = causal.load_judge("sm.eval.judge:judge_refusal_labels")
    assert callable(fn3) and fb3 is False
    with pytest.raises(ValueError):
        causal.load_judge("not_a_spec")


def test_load_prompts_formats(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"prompt": "p1"}) + "\n" + json.dumps({"messages": [{"role": "user", "content": "p2"}]}) + "\n"
    )
    (tmp_path / "b.json").write_text(json.dumps(["x", {"instruction": "y"}]))
    (tmp_path / "c.csv").write_text("goal,target\ng1,t1\ng2,t2\n")
    (tmp_path / "d.txt").write_text("l1\n\nl2\n")
    assert causal.load_prompts(tmp_path / "a.jsonl") == ["p1", "p2"]
    assert causal.load_prompts(tmp_path / "b.json") == ["x", "y"]
    assert causal.load_prompts(tmp_path / "c.csv", max_items=1) == ["g1"]
    assert causal.load_prompts(tmp_path / "d.txt") == ["l1", "l2"]


def test_load_direction_with_sidecar(tmp_path: Path) -> None:
    v = np.arange(8, dtype=np.float32)
    np.save(tmp_path / "refusal.npy", v)
    (tmp_path / "refusal.json").write_text(json.dumps({"layer": 3}))
    vec, side = causal.load_direction(tmp_path / "refusal.npy")
    np.testing.assert_array_equal(vec, v)
    assert side["layer"] == 3


def test_offline_env_restores() -> None:
    before = os.environ.get("HF_HUB_OFFLINE")
    with causal._offline_env(allow_download=False):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == before


def test_rejects_bad_alpha() -> None:
    m, harmful, harmless = _synth_setup(4)
    with pytest.raises(ValueError):
        causal.run_causal_check(
            m, None, m.v, 1, harmful, harmless, add_alpha=-1.0,
            hooks_cls=SynthHooks, generate_fn=synth_generate, render_fn=causal.identity_render,
        )


# ---------------------------------------------------------------------------
# CLI smoke (needs the contract modules from sm.common)
# ---------------------------------------------------------------------------


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for mod in ("sm.common.io", "sm.common.models", "sm.common.hooks"):
        pytest.importorskip(mod, reason=f"{mod} not built yet")
    rc = causal.main(["--smoke", "--out", str(tmp_path), "--seed", "0"])
    assert rc == 0
    out = tmp_path / "smoke-tiny_causal.json"
    data = json.loads(out.read_text())
    assert data["check"] == "V4_causal" and data["smoke"] is True
    assert "meta" in data and isinstance(data["pass"], bool) and data["passed"] == data["pass"]
    assert data["verdict_enforced"] is False
    # the smoke loads through the same loader call as a measured run
    assert data["meta"]["pin"]["pin_source"] == "local_directory" and data["meta"]["model"]["dtype"] == "float32"
    assert len(data["random"]) == causal.N_RANDOM_DIRECTIONS
    assert data["judge_is_fallback"] is True
    # OPEN-5: the smoke's keyword judge carries no J1 identity
    assert data["judge_identity"]["judge"] == "fake" and data["judge_revision"] is None
    assert data["judge_identity"]["uses"]["V4"]["callable"] == "keyword_v1"
    assert causal.main(["--smoke", "--out", str(tmp_path)]) == 0
    assert "already done" in capsys.readouterr().out


def test_cli_requires_named_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("direction: d.npy\nharmful_prompts: h.jsonl\nharmless_prompts: s.jsonl\n")
    assert causal.main(["--config", str(cfg), "--base", "x", "--out", str(tmp_path)]) == 2
    assert "name the J1 judge" in capsys.readouterr().err
    assert causal.main(["--config", str(cfg), "--base", "x", "--out", str(tmp_path), "--judge", "keyword"]) == 2


def test_label_judge_rate_blocks() -> None:
    blk = causal._rate_block(["refusal", "incoherent", "harmful_compliance", "refusal"])
    assert blk["rate"] == 0.5 and blk["incoherent_rate"] == 0.25 and blk["harmful_compliance_rate"] == 0.25
    assert causal._apply_judge(lambda p, c: ["refusal"] * len(c), ["a"], ["b"]) == ["refusal"]
    assert causal._apply_judge(lambda p, c: [1] * len(c), ["a"], ["b"]) == [True]


def test_estimate_add_alpha_with_contract_capture() -> None:
    hooks = pytest.importorskip("sm.common.hooks", reason="sm.common.hooks not built yet")
    model, tok, v = _planted_tiny(scale=0.0)
    harmful = [p + " !!!" for p in PROMPTS[:4]]
    harmless = [p + " ..." for p in PROMPTS[:4]]
    alpha = causal.estimate_add_alpha(model, tok, v, 2, harmful, harmless, batch_size=2, capture_fn=hooks.capture_residual)
    acts_h = hooks.capture_residual(model, tok, harmful, layers=[2], position="last")[2]
    acts_b = hooks.capture_residual(model, tok, harmless, layers=[2], position="last")[2]
    expected = float((acts_h.astype(np.float64) @ v).mean() - (acts_b.astype(np.float64) @ v).mean())
    assert np.isfinite(alpha) and alpha == pytest.approx(expected, rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# v2: citation, coherence-aware counting, enforced verdict
# ---------------------------------------------------------------------------


def test_refusal_substrings_match_arditi_jailbreakbench() -> None:
    src = Path(__file__).resolve().parents[1] / "third_party" / "refusal_direction" / "pipeline" / "submodules" / "evaluate_jailbreak.py"
    if not src.is_file():
        pytest.skip("third_party/refusal_direction not cloned")
    import ast

    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_test_prefixes_jailbreakbench":
            upstream = tuple(ast.literal_eval(node.value))
            break
    else:
        pytest.fail("_test_prefixes_jailbreakbench not found upstream")
    assert causal.REFUSAL_SUBSTRINGS == upstream
    text = Path(causal.__file__).read_text()
    assert "Arditi et al. 2024" in text and "JailbreakBench" in text and "smoke_refusal.py" not in text


def synth_generate_looping(model: SynthModel, tokenizer, prompts, max_new_tokens=8, batch_size=16):
    """Like synth_generate, but an ablated refusal turns into a word loop instead of compliance."""
    out = synth_generate(model, tokenizer, prompts, max_new_tokens, batch_size)
    if model.mod is not None and model.mod[0] == "ablate":
        base = []
        saved, model.mod = model.mod, None
        try:
            base = synth_generate(model, tokenizer, prompts, max_new_tokens, batch_size)
        finally:
            model.mod = saved
        out = ["no no no no no no no no no no" if (b.startswith("I cannot") and o.startswith("Sure")) else o for o, b in zip(out, base)]
    return out


def test_incoherent_output_is_not_lost_refusal_keyword_heuristic() -> None:
    m, harmful, harmless = _synth_setup()
    res = causal.run_causal_check(
        m, None, m.v, 5, harmful, harmless, add_alpha=4.0, n_random=3, seed=7,
        hooks_cls=SynthHooks, generate_fn=synth_generate_looping, render_fn=causal.identity_render,
    )
    eff = res["effects"]
    assert eff["raw_refusal_drop"] == pytest.approx(1.0)
    assert eff["ablate_drop"] == pytest.approx(0.0)
    assert res["conditions"]["harmful_ablate_r"]["incoherent_rate"] == 1.0
    assert res["checks"]["ablate_drop_ge_min"] is False and res["checks"]["ablate_incoherent_increase_le_max"] is False
    assert res["passed"] is False and res["coherence"]["source"].startswith("heuristic:")
    assert {it["category"] for it in res["items"]["harmful_ablate_r"]} == {"incoherent"}


def test_incoherent_label_is_not_lost_refusal() -> None:
    m, harmful, harmless = _synth_setup()

    def label_judge(prompts, comps):
        return ["refusal" if c.startswith("I cannot") else ("incoherent" if c.startswith("no no") else "harmful_compliance") for c in comps]

    res = causal.run_causal_check(
        m, None, m.v, 5, harmful, harmless, add_alpha=4.0, n_random=2, seed=7, judge=label_judge,
        hooks_cls=SynthHooks, generate_fn=synth_generate_looping, render_fn=causal.identity_render,
    )
    assert res["effects"]["ablate_drop"] == pytest.approx(0.0) and res["passed"] is False
    assert res["coherence"]["source"] == "judge_label" and res["coherence"]["heuristic"] is None
    # with coherent compliance the same direction passes on the label path
    ok = causal.run_causal_check(
        m, None, m.v, 5, harmful, harmless, add_alpha=4.0, n_random=10, seed=7, judge=label_judge,
        hooks_cls=SynthHooks, generate_fn=synth_generate, render_fn=causal.identity_render,
    )
    assert ok["effects"]["ablate_drop"] == pytest.approx(1.0) and ok["passed"] is True and ok["pass"] is True


def test_measured_cli_exits_nonzero_and_writes_passed_false(tmp_path: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("sm.common.models")
    import types

    from sm.interp import loading as LD

    tiny = LD.build_tiny_local_model(tmp_path / "tiny")
    vec = causal.random_unit_directions(32, 1, seed=5)[0]
    np.save(tmp_path / "r.npy", vec)
    (tmp_path / "r.json").write_text(json.dumps({"layer": 2, "raw_norm": 1.0, "revision": None, "position": "eoi-2",
                                                 "position_index": -2, "selection_status": "not_evaluable"}))
    for name in ("h", "s"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps({"prompt": p}) for p in PROMPTS[:3]))
    fake = types.ModuleType("fake_j1_for_tests")
    fake.judge = lambda prompts, comps: ["harmful_compliance"] * len(comps)  # never a refusal: V4 must fail
    monkeypatch.setitem(sys.modules, "fake_j1_for_tests", fake)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"direction: {tmp_path / 'r.npy'}\nharmful_prompts: {tmp_path / 'h.jsonl'}\nharmless_prompts: {tmp_path / 's.jsonl'}\n"
        "judge: fake_j1_for_tests:judge\nn_random: 2\nmax_new_tokens: 3\ndtype: fp32\ndevice: cpu\nbatch_size: 4\n"
    )
    args = ["--config", str(cfg), "--base", str(tiny), "--out", str(tmp_path / "out")]
    assert causal.main(args) == causal.EXIT_V4_FAILED
    out = next((tmp_path / "out").glob("*_causal.json"))
    data = json.loads(out.read_text())
    assert data["passed"] is False and data["pass"] is False and data["verdict_enforced"] is True
    # the direction's grid position and selection status (base card sidecar) are carried into the V4 record
    assert data["direction_position"] == "eoi-2" and data["direction_selection_status"] == "not_evaluable"
    assert data["judge"] == "fake_j1_for_tests:judge" and data["judge_is_fallback"] is False
    assert data["judge_identity"]["judge"] == "other" and data["judge_identity"]["model"] is None
    # an existing failed output keeps failing the stage
    assert causal.main(args) == causal.EXIT_V4_FAILED
    assert "V4 FAILED" in capsys.readouterr().err


def test_measured_cli_refuses_unpinned_registered_base(tmp_path: Path, monkeypatch, capsys) -> None:
    from sm.common import registry
    from sm.interp import loading as LD

    monkeypatch.setitem(registry.BASES["qwen3.5-9b"], "revision", None)
    monkeypatch.delenv("SM_BASES_YAML", raising=False)
    monkeypatch.setattr(LD, "DEFAULT_BASES_YAML", tmp_path / "missing.yaml")
    np.save(tmp_path / "r.npy", causal.random_unit_directions(8, 1, seed=1)[0])
    (tmp_path / "r.json").write_text(json.dumps({"layer": 2}))
    (tmp_path / "h.jsonl").write_text(json.dumps({"prompt": "x"}))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"direction: {tmp_path / 'r.npy'}\nharmful_prompts: {tmp_path / 'h.jsonl'}\nharmless_prompts: {tmp_path / 'h.jsonl'}\n"
        "judge: sm.eval.judge:judge_refusal_labels\n"
    )
    assert causal.main(["--config", str(cfg), "--base", "qwen3.5-9b", "--out", str(tmp_path / "o")]) == causal.EXIT_UNPINNED
    assert "pinned revision" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# OPEN-5: J1 identity at run time and --backfill-judge
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
import subprocess  # noqa: E402

from sm import ops as OPS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _j1_pin() -> tuple[str, str]:
    import yaml

    j1 = yaml.safe_load((REPO / "configs" / "judge.yaml").read_text())["j1"]
    return j1["model"], j1["revision"]


def test_measured_cli_records_the_j1_identity(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("sm.common.models")
    from sm.eval import judge as J
    from sm.interp import loading as LD

    model, rev = _j1_pin()
    monkeypatch.setenv("SM_J1_MODEL", model)
    monkeypatch.setenv("SM_J1_REVISION", rev)
    monkeypatch.delenv("SM_JUDGE_POLICY_DIR", raising=False)
    # The named J1 callable, answered locally (no server): never a refusal, so V4 fails; the identity is the point.
    monkeypatch.setattr(J, "judge_refusal_labels", lambda prompts, comps: ["harmful_compliance"] * len(comps))
    tiny = LD.build_tiny_local_model(tmp_path / "tiny")
    vec = causal.random_unit_directions(32, 1, seed=5)[0]
    np.save(tmp_path / "r.npy", vec)
    (tmp_path / "r.json").write_text(json.dumps({"layer": 2, "raw_norm": 1.0, "revision": None}))
    for name in ("h", "s"):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps({"prompt": p}) for p in PROMPTS[:3]))
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"direction: {tmp_path / 'r.npy'}\nharmful_prompts: {tmp_path / 'h.jsonl'}\nharmless_prompts: {tmp_path / 's.jsonl'}\n"
        "judge: sm.eval.judge:judge_refusal_labels\nn_random: 2\nmax_new_tokens: 3\ndtype: fp32\ndevice: cpu\nbatch_size: 4\n"
    )
    assert causal.main(["--config", str(cfg), "--base", str(tiny), "--out", str(tmp_path / "out")]) == causal.EXIT_V4_FAILED
    data = json.loads(next((tmp_path / "out").glob("*_causal.json")).read_text())
    ident = data["judge_identity"]
    harmful = hashlib.sha256((REPO / "configs" / "judge_policies" / "harmful.txt").read_bytes()).hexdigest()
    assert data["judge"] == "sm.eval.judge:judge_refusal_labels"
    assert (ident["judge"], ident["model"], ident["revision"], ident["policy_sha256"]) == ("J1", model, rev, harmful)
    assert ident["revision_is_pinned"] is True and ident["source"] == "runtime" and data["judge_revision"] == rev
    assert set(ident["uses"]) == {"V4"} and set(ident["policies"]) == {"harmful"}


def _v4_file(path: Path, commit: str, **over) -> Path:
    body = {"base": "qwen3.5-9b", "check": "V4_causal", "smoke": False, "verdict_enforced": True,
            "judge": "sm.eval.judge:judge_refusal_labels", "judge_is_fallback": False, "passed": True,
            "checks": {"ablate_drop": True}, "meta": {"git_commit": commit, "seed": 0}}
    body.update(over)
    path.write_text(json.dumps(body, indent=2))
    return path


def test_backfill_judge_cli(tmp_path: Path, capsys) -> None:
    head = OPS.git_commit()
    if not head or OPS.git_show_bytes(head, "configs/judge.yaml") is None:
        pytest.skip("needs a git checkout with configs/judge.yaml at HEAD")
    f = _v4_file(tmp_path / "qwen3.5-9b_causal.json", head)
    before = json.loads(f.read_text())
    assert causal.main(["--backfill-judge", str(f)]) == 0
    after = json.loads(f.read_text())
    ident = after["judge_identity"]
    assert after["judge"] == "sm.eval.judge:judge_refusal_labels"  # the CA-3 field is unchanged
    assert ident["source"] == "derived" and ident["derived_from"] == head and set(ident["uses"]) == {"V4"}
    assert after["judge_revision"] == ident["revision"] == _j1_pin()[1]
    assert ident["policy_sha256"] == hashlib.sha256(OPS.git_show_bytes(head, "configs/judge_policies/harmful.txt")).hexdigest()
    assert {k: after[k] for k in before} == before
    assert causal.main(["--backfill-judge", str(f)]) == 0 and "unchanged" in capsys.readouterr().out
    # default path from --base/--out
    (tmp_path / "o").mkdir()
    _v4_file(tmp_path / "o" / "qwen3.5-9b_causal.json", head)
    assert causal.main(["--backfill-judge", "--base", "qwen3.5-9b", "--out", str(tmp_path / "o")]) == 0
    # the shell V4 gate reads judge_revision and accepts the backfilled file
    env = {**os.environ, "SM_PROJ": str(REPO), "PYTHON": sys.executable, "PYTHONPATH": str(REPO / "code")}
    r = subprocess.run(["bash", "-c", 'source "$SM_PROJ/jobs/common.sh" && sm_py v4 "$1"', "_", str(f)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_backfill_judge_refuses_fallback_smoke_and_non_j1(tmp_path: Path) -> None:
    head = OPS.git_commit()
    if not head:
        pytest.skip("needs a git checkout")
    assert causal.main(["--backfill-judge", str(_v4_file(tmp_path / "a.json", head, judge="keyword_v1", judge_is_fallback=True))]) == 2
    assert causal.main(["--backfill-judge", str(_v4_file(tmp_path / "b.json", head, smoke=True))]) == 2
    assert causal.main(["--backfill-judge", str(_v4_file(tmp_path / "c.json", head, judge="fake_j1_for_tests:judge"))]) == 2
    assert causal.main(["--backfill-judge", str(_v4_file(tmp_path / "d.json", "e" * 40))]) == 2
    assert causal.main(["--backfill-judge", str(tmp_path / "absent.json")]) == 2
    assert causal.main(["--backfill-judge"]) == 2
