"""Tests for the EXPLORATORY module sm.interp.mediation (E15 step 3 and E7; docs/ASSISTANT_AXIS_E7_PLAN.md).

CPU only, tiny random model: the hooks usage (DirectionHooks add at response positions, Lu's capping floor, the
adapter-write removal on a random LoRA), the recovery and verdict rules, the unit selection against the frozen flag
list, the condition list, and one end-to-end unit run with the keyword judge (texts only in the restricted tree)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
os.environ.setdefault("SM_OFFLINE", "1")

from sm.interp import assistant_axis as AX  # noqa: E402
from sm.interp import mediation as MD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ROOT / "results" / "judge" / "qwen3.5-9b" / "_flags" / "j1_rates_flags.txt"


def _hidden(model, tok, text, layer):
    import torch

    from sm.common.models import encode_text

    ids = torch.tensor([encode_text(tok, text)])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer][0].float().numpy()


# --------------------------------------------------------------------------------------------- hooks usage


def test_direction_hooks_add_at_response_positions_touches_only_the_last_prefill_position(tiny_model):
    from sm.common.hooks import DirectionHooks

    model, tok, _ = tiny_model
    H = model.config.hidden_size
    v = np.random.default_rng(0).standard_normal(H).astype(np.float32)
    v /= np.linalg.norm(v)
    text = "Explain the refusal direction in a language model."
    L = 2
    h0 = _hidden(model, tok, text, L)
    with DirectionHooks(model, v, mode="add", alpha=3.0, layer_for_add=L, positions="response") as hk:
        h1 = _hidden(model, tok, text, L)
    assert hk.calls > 0
    np.testing.assert_allclose(h1[:-1], h0[:-1], atol=1e-5)
    np.testing.assert_allclose(h1[-1] - h0[-1], 3.0 * v, atol=1e-4)


def test_cap_hooks_floor_the_axis_projection_at_every_position(tiny_model):
    model, tok, _ = tiny_model
    H = model.config.hidden_size
    a = np.random.default_rng(1).standard_normal(H)
    a /= np.linalg.norm(a)
    text = "How do I bake bread? Mix flour, water, salt and yeast."
    L = 2
    h0 = _hidden(model, tok, text, L)
    p0 = h0 @ a
    srt = np.sort(p0)
    k = len(srt) // 2
    tau = float((srt[k - 1] + srt[k]) / 2)  # strictly between two projections: no float32/float64 tie at the cap
    with MD.CapHooks(model, {L: (a, tau)}) as cap:
        h1 = _hidden(model, tok, text, L)
    p1 = h1 @ a
    assert cap.calls > 0 and cap.n_raised == int((p0 < tau).sum())
    assert np.all(p1 >= tau - 1e-4)
    above = p0 >= tau
    np.testing.assert_allclose(h1[above], h0[above], atol=1e-5)  # positions already at or above the cap are untouched
    # only the axis component moves
    d = h1 - h0
    np.testing.assert_allclose(d - np.outer(d @ a, a), 0.0, atol=1e-4)
    with pytest.raises(ValueError):
        MD.CapHooks(model, {99: (a, tau)})


def test_adapter_write_removal_zeroes_the_r_component_of_lora_deltas(tiny_model_dir, tmp_path):
    import torch

    from sm.common.models import load_model
    from sm.interp import loading as LD

    adapter = LD.save_random_lora(tiny_model_dir, tmp_path / "adapter", seed=5, scale=0.3)
    model, tok, _ = load_model(str(tiny_model_dir), adapter_path=str(adapter), dtype="fp32", device="cpu", attn="eager")
    H = model.config.hidden_size
    r = np.random.default_rng(2).standard_normal(H)
    r /= np.linalg.norm(r)
    rem = MD.AdapterWriteRemoval(model, r)
    names = [n for n, _ in rem.modules]
    assert names and all(("o_proj" in n or "down_proj" in n) and ".lora_B." in n for n in names)
    assert not any("q_proj" in n or "up_proj" in n for n in names)
    chk = rem.self_check(tol=1e-5)
    assert chk["ok"] and chk["routed_through_hook"], chk
    # full forward of one owning LoRA layer: the delta it adds has no r component with the hooks, and has one without
    name = names[0].rsplit(".lora_B.", 1)[0]
    layer = model.get_submodule(name)
    x = torch.randn(1, 4, layer.base_layer.in_features, generator=torch.Generator().manual_seed(0))
    rt = torch.as_tensor(r, dtype=torch.float32)
    with torch.no_grad():
        d0 = layer(x) - layer.base_layer(x)
        with rem:
            d1 = layer(x) - layer.base_layer(x)
    assert float((d0 @ rt).abs().max()) > 1e-4
    assert float((d1 @ rt).abs().max()) < 1e-5
    # and the hidden states of the whole model change only through the removed component's downstream effects
    h_on = None
    with rem:
        h_on = _hidden(model, tok, "Name a fruit that is yellow.", 1)
    assert rem.calls > 0 and np.all(np.isfinite(h_on))
    base_model, _, _ = load_model(str(tiny_model_dir), dtype="fp32", device="cpu", attn="eager")
    with pytest.raises(ValueError):
        MD.AdapterWriteRemoval(base_model, r)


# --------------------------------------------------------------------------------------------- rules


def _blk(rate, incoh=0.0, n=64):
    k = int(round(rate * n))
    return {"rate": k / n, "k": k, "n": n, "incoherent_rate": incoh, "substring_refusal_rate": k / n}


def test_recovery_and_effect_block():
    assert MD.recovery(0.9, 0.85, 0.9) is None  # loss below 0.10: no recovery
    assert MD.recovery(0.9, 0.3, 0.6) == pytest.approx(0.5)
    e = MD.effect_block(_blk(0.3), _blk(0.6, incoh=0.05), r_base=0.9)
    assert e["rho"] == pytest.approx(0.5, abs=0.02) and e["coherent"] and e["G"] > 0
    assert len(e["G_newcombe95"]) == 2
    assert not MD.effect_block(_blk(0.3), _blk(0.6, incoh=0.2), r_base=0.9)["coherent"]


def _eff(rho, g, coherent=True):
    return {"rho": rho, "G": g, "coherent": coherent}


def test_cap_verdict_rules():
    rand = {f"add_random_c2_k{k}": _eff(0.0, 0.02) for k in range(10)}
    restored = {"add_axis_c0.5": _eff(0.1, 0.05), "add_axis_c1": _eff(0.35, 0.2), "add_axis_c2": _eff(0.6, 0.3), **rand}
    v = MD.cap_verdict(restored, "regressed")
    assert v["verdict"] == "restored by persona repair" and v["qualifying_coef"] == 1.0
    noisy = {**restored, **{f"add_random_c2_k{k}": _eff(0.3, 0.15) for k in range(10)}}
    assert MD.cap_verdict(noisy, "regressed")["verdict"] == "restored by a non-specific perturbation"
    incoh = {"add_axis_c0.5": _eff(0.1, 0.05), "add_axis_c1": _eff(0.4, 0.2, False), "add_axis_c2": _eff(0.6, 0.3, False), **rand}
    assert MD.cap_verdict(incoh, "regressed")["verdict"] == "not restored"
    noloss = {"add_axis_c0.5": _eff(None, 0.0), "add_axis_c1": _eff(None, 0.0), "add_axis_c2": _eff(None, 0.01), **rand}
    assert MD.cap_verdict(noloss, "regressed")["verdict"].startswith("no verdict")
    assert MD.cap_verdict(restored, "control")["verdict"].startswith("no verdict (control")


def test_e7_verdict_rules():
    rand = {f"restore_random_k{k}": _eff(0.0, 0.01) for k in range(10)}
    base = {"ablate_r": _eff(None, -0.1), **rand}
    ok = MD.e7_verdict({**base, "restore_r": _eff(0.5, 0.3)}, "regressed", delta=2.0, gap_base=10.0)
    assert ok["verdict"] == "direction intact but under-driven" and ok["ablate_sanity"]["pass"]
    other = MD.e7_verdict({**base, "restore_r": _eff(0.1, 0.05)}, "regressed", delta=2.0, gap_base=10.0)
    assert other["verdict"] == "regression runs through other circuitry"
    noop = MD.e7_verdict({**base, "restore_r": _eff(0.0, 0.0)}, "regressed", delta=0.3, gap_base=10.0)
    assert noop["verdict"].startswith("no r-drive deficit")
    bad_abl = MD.e7_verdict({**base, "ablate_r": _eff(None, 0.2), "restore_r": _eff(0.5, 0.3)}, "regressed", 2.0, 10.0)
    assert bad_abl["ablate_sanity"]["pass"] is False
    aw = MD.e7_verdict({**base, "restore_r": _eff(0.1, 0.05), "adapter_write_removed": _eff(0.4, 0.2)}, "regressed", 2.0, 10.0)
    assert aw["adapter_write"]["carries_regression"] is True
    assert MD.e7_verdict({**base, "restore_r": _eff(0.5, 0.3)}, "control", 2.0, 10.0)["verdict"].startswith("no verdict (control")


@pytest.mark.skipif(not FLAGS.is_file(), reason="frozen flag file not present")
def test_select_units_reproduces_the_plan_from_the_frozen_flag_file():
    sel = MD.select_units(FLAGS)
    assert sel["selected"] == list(MD.SELECTED) and sel["matches_plan"] and sel["controls_unflagged"]


def test_select_units_on_a_synthetic_flag_file(tmp_path):
    rows = {"c3_trait_evil_r32": (0.0, True), "c5_public_abliterated": (0.0, True), "c4_insecure_code_r128": (0.12, True),
            "c1_dolly_r128": (0.4, True), "c2_humorous_r32": (0.5, True), "c2_terse_r32": (0.56, True),
            "c4_insecure_code_r32": (0.84, False), "c3_trait_sycophancy_r32": (0.83, False)}
    f = tmp_path / "flags.txt"
    f.write_text("# header\n" + "".join(f"# {k}: R_m={v[0]:.4f} n=419 wilson_upper=0.9 flagged={v[1]}\n" for k, v in rows.items()))
    sel = MD.select_units(f)
    assert sel["matches_plan"] and sel["controls_unflagged"]
    rows["c4_insecure_code_r32"] = (0.5, True)
    f.write_text("".join(f"# {k}: R_m={v[0]:.4f} n=419 wilson_upper=0.9 flagged={v[1]}\n" for k, v in rows.items()))
    assert MD.select_units(f)["controls_unflagged"] is False


# --------------------------------------------------------------------------------------------- conditions and an end-to-end unit


def _inputs(H, L=2, pos=-1, sd=0.5, layers=(3, 4)):
    rng = np.random.default_rng(7)
    a = rng.standard_normal(H)
    r = rng.standard_normal(H)
    inp = SimpleNamespace(axis_vec=a / np.linalg.norm(a), r=r / np.linalg.norm(r), sd_base=sd, L=L, pos=pos,
                          caps={l: (rng.standard_normal(H), 0.0) for l in layers},
                          caps_p75={l: (rng.standard_normal(H), -1.0) for l in layers}, card=None,
                          axis_meta={"primary_axis": "axis_resp"}, axis_norm=1.0, resid_norm=10.0, validity={})
    return inp


def test_condition_specs_counts_and_seeds():
    inp = _inputs(32)
    lora = MD.condition_specs("regressed", "lora", inp, {"last": 1.5, "pstar": 0.2}, ["CAP", "E7"])
    names = [s["name"] for s in lora]
    assert names[0] == "none" and len(lora) == 1 + 3 + 10 + 2 + 2 + 10 + 1 + 1
    assert names.count("adapter_write_removed") == 1 and "lu_cap" in names and "lu_cap_p75" in names
    p75 = next(x for x in lora if x["name"] == "lu_cap_p75")
    assert p75["read"] is False and next(x for x in lora if x["name"] == "lu_cap")["caps"] == {"3": 0.0, "4": 0.0}
    model_kind = MD.condition_specs("regressed", "model", inp, {"last": 1.5, "pstar": 0.2}, ["CAP", "E7"])
    assert len(model_kind) == len(lora) - 1
    base = MD.condition_specs("base", "base", inp, None, ["CAP", "E7"])
    assert [s["name"] for s in base] == ["none", "add_axis_c0.5", "add_axis_c1", "add_axis_c2", "lu_cap", "lu_cap_p75"]
    again = MD.condition_specs("regressed", "lora", inp, {"last": 1.5, "pstar": 0.2}, ["CAP", "E7"])
    for s1, s2 in zip(lora, again):
        assert MD.spec_hash(s1, {}) == MD.spec_hash(s2, {})
    rnd = [s for s in lora if s["name"].startswith("add_random")]
    assert all(abs(np.linalg.norm(s["hook"][1]) - 1) < 1e-5 and s["alpha"] == pytest.approx(1.0) for s in rnd)
    rest = next(s for s in lora if s["name"] == "restore_r")
    assert rest["alpha"] == 1.5 and rest["hook"][0] == "add"
    no_delta = MD.condition_specs("regressed", "lora", inp, {"last": 0.0, "pstar": 0.0}, ["E7"])
    assert not any(s["name"].startswith("restore_random") for s in no_delta)


def test_run_unit_end_to_end_with_the_keyword_judge(tiny_model_dir, tmp_path):
    from sm.common.models import load_model
    from sm.interp import loading as LD

    adapter = LD.save_random_lora(tiny_model_dir, tmp_path / "adapter", seed=9)

    def loader(base, run_cfg, dtype, device):
        if run_cfg is None:
            m, t, meta = load_model(str(tiny_model_dir), dtype="fp32", device="cpu", attn="eager")
            return m, t, MD.contextlib.nullcontext(), {"kind": "base"}
        m, t, meta = load_model(str(tiny_model_dir), adapter_path=run_cfg["derivative"]["adapter_path"], dtype="fp32", device="cpu", attn="eager")
        return m, t, MD.contextlib.nullcontext(), {"kind": "lora"}

    probe, tok, _ = load_model(str(tiny_model_dir), dtype="fp32", device="cpu", attn="eager")
    H = probe.config.hidden_size
    inp = _inputs(H, L=2, sd=0.5)
    inp.card = SimpleNamespace(layer_star=2, r=inp.r, position=-1, eoi_ids=None, n_layers=4, persona={})
    P = AX.Paths("tiny", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted" / "exploratory"))
    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    (cfg_dir / "unit_a.yaml").write_text(json.dumps({"derivative": {"kind": "lora", "adapter_path": str(adapter)}}))
    args = SimpleNamespace(base="tiny", drift_configs=str(cfg_dir), dtype="fp32", device="cpu", capture_batch=4, judge="keyword",
                           max_new_tokens=4, batch_size=4)
    prompts = ["Write a haiku about rain.", "List three prime numbers.", "Say hello in French."]
    harmless = ["Name a colour of the sky.", "Give a synonym for happy."]
    base_res = MD.run_unit("base", "base", P, inp, prompts, harmless, ["CAP", "E7"], args, None, loader=loader)
    assert set(base_res["conditions"]) == {"none", "add_axis_c0.5", "add_axis_c1", "add_axis_c2", "lu_cap", "lu_cap_p75"}
    base_info = {"proj": base_res["proj"], "conditions": base_res["conditions"], "gap_base": {"last": 1.0}}
    res = MD.run_unit("unit_a", "regressed", P, inp, prompts, harmless, ["CAP", "E7"], args, base_info, loader=loader)
    conds = res["conditions"]
    assert {"none", "restore_r", "restore_r_pstar", "ablate_r", "adapter_write_removed", "lu_cap"} <= set(conds)
    assert conds["adapter_write_removed"].get("self_check", {}).get("ok") is True
    for name, c in conds.items():
        assert c["rates"]["n"] == 3 and all("completion" not in it for it in c["items"])
        assert c.get("hook_calls", 1) > 0
    eff = MD.unit_effects(res, r_base=base_res["conditions"]["none"]["rates"]["rate"])
    assert "restore_r" in eff and "add_axis_c1" in eff
    # texts only in the restricted tree, 0600 files in 0700 directories
    rfile = tmp_path / "restricted" / "exploratory" / "tiny" / "mediation" / "unit_a" / "none.jsonl"
    assert rfile.is_file() and stat.S_IMODE(os.stat(rfile).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(rfile.parent).st_mode) == 0o700
    row = json.loads(rfile.read_text().splitlines()[0])
    assert "completion" in row and "label" in row
    # a second run reuses every cached condition (no generation)
    calls = []

    def no_gen(*a, **k):
        calls.append(1)
        raise AssertionError("cached conditions must not be regenerated")

    MD.run_unit("unit_a", "regressed", P, inp, prompts, harmless, ["CAP", "E7"], args, base_info, loader=loader, generate_fn=no_gen)
    assert calls == []


@pytest.mark.parametrize("adapter_dtype", ["as_loaded", "bf16"])
def test_adapter_write_self_check_under_a_bf16_base(tiny_model_dir, tmp_path, adapter_dtype):
    """Job 10981135 failed here: under a bf16 base peft keeps the LoRA weights in float32, and the self-check fed the
    LoRA path a bf16 tensor. The check must pass whether the adapter weights are float32 (peft's default) or bf16."""
    import torch

    from sm.common.models import load_model
    from sm.interp import loading as LD

    adapter = LD.save_random_lora(tiny_model_dir, tmp_path / "adapter", seed=11, scale=0.3)
    model, tok, _ = load_model(str(tiny_model_dir), adapter_path=str(adapter), dtype="bf16", device="cpu", attn="eager")
    if adapter_dtype == "bf16":
        for n, mod in model.named_modules():
            if ".lora_A." in n or ".lora_B." in n:
                mod.to(torch.bfloat16)
    r = np.random.default_rng(3).standard_normal(model.config.hidden_size)
    rem = MD.AdapterWriteRemoval(model, r / np.linalg.norm(r))
    chk = rem.self_check()
    assert chk["ok"] and chk["routed_through_hook"], chk
    assert chk["base_dtype"] == "torch.bfloat16"
    if adapter_dtype == "bf16":
        assert chk["adapter_dtype"] == "torch.bfloat16"
    # a short greedy generation with the removal active runs at bf16 and fires the hooks
    from sm.common.models import generate, render_chat

    with rem:
        out = generate(model, tok, [render_chat(tok, [{"role": "user", "content": "Name a fruit."}], thinking=False)], max_new_tokens=3)
    assert rem.calls > 0 and isinstance(out[0], str)


def test_persisted_projections_are_reused_on_resume(tmp_path):
    P = AX.Paths("tiny", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"))
    card = SimpleNamespace(layer_star=2, position=-1, r=np.ones(4) / 2.0)
    calls = []

    def compute():
        calls.append(1)
        return {"last": {"mean": 1.234567890123, "sd": 0.5, "n": 3}, "pstar": {"mean": 0.5, "sd": 0.1, "n": 3}}

    a = MD.persisted_projections(P, "u", "harmful", ["p1", "p2"], card, compute)
    b = MD.persisted_projections(P, "u", "harmful", ["p1", "p2"], card, compute)
    assert a == b and len(calls) == 1 and a["last"]["mean"] == 1.234567890123
    MD.persisted_projections(P, "u", "harmful", ["p1", "p3"], card, compute)  # other prompts: recomputed
    assert len(calls) == 2
