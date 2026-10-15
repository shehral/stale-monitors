"""CPU tests for sm.interp.mechanism (docs/MECHANISM_PLAN.md, exploratory lines M-A to M-C).

Unit tests: the first-token margin, the recovery formula (undefined when the loss is not above the seed noise), the
token-set rule on the Qwen3.5-9B vocabulary (docs/JLENS_READOUT_PLAN.md section 3.3; skipped when the tokenizer is not
cached offline), band windowing, and the summarize rules on a synthetic table. Smoke: every subcommand on the tiny
random model (2 layers, 4 items), the instrument checks, the restricted tree's modes and the absence of text in open
outputs.
"""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from sm.interp import mechanism as M

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ metric, recovery


def test_margin_metric_is_logsumexp_refusal_minus_logsumexp_compliance():
    torch = pytest.importorskip("torch")
    logits = np.array([[1.0, 2.0, 0.0, -5.0, 9.0], [0.0, 0.0, 3.0, 3.0, -9.0]])
    m = M.margin_np(logits, [0, 1], [2, 3])
    want = [math.log(math.e + math.e ** 2) - math.log(1.0 + math.e ** -5), math.log(2.0) - (3.0 + math.log(2.0))]
    np.testing.assert_allclose(m, want, rtol=1e-12)
    np.testing.assert_allclose(M.margin_t(torch.tensor(logits), [0, 1], [2, 3]).double().numpy(), want, rtol=1e-6)
    # a single row, and invariance to tokens outside both sets
    np.testing.assert_allclose(M.margin_np(logits[0], [0, 1], [2, 3]), want[:1])
    shifted = logits.copy()
    shifted[:, 4] += 100.0
    np.testing.assert_allclose(M.margin_np(shifted, [0, 1], [2, 3]), want, rtol=1e-12)


def test_recovery_formula_and_its_undefined_cases():
    assert M.recovery(1.0, 0.0, 2.0) == pytest.approx(0.5)
    assert M.recovery(2.0, 0.0, 2.0, noise=0.1) == pytest.approx(1.0)
    assert M.recovery(-1.0, 0.0, 2.0) == pytest.approx(-0.5)
    assert M.recovery(1.0, 0.0, 0.05, noise=0.1) is None  # loss below the seed noise of m
    assert M.recovery(1.0, 0.0, 0.1, noise=0.1) == pytest.approx(10.0)  # loss equal to the noise: defined
    assert M.recovery(1.0, 1.0, 0.5) is None  # the unit refuses more than the base (negative loss)
    assert M.recovery(1.0, 1.0, 1.0) is None  # zero loss (base against itself)
    assert M.recovery(float("nan"), 0.0, 1.0) is None
    assert M.recovery(None, 0.0, 1.0) is None
    assert M.induced(0.5, 0.0, 2.0) == pytest.approx(0.75)
    assert M.induced(0.5, 0.0, 0.01, noise=1.0) is None


def test_band_windows_and_layer_parsing():
    assert M.band_windows([0, 1, 2, 3, 4], 4) == [[0, 1, 2, 3], [1, 2, 3, 4]]
    assert M.band_windows([0, 1, 3, 4, 5, 6], 4) == [[3, 4, 5, 6]]  # windows never span a gap in the selection
    assert M.band_windows([2, 5], 1) == [[2], [5]]
    assert M.band_windows([0, 1, 2], 4) == []
    with pytest.raises(ValueError):
        M.band_windows([1, 2], 0)
    assert M.parse_layers("all", 3) == [0, 1, 2, 3]
    assert M.parse_layers("1-2,0", 3) == [0, 1, 2]
    assert M.parse_layers(" 14-16 , 20", 32) == [14, 15, 16, 20]
    with pytest.raises(ValueError):
        M.parse_layers("5", 3)
    assert M.parse_positions("last,gen1,last") == ["last", "gen1"]
    with pytest.raises(ValueError):
        M.parse_positions("first")
    assert M.position_offset("pstar", -9) == -9 and M.position_offset("last", -9) == -1 and M.position_offset("gen1", -9) == -1


# ------------------------------------------------------------------------------------------------ token sets


class _FakeTok:
    def __init__(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def batch_decode(self, ids):
        return [self.vocab[i[0]] for i in ids]


def test_token_set_rule_on_a_small_vocab_and_the_registered_check():
    tok = _FakeTok(["I", " I", "Sure", " Here", "here", "cannot ", "x", "As"])
    sets = M.margin_sets(tok, check=False)
    assert sets["refusal"] == [0, 1, 5, 7] and sets["compliance"] == [2, 3]  # exact, case-sensitive, stripped
    assert sets["card_refusal_token_ids"] == [] and not sets["registered_check"]
    with pytest.raises(ValueError, match="registered"):
        M.margin_sets(tok, check=True)


def _qwen35_tokenizer():
    """The Qwen3.5-9B tokenizer from the local HF cache only (the hub id, else any cached snapshot directory)."""
    from transformers import AutoTokenizer

    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    cands = ["Qwen/Qwen3.5-9B"] + [str(p.parent) for p in sorted(hub.glob("models--Qwen--Qwen3.5-9B/snapshots/*/tokenizer.json"))]
    errors = []
    for c in cands:
        try:
            return AutoTokenizer.from_pretrained(c, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
    pytest.skip(f"Qwen3.5-9B tokenizer not available offline ({', '.join(errors)})")


def test_token_set_rule_reproduces_the_registered_hashes_on_the_qwen35_vocab():
    tok = _qwen35_tokenizer()
    from sm.interp import jlens as JL

    doc = (ROOT / "docs" / "JLENS_READOUT_PLAN.md").read_text()
    sets = M.margin_sets(tok, check=True)
    for name, key in (("refusal_vocab", "refusal_vocab_sha256"), ("compliance_onset", "compliance_onset_sha256")):
        assert sets[key] == JL.TOKEN_SET_SHA256[name]
        assert f"`{sets[key]}`" in doc  # the hash printed in section 3.3
    assert sets["compliance"] == list(JL.TOKEN_SET_IDS["compliance_onset"])
    # the card's refusal_token_ids (40 "I", 2053 "As") are inside refusal_vocab, so the refusal side is that set
    assert sets["refusal"] == list(JL.TOKEN_SET_IDS["refusal_vocab"]) and {40, 2053} <= set(sets["refusal"])


# ------------------------------------------------------------------------------------------------ summarize rules


def _sweep(unit, role, single, band, m_base=2.0, m_unit=0.0, L=13, w=4, n=20, noise=0.0, fail=None):
    """A patch-sweep JSON with mean margins set from target recoveries (unlisted cells recover 0)."""
    loss = m_base - m_unit

    def cell(layers, rec):
        return {"layers": layers, "m_mean": m_unit + rec * loss}
    cells = {"single": [cell([l], single.get(l, 0.0)) for l in range(n + 1)],
             "band": [cell(list(range(s, s + w)), band.get(s, 0.0)) for s in range(n + 2 - w)]}
    ic = {"IC1": {"status": "skipped"}, "IC2": {"status": "pass"}, "IC4": {"status": "pass"}}
    if fail:
        ic[fail] = {"status": "fail"}
    return {"unit": unit, "role": role, "layer_star": L, "window": w, "positions": {"last": {"offset": -1, "family": "prompt"}},
            "noise": {"harmful": {"prompt": {"mean_abs": noise}}},
            "baseline": {"harmful": {"prompt": {"m_base_mean": m_base, "m_unit_mean": m_unit, "loss": loss}}},
            "cells": {"base_to_unit": {"harmful": {"last": cells}}}, "instrument_checks": ic}


def _write_table(out: Path, sweeps: dict, heads=None, deltas=None, steers=None):
    for sub, items in (("patch_sweep", sweeps), ("heads", heads or {}), ("delta", deltas or {}), ("steer", steers or {})):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for u, js in items.items():
            (out / sub / f"{u}.json").write_text(json.dumps(js))


def test_largest_overlap_group_and_ma_verdict():
    grp, span = M.largest_overlap_group({"a": 15, "b": 16, "c": 18, "d": 25, "e": None}, 4)
    assert grp == ["a", "b", "c"] and span == [15, 21]
    assert M.ma_verdict({"a": 15, "b": 16, "c": 17, "d": None}, 4)["verdict"] == "supported"
    assert M.ma_verdict({"a": 15, "b": 22, "c": 30, "d": None}, 4)["verdict"] == "mixed"
    assert M.ma_verdict({"a": 15, "b": 16, "c": None, "d": None}, 4)["verdict"] == "mixed"
    assert M.ma_verdict({"a": 15, "b": None, "c": None, "d": None}, 4)["verdict"] == "not supported"
    assert M.ma_verdict({}, 4)["verdict"] == "not supported"


def test_summarize_rules_on_a_synthetic_table(tmp_path):
    out = tmp_path / "mechanism"
    sweeps = {
        "c1_dolly_r128": _sweep("c1_dolly_r128", "benign_regressed", {13: 0.1, 20: 1.0}, {15: 0.6, 16: 0.8}),
        "c2_humorous_r32": _sweep("c2_humorous_r32", "benign_regressed", {13: 0.2}, {16: 0.7}),
        "c2_sycophantic_r32": _sweep("c2_sycophantic_r32", "benign_regressed", {13: 0.0}, {17: 0.55}),
        "c1_dolly_r8": _sweep("c1_dolly_r8", "benign_regressed", {13: 0.0}, {16: 0.4}),
        # writer control: recovers at L* already (its loss is at L* by construction), so it does not localize
        "c5_public_abliterated": _sweep("c5_public_abliterated", "control_writer", {13: 0.6}, {14: 0.9, 15: 0.9}),
        "c6_base_vs_base": _sweep("c6_base_vs_base", "control_null", {}, {}, m_base=2.0, m_unit=2.0),
        # a band before L* never localizes; an early-only unit
        "c3_trait_evil_r32": _sweep("c3_trait_evil_r32", "harmful_regressed", {13: 0.0}, {5: 0.9}),
    }
    heads = {"c1_dolly_r128": {"top_k": [{"block": 14, "head": 0}, {"block": 14, "head": 1}],
                               "recovery": {"band_residual": 0.8, "joint_topk": 0.5, "share_joint_over_band": 0.625}}}
    deltas = {"c1_dolly_r128": {"totals": {"mass": 10.0, "params": 100.0}, "head_geometry": {"n_heads": 4, "n_kv": 2},
                                "per_head": {"14": {"q_proj": {"0": {"mass": 3.0, "params": 5}},
                                                    "o_proj": {"0": {"mass": 1.0, "params": 5}},
                                                    "k_proj": {"0": {"mass": 0.5, "params": 2}},
                                                    "v_proj": {"0": {"mass": 0.5, "params": 2}}}}}}
    steers = {"c1_dolly_r128": {"judged": {"recovery": 0.7}, "ifeval": dict(M.IFEVAL_TODO)},
              "c2_humorous_r32": {"judged": {"recovery": 0.2}, "ifeval": dict(M.IFEVAL_TODO)},
              "c1_dolly_r8": {"judged": None}}
    _write_table(out, sweeps, heads, deltas, steers)
    s = M.summarize(out)
    pu = s["per_unit"]
    assert [pu[u]["M-A"]["last"]["start"] for u in ("c1_dolly_r128", "c2_humorous_r32", "c2_sycophantic_r32", "c1_dolly_r8")] == [15, 16, 17, None]
    assert pu["c1_dolly_r128"]["M-A"]["last"]["first_single_layer_ge_half"] == 20
    assert pu["c3_trait_evil_r32"]["M-A"]["last"]["localizes"] is False
    ma = s["M-A"]["last"]
    assert ma["verdict"] == "supported" and ma["group_size"] == 3 and ma["band"] == [15, 20]
    assert ma["control_writer"] == "pass" and ma["control_null"] == "pass"
    assert s["noise"]["prompt"]["noise"] == 0.0
    # M-B: share 0.625 >= 0.5 and enrichment (5 / 10) / (14 / 100) = 3.57 >= 2 (kv head 0 counted once)
    assert pu["c1_dolly_r128"]["M-B"]["verdict"] == "supported"
    assert pu["c1_dolly_r128"]["M-B"]["enrichment"] == pytest.approx((5 / 10) / (14 / 100))
    assert pu["c2_humorous_r32"]["M-B"]["verdict"] == "not run"
    # M-C: judged recovery 0.7 meets the clause, IFEval pending; 0.2 does not; no judged block is not run
    assert pu["c1_dolly_r128"]["M-C"]["verdict"].startswith("judged clause met")
    assert pu["c2_humorous_r32"]["M-C"]["verdict"] == "not supported"
    assert pu["c1_dolly_r8"]["M-C"]["verdict"].startswith("not run")
    assert s["counts"]["M-B"] == {"n_run": 1, "n_supported": 1}
    assert s["counts"]["M-C"]["n_not_supported"] == 1 and s["counts"]["M-C"]["n_judged_met"] == 1

    # an instrument-check failure gates the unit: two units left in one band -> mixed
    sweeps["c2_sycophantic_r32"] = _sweep("c2_sycophantic_r32", "benign_regressed", {13: 0.0}, {17: 0.55}, fail="IC2")
    _write_table(out, sweeps)
    s2 = M.summarize(out)
    assert s2["M-A"]["last"]["verdict"] == "mixed" and s2["counts"]["ic_failed_units"] == ["c2_sycophantic_r32"]
    assert "not read: instrument check IC2 failed" in s2["per_unit"]["c2_sycophantic_r32"]["M-A"]["last"]["reason"]

    # the writer control localizing inside the band fails its check
    sweeps["c2_sycophantic_r32"] = _sweep("c2_sycophantic_r32", "benign_regressed", {13: 0.0}, {17: 0.55})
    sweeps["c5_public_abliterated"] = _sweep("c5_public_abliterated", "control_writer", {13: 0.0}, {16: 0.9})
    _write_table(out, sweeps)
    assert M.summarize(out)["M-A"]["last"]["control_writer"].startswith("fail")

    # a base-vs-base pair whose |loss| exceeds every unit's loss makes every recovery undefined -> not supported
    sweeps["c6_base_vs_base"] = _sweep("c6_base_vs_base", "control_null", {}, {}, m_base=5.0, m_unit=0.0)
    _write_table(out, sweeps)
    s4 = M.summarize(out)
    assert s4["noise"]["prompt"]["base_vs_base"] == 5.0 and s4["M-A"]["last"]["verdict"] == "not supported"
    assert s4["per_unit"]["c1_dolly_r128"]["M-A"]["last"]["defined"] is False


# ------------------------------------------------------------------------------------------------ smoke


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mech_smoke")
    assert M.main(["summarize", "--smoke", "--out", str(d)]) == 0  # the whole pipeline
    return d


@pytest.mark.parametrize("cmd", M.SUBCOMMANDS)
def test_every_subcommand_runs_on_the_tiny_model(smoke_dir, cmd):
    assert M.main([cmd, "--smoke", "--out", str(smoke_dir), "--force"]) == 0
    sub = {"offset-steer": "offset"}.get(cmd, cmd.replace("-", "_"))
    path = smoke_dir / "open" / ("summary.json" if cmd == "summarize" else f"{sub}/smoke_unit.json")
    body = json.loads(path.read_text())
    prov = body["provenance"]
    assert prov["config_sha256"] and prov["card_sha256"] and "git_commit" in prov and "file_sha256" in prov
    if cmd not in ("gain", "summarize"):
        assert prov["adapter_sha256"]


def test_smoke_instrument_checks_and_patch_exactness(smoke_dir):
    s = json.loads((smoke_dir / "open" / "patch_sweep" / "smoke_unit.json").read_text())
    ic = s["instrument_checks"]
    assert ic["IC0_capture_agreement"]["status"] == "pass" and ic["IC2"]["status"] == "pass"
    assert ic["IC4"]["status"] == "pass" and ic["IC1"]["status"] == "skipped"
    n = s["n_layers"]
    assert n == 2 and set(s["positions"]) == {"last", "gen1"}
    for p, f in (("last", "prompt"), ("gen1", "gen1")):
        bl = s["baseline"]["harmful"][f]
        single = {c["layers"][0]: c for c in s["cells"]["base_to_unit"]["harmful"][p]["single"]}
        # patching the last block's output at the readout column reproduces the base exactly; the embeddings are shared
        assert single[n]["m_mean"] == pytest.approx(bl["m_base_mean"], abs=1e-5)
        assert single[0]["m_mean"] == pytest.approx(bl["m_unit_mean"], abs=1e-5)
        back = {c["layers"][0]: c for c in s["cells"]["unit_to_base"]["harmful"][p]["single"]}
        assert back[n]["m_mean"] == pytest.approx(bl["m_unit_mean"], abs=1e-5)
        assert len(s["cells"]["base_to_unit"]["harmful"][p]["band"]) == 2  # windows [0, 1] and [1, 2] at w = 2
        assert len(single[n]["m_items"]) == 4
    assert "m_items" not in s["cells"]["base_to_unit"]["harmless"]["last"]["single"][0]
    z = np.load(smoke_dir / "open" / "acts" / "smoke_unit.npz")
    assert z["harmful@last"].dtype == np.float16 and z["harmful@last"].shape == (4, n + 1, 32)
    gain = json.loads((smoke_dir / "open" / "gain" / "smoke_unit.json").read_text())
    assert set(gain["table"]) == {"harmful@last", "harmful@gen1", "harmless@last", "harmless@gen1"}
    delta = json.loads((smoke_dir / "open" / "delta" / "smoke_unit.json").read_text())
    full_blocks = list(delta["per_head"])  # per-head masses exist for the full-attention blocks only
    assert delta["head_geometry"]["n_heads"] == 4 and full_blocks
    assert set(delta["per_head"][full_blocks[0]]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert sum(delta["share_by_type"].values()) == pytest.approx(1.0)
    steer = json.loads((smoke_dir / "open" / "steer" / "smoke_unit.json").read_text())
    assert steer["doses"][0]["hook_calls"] > 0 and steer["ifeval"]["status"] == "TODO"
    assert set(steer["judged"]["rates"]) == {"base", "unit", "steered"}


def _open(smoke_dir, sub):
    return json.loads((smoke_dir / "open" / sub / "smoke_unit.json").read_text())


def _hybrid_available() -> bool:
    try:
        from transformers import Qwen3_5ForCausalLM  # noqa: F401
    except ImportError:
        return False
    return True


def test_component_patch_maps_mixers_and_passes_its_checks(smoke_dir):
    cp = _open(smoke_dir, "component_patch")
    kinds = [e["mixer_type"] for e in cp["component_map"]]
    if _hybrid_available():  # the smoke model is the hybrid Qwen3.5: block 0 gated DeltaNet, block 1 full attention
        assert kinds == ["linear_attn", "full_attn"] and cp["component_map"][0]["mixer"] == "linear_attn"
        assert cp["layer_types"]["consistent_with_modules"] is True
        assert cp["layer_types"]["config_layer_types"] == ["linear_attention", "full_attention"]
    assert cp["instrument_checks"]["IC2"]["status"] == "pass" and cp["instrument_checks"]["IC4"]["status"] == "pass"
    for d in ("base_to_unit", "unit_to_base"):
        for p in ("last", "gen1"):
            cells = cp["cells"][d]["harmful"][p]
            assert [(c["block"], c["component"]) for c in cells["single"]] == [(0, "mixer"), (0, "mlp"), (1, "mixer"), (1, "mlp")]
            assert all(c["layers"] == [c["block"] + 1] for c in cells["single"])
            assert [(c["blocks"], c["component"]) for c in cells["band"]] == [([0, 1], "mixer"), ([0, 1], "mlp")]
    assert "recovery" in cp["cells"]["base_to_unit"]["harmful"]["last"]["single"][0]
    assert "induced" in cp["cells"]["unit_to_base"]["harmful"]["last"]["single"][0]


def test_revert_and_svd_remove_checks(smoke_dir):
    rv = _open(smoke_dir, "revert")
    ic = rv["instrument_checks"]
    assert ic["RV0"]["status"] == "pass" and ic["RV1"]["status"] == "pass"
    assert ic["mass_keys"] == {"live_without_mass": [], "mass_without_live": []}
    conds = {c["set"]: c for c in rv["conditions"]}
    types = [conds[f"type={t}"] for t in ("self_attn", "linear_attn", "mlp") if not conds[f"type={t}"].get("skipped")]
    assert sum(c["mass_fraction"] for c in types) == pytest.approx(1.0)
    assert conds["all"]["mass_fraction"] == pytest.approx(1.0) and "keep_only_set" not in conds["all"]
    assert conds["all"]["revert_set"]["m_mean"] == pytest.approx(rv["m_base_mean"], abs=1e-6)
    mlp = conds["type=mlp"]
    assert mlp["keep_only_set"]["mass_fraction_kept"] == pytest.approx(1.0 - mlp["mass_fraction"])
    sv = _open(smoke_dir, "svd_remove")
    assert {k: v["status"] for k, v in sv["instrument_checks"].items()} == {"SV0": "pass", "SVALL": "pass", "RV1": "pass"}
    for mod in sv["modules"]:
        s = mod["singular_values"]
        assert s == sorted(s, reverse=True) and mod["numerical_rank"] <= mod["rank"]
    rows = {str(x["k"]): x for x in sv["conditions"]}
    assert rows["0"]["m_mean"] == pytest.approx(sv["m_unit_mean"], abs=1e-5)
    assert rows["all"]["m_mean"] == pytest.approx(rv["m_base_mean"], abs=1e-5)  # every module, full revert


def test_selectors_and_default_revert_sets():
    keys = [(0, "linear_attn.out_proj"), (0, "mlp.down_proj"), (3, "self_attn.q_proj"), (3, "mlp.up_proj"), (7, "mlp.down_proj")]
    assert M.select_modules(keys, M.parse_selector("all")) == sorted(keys)
    assert M.select_modules(keys, M.parse_selector("type=mlp")) == [(0, "mlp.down_proj"), (3, "mlp.up_proj"), (7, "mlp.down_proj")]
    assert M.select_modules(keys, M.parse_selector("type=attention+linear_attn")) == [(0, "linear_attn.out_proj"), (3, "self_attn.q_proj")]
    assert M.select_modules(keys, M.parse_selector("type=mlp,blocks=3-7")) == [(3, "mlp.up_proj"), (7, "mlp.down_proj")]
    assert M.select_modules(keys, M.parse_selector(r"module=down_proj$,blocks=0+7")) == [(0, "mlp.down_proj"), (7, "mlp.down_proj")]
    with pytest.raises(ValueError):
        M.parse_selector("type=convolution")
    with pytest.raises(ValueError):
        M.parse_selector("layer=3")
    assert M.default_revert_sets(8, 4) == ["type=self_attn", "type=linear_attn", "type=mlp", "blocks=0-3", "blocks=4-7", "all"]
    assert M.default_revert_sets(5, 4)[-2:] == ["blocks=4-4", "all"]


def test_addback_joins_the_full_residual_patch(smoke_dir):
    ab = _open(smoke_dir, "addback_sweep")
    ps = _open(smoke_dir, "patch_sweep")
    full = {c["layers"][0]: c for c in ps["cells"]["base_to_unit"]["harmful"]["last"]["single"]}
    assert ab["table"] and all(r["layer"] > ab["layer_star"] for r in ab["table"])
    for r in ab["table"]:
        assert r["full"]["m_mean"] == pytest.approx(full[r["layer"]]["m_mean"])
        assert [x["gain"] for x in r["rank1"]] == [0.5, 1.0]
        assert r["delta_proj"] == pytest.approx(r["proj_base"] - r["proj_unit"])
    assert set(ab["judged"]["rates"]) == {"base", "unit", "steered"}
    assert (smoke_dir / "restricted" / "smoke_unit" / "addback").is_dir()


def test_gain_harmfulness_control_and_its_in_sample_fallback(smoke_dir, tmp_path):
    hc = _open(smoke_dir, "gain")["harmfulness_control"]
    assert set(hc["table"]) == {"r", "r_last"} and "1%-FPR" in hc["threshold_rule"]
    row = hc["table"]["r_last"][0]
    assert {"auroc", "recall", "fpr", "accuracy", "balanced_accuracy"} <= set(row["base"]) and "threshold" in row
    # no card: the last-token direction is fitted on the base's cached sets at (L*, last)
    rng = np.random.default_rng(0)
    shift = np.zeros(8)
    shift[0] = 3.0
    zb = {"harmful@last": rng.standard_normal((20, 2, 8)) + shift, "harmless@last": rng.standard_normal((20, 2, 8))}
    zu = {"harmful@last": rng.standard_normal((20, 2, 8)) + shift / 3, "harmless@last": rng.standard_normal((20, 2, 8))}
    np.savez(tmp_path / "b.npz", **zb)
    np.savez(tmp_path / "u.npz", **zu)
    out = M.harmfulness_control(M.SimpleNamespace(card=None), np.load(tmp_path / "b.npz"), np.load(tmp_path / "u.npz"), [0, 1],
                                [0, 1], [0, 1], np.eye(8)[1], 1)
    assert out["directions"]["r_last"].startswith("fitted in-sample")
    base, unit = out["table"]["r_last"][1]["base"], out["table"]["r_last"][1]["unit"]
    assert base["auroc"] > 0.9 and unit["recall"] < base["recall"] + 1e-12 and base["fpr"] <= 0.05
    assert base["accuracy"] == pytest.approx((base["recall"] * 20 + (1 - base["fpr"]) * 20) / 40)


def test_component_and_revert_summaries():
    comp = {"cells": {"base_to_unit": {"harmful": {"last": {
        "single": [{"component": "mixer", "recovery": 0.7}, {"component": "mlp", "recovery": 0.1}],
        "band": [{"blocks": [3, 4], "component": "mixer", "recovery": 0.4}, {"blocks": [5, 6], "component": "mixer", "recovery": 0.6},
                 {"blocks": [5, 6], "component": "mlp", "recovery": None}]}}}}, "instrument_checks": {"IC2": {"status": "pass"}}}
    s = M.component_summary(comp)
    assert s["last:mixer"] == {"earliest_band_ge_half": [5, 6], "n_single_blocks_ge_half": 1}
    assert s["last:mlp"] == {"earliest_band_ge_half": None, "n_single_blocks_ge_half": 0}
    assert M.component_summary({**comp, "instrument_checks": {"IC4": {"status": "fail"}}}) == {"not_read": "instrument check failed"}
    rv = {"conditions": [{"set": "type=mlp", "revert_set": {"recovery": 0.8}, "keep_only_set": {"loss_share_kept": 0.9}},
                         {"set": "type=self_attn", "revert_set": {"recovery": 0.0}, "keep_only_set": {"loss_share_kept": 0.2}},
                         {"set": "type=linear_attn", "skipped": "no LoRA module matches"}],
          "instrument_checks": {"RV0": {"status": "pass"}}}
    assert M.revert_summary(rv) == {"necessary_ge_half": ["type=mlp"], "sufficient_ge_half": ["type=mlp"],
                                    "instrument_checks": {"RV0": "pass"}}


def test_add_positions_on_the_first_token_margin(tiny_model):
    """The margin is read at the last prompt token of one prefill pass. `last` and `response` both edit exactly that
    position there, so they move the first-token margin identically; `all` also edits every earlier prompt position.
    The two differ on decoding passes only: `response` adds at each generated token, `last` leaves them alone."""
    torch = pytest.importorskip("torch")
    from sm.common.models import hidden_size, pad_token_id

    model, tok, _ = tiny_model
    tsets = M.margin_sets(tok, check=False, smoke=True)
    ids = M.encode_prompts(tok, ["How do I bake bread", "Describe the sea", "What is the capital of France"])
    eng = M.Engine(model, None, ids, 2, tsets, pad_token_id(tok))
    H = hidden_size(model)
    d = np.random.default_rng(3).standard_normal(H)
    d /= np.linalg.norm(d)
    m0 = eng.run()[0]
    m = {}
    for mode in M.ADD_POSITIONS:
        with M.add_hook(model, d, 8.0, 2, mode) as hk:
            m[mode] = eng.run()[0]
        assert hk.calls > 0, mode
    assert np.abs(m["last"] - m0).max() > 1e-4  # an add at the last prompt token moves the first-token margin
    np.testing.assert_allclose(m["response"], m["last"], rtol=0, atol=1e-6)  # the same edit in a prefill-only pass
    assert np.abs(m["all"] - m["last"]).max() > 1e-6
    # decoding pass (one new token): response edits it, last does not; prefill pass: both edit only the last column
    one, three = torch.randn(2, 1, H), torch.randn(2, 3, H)
    last, resp = M.add_hook(model, d, 8.0, 2, "last"), M.add_hook(model, d, 8.0, 2, "response")
    assert torch.equal(last._edit(one), one) and not torch.equal(resp._edit(one), one)
    for hk in (last, resp):
        out = hk._edit(three)
        assert torch.equal(out[:, :2], three[:, :2]) and not torch.equal(out[:, 2], three[:, 2])
    with pytest.raises(ValueError):
        M.add_hook(model, d, 1.0, 2, "prompt")


def test_smoke_records_the_add_positions(smoke_dir):
    steer, ab = _open(smoke_dir, "steer"), _open(smoke_dir, "addback_sweep")
    assert steer["add_positions"] == "last" and steer["provenance"]["config"]["add_positions"] == "last"
    assert ab["add_positions"] == "all" and ab["provenance"]["config"]["add_positions"] == "all"
    assert all(x["hook_calls"] > 0 for x in steer["doses"])
    steered = steer["judged"]["conditions"]["smoke_unit"]["steer_L1"]
    assert steered["spec"]["add_positions"] == "last" and "generated_now" in steered
    assert steered["add_hook_calls"] > 0 or not steered["generated_now"]  # a cached rerun generates nothing


@pytest.mark.parametrize("rank", [8, 32, 128])
@pytest.mark.parametrize("out_f,in_f", [(32, 4096), (4096, 4096), (8, 32)])
def test_lowrank_spaces_when_a_dimension_is_below_the_lora_rank(rank, out_f, in_f):
    """The 9B's linear_attn.in_proj_b / in_proj_a have out = 32 < r = 128 (the c1_dolly_r128 crash); the bases and the
    singular values must agree in size, min(out, in, r), and reproduce B A."""
    rng = np.random.default_rng(rank + out_f)
    B, A = rng.standard_normal((out_f, rank)), rng.standard_normal((rank, in_f))
    col, row, s = M.lowrank_spaces(B, A)
    k = min(out_f, in_f, rank)
    assert col.shape == (out_f, k) and row.shape == (in_f, k) and s.shape == (k,)
    np.testing.assert_allclose(col.T @ col, np.eye(k), atol=1e-8)
    np.testing.assert_allclose(row.T @ row, np.eye(k), atol=1e-8)
    assert np.all(np.diff(s) <= 0)
    X = rng.standard_normal((in_f, 3))
    want = B @ (A @ X)
    np.testing.assert_allclose(col @ (s[:, None] * (row.T @ X)), want, atol=1e-8 * np.abs(want).max())


@pytest.mark.parametrize("rank", [8, 128])
def test_svd_remove_and_delta_on_rank_8_and_rank_128_adapters(tmp_path, rank):
    """A rank above some module dimensions (128 here, as c1_dolly_r128; 8 as c1_dolly_r8) on the tiny model."""
    from sm.interp import loading as LD

    assets = tmp_path / "_smoke_assets"
    tiny = M.build_tiny_hybrid(assets / "tiny_hybrid", LD.build_tiny_local_model(assets / "tiny_model", n_layers=2, hidden=32,
                                                                                  seed=0)) or assets / "tiny_model"
    M.save_smoke_lora(tiny, assets / "adapter", rank=rank)  # smoke_setup keeps an existing adapter
    assert M.main(["svd-remove", "--smoke", "--out", str(tmp_path)]) == 0  # runs delta and revert first
    sv = json.loads((tmp_path / "open" / "svd_remove" / "smoke_unit.json").read_text())
    assert {k: v["status"] for k, v in sv["instrument_checks"].items()} == {"SV0": "pass", "SVALL": "pass", "RV1": "pass"}
    assert all(m["rank"] == rank and 0 < m["numerical_rank"] <= min(rank, 64) for m in sv["modules"])
    assert len({m["numerical_rank"] for m in sv["modules"]}) > 1 or rank == 8  # modules of different ranks, decomposed jointly
    rv = json.loads((tmp_path / "open" / "revert" / "smoke_unit.json").read_text())
    rows = {str(x["k"]): x for x in sv["conditions"]}
    assert rows["all"]["m_mean"] == pytest.approx(rv["m_base_mean"], abs=1e-5)
    delta = json.loads((tmp_path / "open" / "delta" / "smoke_unit.json").read_text())
    assert all(0 < m["cos2_random_baseline"] * delta["hidden"] <= min(rank, 64) for m in delta["modules"])


def test_patch_sweep_judged_band_confirmation(smoke_dir):
    js = _open(smoke_dir, "patch_sweep")["judged"]
    c = js["conditions"]["patch_band_L1_L2"]
    assert js["band_layers"] == [1, 2] and js["judge"] == "keyword_v1" and js["judge_is_fallback"] is True
    assert {"rate", "k", "n", "wilson95", "recovery"} <= set(c) and c["n"] == 4
    assert {"rate", "wilson95"} <= set(js["base"]) and {"rate", "wilson95"} <= set(js["unit"]) and js["greedy"] is True
    assert js["IC3"]["status"] == "skipped" and js["first_token"]["position"] == "last"
    det = js["details"]["unit"]["patch_band_L1_L2"]
    assert det["spec"]["band_layers"] == [1, 2] and (det["hook_calls"] > 0 or not det["generated_now"])
    for who in ("base", "unit"):
        p = smoke_dir / "restricted" / "patch_judged" / "smoke_unit" / who
        assert p.is_dir() and stat.S_IMODE(os.stat(p).st_mode) == 0o700


def test_revert_judged_sets_and_complements(smoke_dir):
    rv = _open(smoke_dir, "revert")
    names = set(rv["judged"]["conditions"])
    assert any(x.startswith("revert_set_") for x in names) and any(x.startswith("keep_only_set_") for x in names)
    for row in rv["conditions"]:
        if row.get("skipped"):
            continue
        assert "wilson95" in row["judged"]["revert_set"] and "recovery" in row["judged"]["revert_set"]
        assert ("keep_only_set" in row["judged"]) == (row["n_complement"] > 0)
    p = smoke_dir / "restricted" / "revert_judged" / "smoke_unit"
    assert (p / "base").is_dir() and (p / "unit").is_dir() and stat.S_IMODE(os.stat(p).st_mode) == 0o700


def test_patched_generation_is_greedy_and_a_full_band_reproduces_the_base(smoke_dir):
    from sm.common.models import generate, num_layers
    from sm.interp.causal import _default_render

    assets = smoke_dir / "_smoke_assets"
    tiny = assets / "tiny_hybrid" if (assets / "tiny_hybrid").is_dir() else assets / "tiny_model"
    base = M._smoke_loader(str(tiny), None, "fp32", "cpu")
    unit = M._smoke_loader(str(tiny), {"derivative": {"kind": "lora", "adapter_path": str(assets / "adapter")}}, "fp32", "cpu")
    rendered = [_default_render(base[1], p) for p in M.SMOKE_HARMFUL]
    ref_b = generate(base[0], base[1], rendered, max_new_tokens=6, batch_size=2)
    ref_u = generate(unit[0], unit[1], rendered, max_new_tokens=6, batch_size=2)
    empty = M.PatchedGenerate(base[0], None, [])
    assert empty(unit[0], unit[1], rendered, max_new_tokens=6, batch_size=2) == ref_u and empty.calls == 0
    full = M.PatchedGenerate(base[0], None, list(range(1, num_layers(base[0]) + 1)))
    assert full(unit[0], unit[1], rendered, max_new_tokens=6, batch_size=2) == ref_b and full.calls > 0


def test_rescore_joins_saved_completions_by_their_recorded_hashes(smoke_dir):
    from sm.interp import assistant_axis as AX

    a, b = "I cannot help with that.", "Sure, here is"
    items = [{"i": 1, "completion_sha256": AX.sha256_text(b)}, {"i": 0, "completion_sha256": AX.sha256_text(a)}]
    assert M.join_by_hash(items, [{"i": 0, "completion": a}, {"i": 1, "completion": b}]) == ([a, b], [])
    texts, bad = M.join_by_hash(items, [{"i": 0, "completion": a}, {"i": 1, "completion": b + "!"}])
    assert texts is None and bad == [{"i": 1, "recorded": AX.sha256_text(b), "found": AX.sha256_text(b + "!")}]
    assert M.join_by_hash(items, [{"i": 0, "completion": a}])[1] == [{"i": 1, "recorded": AX.sha256_text(b), "found": None}]
    # the smoke rescored every judged source with the keyword reader, the same reader that labelled them: same rates
    for sub in ("revert", "patch_sweep", "steer", "addback_sweep"):
        src = _open(smoke_dir, sub)["judged"]
        rs = json.loads((smoke_dir / "open" / sub / "wildguard" / "smoke_unit.json").read_text())
        assert rs["join"]["status"] == "pass" and rs["join"]["n_unique_scored"] > 0 and rs["reader"]["spec"] == "keyword"
        assert rs["base"]["wg"]["rate"] == rs["base"]["j1"]["rate"] and rs["unit"]["wg"]["rate"] == rs["unit"]["j1"]["rate"]
        assert rs["conditions"] and all(c["wg"]["rate"] == c["j1"]["rate"] and c["wg"]["recovery"] == c["j1"]["recovery"]
                                        for c in rs["conditions"].values())
        details = src.get("details") or {("base" if k == "base" else "unit"): v for k, v in src["conditions"].items()}
        for who, conds in rs["items"].items():
            for cond, rows in conds.items():
                want = {x["i"]: x["completion_sha256"] for x in details[who][cond]["items"]}
                assert {r["i"]: r["completion_sha256"] for r in rows} == want
                assert all(r["wg_label"] == r["j1_label"] for r in rows)
    idx = _open(smoke_dir, "rescore_judged")
    assert all(idx["written"][s] for s in ("revert", "patch-sweep", "steer", "addback-sweep"))


def test_smoke_texts_stay_restricted(smoke_dir):
    rdir = smoke_dir / "restricted"
    files = [p for p in rdir.rglob("*") if p.is_file()]
    assert files, "the judged steer path wrote no restricted files"
    for p in [rdir, *[q for q in rdir.rglob("*") if q.is_dir()]]:
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o700, p
    for p in files:
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, p
    for p in (smoke_dir / "open").rglob("*.json"):
        text = p.read_text()
        assert not any(x in text for x in M.SMOKE_HARMFUL + M.SMOKE_HARMLESS), p
        assert '"completion"' not in text, p


# ------------------------------------------------------------------------------------------------ offset-steer (R2)


def _fake_acts(path, harmful, harmless=None, layers=(0, 1, 2)):
    arrays = {"harmful@last": np.asarray(harmful, np.float16)}
    if harmless is not None:
        arrays["harmless@last"] = np.asarray(harmless, np.float16)
    np.savez(path, layers=np.asarray(layers, np.int64), **arrays)
    return path


def test_offset_vectors_are_the_mean_paired_difference_with_a_matched_norm_random_control(tmp_path):
    rng = np.random.default_rng(0)
    hb = rng.standard_normal((6, 3, 8))
    shift = np.zeros(8)
    shift[0] = 2.0
    hu = hb + shift + 0.01 * rng.standard_normal((6, 3, 8))
    hh = hb - np.eye(8)[1] * 3.0
    pu = _fake_acts(tmp_path / "u.npz", hu)
    pb = _fake_acts(tmp_path / "b.npz", hb, hh)
    vecs, meta = M.offset_vectors(pu, pb, [1, 2], seed=5)
    hu16, hb16, hh16 = (np.asarray(x, np.float16).astype(np.float64) for x in (hu, hb, hh))
    for L in (1, 2):
        d = (hu16[:, L] - hb16[:, L]).mean(0)
        assert np.allclose(vecs[L]["offset"], d) and np.allclose(vecs[L]["harm"], hb16[:, L].mean(0) - hh16[:, L].mean(0))
        assert np.linalg.norm(vecs[L]["random"]) == pytest.approx(np.linalg.norm(d))
        v = meta["vectors"][f"L{L}"]
        assert v["offset_norm"] == pytest.approx(np.linalg.norm(d)) and v["random_norm"] == pytest.approx(v["offset_norm"])
        assert abs(v["cos_offset_harm"]) < 0.1 and v["per_item_cos_with_offset"]["frac_positive"] == 1.0
    # seeded per (seed, layer): reproducible, and different across layers and seeds
    again, _ = M.offset_vectors(pu, pb, [1, 2], seed=5)
    other, _ = M.offset_vectors(pu, pb, [1], seed=6)
    assert np.array_equal(again[1]["random"], vecs[1]["random"]) and not np.allclose(vecs[1]["random"], vecs[2]["random"])
    assert not np.allclose(other[1]["random"], vecs[1]["random"])
    assert meta["n_items"] == 6 and meta["hidden"] == 8 and len(meta["unit"]["sha256"]) == 64
    with pytest.raises(ValueError):
        M.offset_vectors(pu, pb, [3], seed=5)  # not in the saved layers
    with pytest.raises(ValueError):
        M.offset_vectors(pu, _fake_acts(tmp_path / "b2.npz", hb), [1], seed=5)  # the base lacks harmless@last


def _row(mode, direction, layer, alpha, rate, rb=0.86, ru=0.47):
    specs = {"x": {"mode": mode, "direction": direction, "layer": layer, "alpha": alpha}}
    return M.offset_table(specs, {"x": {"rate": rate, "k": round(rate * 64), "n": 64, "wilson95": [0, 1]}}, rb, ru)["x"]


def _table(rows):
    return {M.offset_cond_name(r["mode"], r["direction"], r["layer"], r["alpha"]): r for r in rows}


def test_offset_table_and_the_r2_reading():
    add = _row("add_to_base", "offset", 24, 1.0, 0.56)
    assert add["drop_vs_base"] == pytest.approx(0.30) and add["induced_share_of_unit_loss"] == pytest.approx(0.30 / 0.39)
    sub = _row("subtract_from_unit", "offset", 24, -1.0, 0.70)
    assert sub["recovery"] == pytest.approx(0.23 / 0.39) and sub["model"] == "unit"
    assert _row("subtract_from_unit", "offset", 24, -1.0, 0.70, rb=0.86, ru=0.80)["recovery"] is None  # loss < 0.10
    ok = _table([add, _row("add_to_base", "random", 24, 1.0, 0.83), sub, _row("subtract_from_unit", "random", 24, -1.0, 0.48)])
    pr = M.offset_predictions(ok, 0.86, 0.47)
    assert pr["add"]["status"] == "met" and pr["subtract"]["status"] == "met" and pr["verdict"] == "supported"
    assert pr["falsification"]["status"] == "not falsified" and pr["add"]["layer"] == 24
    # the random control matching the offset falsifies; so does a negligible move everywhere
    rnd = _table([add, _row("add_to_base", "random", 24, 1.0, 0.70), sub])
    assert M.offset_predictions(rnd, 0.86, 0.47)["verdict"] == "falsified"
    flat = _table([_row("add_to_base", "offset", L, a, 0.82) for L in (20, 24, 28) for a in (0.5, 1.0, 1.5)] + [sub])
    fl = M.offset_predictions(flat, 0.86, 0.47)
    assert fl["falsification"]["all_moves_small"] is True and fl["verdict"] == "falsified"
    weak = _table([add, _row("add_to_base", "random", 24, 1.0, 0.83), _row("subtract_from_unit", "offset", 24, -1.0, 0.50)])
    assert M.offset_predictions(weak, 0.86, 0.47)["verdict"] == "partial"
    assert M.offset_predictions(_table([add]), 0.86, 0.47)["verdict"] == "incomplete"


def test_offset_add_hook_shifts_block_output_at_the_last_prompt_token_only(smoke_dir):
    torch = pytest.importorskip("torch")
    from sm.common.models import left_pad

    assets = smoke_dir / "_smoke_assets"
    tiny = assets / "tiny_hybrid" if (assets / "tiny_hybrid").is_dir() else assets / "tiny_model"
    model, tok, _ctx, _ = M._smoke_loader(str(tiny), None, "fp32", "cpu")
    seqs = M.encode_prompts(tok, M.SMOKE_HARMFUL[:2])
    ids, mask, pos = left_pad(seqs, 0, torch.device("cpu"))
    d = np.random.default_rng(3).standard_normal(32)

    def hs(hook=None):
        with torch.inference_mode(), (hook if hook is not None else __import__("contextlib").nullcontext()):
            return model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states[1]
    h0 = hs()
    hook = M.add_hook(model, d, 0.5, 1, M.OFFSET_ADD_POSITIONS)
    h1 = hs(hook)
    assert hook.calls == 1
    assert torch.allclose(h1[:, -1] - h0[:, -1], torch.as_tensor(0.5 * d, dtype=h0.dtype), atol=1e-5)
    assert torch.equal(h1[:, :-1], h0[:, :-1])  # earlier prompt positions untouched


def test_offset_steer_smoke_output(smoke_dir):
    js = _open_offset(smoke_dir)
    assert js["layers"] == [1, 2] and js["alphas"] == [0.5, 1.0] and js["OS0"]["status"] == "pass"
    assert set(js["modes"]) == set(M.OFFSET_MODES) and set(js["directions"]) == set(M.OFFSET_DIRECTIONS)
    for L in ("L1", "L2"):
        v = js["vectors"][L]
        assert v["random_norm"] == pytest.approx(v["offset_norm"]) and v["cos_offset_harm"] is not None
    jd = js["judged"]
    conds = jd["conditions"]
    want = {M.offset_cond_name("add_to_base", dn, L, a) for L in (1, 2) for a in (0.5, 1.0) for dn in M.OFFSET_DIRECTIONS}
    want |= {M.offset_cond_name("subtract_from_unit", dn, L, 1.0) for L in (1, 2) for dn in M.OFFSET_DIRECTIONS}
    assert set(conds) == want
    for name, c in conds.items():
        assert {"rate", "k", "n", "wilson95", "shift_vs_own_unhooked"} <= set(c) and c["n"] == 4
        assert (c["alpha"] < 0) == (c["mode"] == "subtract_from_unit")
        det = jd["details"][c["model"]][name]
        assert det["hook_calls"] > 0 or not det["generated_now"]
        assert det["spec"]["add_positions"] == "response" and det["spec"]["vec_norm"] > 0
    assert set(jd["details"]["base"]) >= {"base_none"} and set(jd["details"]["unit"]) >= {"unit_none"}
    assert jd["IC3"]["status"] == "skipped" and jd["IC3_base"]["status"] == "skipped"
    assert js["predictions"]["verdict"] in ("supported", "partial", "falsified", "incomplete")
    p = smoke_dir / "restricted" / "offset_judged" / "smoke_unit"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o700
    assert {f.name for f in p.glob("*.jsonl")} >= {"base_none.jsonl", "unit_none.jsonl", "add_offset_L1_a1.jsonl",
                                                   "sub_random_L2_a1.jsonl"}
    assert js["provenance"]["config"]["acts"]["unit"]["sha256"] and js["provenance"]["adapter_sha256"]


def _open_offset(smoke_dir):
    return json.loads((smoke_dir / "open" / "offset" / "smoke_unit.json").read_text())


def test_offset_rescore_joins_by_hash_and_matches_the_same_reader(smoke_dir):
    rs = json.loads((smoke_dir / "open" / "offset" / "wildguard" / "smoke_unit.json").read_text())
    src = _open_offset(smoke_dir)
    assert rs["join"]["status"] == "pass" and rs["source"] == "offset-steer"
    assert rs["join"]["n_conditions_joined"] == 2 + len(src["judged"]["conditions"])
    assert rs["base"]["wg"]["rate"] == rs["base"]["j1"]["rate"] and rs["unit"]["wg"]["rate"] == rs["unit"]["j1"]["rate"]
    for name, c in rs["conditions"].items():
        assert c["wg"]["rate"] == c["j1"]["rate"] and c["wg"].get("recovery") == c["j1"].get("recovery")
    assert rs["predictions"]["wg"]["verdict"] == rs["predictions"]["j1"]["verdict"]
    for who, conds in rs["items"].items():
        for cond, rows in conds.items():
            want = {x["i"]: x["completion_sha256"] for x in src["judged"]["details"][who][cond]["items"]}
            assert {r["i"]: r["completion_sha256"] for r in rows} == want
    assert _open(smoke_dir, "rescore_judged")["written"]["offset-steer"]
