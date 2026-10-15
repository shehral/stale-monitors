"""Tests for the EXPLORATORY E15 module sm.interp.assistant_axis (docs/ASSISTANT_AXIS_E7_PLAN.md).

CPU only: the axis maths on planted data, the plan reproduction against the protocol's hashes, the capping and layer
translations, the displacement, the H3' rule, restricted-storage permissions, and the activation capture and unit
loading on the tiny random model fixture (with a random LoRA)."""

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

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "assistant-axis"
PLAN_ROLES = [
    "aberration", "actor", "addict", "anarchist", "auditor", "biologist", "chef", "chemist", "collaborator", "contrarian",
    "criminal", "crystalline", "cynic", "debugger", "dilettante", "economist", "egregore", "eldritch", "engineer", "flaneur",
    "generalist", "gossip", "graduate", "guardian", "hedonist", "idealist", "immigrant", "interpreter", "librarian", "merchant",
    "narcissist", "networker", "optimist", "orphan", "parasite", "parent", "peacekeeper", "pirate", "planner", "playwright",
    "presenter", "prey", "procrastinator", "prophet", "publisher", "reviewer", "scholar", "shaman", "smuggler", "sociologist",
    "sommelier", "therapist", "traditionalist", "tree", "trickster", "tulpa", "visionary", "warrior", "widow", "witch",
]
PLAN_QIDS = [4, 21, 24, 27, 33, 43, 47, 61, 68, 79, 87, 106, 127, 172, 178, 188, 191, 194, 199, 203]
PLAN_SHA = "f0a226a4e0fab9eadb6cbf7aa2156a1c6b5d7524dd073005aed6cc875a0d8b9b"


# --------------------------------------------------------------------------------------------- pure maths


def test_contrast_axis_is_default_minus_mean_of_role_vectors():
    d = np.array([1.0, 2.0, 3.0])
    roles = [np.array([0.0, 0.0, 1.0]), np.array([2.0, 0.0, 1.0])]
    np.testing.assert_allclose(AX.contrast_axis(d, roles), d - np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        AX.contrast_axis(d, [])
    assert AX.cosine(np.array([1.0, 0, 0]), np.array([2.0, 0, 0])) == pytest.approx(1.0)


def _planted(n_roles=6, n_q=4, n_p=5, H=16, L=3, seed=0, noise=0.05):
    """Rollouts whose response means are +a (default) or -a + role offset + noise (roles), at every layer."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(H)
    a /= np.linalg.norm(a)
    rows, keys = {}, []
    resp, pst = [], []
    scores = {}
    qids = list(range(n_q))
    offsets = {f"r{k}": rng.standard_normal(H) * 0.05 for k in range(n_roles)}
    for role in ["default"] + list(offsets):
        for pi in range(n_p):
            for q in qids:
                key = f"{role}|p{pi}|q{q}"
                kind = "default" if role == "default" else "role"
                rows[key] = {"key": key, "kind": kind, "role": role, "prompt_index": pi, "question_id": q}
                keys.append(key)
                base = a if kind == "default" else -a + offsets[role]
                v = np.stack([base * (l + 1) + noise * rng.standard_normal(H) for l in range(L)])
                resp.append(v)
                pst.append(v * 0.5)
                if kind == "role":
                    # role r0 never fully role-plays (score 2); the others mostly score 3
                    scores[key] = {"score": 2 if role == "r0" else (3 if (pi + q) % 5 else 1)}
    return a, np.stack(resp).astype(np.float32), np.stack(pst).astype(np.float32), keys, rows, scores, qids


def test_compute_axis_bundle_on_planted_directions():
    a, resp, pst, keys, rows, scores, qids = _planted()
    layers = [1, 2, 3]
    halves = (qids[:2], qids[2:])
    card = SimpleNamespace(layer_star=2, r=np.eye(16)[0], persona={})
    arrays, summ = AX.compute_axis_bundle(resp, pst, keys, rows, scores, layers, halves, card, min_score3=10)
    # r0 has no score-3 rollout, so it is dropped; the five others have 16 score-3 rollouts each (of 20)
    assert summ["roles_kept"] == ["r1", "r2", "r3", "r4", "r5"]
    assert summ["roles_score3_counts"]["r1"] == 16 and summ["roles_score3_counts"]["r0"] == 0
    assert summ["parse_rate"] == 1.0
    assert summ["v1_judge_ok"] is False  # 5 roles kept < 20: the unfiltered axis is primary
    assert summ["primary_axis_note"].startswith("axis_resp_unfiltered")
    for l in layers:
        assert AX.cosine(arrays["axis_resp"][l], a) > 0.99
        assert AX.cosine(arrays["axis_resp_unfiltered"][l], a) > 0.99
        assert AX.cosine(arrays["axis_pstar"][l], a) > 0.99
        rec = summ["per_layer"][str(l)]
        assert rec["split_half_cos"] > 0.99
        assert rec["default_proj_mean"] > rec["role_proj_mean"]
        assert abs(rec["pca"]["pc1_cos_axis"]) > 0.0
    assert np.all(arrays["axis_resp"][0] == 0)
    assert "cos_refusal_direction" in summ["per_layer"]["2"]


def test_role_vectors_and_group_rows_filter_scores_and_counts():
    keys = ["d|p0|q0", "a|p0|q0", "a|p1|q0", "b|p0|q0"]
    rows = {k: {"kind": "default" if k.startswith("d") else "role", "role": k.split("|")[0], "question_id": 0} for k in keys}
    valid = np.array([True, True, True, False])
    scores = {"a|p0|q0": 3, "a|p1|q0": 2, "b|p0|q0": 3}
    g3 = AX.group_rows(keys, rows, scores, valid, 3)
    assert g3 == {"a": [1]}  # b's rollout is invalid (empty response)
    gall = AX.group_rows(keys, rows, scores, valid, None)
    assert gall == {"a": [1, 2]}
    acts = np.arange(8, dtype=np.float32).reshape(4, 2)
    rv = AX.role_vectors(acts, gall, 2)
    np.testing.assert_allclose(rv["a"], acts[[1, 2]].mean(0))
    assert AX.role_vectors(acts, gall, 3) == {}


def test_parse_judge_score_matches_the_official_parser():
    assert AX.parse_judge_score("3") == 3
    assert AX.parse_judge_score(" 2\n") == 2
    assert AX.parse_judge_score("Score: 1") == 1
    assert AX.parse_judge_score("I would say 0.") == 0
    assert AX.parse_judge_score("5") is None
    assert AX.parse_judge_score("10") is None
    assert AX.parse_judge_score("") is None
    assert AX.parse_judge_score(None) is None
    assert AX.parse_judge_score("no number") is None


def test_judge_prompt_fills_the_official_template():
    tmpl = "Role: x.\n[QUESTION START]\n{question}\n[QUESTION END]\n[ANSWER START]\n{answer}\n[ANSWER END]"
    out = AX.judge_prompt(tmpl, "Q?", "A {with braces}")
    assert "Q?" in out and "A {with braces}" in out


def test_capping_and_middle_layer_translation():
    assert AX.capping_layers(64) == list(range(47, 55))  # Lu's blocks 46..53 as hidden_states indices
    assert AX.capping_layers(32) == [24, 25, 26, 27]
    assert AX.lu_middle_layer(64) == 33 and AX.lu_middle_layer(32) == 17
    proj = np.arange(100, dtype=float)
    assert AX.cap_threshold(proj) == pytest.approx(np.percentile(proj, 75))


def test_displacement_uses_population_sd():
    b = np.array([0.0, 1.0, 2.0, 3.0])
    m = b + 1.5
    d = AX.displacement(b, m)
    assert d["shift"] == pytest.approx(1.5)
    assert d["base_sd"] == pytest.approx(np.std(b))
    assert d["z"] == pytest.approx(1.5 / np.std(b))
    assert np.isnan(AX.displacement(np.ones(3), np.ones(3) * 2)["z"])


def test_h3prime_verdict_rule():
    assert AX.h3prime_verdict((-0.04, 0.3), (-0.01, 0.4)) == "supported"
    assert AX.h3prime_verdict((-0.3, -0.01), (0.1, 0.5)) == "falsified"
    assert AX.h3prime_verdict((0.1, 0.5), (-0.6, -0.1)) == "falsified"
    assert AX.h3prime_verdict((-0.2, 0.3), (-0.04, 0.2)) == "undecided"
    assert AX.h3prime_verdict((float("nan"), 0.3), (0.0, 0.1)).startswith("undecided")


def test_paired_spearman_diff_is_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(15)
    y = x + rng.standard_normal(15)
    out = AX.paired_spearman_diff(x, x, y, n_boot=200, seed=0)
    assert out["diff"] == pytest.approx(0.0)
    assert out["ci_low"] == pytest.approx(0.0) and out["ci_high"] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------- plan


@pytest.mark.skipif(not (OFFICIAL / "data" / "extraction_questions.jsonl").is_file(), reason="official clone not present")
def test_plan_reproduces_the_protocol():
    off = AX.load_official(OFFICIAL)
    assert len(off["roles"]) == 275 and len(off["questions"]) == 240
    plan = AX.build_plan(off)
    assert plan["roles"] == PLAN_ROLES
    assert plan["question_ids"] == PLAN_QIDS
    assert plan["n_rollouts"] == 6100
    assert plan["plan_sha256"] == PLAN_SHA
    assert plan["default_prompts"] == ["", "You are an AI assistant.", "You are a large language model.", "You are Qwen.", "Respond as yourself."]
    assert AX.split_halves(plan["question_ids"]) == ([4, 27, 47, 68, 79, 172, 188, 191, 194, 199], [21, 24, 33, 43, 61, 87, 106, 127, 178, 203])
    pub = AX.public_plan(plan)
    assert all("system_prompt" not in r and "question" not in r for r in pub["rollouts"])
    first = plan["rollouts"][0]
    assert first["key"] == "default|p0|q4" and AX.messages_for(first["system_prompt"], first["question"])[0]["role"] == "user"


def test_select_is_deterministic_and_sorted():
    names = [f"r{i:03d}" for i in range(50)]
    a = AX.select_roles(names, 7)
    assert a == AX.select_roles(list(reversed(names)), 7) and a == sorted(a) and len(set(a)) == 7
    assert AX.select_questions(range(30), 5) == AX.select_questions(range(30), 5)


# --------------------------------------------------------------------------------------------- storage


def test_restricted_writes_are_0700_and_0600(tmp_path):
    root = tmp_path / "restricted" / "exploratory" / "b"
    f = root / "axis_rollouts" / "rollouts.jsonl"
    AX.append_restricted_jsonl(f, [{"key": "k", "text": "t"}])
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    for d in (f.parent, root, root.parent):
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    g = root / "x" / "y.json"
    AX.write_restricted_text(g, "{}")
    assert stat.S_IMODE(os.stat(g).st_mode) == 0o600
    assert AX.read_jsonl(f) == [{"key": "k", "text": "t"}]


def test_generate_and_judge_are_resumable(tmp_path):
    plan = {"n_rollouts": 3, "plan_sha256": "x", "rollouts": [
        {"key": "default|p0|q1", "kind": "default", "role": "default", "system_prompt": "", "question": "q1?", "seed": 1},
        {"key": "a|p0|q1", "kind": "role", "role": "a", "system_prompt": "Be a.", "question": "q1?", "seed": 2},
        {"key": "a|p1|q1", "kind": "role", "role": "a", "system_prompt": "Act as a.", "question": "q1?", "seed": 3}]}
    calls = []

    def sampler(rows):
        calls.append(len(rows))
        return [(f"answer to {r['key']}", False) for r in rows]

    out = tmp_path / "r" / "rollouts.jsonl"
    s1 = AX.generate_rollouts(plan, out, sampler, chunk=2)
    s2 = AX.generate_rollouts(plan, out, sampler, chunk=2)
    assert s1["n_new"] == 3 and s2["n_new"] == 0 and calls == [2, 1]
    texts = {r["key"]: r["text"] for r in AX.read_jsonl(out)}
    official = {"roles": {"a": {"eval_prompt": "Is it a? {question} / {answer}. Respond 0-3."}}}
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return "3"

    raw = tmp_path / "r" / "raw.jsonl"
    sc = AX.judge_rollouts(plan, official, texts, ask, raw, {}, workers=2)
    assert {k: v["score"] for k, v in sc.items()} == {"a|p0|q1": 3, "a|p1|q1": 3}  # the default is never judged
    sc2 = AX.judge_rollouts(plan, official, texts, ask, raw, sc, workers=2)
    assert len(asked) == 2 and sc2 == sc
    assert all("text" not in v for v in sc.values())


# --------------------------------------------------------------------------------------------- tiny model


def _tiny_card(model, layer_star=2, seed=0):
    from sm.common.models import hidden_size, num_layers

    rng = np.random.default_rng(seed)
    r = rng.standard_normal(hidden_size(model))
    return SimpleNamespace(layer_star=layer_star, r=r / np.linalg.norm(r), position=-1, eoi_ids=None,
                           n_layers=num_layers(model), persona={})


def test_capture_rollout_acts_matches_the_contract_capture(tiny_model, tmp_path):
    from sm.common.hooks import capture_residual
    from sm.common.models import render_chat

    model, tok, _ = tiny_model
    card = _tiny_card(model)
    plan = {"plan_sha256": "p", "rollouts": [
        {"key": "default|p0|q0", "kind": "default", "role": "default", "system_prompt": "", "question": "What is bread?"},
        {"key": "a|p0|q0", "kind": "role", "role": "a", "system_prompt": "You are a pirate.", "question": "What is bread?"},
        {"key": "a|p1|q0", "kind": "role", "role": "a", "system_prompt": "Act as a sailor.", "question": "What is bread?"}]}
    texts = {"default|p0|q0": "Bread is food.", "a|p0|q0": "Arr, bread be food.", "a|p1|q0": "   "}
    layers = [1, 2, 3, 4]
    stats = AX.capture_rollout_acts(model, tok, plan, texts, card, tmp_path / "cache", layers, batch_size=2, chunk=2)
    assert stats["n_valid"] == 2 and stats["n_empty"] == 1
    resp = np.load(tmp_path / "cache" / "resp_mean.npy", mmap_mode="r")
    assert resp.shape == (3, 4, model.config.hidden_size)
    assert np.isnan(resp[2]).all()
    rendered = [render_chat(tok, AX.messages_for(r["system_prompt"], r["question"]), thinking=False) for r in plan["rollouts"][:2]]
    ref = capture_residual(model, tok, rendered, layers=layers, position="resp_mean", completions=[texts["default|p0|q0"], texts["a|p0|q0"]])
    for li, l in enumerate(layers):
        np.testing.assert_allclose(resp[:2, li, :], ref[l], rtol=1e-5, atol=1e-5)
    # resumable: a second call does no work and leaves the arrays unchanged
    before = np.array(resp)
    AX.capture_rollout_acts(model, tok, plan, texts, card, tmp_path / "cache", layers, batch_size=2, chunk=2)
    np.testing.assert_array_equal(np.load(tmp_path / "cache" / "resp_mean.npy"), before)


def test_load_unit_lora_unmerged_and_projection(tiny_model_dir, tmp_path):
    from sm.interp import loading as LD

    adapter = LD.save_random_lora(tiny_model_dir, tmp_path / "adapter", seed=3)
    cfg = {"derivative": {"kind": "lora", "adapter_path": str(adapter)}}
    model, tok, ctx, meta = AX.load_unit(str(tiny_model_dir), cfg, "fp32", "cpu")
    assert meta["kind"] == "lora" and type(model).__name__.startswith("Peft")
    card = _tiny_card(model)
    from sm.interp import drift as DR

    fn, _ = DR.get_capture_fn(-1, None)
    prompts = [DR._render(tok, p) for p in ("How do I bake bread?", "Name a fruit.")]
    acts = {"test_harmful@-1": fn(model, tok, prompts, [1, 2, 3, 4], batch_size=2)}
    H = model.config.hidden_size if hasattr(model, "config") else None
    axes = {"axis_primary": np.vstack([np.zeros((1, H)), np.random.default_rng(1).standard_normal((4, H))])}
    proj = AX.project_sets(acts, axes, [1, 2, 3, 4], card.r, card.layer_star)
    mat = proj["test_harmful@-1"]["axis_primary"]
    assert mat.shape == (2, 5) and np.all(np.isfinite(mat[:, 1:]))
    np.testing.assert_allclose(mat[:, 2], acts["test_harmful@-1"][2] @ AX.unit(axes["axis_primary"][2]), rtol=1e-6)
    np.testing.assert_allclose(proj["test_harmful@-1"]["r_at_L_star"], acts["test_harmful@-1"][2] @ card.r, rtol=1e-6)
    # the base loads without peft
    base, _, _, bmeta = AX.load_unit(str(tiny_model_dir), None, "fp32", "cpu")
    assert bmeta["kind"] == "base" and not type(base).__name__.startswith("Peft")


@pytest.mark.skipif(not (ROOT / "results" / "judge" / "qwen3.5-9b" / "_flags" / "j1_rates_flags.txt").is_file()
                    or not (ROOT / "results" / "drift" / "qwen3.5-9b" / "c1_dolly_r8.json").is_file()
                    or not (ROOT / "results" / "base_cards" / "qwen3.5-9b.json").is_file(), reason="accepted artifacts not present")
def test_h3prime_wiring_on_synthetic_displacements(tmp_path):
    """phase_h3prime on the real J1 rates, flag file, drift JSONs and card, with synthetic displacement files."""
    import argparse

    card = AX.load_card(ROOT / "results" / "base_cards" / "qwen3.5-9b.json")
    P = AX.Paths("qwen3.5-9b", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"))
    P.displace_dir.mkdir(parents=True)
    P.axis_dir.mkdir(parents=True)
    AX.write_open_json(P.axis_json, {"primary_axis": "axis_resp", "validity": {"V2_split_half": {"pass": True}, "V4_overlap_r": {"informative": True}}})
    rng = np.random.default_rng(0)
    units = ["base"] + [l.split(":")[0][2:] for l in (ROOT / "results/judge/qwen3.5-9b/_flags/j1_rates_flags.txt").read_text().splitlines()
                        if l.startswith("# c")]
    pos = int(card.position)
    for u in units:
        arrs = {}
        for set_name, n in (("test_harmful", 128), ("test_harmless", 128), ("construction_harmful", 256), ("construction_harmless", 256)):
            for p in (pos, -1):
                for av in AX.AXIS_VARIANTS:
                    arrs[f"{set_name}@{p}::{av}"] = rng.standard_normal((n, 33)).astype(np.float32) + (0 if u == "base" else rng.normal())
                arrs[f"{set_name}@{p}::r_at_L_star"] = rng.standard_normal(n).astype(np.float32) + (3.0 if "harmful" in set_name and "harmless" not in set_name else 0.0)
        np.savez(P.displace_dir / f"{u}.npz", **arrs)
    args = argparse.Namespace(base="qwen3.5-9b", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"), cache=None,
                              card=str(ROOT / "results/base_cards/qwen3.5-9b.json"), judge_root=str(ROOT / "results/judge/qwen3.5-9b"),
                              flag_file=str(ROOT / "results/judge/qwen3.5-9b/_flags/j1_rates_flags.txt"),
                              drift_dir=str(ROOT / "results/drift/qwen3.5-9b"), run="run1")
    assert AX.phase_h3prime(args) == 0
    out = json.loads(P.h3prime_json.read_text())
    assert out["n"] == 15 and out["n_flagged"] == 11 and out["flag_file_mismatches"] == [] and out["missing_units"] == []
    # the flag file may have been regenerated with more runs since the protocol froze it: those rows are an addendum
    assert out["flag_file_is_frozen"] == (out["flag_file_sha256"] == AX.FROZEN_FLAG_SHA256)
    assert {r["run_id"] for r in out["addendum"]} == set(units[1:]) - set(AX.H3P_UNITS)
    assert out["verdict"] in ("supported", "falsified", "undecided", "undecided (undefined interval)")
    assert {r["run_id"] for r in out["table"]} == set(AX.H3P_UNITS)
    assert all(len(r["z_by_layer"]) == 32 for r in out["table"])


@pytest.mark.skipif(not (OFFICIAL / "data" / "extraction_questions.jsonl").is_file(), reason="official clone not present")
def test_phase_axis_and_mediation_inputs_wiring(tmp_path, monkeypatch):
    """phase_axis on a synthetic activation cache for the real 6,100-rollout plan (hidden 8, 32 layers), then the
    mediation Inputs that read its outputs."""
    import argparse

    from sm.interp import mediation as MD

    H, NL = 8, 32
    rng = np.random.default_rng(0)
    r = rng.standard_normal(H)
    card = SimpleNamespace(layer_star=13, r=r / np.linalg.norm(r), position=-9, eoi_ids=None, n_layers=NL, persona={})
    monkeypatch.setattr(AX, "load_card", lambda path: card)
    off = AX.load_official(OFFICIAL)
    plan = AX.build_plan(off)
    keys = [x["key"] for x in plan["rollouts"]]
    a = rng.standard_normal(H)
    a /= np.linalg.norm(a)
    P = AX.Paths("qwen3.5-9b", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"), cache=str(tmp_path / "cache"))
    P.cache.mkdir(parents=True)
    resp = np.lib.format.open_memmap(P.cache / "resp_mean.npy", mode="w+", dtype=np.float32, shape=(len(keys), NL, H))
    pst = np.lib.format.open_memmap(P.cache / "pstar.npy", mode="w+", dtype=np.float32, shape=(len(keys), NL, H))
    scores = {}
    for i, x in enumerate(plan["rollouts"]):
        sign = 1.0 if x["kind"] == "default" else -1.0
        resp[i] = sign * a[None, :] * np.arange(1, NL + 1)[:, None] + 0.1 * rng.standard_normal((NL, H))
        pst[i] = 0.5 * resp[i]
        if x["kind"] == "role":
            scores[x["key"]] = {"score": 3 if (x["question_id"] + x["prompt_index"]) % 4 else 2}
    resp.flush()
    pst.flush()
    np.save(P.cache / "done.npy", np.ones(len(keys), dtype=bool))
    (P.cache / "acts_meta.json").write_text(json.dumps({"plan_sha256": plan["plan_sha256"], "layers": list(range(1, NL + 1))}))
    AX.write_open_json(P.scores_json, {"scores": scores})
    args = argparse.Namespace(base="qwen3.5-9b", out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"),
                              cache=str(tmp_path / "cache"), card="unused", official_dir=str(OFFICIAL), n_roles=AX.N_ROLES,
                              n_questions=AX.N_QUESTIONS, run="run1")
    assert AX.phase_axis(args) == 0
    meta = json.loads(P.axis_json.read_text())
    assert meta["primary_axis"] == "axis_resp" and meta["validity"]["V1_judge"]["pass"] is True
    assert meta["n_roles_kept"] == 60 and meta["validity"]["V2_split_half"]["pass"] is True
    assert sorted(int(k) for k in meta["caps"]) == [24, 25, 26, 27]
    axes = AX.load_axis(P.axis_npz)
    assert AX.cosine(axes["axis_primary"][13], a) > 0.99
    # mediation Inputs read the axis, the caps and the base displacement spread
    P.displace_dir.mkdir(parents=True)
    base_proj = rng.standard_normal((128, NL + 1)).astype(np.float32)
    np.savez(P.displace_dir / "base.npz", **{"test_harmful@-9::axis_primary": base_proj})
    inp = MD.Inputs(P, card)
    assert inp.sd_base == pytest.approx(float(np.std(base_proj[:, 13].astype(np.float64))), rel=1e-6)
    assert set(inp.caps) == {24, 25, 26, 27} and abs(np.linalg.norm(inp.axis_vec) - 1) < 1e-9
    tau24 = meta["caps"]["24"]["tau"]
    assert tau24 == pytest.approx(meta["per_layer"]["24"]["rollout_proj_p75"], rel=1e-6)
    # plan section 13: lu_cap floors at the default's mean projection, lu_cap_p75 at the 75th percentile
    assert inp.caps[24][1] == pytest.approx(meta["caps"]["24"]["default_proj_mean"], rel=1e-9)
    assert inp.caps_p75[24][1] == pytest.approx(tau24, rel=1e-9)


@pytest.mark.skipif(not (OFFICIAL / "data" / "extraction_questions.jsonl").is_file(), reason="official clone not present")
def test_phase_generate_falls_back_to_hf_when_vllm_fails(tiny_model_dir, tmp_path):
    """No vLLM on the laptop: the vllm backend raises, HF sampling continues and generate.json records the reason."""
    import argparse

    args = argparse.Namespace(base=str(tiny_model_dir), out=str(tmp_path / "open"), restricted=str(tmp_path / "restricted"),
                              cache=None, official_dir=str(OFFICIAL), n_roles=1, n_questions=1, backend="vllm", gpu_util=0.4,
                              chunk=4, max_new_tokens=4, dtype="fp32", device="cpu", batch_size=4, card="unused", run="run1")
    assert AX.phase_generate(args) == 0
    P = AX.Paths(str(tiny_model_dir), args.out, args.restricted)
    rows = AX.read_jsonl(P.rollouts)
    assert len(rows) == 10 and all("text" in r and "sha256" in r for r in rows)  # 5 default + 1 role x 5 prompts, 1 question
    gen = json.loads((P.axis_dir / "generate.json").read_text())
    assert gen["generator"]["backend"] == "hf" and gen["vllm_fallback_reason"]
    assert stat.S_IMODE(os.stat(P.rollouts).st_mode) == 0o600


RUN2_SHA = "ea60bd7ee119d37ded427c74de1d2278f5dedfae6b3cfe1f00c9648211af6691"
RUN2_EXTRA = [9, 39, 44, 69, 81, 89, 95, 105, 111, 112, 122, 131, 135, 136, 153, 155, 174, 195, 204, 221]


@pytest.mark.skipif(not (OFFICIAL / "data" / "extraction_questions.jsonl").is_file(), reason="official clone not present")
def test_run2_plan_reproduces_the_addendum():
    off = AX.load_official(OFFICIAL)
    p1 = AX.build_plan(off)
    p2 = AX.build_plan(off, run="run2")
    assert p1["plan_sha256"] == PLAN_SHA  # run 1 unchanged
    assert p2["plan_sha256"] == RUN2_SHA and p2["n_rollouts"] == 12200
    assert p2["roles"] == PLAN_ROLES and p2["role_prompt_sha256"] == p1["role_prompt_sha256"]
    assert p2["question_ids"] == sorted(PLAN_QIDS + RUN2_EXTRA)
    a, b = AX.split_halves(p2["question_ids"], AX.run_split_seed("run2"))
    assert len(a) == len(b) == 20 and not set(a) & set(b)
    s1 = {x["key"]: x["seed"] for x in p1["rollouts"]}
    shared = [x for x in p2["rollouts"] if x["key"] in s1]
    assert len(shared) == 6100 and all(x["seed"] != s1[x["key"]] for x in shared[:200])  # fresh sampling seeds
    assert p2["run"] == "run2" and "run" not in p1


def test_paths_keep_run1_and_put_run2_in_subdirectories(tmp_path, monkeypatch):
    monkeypatch.delenv("SM_AXIS_CACHE", raising=False)
    r1 = AX.Paths("qwen3.5-9b", out=str(tmp_path / "o"), restricted=str(tmp_path / "r"))
    r2 = AX.Paths("qwen3.5-9b", out=str(tmp_path / "o"), restricted=str(tmp_path / "r"), run="run2")
    assert r1.open == tmp_path / "o" / "qwen3.5-9b" and r2.open == tmp_path / "o" / "qwen3.5-9b" / "run2"
    assert r2.restricted == tmp_path / "r" / "qwen3.5-9b" / "run2" and r2.axis_npz.parent == r2.open / "axis"
    assert r2.cache == r2.open / "_cache"
    assert AX.Paths("b", cache=str(tmp_path / "c"), run="run2").cache == tmp_path / "c" / "run2"
    monkeypatch.setenv("SM_AXIS_CACHE", str(tmp_path / "env"))
    assert AX.Paths("b").cache == tmp_path / "env"
    assert AX.Paths("b", run="run2").cache == tmp_path / "env" / "run2"
    with pytest.raises(ValueError):
        AX.Paths("b", run="run3")
