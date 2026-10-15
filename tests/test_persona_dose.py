"""Tests for sm.interp.persona_dose (FIDELITY_REVIEW R2): sign-and-dose statistics on a fake backend whose trait
expression follows the steering coefficient, and the CLI end to end on the tiny model (smoke base card) with a
stubbed J1 trait judge. CPU only.

Run: .venv/bin/python -m pytest -q tests/test_persona_dose.py
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if importlib.util.find_spec("sm") is None and str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sm.eval.judge import TraitScore  # noqa: E402
from sm.interp import base_card as BC  # noqa: E402
from sm.interp import persona_dose as PD  # noqa: E402


class DoseBackend:
    """Sampling yields 'TRAIT<k>' where k follows the coefficient of the active add hook (0 unsteered)."""

    meta = {"revision": "r"}

    def __init__(self, sign: float = 1.0):
        self.sign = sign
        self.alpha = 0.0
        self.calls: list[tuple[float, str, int]] = []

    def render(self, user, system=None):
        return f"<user>{user}"

    @contextlib.contextmanager
    def _ctx(self, alpha):
        self.alpha = alpha
        try:
            yield
        finally:
            self.alpha = 0.0

    def add(self, vector, layer, alpha, positions="all"):
        self.calls.append((float(alpha), positions, int(layer)))
        return self._ctx(float(alpha))

    def sample(self, prompts, max_new_tokens, temperature=1.0, top_p=1.0, seed=0):
        return [f"TRAIT{self.sign * self.alpha:+g}" for _ in prompts]

    def generate(self, prompts, max_new_tokens):
        return self.sample(prompts, max_new_tokens)


def _graded_by_level(trait, questions, responses):
    out = []
    for i, r in enumerate(responses):
        k = float(r.replace("TRAIT", ""))
        if k <= -4 and i % 5 == 0:
            out.append(TraitScore(float("nan"), "incoherent", float("nan")))
        else:
            g = max(0.0, min(100.0, 40.0 + 10.0 * k + (i % 3)))
            out.append(TraitScore(100.0 if g >= 50 else 0.0, "trait_expressed" if g >= 50 else "trait_absent", g))
    return out


def test_dose_summary_monotone_pass_and_incoherent_rate():
    be = DoseBackend()
    res = PD.run_trait_dose(be, "evil", np.ones(4, np.float32), 2, [f"q{i}?" for i in range(10)], _graded_by_level,
                            PD.DEFAULT_COEFS, n_samples=5, temperature=1.0, top_p=1.0, max_new_tokens=128,
                            positions="response", seed=3)
    assert set(res["by_coef"]) == {"-4", "-2", "-1", "0", "1", "2", "4"}
    assert all(v["n"] == 50 for v in res["by_coef"].values())
    assert res["unsteered"]["mean_score"] == pytest.approx(41.0, abs=0.1)
    assert res["by_coef"]["-4"]["incoherent_rate"] == pytest.approx(0.2) and res["by_coef"]["-4"]["n_scored"] == 40
    assert res["by_coef"]["2"]["incoherent_rate"] == 0.0
    assert res["spearman_item_level"]["rho"] > 0.9 and res["spearman_item_level"]["n"] == 340
    assert res["spearman_condition_means"]["rho"] == pytest.approx(1.0)
    assert res["all_negative_below_unsteered"] and res["all_positive_above_unsteered"] and res["sign_dose_pass"]
    # steering at the response positions at the card's layer, one hook per non-zero coefficient
    assert sorted(a for a, _, _ in be.calls) == sorted(PD.DEFAULT_COEFS)
    assert {p for _, p, _ in be.calls} == {"response"} and {l for _, _, l in be.calls} == {2}
    assert res["sampling"]["n_items_per_condition"] == 50 and res["sampling"]["temperature"] == 1.0


def test_dose_summary_flags_a_reversed_sign():
    res = PD.run_trait_dose(DoseBackend(sign=-1.0), "evil", np.ones(4, np.float32), 1, ["a?", "b?"], _graded_by_level,
                            [-2.0, 2.0], n_samples=3, temperature=1.0, top_p=1.0, max_new_tokens=8,
                            positions="response", seed=0)
    assert res["spearman_item_level"]["rho"] < 0
    assert res["negative_below_unsteered"] == {"-2": False} and not res["sign_dose_pass"]


def test_parse_coefs_and_unsteered_required():
    assert PD.parse_coefs("-4,-2,-1,1,2,4") == [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]
    with pytest.raises(ValueError):
        PD.parse_coefs("0,1")
    with pytest.raises(ValueError):
        PD.dose_summary({1.0: [10.0]})
    # plain floats (no J1 label): incoherent rate is not reported rather than reported as 0
    assert PD.condition_stats(0.0, [10.0, 20.0])["incoherent_rate"] is None


def test_cli_end_to_end_on_tiny_smoke_card(tmp_path, monkeypatch):
    for mod in ("sm.common.models", "sm.common.hooks"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not available")
    for k in ("SM_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(k, "1")
    monkeypatch.delenv("SM_TINY_MODEL", raising=False)
    cards = tmp_path / "cards"
    assert BC.main(["--smoke", "--out", str(cards), "--seed", "0"]) == 0
    card = json.loads((cards / "tiny.json").read_text())
    assert card["persona"]["sycophancy"]["status"] == "ok"

    seen: list[int] = []

    def stub_j1(trait, questions, responses):  # stands in for sm.eval.judge.score_trait
        seen.append(len(responses))
        return [TraitScore(float("nan"), "incoherent", float("nan")) if i % 4 == 0
                else TraitScore(0.0, "trait_absent", float(len(r) % 50)) for i, r in enumerate(responses)]

    monkeypatch.setattr(PD, "resolve_callable", lambda spec: stub_j1)
    out = tmp_path / "tiny_persona_dose.json"
    argv = ["--smoke", "--base", "tiny", "--card", str(cards / "tiny.json"), "--out", str(out),
            "--coefs=-2,2", "--n-samples", "2", "--max-new-tokens", "4"]
    assert PD.main(argv) == 0
    doc = json.loads(out.read_text())
    tr = doc["traits"]["sycophancy"]
    assert tr["status"] == "ok" and tr["layer"] == card["persona"]["sycophancy"]["layer"]
    nq = tr["questions"]["n_used"]
    assert seen == [2 * nq] * 3 and set(tr["by_coef"]) == {"-2", "0", "2"}
    for v in tr["by_coef"].values():
        assert v["n"] == 2 * nq and 0.0 <= v["incoherent_rate"] <= 1.0
    assert tr["by_coef"]["0"]["incoherent_rate"] == pytest.approx(sum(i % 4 == 0 for i in range(2 * nq)) / (2 * nq))
    assert "rho" in tr["spearman_item_level"] and set(tr["negative_below_unsteered"]) == {"-2"}
    assert tr["raw_norm"] == pytest.approx(BC.D.load_direction(cards / "tiny" / "persona_sycophancy")[1]["raw_norm"], rel=1e-5)
    assert doc["coefs"] == [0.0, -2.0, 2.0] and doc["smoke"] is True and doc["settings"]["positions"] == "response"
    # judge identity recorded like the card's (a smoke card: the fake persona judge)
    assert doc["judge"] == doc["judge_identity"] and doc["judge"]["primary_use"] == "persona_dose"
    assert doc["judge"]["uses"]["persona_dose"]["callable"] == "fake:trait"
    assert "meta" in doc and doc["meta"]["smoke"] is True
    # idempotent; unknown trait and a smoke card without --smoke are refused
    assert PD.main(argv) == 0
    assert PD.main([*argv[:7], "--out", str(tmp_path / "x.json"), "--traits", "evil"]) == 2
    assert PD.main(["--base", "tiny", "--card", str(cards / "tiny.json"), "--out", str(tmp_path / "y.json")]) == 2


# --------------------------------------------------------------------------------------------- pre-registered readout
# docs/PERSONA_FIDELITY_RERUN.md section 7: coherence window > 80, addition and subtraction rules, question clusters.


def _items(spec: dict[float, tuple[float, float]], n_q: int = 6, n_s: int = 5, jitter: float = 3.0):
    """Rows {coef, q, trait, coherence}: per coefficient (mean trait, mean coherence) with a small per-question spread."""
    rows = []
    for c, (t, h) in spec.items():
        for q in range(n_q):
            for s in range(n_s):
                rows.append({"coef": c, "q": q, "trait": max(0.0, min(100.0, t + jitter * ((q + s) % 3 - 1))),
                             "coherence": h})
    return rows


def test_readout_addition_replicates_and_subtraction_lowers():
    rows = _items({0.0: (20, 95), 0.5: (25, 95), 1.0: (35, 92), 2.0: (60, 85), 2.5: (75, 70), -1.0: (12, 95), -2.0: (5, 90)})
    r = PD.dose_readout(rows, n_boot=300, seed=1)
    assert r["by_coef"]["2.5"]["in_coherence_window"] is False and r["by_coef"]["2"]["in_coherence_window"] is True
    assert r["by_coef"]["0"]["in_coherence_window"] is None and r["in_window_positive"] == ["0.5", "1", "2"]
    add = r["addition"]
    assert add["verdict"] == "replicates" and add["largest_in_window"] == "2" and add["monotone_point_estimates"]
    assert add["slope"]["ci95"][0] > 0 and add["diff_largest_minus_unsteered"]["estimate"] == pytest.approx(40.0, abs=1.5)
    sub = r["subtraction"]
    assert sub["verdict"] == "lowers_expression" and sub["most_negative_in_window"] == "-2"
    assert r["unsteered_mean_trait"] == pytest.approx(20.0, abs=1.0) and r["n_questions"] == 6


def test_readout_floor_window_and_missing_coherence():
    floor = PD.dose_readout(_items({0.0: (4, 95), 1.0: (30, 90), -1.0: (2, 95)}), n_boot=200)
    assert floor["subtraction"]["verdict"] == "at_floor" and floor["addition"]["verdict"] == "replicates"
    broken = PD.dose_readout(_items({0.0: (20, 95), 1.0: (60, 60), -1.0: (10, 40)}), n_boot=200)
    assert broken["addition"]["verdict"] == "not_evaluable" and broken["subtraction"]["verdict"] == "not_evaluable"
    flat = PD.dose_readout(_items({0.0: (20, 95), 1.0: (20, 95), -1.0: (20, 95)}), n_boot=200)
    assert flat["addition"]["verdict"] == "not_replicated" and flat["subtraction"]["verdict"] == "no_detectable_change"
    rows = [{**r, "coherence": None} for r in _items({0.0: (20, 95), 1.0: (60, 95)})]
    none = PD.dose_readout(rows, n_boot=50)
    assert none["addition"]["verdict"] == "not_evaluable" and "no coherence" in none["addition"]["reason"]
    half = PD.dose_readout(_items({0.0: (20, 95), 1.0: (60, 95)}), n_boot=50, question_ids=[0, 1, 2])
    assert half["n_questions"] == 3 and half["window_source"] == "these questions"
    fixed = PD.dose_readout(_items({0.0: (20, 95), 1.0: (60, 95)}), n_boot=50, question_ids=[0, 1, 2],
                            window_flags={"1": False})
    assert fixed["by_coef"]["1"]["in_coherence_window"] is False and fixed["addition"]["verdict"] == "not_evaluable"
    with pytest.raises(ValueError):
        PD.dose_readout([{"coef": 1.0, "q": 0, "trait": 1.0, "coherence": 90.0}])


def _chen_by_level(trait, questions, responses):
    from sm.interp.persona_grader import ChenScore

    out = []
    for r in responses:
        k = float(r.replace("TRAIT", ""))
        out.append(ChenScore(max(0.0, min(100.0, 30.0 + 10.0 * k)), 95.0 - 5.0 * abs(k)))
    return out


def test_run_trait_dose_records_items_hashes_and_halves():
    be = DoseBackend()
    res = PD.run_trait_dose(be, "evil", np.ones(4, np.float32), 2, ["a?", "b?", "c?", "d?"], _chen_by_level,
                            [-1.0, 1.0, 2.0], n_samples=3, temperature=1.0, top_p=1.0, max_new_tokens=8,
                            positions="response", seed=0, question_ids=[3, 7, 11, 15], halves=["select", "score", "score", "select"])
    items = res["items"]
    assert len(items) == 4 * 3 * 4 and {i["q"] for i in items} == {3, 7, 11, 15}
    assert all(set(i) >= {"coef", "q", "half", "sample", "trait", "coherence", "completion_sha256"} for i in items)
    assert not any("completion" == k or "text" == k for i in items for k in i)
    one = next(i for i in items if i["coef"] == 1.0)
    assert one["completion_sha256"] == hashlib.sha256(b"TRAIT+1").hexdigest() and one["coherence"] == 90.0
    assert res["by_coef"]["1"]["mean_coherence"] == 90.0 and res["by_coef"]["1"]["in_coherence_window"] is True
    assert res["by_coef"]["2"]["in_coherence_window"] is True and res["by_coef"]["0"]["in_coherence_window"] is None
    assert res["readout"]["addition"]["verdict"] == "replicates"
    assert set(res["readout_by_half"]) == {"select", "score"} and res["readout_by_half"]["score"]["n_questions"] == 2
    assert res["readout_by_half"]["score"]["window_source"] == "given (all questions)"
    assert "legacy_rule_note" in res


def test_cli_eval_questions_with_the_fake_chen_grader(tmp_path, monkeypatch):
    for mod in ("sm.common.models", "sm.common.hooks"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not available")
    for k in ("SM_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(k, "1")
    monkeypatch.delenv("SM_TINY_MODEL", raising=False)
    cards = tmp_path / "cards"
    assert BC.main(["--smoke", "--out", str(cards), "--seed", "0"]) == 0
    out = tmp_path / "tiny_persona_dose_chen.json"
    argv = ["--smoke", "--base", "tiny", "--card", str(cards / "tiny.json"), "--out", str(out), "--judge", "fake:chen",
            "--questions", "eval", "--n-questions", "20", "--coefs=-1,1", "--n-samples", "2", "--max-new-tokens", "4"]
    assert PD.main(argv) == 0
    doc = json.loads(out.read_text())
    tr = doc["traits"]["sycophancy"]
    assert tr["questions"]["mode"] == "eval" and tr["questions"]["n_used"] == tr["questions"]["n_questions"]
    assert len(tr["items"]) == 3 * 2 * tr["questions"]["n_used"] and "readout" in tr
    assert doc["grader"]["grader"] == "fake:chen" and doc["settings"]["questions"] == "eval"
    assert doc["settings"]["readout"]["window_coherence_gt"] == 80.0 and "Chen" in doc["settings"]["score"]
    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)

    assert not {"completion", "completions", "response", "responses", "text"} & set(keys(doc))  # numbers and hashes only


def test_failed_grader_self_test_exits_2_before_any_generation(tmp_path, monkeypatch):
    card = {"base": "x", "persona": {"evil": {"status": "ok", "layer": 1}}, "config": {"persona": {}}, "seed": 0,
            "meta": {"smoke": False}}
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card))
    from sm.interp import persona_grader as PG

    def scorer(trait, questions, responses):
        raise AssertionError("must not grade after a failed self-test")

    def failing_preflight():
        raise PG.GraderSelfTestError("persona grader self-test failed: test")

    scorer.preflight = failing_preflight  # type: ignore[attr-defined]
    monkeypatch.setattr(PD, "resolve_callable", lambda spec: scorer)
    out = tmp_path / "dose.json"
    assert PD.main(["--base", "x", "--card", str(path), "--out", str(out), "--judge", "a.b:c"]) == 2
    assert not out.exists()
