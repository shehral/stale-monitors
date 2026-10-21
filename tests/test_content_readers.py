"""Tests for sm.exploratory.content_readers (EXPLORATORY R1-C, docs/HYPOTHESIS_LEDGER.md, registered at 69bfdc0).

CPU only, tiny synthetic arrays and a synthetic results tree laid out as the real one; no model, no text."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sm.analysis import stats
from sm.exploratory import content_readers as CR
from sm.exploratory import decision_probe_fit as F

H_SIZES = {"strongreject": 20, "harmbench": 30, "xstest_unsafe": 30}  # 80: the registered E2 draw takes 64
N_XS, N_XS_J2, N_HARMLESS, DIM = 10, 6, 24, 12
LAYERS, N_LAYERS = [2, 3, 4], 4  # deep layers (3 l >= 2 n): 3 and 4
GRID = (0.01, 1.0)
N_BOOT = 60
# unit: (share of the base's H refusals it loses, confirmed regression); "u_evil" refuses no H item (undefined (b))
UNITS9 = {"u_a": (0.6, True), "u_b": (0.4, True), "u_c": (0.3, True), "u_evil": (1.0, True), "u_null": (0.0, False),
          "u_null2": (0.05, False)}
UNITS27 = {"u_a": (0.6, True), "u_b": (0.4, True), "u_c": (0.3, True), "u_evil": (1.0, True)}


def _rows():
    rows = [(b, f"{b}-{i}") for b, n in H_SIZES.items() for i in range(n)]
    rows += [("xstest_safe", f"xstest-{i}") for i in range(N_XS)]
    rows += [("test_harmless", f"test_harmless-{i:05d}") for i in range(N_HARMLESS)]
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _labels(rng, rows, units):
    """The base refuses the H items except the first six and over-refuses two XSTest-safe items; each unit then loses
    its share of the base's H refusals."""
    base = np.array([1 if b in H_SIZES else 0 for b, _ in rows])
    h_idx = [i for i, (b, _) in enumerate(rows) if b in H_SIZES]
    base[h_idx[:6]] = 0
    base[[i for i, (b, _) in enumerate(rows) if b == "xstest_safe"][:2]] = 1
    out = {"base": base}
    for u, (loss, _) in units.items():
        lab = base.copy()
        ref = [i for i in h_idx if base[i] == 1]
        lab[rng.choice(ref, size=int(round(loss * len(ref))), replace=False)] = 0
        out[u] = lab
    return out


def make_tree(root: Path, base: str, units: dict, seed: int, with_e2: bool) -> tuple[Path, dict]:
    """A synthetic results tree: a content (harm) direction g on every H item and a decision (refusal) direction v
    where the model refuses; a unit that stops refusing loses v there and keeps g. Writes the captures (+ the capture
    sidecar with n_layers), the text-free item-label table, XSTest-safe labels for both readers (one J1 parse error on
    the base), the bank table and (9B-like) E2's file, then runs R1's fit so probe_results.json exists."""
    rng = np.random.default_rng(seed)
    rows = _rows()
    g, v = rng.standard_normal(DIM), rng.standard_normal(DIM)
    g, v = g / np.linalg.norm(g), v / np.linalg.norm(v)
    labels = _labels(rng, rows, units)
    is_h = np.array([b in H_SIZES for b, _ in rows], float)
    acts_dir = root / "exploratory" / base / "decision_probe" / "acts"
    acts_dir.mkdir(parents=True)
    for u, lab in labels.items():
        a = {}
        for pos in F.POSITIONS:
            X = rng.standard_normal((len(rows), len(LAYERS), DIM)) * 0.6
            X += (1.2 * is_h)[:, None, None] * g[None, None, :] + (1.6 * lab)[:, None, None] * v[None, None, :]
            a[f"P@{pos}"] = X.astype(np.float16)
        np.savez(acts_dir / f"{u}.npz", layers=np.array(LAYERS), rows=np.asarray([list(r) for r in rows], dtype=str), **a)
    (acts_dir / "base.json").write_text(json.dumps({"n_layers": N_LAYERS, "layers": LAYERS, "hidden_size": DIM, "base": base}))
    il = []
    for u, lab in labels.items():
        for i, (b, item) in enumerate(rows):
            if b not in H_SIZES:
                continue
            for judge in ("J1", "J2"):
                lbl, ok = ("refusal" if lab[i] else ("harmful_compliance" if i % 2 else "non_harmful_deflection")), True
                if u == "base" and i == 1 and judge == "J1":
                    lbl, ok = "parse_error", False
                il.append({"run_id": u, "battery": b, "item_id": item, "sample_idx": 0, "judge": judge, "label": lbl,
                           "parse_ok": ok, "completion_sha256": "0" * 64})
        xs = [(i, item) for i, (b, item) in enumerate(rows) if b == "xstest_safe"]
        for judge, keep in (("J2", xs[:N_XS_J2]), ("J1", xs)):
            _write_jsonl(root / "judge" / base / u / "labels" / f"xstest_safe.{judge}.jsonl",
                         [{"item_id": item, "sample_idx": 0, "battery": "xstest_safe", "judge": judge, "parse_ok": True,
                           "label": "full_refusal" if lab[i] else "full_compliance"} for i, item in keep])
    _write_jsonl(root / "exploratory" / base / "cpu" / "item_labels.jsonl", il)
    d2 = {u: float(rng.normal()) for u in units}
    bank = root / "tables" / "subsets" / base / f"bank_{base}.csv"
    bank.parent.mkdir(parents=True)
    with bank.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "D2", "j1_regression", "j2_regression", "regression_status"])
        for u, (_, conf) in units.items():
            w.writerow([u, d2[u], conf, conf, "confirmed" if conf else "not_flagged"])
    if with_e2:
        us = list(units)
        can = {rd: F.registered_canary(il, us, rd) for rd in ("J1", "J2")}
        e2 = {"units": us, "confirmed": [u for u, (_, c) in units.items() if c], "D2": d2,
              "registered": {"J1_canary": {"canary": can["J1"]},
                             "WildGuard_canary": {"canary": can["J2"], "paired_bootstrap": {"auroc_a": 1.0}}}}
        (root / "exploratory" / base / "cpu" / "e2.json").write_text(json.dumps(e2))
    F.run_fit(root, base, n_boot=N_BOOT, grid=GRID, quiet=True)
    return root, labels


@pytest.fixture(scope="module")
def toy9(tmp_path_factory):
    root = tmp_path_factory.mktemp("cr9") / "results"
    return make_tree(root, "toy9", UNITS9, 5, with_e2=True)


@pytest.fixture(scope="module")
def body9(toy9):
    root, _ = toy9
    return CR.compute(root, "toy9", n_boot=N_BOOT, grid=GRID, rule_key="qwen3.8-27b", quiet=True)


# ------------------------------------------------------------------------------------------------ rules (pure)


def test_registered_constants():
    assert CR.N_BOOT == 2000 and CR.BOOT_SEED == 0 and CR.SETD_TOL == 0.005 and CR.REPRO_TOL == 1e-9
    assert CR.REGISTRATION["commit"] == "69bfdc0" and CR.LABEL.startswith("EXPLORATORY R1-C")
    r9, r27 = CR.RULES["qwen3.5-9b"], CR.RULES["qwen3.8-27b"]
    assert (r9["n_units"], r9["holds_min_interval_above_0"], r9["falsify_median_below"], r9["falsify_min_delta_le_0"]) == (7, 5, 0.02, 3)
    assert (r27["n_units"], r27["holds_min_interval_above_0"], r27["falsify_median_below"], r27["falsify_min_delta_le_0"]) == (3, 3, 0.02, 2)


def _pu(deltas, los):
    return {f"u{i}": {"delta": d, "ci": [lo, d + 0.05]} for i, (d, lo) in enumerate(zip(deltas, los))}


def test_verdict_9b_rule_and_boundaries():
    rule = CR.RULES["qwen3.5-9b"]
    up = [0.01] * 7
    # 5 of 7 intervals above 0, median >= 0.02, fewer than 3 at or below 0: holds
    v = CR.verdict(_pu([0.05, 0.06, 0.07, 0.08, 0.09, -0.01, 0.0], [0.01, 0.01, 0.01, 0.01, 0.01, -0.05, -0.04]), rule)
    assert (v["n_interval_above_0"], v["n_delta_le_0"], v["median_delta"]) == (5, 2, 0.06)
    assert v["holds_condition"] and not v["falsified_condition"] and v["verdict"] == "holds"
    # 4 of 7 above 0, median 0.05, 2 at or below 0: neither
    v = CR.verdict(_pu([0.05, 0.06, 0.07, 0.08, 0.04, -0.01, 0.0], [0.01, 0.01, 0.01, 0.01, -0.01, -0.05, -0.04]), rule)
    assert (v["n_interval_above_0"], v["verdict"]) == (4, "neither")
    # median EXACTLY 0.02 is not below 0.02: not falsified by the median (all 7 intervals above 0: holds)
    v = CR.verdict(_pu([0.01, 0.015, 0.018, 0.02, 0.05, 0.06, 0.07], up), rule)
    assert v["median_delta"] == 0.02 and not v["falsified_by_median"] and v["verdict"] == "holds"
    v = CR.verdict(_pu([0.01, 0.015, 0.018, 0.0199, 0.05, 0.06, 0.07], up), rule)
    assert v["falsified_by_median"] and v["verdict"] == "falsified"
    # a delta of EXACTLY 0 counts as <= 0: three zeros falsify; two do not (the rest well above 0: holds)
    v = CR.verdict(_pu([0.0, 0.0, 0.0, 0.05, 0.06, 0.07, 0.08], [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.01]), rule)
    assert v["n_delta_le_0"] == 3 and v["falsified_by_count"] and v["verdict"] == "falsified"
    v = CR.verdict(_pu([0.0, 0.0, 0.05, 0.06, 0.07, 0.08, 0.09], [-0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), rule)
    assert v["n_delta_le_0"] == 2 and not v["falsified_condition"] and v["verdict"] == "holds"
    # an interval whose lower end is exactly 0 is not above 0
    v = CR.verdict(_pu([0.05] * 7, [0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01]), rule)
    assert v["n_interval_above_0"] == 4 and v["verdict"] == "neither"
    # both conditions met (5 intervals above 0, median below 0.02): the falsifier takes precedence and is flagged
    v = CR.verdict(_pu([0.011, 0.012, 0.013, 0.015, 0.03, 0.04, 0.05], up), rule)
    assert v["holds_condition"] and v["falsified_condition"] and v["both_conditions_met"] and v["verdict"] == "falsified"
    # the wrong number of defined units: no verdict
    v = CR.verdict(_pu([0.05] * 6, [0.01] * 6), rule)
    assert v["verdict"] == "not_computed" and "6 defined" in v["reason"]
    assert CR.verdict(_pu([0.05] * 7, up), None)["verdict"] == "not_computed"


def test_verdict_27b_rule_and_boundaries():
    rule = CR.RULES["qwen3.8-27b"]
    assert CR.verdict(_pu([0.05, 0.06, 0.04], [0.01, 0.02, 0.01]), rule)["verdict"] == "holds"
    # 2 of 3 intervals above 0 (none at or below 0, median >= 0.02): neither
    v = CR.verdict(_pu([0.05, 0.06, 0.04], [0.01, 0.02, -0.01]), rule)
    assert (v["n_interval_above_0"], v["verdict"]) == (2, "neither")
    # delta <= 0 on 2 of 3 falsifies, a delta of exactly 0 included
    v = CR.verdict(_pu([0.05, -0.01, 0.0], [0.01, -0.05, -0.02]), rule)
    assert v["n_delta_le_0"] == 2 and v["verdict"] == "falsified"
    v = CR.verdict(_pu([0.05, 0.03, 0.0], [0.01, 0.01, -0.02]), rule)
    assert v["n_delta_le_0"] == 1 and v["verdict"] == "neither"
    # median exactly 0.02 over 3 units is not below
    v = CR.verdict(_pu([0.01, 0.02, 0.05], [0.001, 0.001, 0.001]), rule)
    assert v["median_delta"] == 0.02 and v["verdict"] == "holds"
    v = CR.verdict(_pu([0.01, 0.019, 0.05], [0.001, 0.001, 0.001]), rule)
    assert v["verdict"] == "falsified" and v["both_conditions_met"]


def test_headline_rule_is_setting_d():
    assert CR.deep_layers([13, 16, 20, 24, 28, 31, 32], 32) == [24, 28, 31, 32]
    assert CR.deep_layers([32, 37, 40, 48, 56, 63, 64], 64) == [48, 56, 63, 64]
    best = 0.992617
    a = {13: 0.99, 16: best, 20: 0.98, 24: best - 0.005, 28: 0.97, 31: 0.96, 32: 0.95}
    sel = CR.headline_layer(a, 32)
    assert sel["layer"] == 24 and not sel["fallback"] and sel["eligible"] == [24]  # inclusive at exactly best - 0.005
    a[32] = best - 0.004
    assert CR.headline_layer(a, 32)["layer"] == 32  # the deepest eligible
    # nothing deep within 0.005: the deep layer with the highest (a), ties to the deeper, flagged
    sel = CR.headline_layer({16: 0.99, 24: 0.90, 28: 0.95, 31: 0.95, 32: 0.93}, 32)
    assert sel["layer"] == 31 and sel["fallback"] and sel["eligible"] == []
    sel = CR.headline_layer({32: 0.95, 37: 0.99, 40: 0.97, 48: 0.986, 56: 0.9851, 63: 0.97, 64: 0.98}, 64)
    assert sel["layer"] == 56 and sel["eligible"] == [48, 56] and sel["deep_layers"] == [48, 56, 63, 64]


def test_upper_bound_rule():
    b = {3: {"x": 0.7, "y": 0.8, "z": 0.9}, 4: {"x": 0.8, "y": 0.8, "z": 0.6}}
    ub = CR.upper_bound_layer(b, [3, 4], ["x", "y", "z"])
    assert ub["layer"] == 4 and ub["median_by_layer"] == {"3": 0.8, "4": 0.8}  # tie on the median: the deeper
    b[3]["y"] = 0.81
    assert CR.upper_bound_layer(b, [3, 4], ["x", "y", "z"])["layer"] == 3
    assert CR.upper_bound_layer({3: {}, 4: {}}, [3, 4], [])["layer"] is None  # no defined unit: no cell


# ------------------------------------------------------------------------------------------------ bootstrap


def test_paired_delta_bootstrap_is_paired_and_seeded():
    rng = np.random.default_rng(3)
    y = rng.random(150) < 0.4
    sA, sB, sC = y + rng.normal(0, 0.8, 150), y + rng.normal(0, 1.2, 150), rng.normal(0, 1, 150)
    r1 = CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=400, seed=0)
    assert r1 == CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=400, seed=0)  # seeded: identical
    assert r1["ci"] != CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=400, seed=1)["ci"]
    assert r1["delta"] == pytest.approx(stats.auroc(y, sA) - max(stats.auroc(y, sB), stats.auroc(y, sC)), abs=1e-15)
    assert r1["n_boot"] == 400 and r1["seed"] == 0 and r1["n_boot_undefined"] == 0
    # paired: with C identical to B the draw is the codebase's paired AUROC bootstrap, value for value
    r = CR.paired_delta_bootstrap(y, sA, [sB, sB], n_boot=400, seed=0)
    ref = stats.paired_bootstrap_auroc_diff(y, sA, sB, n_boot=400, seed=0, secondary=False)
    assert r["ci"] == [ref["ci_low"], ref["ci_high"]] and r["delta"] == ref["diff"]
    # paired: a reader against itself gives exactly 0 in every draw (independent draws per reader would not)
    r0 = CR.paired_delta_bootstrap(y, sA, [sA], n_boot=400, seed=0)
    assert r0["delta"] == 0.0 and r0["ci"] == [0.0, 0.0]
    # the i.i.d. item draw (sensitivity line) is seeded too and differs from the stratified one
    i1 = CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=400, seed=0, scheme="iid")
    assert i1 == CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=400, seed=0, scheme="iid")
    assert i1["ci"] != r1["ci"] and i1["scheme"] == "iid" and r1["scheme"] == "stratified"
    with pytest.raises(ValueError):
        CR.paired_delta_bootstrap(y, sA, [sB], scheme="block")


def test_projection_and_mean_difference():
    X = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]], np.float16)
    w = CR.mean_difference(X, [0, 1], [2, 3])
    np.testing.assert_allclose(w, [2.0, -3.0])
    np.testing.assert_allclose(CR.projection(X, w), (X.astype(float) @ w) / np.linalg.norm(w))


# ------------------------------------------------------------------------------------------------ readers


def test_probe_reader_is_the_fit():
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((70, 6)) + 3).astype(np.float16)
    idx = np.arange(5, 65)
    y = (X[idx, 0] > 3).astype(int)
    r = CR.probe_reader(X, idx, y, 11, GRID)
    nested = F.nested_cv(X[idx], y, GRID, F.N_FOLDS, 11)
    C, _ = F.choose_C(X[idx], y, GRID, F.N_FOLDS, 11)
    full = F.fit_probe(X[idx], y, C)
    np.testing.assert_array_equal(r["nested"]["oof"], nested["oof"])
    assert r["C"] == C and r["seed"] == 11
    np.testing.assert_array_equal(r["probe"].coef, full.coef)
    np.testing.assert_array_equal(CR.probe_scores(r, X), F.cross_fit_prob(full, nested["fold_probes"], idx, X))
    np.testing.assert_array_equal(CR.oof_on(r, idx[::-1]), nested["oof"][::-1])


def test_prompt_labels_are_the_registered_ones():
    rows = _rows()
    keep = {f"xstest-{i}" for i in range(N_XS_J2)}
    idx, y = CR.prompt_label_rows(rows, keep)
    n_h = sum(H_SIZES.values())
    assert len(idx) == n_h + N_XS_J2 + N_HARMLESS and int(y.sum()) == n_h
    assert all((rows[i][0] in F.H_BATTERIES) == bool(t) for i, t in zip(idx, y))
    assert {rows[i][1] for i in idx if rows[i][0] == "xstest_safe"} == keep


def test_A_reproduces_the_fit_and_the_gate_stops_a_mismatch(toy9, tmp_path):
    root, _ = toy9
    rep = CR.compute(root, "toy9", n_boot=N_BOOT, grid=GRID, reproduce_only=True, quiet=True)["reproduction"]
    assert rep["pass"] and rep["required"]["pass"] and rep["n_values_required"] > 0
    assert rep["required"]["max_abs_diff"] == 0.0 and rep["extra"]["pass"]
    assert rep["extra"]["directions_max_abs_diff"] == 0.0
    # a probe_results.json that differs at one per-unit (b) by 1e-6 stops the run before any content reader; nothing
    # is written
    alt = tmp_path / "results"
    shutil.copytree(root, alt)
    pr = alt / "exploratory" / "toy9" / "decision_probe" / "probe_results.json"
    d = json.loads(pr.read_text())
    d["readers"]["J2"]["positions"]["last"]["layers"]["3"]["b"]["u_b"]["auroc_H_only"] += 1e-6
    pr.write_text(json.dumps(d))
    out = alt / "exploratory" / "toy9" / "decision_probe" / "content_readers.json"
    with pytest.raises(CR.ReproductionError, match="auroc_H_only"):
        CR.run(alt, "toy9", n_boot=N_BOOT, grid=GRID, rule_key="qwen3.8-27b", quiet=True)
    assert not out.exists()


def test_B_differs_from_A_only_through_its_labels(toy9, body9, monkeypatch):
    root, _ = toy9
    cells = body9["cells"]["J2"]
    # as registered, (B) reads the prompt labels: its records differ from (A)'s
    assert any(cells[p][str(L)]["B"]["b"]["u_a"]["auroc_H_only"] != cells[p][str(L)]["A"]["b"]["u_a"]["auroc_H_only"]
               for p in F.POSITIONS for L in LAYERS)
    assert all(cells[p][str(L)]["B"]["fit"]["labels"] == "prompt" for p in F.POSITIONS for L in LAYERS)
    # swap ONLY (B)'s labels (and rows) for the base's WildGuard decision labels: (B) is then (A) under WildGuard,
    # number for number (same fit, seeds, folds, C, threshold, cross-fitting)
    rows, _, _ = F.load_acts(root / "exploratory" / "toy9" / "decision_probe" / "acts" / "base.npz")
    judge = root / "judge" / "toy9"
    il = F.read_jsonl(root / "exploratory" / "toy9" / "cpu" / "item_labels.jsonl")
    keep = {str(r["item_id"]) for r in F.read_jsonl(judge / "base" / "labels" / "xstest_safe.J2.jsonl")}
    blab = F.unit_labels(il, judge, "base", "J2", keep, [k for k in rows if k[0] == F.HARMLESS])
    b_idx = np.array([i for i, k in enumerate(rows) if k in blab])
    y_b = np.array([blab[rows[i]] for i in b_idx])
    monkeypatch.setattr(CR, "prompt_label_rows", lambda rows_, keep_: (b_idx, y_b))
    swapped = CR.compute(root, "toy9", n_boot=N_BOOT, grid=GRID, rule_key="qwen3.8-27b", quiet=True)
    for p in F.POSITIONS:
        for L in map(str, LAYERS):
            A, B = swapped["cells"]["J2"][p][L]["A"], swapped["cells"]["J2"][p][L]["B"]
            for k in ("a", "threshold", "base_false_alarm_rate", "base_decision_score_B0", "b", "c", "c_auroc", "C",
                      "seed", "frozen_full_probe"):
                assert A[k] == B[k], (p, L, k)
            assert swapped["cells"]["J2"][p][L]["cosines"]["A_B"] == pytest.approx(1.0, abs=1e-12)
            # the unswapped (A) is untouched by the patch
            assert A["b"] == body9["cells"]["J2"][p][L]["A"]["b"]


def test_content_readers_on_the_toy_tree(toy9, body9):
    root, labels = toy9
    b = body9
    assert b["label"].startswith("EXPLORATORY R1-C") and b["registration"]["commit"] == "69bfdc0"
    assert b["reproduction"]["pass"] and b["n_layers"] == N_LAYERS and b["deep_layers"] == [3, 4]
    assert b["units"]["regressed"] == ["u_a", "u_b", "u_c", "u_evil"]
    assert b["units"]["defined"]["J2"] == ["u_a", "u_b", "u_c"] and b["units"]["undefined"]["J2"] == ["u_evil"]
    # inputs are hashed
    acts = root / "exploratory" / "toy9" / "decision_probe" / "acts"
    assert b["inputs"]["acts"]["base"] == CR._sha_file(acts / "base.npz")
    assert b["inputs"]["probe_results"]["sha256"] and b["inputs"]["item_labels"]["sha256"]
    for J in F.READERS:
        hc = b["headline_cells"][J]
        assert set(hc) == set(CR.READERS) and all(hc[R]["layer"] in (3, 4) for R in CR.READERS)
        for R in CR.READERS:  # chosen on the base's (a) at the last prompt token only
            a = {L: b["cells"][J]["last"][str(L)][R]["a"]["auroc"] for L in LAYERS}
            assert hc[R] == CR.headline_layer(a, N_LAYERS)
        for L in map(str, LAYERS):
            for p in F.POSITIONS:
                cell = b["cells"][J][p][L]
                assert cell["A"]["a"]["footing"].startswith("out of fold") and cell["B"]["a"]["footing"].startswith("out of fold")
                assert cell["C"]["a"]["footing"].startswith("in-sample") and cell["D"]["a"]["footing"].startswith("in-sample")
                for R in CR.READERS:
                    assert cell[R]["base_false_alarm_rate"] <= F.FRR + 1e-12
                    assert set(cell[R]["b"]) >= {"u_a", "u_b", "u_c"}
                    assert "c_auroc" in cell[R] and "labelled" in cell[R]["c_auroc"]
                assert all(-1 - 1e-12 <= v <= 1 + 1e-12 for v in cell["cosines"].values())
                assert set(cell["cosines"]) == {"A_B", "A_C", "A_D", "B_C", "B_D", "C_D"}
        comp = b["comparison"][J]
        assert set(comp["per_unit"]) == {"u_a", "u_b", "u_c"}
        for u, rec in comp["per_unit"].items():
            hA, hB, hC = (hc[R]["layer"] for R in ("A", "B", "C"))
            bA = b["cells"][J]["last"][str(hA)]["A"]["b"][u]["auroc_H_only"]
            bB = b["cells"][J]["last"][str(hB)]["B"]["b"][u]["auroc_H_only"]
            bC = b["cells"][J]["last"][str(hC)]["C"]["b"][u]["auroc_H_only"]
            assert (rec["b_A"], rec["b_B"], rec["b_C"]) == (bA, bB, bC)
            assert rec["delta"] == pytest.approx(bA - max(bB, bC), abs=1e-15)
            assert rec["ci"][0] <= rec["ci"][1] and rec["bootstrap"]["n_boot"] == N_BOOT
            assert rec["iid_sensitivity"]["ci"][0] <= rec["iid_sensitivity"]["ci"][1]
    assert b["verdict"]["judge"] == "J2" and b["verdict"]["rule"]["base"] == "qwen3.8-27b"
    assert b["verdict"]["verdict"] in ("holds", "falsified", "neither")
    assert b["J1_alongside"]["decision"].startswith("none")
    assert set(b["upper_bound"]["J2"]) >= {"B", "C", "per_unit", "arithmetic"}
    assert b["upper_bound"]["J2"]["B"]["layer"] in (3, 4)
    assert b["first_generated_token"]["J2"]["decides"] == "nothing (descriptive)"
    assert set(b["estimator_D_vs_A"]["J2"]) == {"u_a", "u_b", "u_c"}
    assert set(b["operating_points"]["J2"]) == set(CR.READERS)


def test_cli_writes_the_output(toy9, tmp_path):
    root, _ = toy9
    alt = tmp_path / "results"
    shutil.copytree(root, alt)
    rc = CR.main(["--base", "toy9", "--results", str(alt), "--n-boot", str(N_BOOT), "--grid", "0.01,1.0", "--rule",
                  "qwen3.8-27b", "--expect-defined", "u_c,u_a,u_b", "--quiet"])
    assert rc == 0
    out = alt / "exploratory" / "toy9" / "decision_probe" / "content_readers.json"
    d = json.loads(out.read_text())
    assert d["label"].startswith("EXPLORATORY R1-C") and d["meta"]["stage"] == "content_readers"
    assert d["code_sha256"] == CR._sha_file(Path(CR.__file__)) and d["fit_code_sha256"] == CR._sha_file(Path(F.__file__))
    # a wrong expected unit list stops before any fit
    with pytest.raises(ValueError, match="defined units"):
        CR.compute(alt, "toy9", n_boot=N_BOOT, grid=GRID, expect_defined=["u_a", "u_b"], quiet=True)


def test_restricted_paths_are_refused(tmp_path):
    with pytest.raises(ValueError, match="restricted"):
        CR.compute(tmp_path / "results" / "restricted", "toy9", quiet=True)
    assert CR.main(["--base", "toy9", "--results", str(tmp_path / "results"), "--item-labels",
                    str(tmp_path / "results" / "restricted" / "x.jsonl"), "--quiet"]) == 2


def test_a_27b_style_tree_without_e2(tmp_path):
    """No E2 file (as on 27B): the regressed set is every confirmed labelled unit; a unit that refuses no H item is
    undefined; the 27B rule needs 3 defined units; (c) is undefined (every unit regressed)."""
    root, _ = make_tree(tmp_path / "results", "toy27", UNITS27, 9, with_e2=False)
    b = CR.compute(root, "toy27", n_boot=N_BOOT, grid=GRID, rule_key="qwen3.8-27b", expect_defined=["u_a", "u_b", "u_c"],
                   quiet=True)
    assert b["units"]["e2_units"] == [] and b["units"]["regressed"] == ["u_a", "u_b", "u_c", "u_evil"]
    assert b["units"]["defined"]["J2"] == ["u_a", "u_b", "u_c"]
    assert b["verdict"]["n_defined"] == 3 and b["verdict"]["verdict"] in ("holds", "falsified", "neither")
    cell = b["cells"]["J2"]["last"]["4"]
    assert all(cell[R]["c_auroc"]["labelled"]["auroc"] is None for R in CR.READERS)
    # without a registered rule for the base, every number is reported and no verdict is computed
    nb = CR.compute(root, "toy27", n_boot=N_BOOT, grid=GRID, quiet=True)
    assert nb["verdict"]["verdict"] == "not_computed"
