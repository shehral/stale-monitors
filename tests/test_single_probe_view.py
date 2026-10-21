"""Tests for sm.exploratory.single_probe_view (POST HOC on 9B; on 27B a registered descriptive line, b6408c7).

CPU only, tiny synthetic arrays and a synthetic results tree laid out as the real one (the R1-C test tree, rebuilt
here so this file stands alone); no model, no text."""

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
from sm.exploratory import single_probe_view as SV

H_SIZES = {"strongreject": 20, "harmbench": 30, "xstest_unsafe": 30}  # 80: the registered E2 draw takes 64
N_XS, N_XS_J2, N_HARMLESS, DIM = 10, 6, 24, 12
LAYERS, N_LAYERS = [2, 3, 4], 4  # deep layers (3 l >= 2 n): 3 and 4
GRID = (0.01, 1.0)
N_BOOT = 60
# unit: (share of the base's H refusals it loses, confirmed regression); "u_evil" refuses no H item (undefined (b))
UNITS9 = {"u_a": (0.6, True), "u_b": (0.4, True), "u_c": (0.3, True), "u_evil": (1.0, True), "u_null": (0.0, False),
          "u_null2": (0.05, False)}
UNITS27 = {"u_a": (0.6, True), "u_b": (0.4, True), "u_c": (0.3, True), "u_evil": (1.0, True)}
RULE = "qwen3.8-27b"  # the toy trees have 3 defined units, as the 27B slice


def _rows():
    rows = [(b, f"{b}-{i}") for b, n in H_SIZES.items() for i in range(n)]
    rows += [("xstest_safe", f"xstest-{i}") for i in range(N_XS)]
    rows += [("test_harmless", f"test_harmless-{i:05d}") for i in range(N_HARMLESS)]
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _labels(rng, rows, units):
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


def make_tree(root: Path, base: str, units: dict, seed: int, with_e2: bool, with_cr: bool = True) -> tuple[Path, dict]:
    """A synthetic results tree (content direction g on every H item, decision direction v where the model refuses),
    the text-free label tables (one J1 parse error on the base's H item 1, so that prompt is not one of the J1 probe's
    rows), the bank table, E2's file when asked, R1's fit and, when asked, R1-C's content_readers.json."""
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
    if with_cr:
        CR.run(root, base, n_boot=N_BOOT, grid=GRID, rule_key=RULE, quiet=True)
    return root, labels


@pytest.fixture(scope="module")
def toy9(tmp_path_factory):
    root = tmp_path_factory.mktemp("spv9") / "results"
    return make_tree(root, "toy9", UNITS9, 5, with_e2=True)


@pytest.fixture(scope="module")
def body9(toy9):
    root, _ = toy9
    return SV.compute(root, "toy9", n_boot=N_BOOT, grid=GRID, rule_key=RULE, quiet=True)


# ------------------------------------------------------------------------------------------------ fixtures by hand


class _Spy:
    """A stand-in fold probe that records the prompts (an id in column 0) it is asked to score."""

    def __init__(self, train_ids):
        self.train_ids, self.seen = {int(i) for i in train_ids}, []

    def decision(self, X):
        X = np.asarray(X, float)
        self.seen.extend(X[:, 0].astype(int).tolist())
        return X[:, 1]

    def prob(self, X):
        return 1.0 / (1.0 + np.exp(-self.decision(X)))


def _probe(coef, intercept):
    k = len(coef)
    return F.Probe(np.zeros(k), np.ones(k), np.asarray(coef, float), float(intercept), 1.0)


def _five_equal_folds(probes, n_outside=0):
    """Five folds of five base rows; every fold holds out rows carrying the same values (-2..2 on feature 0), so a
    probe's held-out base scores have the same mean and SD in every fold. n_outside extra rows of P are not the probe's
    rows (the all-data probe scores them in the registered cross-fit)."""
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    Xb = np.column_stack([np.concatenate([np.tile(vals, 5), np.linspace(-1, 1, n_outside)]), np.zeros(25 + n_outside)])
    folds = [np.arange(5 * f, 5 * f + 5) for f in range(5)]
    return {"idx": np.arange(25), "nested": {"fold_probes": list(zip(folds, probes))}, "probe": probes[0]}, Xb


# ------------------------------------------------------------------------------------------------ registered constants


def test_label_definition_and_variants():
    assert SV.LABEL == ("POST HOC on 9B: single-probe view of R1 and R1-C; on 27B the same view is a registered "
                        "descriptive line (b6408c7)")
    assert SV.DEFINITION["registered"] == "0c3a5e5" and SV.DEFINITION["tightened"] == "b6408c7"
    assert SV.VARIANTS == ("pooled", "per_fold", "standardized", "all_data")
    assert SV.SENSITIVITY == ("standardized_prob",)
    assert "POST HOC" in SV.status_for("qwen3.5-9b") and "registered descriptive line" in SV.status_for("qwen3.8-27b")
    assert issubclass(SV.ReproductionError, CR.ReproductionError)


# ------------------------------------------------------------------------------------------------ (i) held-out only


def test_held_out_scoring_never_uses_a_probe_trained_on_the_prompt():
    rng = np.random.default_rng(0)
    n = 40
    X = np.column_stack([np.arange(n), rng.standard_normal(n)])
    ridx = np.arange(3, 37)  # the probe's rows; 0-2 and 37-39 are not (their base label unusable, say)
    folds = F.folds(np.arange(ridx.size) % 2, 5, 0)
    spies = [_Spy(ridx[tr]) for tr, _ in folds]
    full = _Spy(ridx)
    r = {"idx": ridx, "nested": {"fold_probes": [(te, s) for (_, te), s in zip(folds, spies)]}, "probe": full}
    fold, s = SV.held_out_scores(r, X)
    for f, ((tr, te), spy) in enumerate(zip(folds, spies)):
        assert spy.seen and not set(spy.seen) & spy.train_ids  # never a prompt the fold trained on
        assert sorted(spy.seen) == sorted(ridx[te].tolist())  # exactly the prompts it held out, each once
        assert set(np.flatnonzero(fold == f).tolist()) == set(ridx[te].tolist())
    assert full.seen == []  # the all-data probe (trained on every prompt) scores nothing
    outside = np.setdiff1d(np.arange(n), ridx)
    assert (fold[outside] == -1).all() and np.isnan(s[outside]).all() and not np.isnan(s[ridx]).any()
    np.testing.assert_array_equal(SV.fold_index(r, n), fold)
    # the per-fold AUROC reads scored rows only, each fold's items by that fold's probe
    y = (X[:, 1] + rng.normal(0, 1, n)) > 0
    pf = SV.per_fold_auroc(y, fold, s, 5)
    assert pf["n_unscored"] == outside.size and sum(pf["n_per_fold"]) == ridx.size
    for f, (_, te) in enumerate(folds):
        rr = ridx[te]
        assert pf["folds"][f] == F.auroc(y[rr], s[rr])


def test_held_out_scores_are_the_registered_cross_fit_bit_for_bit():
    rng = np.random.default_rng(1)
    X = (rng.standard_normal((70, 6)) + 3).astype(np.float16)
    idx = np.arange(5, 65)
    y = (X[idx, 0] > 3).astype(int)
    r = CR.probe_reader(X, idx, y, 11, GRID)
    Xu = (rng.standard_normal((70, 6)) + 3).astype(np.float16)
    fold, s = SV.held_out_scores(r, Xu)
    for f, (te, _) in enumerate(r["nested"]["fold_probes"]):
        tr = np.setdiff1d(np.arange(idx.size), te)
        scored = set(np.flatnonzero(fold == f).tolist())
        assert scored == set(idx[te].tolist()) and not scored & set(idx[tr].tolist())
    np.testing.assert_array_equal(s[idx], CR.probe_scores(r, Xu)[idx])  # the pooled cross-fit's item scores
    assert np.isnan(s[np.setdiff1d(np.arange(70), idx)]).all()
    # on the base itself they are the nested CV's out-of-fold scores
    fb, sb = SV.held_out_scores(r, X)
    np.testing.assert_array_equal(sb[idx], r["nested"]["oof"])


def test_per_fold_auroc_by_hand():
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.4, 0.6, 0.8, 0.7, 0.2, 0.3, 0.5, 0.95])
    fold = np.array([0, 0, 0, 0, 1, 1, 2, 2, -1, -1])  # fold 1 only positives, fold 2 only negatives
    pf = SV.per_fold_auroc(y, fold, s, 3)
    assert pf["folds"] == [0.75, None, None] and pf["mean"] == 0.75 and pf["n_defined_folds"] == 1
    assert pf["n_per_fold"] == [4, 2, 2] and pf["n_pos_per_fold"] == [2, 2, 0] and pf["n_unscored"] == 2
    stat = SV.per_fold_stat(y, fold, s, 3)
    assert stat(np.arange(10)) == 0.75
    assert np.isnan(stat(np.array([4, 5, 6])))  # no fold with both classes in this draw


# ------------------------------------------------------------------------------------------------ (ii) standardized


def test_standardized_pooling_equals_pooled_scoring_when_all_fold_probes_are_identical():
    P = _probe([1.3, 0.0], 0.4)
    r, Xb = _five_equal_folds([P] * 5, n_outside=5)
    rng = np.random.default_rng(2)
    Xu = rng.standard_normal((30, 2)) * 2
    y = rng.random(30) < 0.5
    pooled = F.cross_fit_prob(P, r["nested"]["fold_probes"], r["idx"], Xu)
    np.testing.assert_array_equal(pooled, P.prob(Xu))  # identical fold probes: pooling is one probe
    for scale in SV.SCALES:
        sz = SV.fold_standardization(r, Xb, scale)
        assert len({tuple(t) for t in sz["folds"]}) == 1  # one location and spread for every fold
        np.testing.assert_allclose(sz["all_data"], sz["folds"][0], rtol=0, atol=1e-12)
        z = SV.standardized_scores(r, Xu, sz)
        mu, sd = sz["folds"][0]
        raw = P.decision(Xu) if scale == "logit" else P.prob(Xu)
        np.testing.assert_allclose(z, (raw - mu) / sd, rtol=0, atol=1e-12)  # one common affine map
        assert stats.auroc(y, z) == stats.auroc(y, pooled)
        # a unit's per-unit AUROC is unchanged on every subset (the map is common to all folds)
        assert stats.auroc(y[5:25], z[5:25]) == stats.auroc(y[5:25], pooled[5:25])


def test_standardization_undoes_a_per_fold_affine_recalibration():
    """Fold probes that share a direction but differ in logit scale and offset (what different inner-CV C do): the
    registered pool mixes their calibrations; z-scoring each against its own held-out base scores gives back the single
    direction's ranking exactly."""
    ks, bs = (0.05, 0.5, 1.0, 4.0, 20.0), (-1.0, 0.0, 0.5, 2.0, -3.0)
    probes = [_probe([k, 0.0], b) for k, b in zip(ks, bs)]
    r, Xb = _five_equal_folds(probes)
    rng = np.random.default_rng(3)
    Xu = rng.standard_normal((25, 2))
    y = (Xu[:, 0] + rng.normal(0, 0.7, 25)) > 0
    z = SV.standardized_scores(r, Xu, SV.fold_standardization(r, Xb, "logit"))
    np.testing.assert_allclose(z, Xu[:, 0] / np.std([-2.0, -1.0, 0.0, 1.0, 2.0]), rtol=0, atol=1e-9)
    assert stats.auroc(y, z) == stats.auroc(y, Xu[:, 0])
    pooled = F.cross_fit_prob(probes[0], r["nested"]["fold_probes"], r["idx"], Xu)
    assert stats.auroc(y, pooled) != stats.auroc(y, Xu[:, 0])  # the raw pool does not


def test_standardization_reads_each_fold_probe_on_its_own_held_out_base_rows():
    rng = np.random.default_rng(5)
    n = 40
    Xb = np.column_stack([np.arange(n), rng.standard_normal(n)])
    ridx = np.arange(2, 38)
    folds = F.folds(np.arange(ridx.size) % 2, 5, 1)
    spies = [_Spy(ridx[tr]) for tr, _ in folds]
    full = _Spy(ridx)
    r = {"idx": ridx, "nested": {"fold_probes": [(te, sp) for (_, te), sp in zip(folds, spies)]}, "probe": full}
    sz = SV.fold_standardization(r, Xb, "logit")
    for f, ((tr, te), sp) in enumerate(zip(folds, spies)):
        assert sorted(sp.seen) == sorted(ridx[te].tolist())  # its held-out base rows only, never a training row
        v = Xb[ridx[te], 1]
        assert sz["folds"][f] == [float(v.mean()), float(v.std())]
    assert sorted(full.seen) == sorted(ridx.tolist())  # the all-data probe: all its rows (in-sample, flagged)
    assert sz["all_data"] == [float(Xb[ridx, 1].mean()), float(Xb[ridx, 1].std())]
    # each row is z-scored by the probe that scores it in the registered cross-fit, with that probe's own location
    z = SV.standardized_scores(r, Xb, sz)
    fold = SV.fold_index(r, n)
    for i in range(n):
        mu, sd = sz["folds"][fold[i]] if fold[i] >= 0 else sz["all_data"]
        assert z[i] == (Xb[i, 1] - mu) / sd


def test_standardization_uses_no_label_and_refuses_a_flat_probe():
    P = _probe([1.0, 0.0], 0.0)
    r, Xb = _five_equal_folds([P] * 5)
    a = SV.fold_standardization(r, Xb, "logit")
    Xb2 = Xb.copy()
    Xb2[:, 1] = 7.0  # a feature the probe ignores: nothing changes, and no label enters anywhere
    assert SV.fold_standardization(r, Xb2, "logit") == a
    flat = _probe([0.0, 0.0], 0.3)
    r2, _ = _five_equal_folds([flat] * 5)
    with pytest.raises(ValueError, match="spread"):
        SV.fold_standardization(r2, Xb, "logit")
    with pytest.raises(ValueError):
        SV.fold_standardization(r, Xb, "rank")


# ------------------------------------------------------------------------------------------------ the bootstrap


def test_generic_paired_bootstrap_is_the_registered_one_for_pooled_scores():
    rng = np.random.default_rng(4)
    y = rng.random(150) < 0.4
    sA, sB, sC = y + rng.normal(0, 0.8, 150), y + rng.normal(0, 1.2, 150), rng.normal(0, 1, 150)
    mine = SV.paired_delta_bootstrap_stats(y, SV.pooled_stat(y, sA), [SV.pooled_stat(y, sB), SV.pooled_stat(y, sC)],
                                           n_boot=300, seed=0)
    ref = CR.paired_delta_bootstrap(y, sA, [sB, sC], n_boot=300, seed=0)
    for k in ("delta", "b_a", "b_others", "ci", "n_boot", "seed", "n_boot_undefined", "n_pos", "n_neg"):
        assert mine[k] == ref[k], k
    # per-fold statistics: the point estimate is the fold mean, every draw is paired and seeded
    fold = rng.integers(0, 5, 150)
    st = SV.per_fold_stat(y, fold, sA, 5)
    pf = SV.paired_delta_bootstrap_stats(y, st, [SV.pooled_stat(y, sC)], n_boot=300, seed=0)
    assert pf["b_a"] == SV.per_fold_auroc(y, fold, sA, 5)["mean"]
    assert pf == SV.paired_delta_bootstrap_stats(y, st, [SV.pooled_stat(y, sC)], n_boot=300, seed=0)
    zero = SV.paired_delta_bootstrap_stats(y, st, [st], n_boot=300, seed=0)
    assert zero["delta"] == 0.0 and zero["ci"] == [0.0, 0.0]


# ------------------------------------------------------------------------------------------------ the gate


@pytest.mark.parametrize("what", ["fit_b", "fit_fold_C", "fit_frozen_c", "cr_B", "cr_ci", "cr_inputs"])
def test_the_gate_catches_a_tamper(toy9, tmp_path, what):
    root, _ = toy9
    alt = tmp_path / "results"
    shutil.copytree(root, alt)
    dp = alt / "exploratory" / "toy9" / "decision_probe"
    path = dp / ("probe_results.json" if what.startswith("fit") else "content_readers.json")
    d = json.loads(path.read_text())
    if what.startswith("fit"):
        rec = d["readers"]["J2"]["positions"]["last"]["layers"]["3"]
        if what == "fit_b":
            rec["b"]["u_b"]["auroc_H_only"] += 1e-6
        elif what == "fit_fold_C":
            rec["a"]["folds"][2]["C"] = 99.0
        else:
            rec["frozen_full_probe"]["c"]["u_a"]["regression_score"] += 1e-6
    elif what == "cr_B":
        d["cells"]["J1"]["gen1"]["4"]["B"]["b"]["u_a"]["auroc_H_only"] += 1e-6
    elif what == "cr_ci":
        d["comparison"]["J2"]["per_unit"]["u_c"]["ci"][0] -= 1e-6
    else:
        d["inputs"]["acts"]["u_b"] = "0" * 64
    path.write_text(json.dumps(d))
    out = dp / "single_probe_view.json"
    which = {"fit_b": "pooled does not reproduce probe_results", "fit_fold_C": "per-fold or frozen records",
             "fit_frozen_c": "per-fold or frozen records"}.get(what, "do not reproduce content_readers.json")
    with pytest.raises(SV.ReproductionError, match=which):
        SV.run(alt, "toy9", n_boot=N_BOOT, grid=GRID, rule_key=RULE, quiet=True)
    assert not out.exists()


def test_the_view_on_the_toy_tree(toy9, body9):
    root, _ = toy9
    b = body9
    dp = root / "exploratory" / "toy9" / "decision_probe"
    fit = json.loads((dp / "probe_results.json").read_text())
    cr = json.loads((dp / "content_readers.json").read_text())
    assert b["label"] == SV.LABEL and b["decides"].startswith("nothing")
    g = b["gate"]
    assert g["pass"] and g["probe_results"]["pass"] and g["probe_results"]["required"]["max_abs_diff"] == 0.0
    assert g["fit_extra"]["pass"] and g["fit_extra"]["max_abs_diff"] == 0.0
    assert g["content_readers"]["available"] and g["content_readers"]["pass"] and g["content_readers"]["max_abs_diff"] == 0.0
    assert b["units"]["defined"]["J2"] == ["u_a", "u_b", "u_c"]
    for rd in F.READERS:
        for pos in F.POSITIONS:
            for L in map(str, LAYERS):
                th = fit["readers"][rd]["positions"][pos]["layers"][L]
                A, B = b["cells"]["A"][rd][pos][L], b["cells"]["B"][rd][pos][L]
                assert A["fit"]["fold_C"] == [f["C"] for f in th["a"]["folds"]] and A["fit"]["C_all_data"] == th["C"]
                assert A["a"]["per_fold"]["mean"] == th["a"]["auroc_fold_mean"]
                for u in ("u_a", "u_b", "u_c"):
                    assert A["b"]["pooled"][u] == th["b"][u]["auroc_H_only"]
                    assert A["b"]["all_data"][u] == th["frozen_full_probe"]["b"][u]["auroc_H_only"]
                    assert B["b"]["pooled"][u] == cr["cells"][rd][pos][L]["B"]["b"][u]["auroc_H_only"]
                    pf = A["b"]["per_fold"][u]
                    vals = [v for v in pf["folds"] if v is not None]
                    assert pf["mean"] == pytest.approx(float(np.mean(vals)), abs=1e-15)
                    # J1's base cannot label H item 1, so no J1 fold of (A) holds it out: never scored per fold
                    assert pf["n_unscored"] == (1 if rd == "J1" else 0)
                    assert B["b"]["per_fold"][u]["n_unscored"] == 0
                assert A["c"]["pooled"]["c_auroc"]["labelled"]["auroc"] == th["c_auroc"]["labelled"]["auroc"]
                assert A["c"]["all_data"]["c_auroc"]["labelled"]["auroc"] == th["frozen_full_probe"]["c_auroc"]["labelled"]["auroc"]
                assert len(A["c"]["per_fold"]["folds"]) == 5 and len(B["fit"]["fold_C"]) == 5
                for v in SV.ALL_VARIANTS:
                    s = A["summary"][v]
                    vals = [A["b"][v][u]["mean"] if v == "per_fold" else A["b"][v][u] for u in ("u_a", "u_b", "u_c")]
                    assert s["n_ge_075"] == sum(x >= 0.75 for x in vals) and s["n_defined"] == 3
                    assert s["median"] == pytest.approx(float(np.median(vals)), abs=1e-15)
                    assert b["r1_summary"]["A"][v][rd][pos][L] == s
        # the R1-C delta: pooled is R1-C's registered comparison, value for value; every variant is b(A) - max(b(B), b(C))
        hl = {R: cr["headline_cells"][rd][R]["layer"] for R in ("A", "B", "C")}
        assert set(b["headline_cells"][rd]) == {"A", "B", "C"}
        assert {R: b["headline_cells"][rd][R]["layer"] for R in ("A", "B", "C")} == hl
        for v in SV.ALL_VARIANTS:
            dv = b["r1c_delta"][v][rd]
            assert dv["layers"] == hl and dv["arithmetic"]["decides"].startswith("nothing")
            for u, rec in dv["per_unit"].items():
                key = (lambda x: x["mean"]) if v == "per_fold" else (lambda x: x)
                bA = key(b["cells"]["A"][rd]["last"][str(hl["A"])]["b"][v][u])
                bB = key(b["cells"]["B"][rd]["last"][str(hl["B"])]["b"][v][u])
                assert (rec["b_A"], rec["b_B"]) == (bA, bB)
                assert rec["b_C"] == cr["cells"][rd]["last"][str(hl["C"])]["C"]["b"][u]["auroc_H_only"]
                assert rec["delta"] == pytest.approx(bA - max(bB, rec["b_C"]), abs=1e-15)
                assert rec["ci"][0] <= rec["ci"][1] and rec["bootstrap"]["n_boot"] == N_BOOT
                if v == "pooled":
                    reg = cr["comparison"][rd]["per_unit"][u]
                    assert (rec["delta"], rec["ci"]) == (reg["delta"], reg["ci"])
    assert b["code_sha256"] == SV._sha_file(Path(SV.__file__))
    assert b["fit_code_sha256"] == SV._sha_file(Path(F.__file__))
    assert b["content_readers_code_sha256"] == SV._sha_file(Path(CR.__file__))


def test_a_27b_style_tree_without_e2_or_r1c(tmp_path):
    """No E2 file and no content_readers.json yet (as on 27B before R1-C runs): every confirmed labelled unit is
    regressed, 3 are defined, (c) is undefined (every unit regressed), and the R1-C check is reported as unavailable."""
    root, _ = make_tree(tmp_path / "results", "toy27", UNITS27, 9, with_e2=False, with_cr=False)
    b = SV.compute(root, "toy27", n_boot=N_BOOT, grid=GRID, rule_key=RULE, expect_defined=["u_a", "u_b", "u_c"],
                   quiet=True)
    assert b["units"]["e2_units"] == [] and b["units"]["regressed"] == ["u_a", "u_b", "u_c", "u_evil"]
    assert b["gate"]["pass"] and not b["gate"]["content_readers"]["available"]
    for v in SV.ALL_VARIANTS:
        assert b["r1_summary"]["A"][v]["J2"]["last"]["4"]["c_labelled"] is None
        assert b["r1c_delta"][v]["J2"]["arithmetic"]["n_defined"] == 3
    with pytest.raises(ValueError, match="defined units"):
        SV.compute(root, "toy27", n_boot=N_BOOT, grid=GRID, expect_defined=["u_a"], quiet=True)


def test_cli_writes_the_output_and_refuses_restricted_paths(toy9, tmp_path):
    root, _ = toy9
    alt = tmp_path / "results"
    shutil.copytree(root, alt)
    rc = SV.main(["--base", "toy9", "--results", str(alt), "--n-boot", str(N_BOOT), "--grid", "0.01,1.0", "--rule", RULE,
                  "--quiet"])
    assert rc == 0
    d = json.loads((alt / "exploratory" / "toy9" / "decision_probe" / "single_probe_view.json").read_text())
    assert d["label"] == SV.LABEL and d["meta"]["stage"] == "single_probe_view"
    assert d["meta"]["code_sha256"] == SV._sha_file(Path(SV.__file__)) and "git_commit" in d["meta"]
    assert SV.main(["--base", "toy9", "--results", str(tmp_path / "results" / "restricted"), "--quiet"]) == 2
