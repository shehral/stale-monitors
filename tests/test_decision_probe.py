"""CPU tests for the exploratory R1 decision probe (docs/PROGRAM_2026-10-13.md R1): the capture on a tiny random model
(sm.exploratory.decision_probe) and the fit on a synthetic results tree (sm.exploratory.decision_probe_fit)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sm.exploratory import decision_probe as DP
from sm.exploratory import decision_probe_fit as F

# ------------------------------------------------------------------------------------------------ capture


@pytest.fixture(scope="module")
def smoke_root(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("dprobe_smoke")
    assert DP.main(["--smoke", "--out", str(root), "--unit", "base"]) == 0
    assert DP.main(["--smoke", "--out", str(root), "--unit", "smoke_unit"]) == 0
    return root


def test_capture_writes_numbers_rows_and_stamps(smoke_root):
    acts = smoke_root / "open" / "acts"
    z = np.load(acts / "smoke_unit.npz", allow_pickle=False)
    n = sum(len(v) for v in DP.SMOKE_PROMPTS.values())
    assert z["P@last"].shape == (n, 3, 32) and z["P@gen1"].shape == (n, 3, 32)
    assert z["P@last"].dtype == np.float16 and list(z["layers"]) == [1, 2, 3]
    rows = [tuple(r) for r in z["rows"]]
    assert [b for b, _ in rows] == [b for b, _ in DP.prompt_files(Path("x")) for _ in DP.SMOKE_PROMPTS[b]]
    for k in ("margin_last", "margin_gen1", "first_token_refusal_set", "first_token_compliance_set", "first_token_eos"):
        assert z[k].shape == (n,)
    side = json.loads((acts / "smoke_unit.json").read_text())
    assert side["checks"]["IC0"]["status"] == "pass"
    assert side["render"]["thinking"] is False and side["render"]["chat_template_sha256"]
    prov = side["provenance"]
    assert prov["config_sha256"] and prov["adapter_sha256"] and "git_commit" in prov
    assert prov["file_sha256"]["code/sm/exploratory/decision_probe.py"]
    assert set(side["prompt_set"]["files"]) == set(DP.BATTERIES) | {DP.HARMLESS_SET}
    # no text: no prompt string appears anywhere in the open outputs
    blob = (acts / "smoke_unit.json").read_text() + (acts / "base.json").read_text()
    for prompts in DP.SMOKE_PROMPTS.values():
        for p in prompts:
            assert p not in blob


def test_capture_matches_a_direct_forward_and_the_adapter_moves_it(smoke_root):
    """The last-token capture equals hidden_states[l] of a plain forward on the battery-rendered prompt, and gen1 is the
    position after the model's own greedy first token."""
    torch = pytest.importorskip("torch")
    from sm.common.models import encode_text, load_model

    assets = smoke_root / "_smoke_assets"
    model, tok, _ = load_model(str(assets / "tiny_model"), dtype="fp32", device="cpu", attn="eager")
    rows, msgs, _ = DP.load_prompt_set(assets / "prompts")
    texts, meta = DP.render_prompts(tok, msgs)
    assert meta["thinking"] is False
    z = np.load(smoke_root / "open" / "acts" / "base.npz")
    for i in (0, len(rows) - 1):
        ids = encode_text(tok, texts[i])
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
        hs = out.hidden_states
        top = int(out.logits[0, -1].argmax())
        with torch.no_grad():
            hs2 = model(input_ids=torch.tensor([ids + [top]]), output_hidden_states=True).hidden_states
        for j, L in enumerate([1, 2]):  # block outputs before the final norm (the last index is post-norm in HF)
            np.testing.assert_allclose(z["P@last"][i, j].astype(np.float32), hs[L][0, -1].numpy(), rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(z["P@gen1"][i, j].astype(np.float32), hs2[L][0, -1].numpy(), rtol=2e-3, atol=2e-3)
    u = np.load(smoke_root / "open" / "acts" / "smoke_unit.npz")
    assert np.abs(u["P@last"].astype(np.float32) - z["P@last"].astype(np.float32)).max() > 1e-3


def test_capture_skips_unless_forced(smoke_root, capsys):
    assert DP.main(["--smoke", "--out", str(smoke_root), "--unit", "base"]) == 0
    assert "already done" in capsys.readouterr().out


def test_a_failed_render_check_saves_no_capture(smoke_root, tmp_path):
    from sm.interp import mechanism as M

    assets = smoke_root / "_smoke_assets"
    meta_dir = tmp_path / "results" / "batteries" / "tiny-smoke" / "base"
    meta_dir.mkdir(parents=True)
    (meta_dir / "harmbench.meta.json").write_text(json.dumps({"meta": {"chat_render": {"chat_template_sha256": "other",
                                                                                         "thinking": False}}}))
    with pytest.raises(RuntimeError, match="battery_render"):
        DP.run_capture("base", base="tiny-smoke", items_dir=assets / "prompts", out_dir=tmp_path / "open",
                       results_root=tmp_path / "results", loader=M._smoke_loader, model_ref=str(assets / "tiny_model"),
                       run_cfg=None, dtype="fp32", device="cpu", layers=[1, 2], batch_size=2, check_registered=False,
                       smoke=True)
    assert not (tmp_path / "open" / "acts" / "base.npz").exists()
    assert not (tmp_path / "open" / "acts" / "base.json").exists()
    side = json.loads((tmp_path / "open" / "acts" / "base.failed.json").read_text())
    assert side["checks"]["battery_render"]["per_battery"]["harmbench"]["status"] == "fail"


def test_manifest_hash_mismatch_is_refused(tmp_path):
    items = DP.write_smoke_items(tmp_path / "prompts")
    (items / "MANIFEST.json").write_text(json.dumps({"batteries": {"harmbench": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="manifest"):
        DP.load_prompt_set(items)


def test_battery_render_check(tmp_path):
    files = {b: {"sha256": f"sha-{b}"} for b in DP.BATTERIES}
    rmeta = {"chat_template_sha256": "T", "thinking": False}
    assert DP.battery_render_check(tmp_path, "toy", "base", rmeta, files)["status"] == "absent"
    for i, b in enumerate(DP.BATTERIES):
        d = tmp_path / ("batteries" if i % 2 else "restricted/batteries") / "toy" / "base"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{b}.meta.json").write_text(json.dumps({"meta": {"chat_render": {"chat_template_sha256": "T", "thinking": False},
                                                               "item_file_sha256": f"sha-{b}"}}))
    assert DP.battery_render_check(tmp_path, "toy", "base", rmeta, files)["status"] == "pass"
    assert DP.battery_render_check(tmp_path, "toy", "base", {**rmeta, "chat_template_sha256": "U"}, files)["status"] == "fail"

# ------------------------------------------------------------------------------------------------ fit


def test_threshold_and_probe_pieces():
    s = np.arange(100, dtype=float)
    t, frr = F.threshold_at_frr(s, 0.05)
    assert frr == 0.05 and t == 94.0
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 5)) * [1, 2, 3, 4, 5] + 10
    y = (X[:, 0] > 10).astype(int)
    p = F.fit_probe(X, y, 1.0)
    w, b = p.raw_direction()
    np.testing.assert_allclose(X @ w + b, p.decision(X), rtol=1e-10, atol=1e-8)
    assert F.auroc(y, p.decision(X)) > 0.95
    C, table = F.choose_C(X, y, (0.01, 1.0))
    assert C in (0.01, 1.0) and len(table) == 2
    assert F.auroc([1, 1], [0.1, 0.2]) is None
    assert F.cv_fixed_C(X, np.r_[np.ones(78), np.zeros(2)].astype(int), 1.0) is None


BASE_KEY = "toy"
H_SIZES = {"strongreject": 20, "harmbench": 30, "xstest_unsafe": 30}
N_XS, N_XS_J2, N_HARMLESS, DIM = 10, 6, 20, 16
LAYERS = [2, 3]
# unit: (share of base refusals lost, confirmed regression)
UNITS = {"u_reg_a": (0.6, True), "u_reg_b": (0.4, True), "u_reg_c": (0.3, True), "u_null_a": (0.0, False),
         "u_null_b": (0.02, False), "u_null_c": (0.0, False)}


def _rows():
    rows = [(b, f"{b}-{i}") for b, n in H_SIZES.items() for i in range(n)]
    rows += [("xstest_safe", f"xstest-{i}") for i in range(N_XS)]
    rows += [("test_harmless", f"test_harmless-{i:05d}") for i in range(N_HARMLESS)]
    return rows


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _toy_labels(rng, rows):
    """The base refuses the H items except the first six and over-refuses two XSTest-safe items; each unit then loses
    its share (UNITS) of the base's H refusals."""
    base_ref = np.array([1 if b in H_SIZES else 0 for b, _ in rows])
    base_ref[[i for i, (b, _) in enumerate(rows) if b in H_SIZES][:6]] = 0  # a few answered H items
    base_ref[[i for i, (b, _) in enumerate(rows) if b == "xstest_safe"][:2]] = 1  # two over-refusals
    labels = {"base": base_ref}
    for u, (loss, _) in UNITS.items():
        lab = base_ref.copy()
        h_ref = [i for i, (b, _) in enumerate(rows) if b in H_SIZES and base_ref[i] == 1]
        lab[rng.choice(h_ref, size=int(round(loss * len(h_ref))), replace=False)] = 0
        labels[u] = lab
    return labels


def _write_tree(root: Path, rows, labels, acts, layers, d2) -> None:
    """A synthetic results tree from the captures (acts: unit -> {"P@<pos>": rows x layers x H}) and the labels: the H
    and XSTest-safe labels of both readers (one J1 parse error on the base), the bank table and E2's file."""
    acts_dir = root / "exploratory" / BASE_KEY / "decision_probe" / "acts"
    acts_dir.mkdir(parents=True)
    for u, a in acts.items():
        np.savez(acts_dir / f"{u}.npz", layers=np.array(layers), rows=np.asarray([list(r) for r in rows], dtype=str), **a)
    il = []
    for u, lab in labels.items():
        for i, (b, item) in enumerate(rows):
            if b not in H_SIZES:
                continue
            for judge in ("J1", "J2"):
                lbl = "refusal" if lab[i] else ("harmful_compliance" if i % 2 else "non_harmful_deflection")
                ok = True
                if u == "base" and i == 1 and judge == "J1":
                    lbl, ok = "parse_error", False
                il.append({"run_id": u, "battery": b, "item_id": item, "sample_idx": 0, "judge": judge, "label": lbl,
                           "parse_ok": ok})
        xs = [(i, item) for i, (b, item) in enumerate(rows) if b == "xstest_safe"]
        for judge, keep in (("J2", xs[:N_XS_J2]), ("J1", xs)):
            _write_jsonl(root / "judge" / BASE_KEY / u / "labels" / f"xstest_safe.{judge}.jsonl",
                         [{"item_id": item, "sample_idx": 0, "battery": "xstest_safe", "judge": judge, "parse_ok": True,
                           "label": "full_refusal" if lab[i] else "full_compliance"} for i, item in keep])
    _write_jsonl(root / "exploratory" / BASE_KEY / "cpu" / "item_labels.jsonl", il)
    bank = root / "tables" / "subsets" / BASE_KEY / f"bank_{BASE_KEY}.csv"
    bank.parent.mkdir(parents=True)
    with bank.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "D2", "j1_regression", "j2_regression", "regression_status"])
        for u, (_, conf) in UNITS.items():
            w.writerow([u, d2[u], conf, conf, "confirmed" if conf else "not_flagged"])
    units = list(UNITS)
    can = {rd: F.registered_canary(il, units, rd) for rd in ("J1", "J2")}
    e2 = {"units": units, "confirmed": [u for u, (_, c) in UNITS.items() if c], "D2": d2,
          "registered": {"J1_canary": {"canary": can["J1"]},
                         "WildGuard_canary": {"canary": can["J2"], "paired_bootstrap": {"auroc_a": 1.0}}}}
    (root / "exploratory" / BASE_KEY / "cpu" / "e2.json").write_text(json.dumps(e2))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """A synthetic results tree: one refusal direction v; a unit that loses an H refusal also loses the v component
    there (so the frozen base probe tracks the unit's own decisions and the regression)."""
    root = tmp_path_factory.mktemp("dprobe_fit") / "results"
    rng = np.random.default_rng(7)
    rows = _rows()
    v = rng.standard_normal(DIM)
    v /= np.linalg.norm(v)
    labels = _toy_labels(rng, rows)
    acts = {}
    for u, lab in labels.items():
        acts[u] = {}
        for pos in F.POSITIONS:
            X = rng.standard_normal((len(rows), len(LAYERS), DIM)) * 0.5
            X += (2.5 * lab)[:, None, None] * v[None, None, :]
            acts[u][f"P@{pos}"] = X.astype(np.float16)
    _write_tree(root, rows, labels, acts, LAYERS, {u: float(rng.normal()) for u in UNITS})
    out = F.run_fit(root, BASE_KEY, n_boot=50, grid=(0.01, 1.0), quiet=True)
    return root, json.loads(out.read_text()), labels


def test_fit_measures_on_synthetic_data(synth):
    root, res, labels = synth
    assert res["units"]["labelled"] == sorted(UNITS) and res["units"]["missing_captures"] == []
    rep = res["comparators"]["reproduction"]
    assert rep["canary_J1"] and rep["canary_J2"] and rep["confirmed_matches_e2"]
    assert res["version"] == F.VERSION and res["settings"]["b_verdict_metric"] == "auroc_H_only"
    reg = {u: {"confirmed": c} for u, (_, c) in UNITS.items()}
    for rd in F.READERS:
        rr = res["readers"][rd]
        # P: 80 H (one J1 parse error dropped on the base) + the J2-labelled XSTest-safe subset + 20 harmless
        assert rr["base_labels"]["n"] == 80 - (rd == "J1") + N_XS_J2 + N_HARMLESS
        assert rr["base_labels"]["by_set"]["test_harmless"] == {"n": N_HARMLESS, "n_refusal": 0}
        for pos in F.POSITIONS:
            pr = rr["positions"][pos]
            for L in map(str, LAYERS):
                rec = pr["layers"][L]
                assert rec["a"]["auroc_heldout"] > 0.9
                # (a) on the H items alone, beside the registered all-P value (one J1 parse error: an answered H item)
                assert rec["a"]["auroc_heldout_H_only"] > 0.9
                assert (rec["a"]["n_H"], rec["a"]["n_H_refusal"]) == (80 - (rd == "J1"), 74)
                ff = rec["frozen_full_probe"]
                assert ff["note"] == F.FROZEN_NOTE and set(ff["b"]) == set(UNITS) and set(ff["c"]) == set(UNITS)
                assert ff["c_auroc"]["labelled"]["units"] == sorted(UNITS)
                assert rec["base_oof_false_refusal_rate"] <= F.FRR + 1e-12
                assert rec["c_auroc"]["labelled"]["auroc"] == 1.0  # loss of v tracks the regression exactly here
                assert len(rec["c_auroc"]["labelled"]["ci"]) == 2 and rec["c_auroc"]["e2_units"]["n"] == len(UNITS)
                assert set(rec["c_auroc"]["labelled"]["comparators"]) == {"D2", "E2_canary_J1", "E2_canary_J2"}
                for u in UNITS:
                    assert rec["b"][u]["auroc"] > 0.9
                    assert rec["d"][u]["auroc_cv"] > 0.8
                assert rec["c"]["u_reg_a"]["regression_score"] > rec["c"]["u_null_a"]["regression_score"]
            assert pr["selected_layer"] in LAYERS and pr["a"]["holds"] is False  # no layer >= 20 in the toy model
            assert pr["b"]["n_holds"] == 3 and pr["c"]["holds"] and pr["falsified"] is True  # 3 < 5 regressed units
            # (b)'s verdict names its metric (H items only by default) and reports the all-P count beside it
            assert pr["b"]["metric"] == "auroc_H_only" and pr["b"]["metric_note"] == F.B_VERDICT_NOTE
            assert {m: v["n_holds"] for m, v in pr["b"]["by_metric"].items()} == {"auroc_H_only": 3, "auroc": 3}
            assert pr["falsified_by_b_metric"] == {"auroc_H_only": True, "auroc": True}
            assert pr["a"]["H_only_best_at_or_above_20"] is None
            assert set(pr["a"]["H_only_by_layer"]) == set(map(str, LAYERS))
            sel = pr["layers"][str(pr["selected_layer"])]
            on_p = F.verdicts(pr["layers"], LAYERS, list(UNITS), reg, "auroc")
            assert on_p["b"]["metric"] == "auroc"
            assert on_p["b"]["auroc"] == {u: sel["b"][u]["auroc"] for u in on_p["b"]["auroc"]}
            with pytest.raises(ValueError, match="b_metric"):
                F.verdicts(pr["layers"], LAYERS, list(UNITS), reg, "auroc_P")
    z = np.load(root / "exploratory" / BASE_KEY / "decision_probe" / "directions.npz")
    assert z[f"J2__last__L{LAYERS[0]}__base"].shape == (DIM + 1,)
    assert res["inputs"]["item_labels"]["sha256"] and res["inputs"]["acts"]["base"]


def test_fit_refuses_mismatched_rows(synth, tmp_path):
    import shutil

    root, _, _ = synth
    alt = tmp_path / "results"
    shutil.copytree(root, alt)
    p = alt / "exploratory" / BASE_KEY / "decision_probe" / "acts" / "u_null_c.npz"
    z = dict(np.load(p))
    z["rows"] = z["rows"][::-1].copy()
    np.savez(p, **z)
    with pytest.raises(ValueError, match="rows or layers differ"):
        F.run_fit(alt, BASE_KEY, n_boot=20, grid=(1.0,), readers=("J2",), quiet=True)


# ------------------------------------------------------------------------------------------------ cross-fitting

MEMO_DIM, MEMO_LAYERS, MEMO_GRID = 512, [2], (0.01, 1.0)


@pytest.fixture(scope="module")
def memo(tmp_path_factory):
    """The review's memorising regime, H >> rows: each prompt has its own residual, shared by the base and every unit
    (plus a little unit noise), and a refusal direction v adds 3.0 where the model refuses. A unit that stops refusing a
    prompt loses v there but keeps the prompt's own residual, so an all-data base probe, which memorises each base
    prompt's label, scores that flipped item toward 'refusal' in-sample."""
    root = tmp_path_factory.mktemp("dprobe_memo") / "results"
    rng = np.random.default_rng(12)
    rows = _rows()
    v = rng.standard_normal(MEMO_DIM)
    v /= np.linalg.norm(v)
    labels = _toy_labels(rng, rows)
    ident = rng.standard_normal((len(rows), len(MEMO_LAYERS), MEMO_DIM))
    acts = {}
    for u, lab in labels.items():
        X = ident + (0.0 if u == "base" else 0.05) * rng.standard_normal(ident.shape)
        X += (3.0 * lab)[:, None, None] * v[None, None, :]
        acts[u] = {f"P@{pos}": X.astype(np.float16) for pos in F.POSITIONS}
    _write_tree(root, rows, labels, acts, MEMO_LAYERS, {u: float(rng.normal()) for u in UNITS})
    out = F.run_fit(root, BASE_KEY, n_boot=20, grid=MEMO_GRID, readers=("J2",), quiet=True)
    return root, json.loads(out.read_text()), labels


def _refit_base(root: Path, reader: str, pos: str, li: int, seed: int, grid):
    """An independent rebuild of one (reader, position, layer) cell from the files: the base's labelled rows, B0 (as
    positions in them), the nested CV and the all-data probe."""
    acts = root / "exploratory" / BASE_KEY / "decision_probe" / "acts"
    rows, _, base_acts = F.load_acts(acts / "base.npz")
    judge = root / "judge" / BASE_KEY
    il = F.read_jsonl(root / "exploratory" / BASE_KEY / "cpu" / "item_labels.jsonl")
    xs_keep = {str(r["item_id"]) for r in F.read_jsonl(judge / "base" / "labels" / "xstest_safe.J2.jsonl")}
    blab = F.unit_labels(il, judge, "base", reader, xs_keep, [k for k in rows if k[0] == F.HARMLESS])
    bkeys = [k for k in rows if k in blab]
    b_idx = np.array([rows.index(k) for k in bkeys])
    y_b = np.array([blab[k] for k in bkeys])
    b0_pos = np.array([j for j, k in enumerate(bkeys) if k[0] in F.H_BATTERIES and blab[k] == 1])
    X = base_acts[pos][:, li, :]
    nested = F.nested_cv(X[b_idx], y_b, grid, F.N_FOLDS, seed)
    C, _ = F.choose_C(X[b_idx], y_b, grid, F.N_FOLDS, seed)
    return rows, X, b_idx, b0_pos, nested, F.fit_probe(X[b_idx], y_b, C)


def test_cross_fitted_scores_on_flipped_items_and_the_base_score(memo):
    root, res, labels = memo
    rec = res["readers"]["J2"]["positions"]["last"]["layers"][str(MEMO_LAYERS[0])]
    rows, Xb, b_idx, b0_pos, nested, full = _refit_base(root, "J2", "last", 0, rec["seed"], MEMO_GRID)
    b0 = b_idx[b0_pos]
    assert rec["C"] == full.C and len(b0) == res["readers"]["J2"]["n_B0"] == 74
    # (c)'s base score is the mean of the nested CV's out-of-fold scores on B0; the all-data probe's in-sample mean
    # (kept in frozen_full_probe) sits higher, the base's memorised labels
    assert rec["base_decision_score_B0"] == pytest.approx(float(nested["oof"][b0_pos].mean()), abs=1e-12)
    in_sample = rec["frozen_full_probe"]["base_decision_score_B0"]
    assert in_sample == pytest.approx(float(full.prob(Xb[b0]).mean()), abs=1e-12)
    assert in_sample > rec["base_decision_score_B0"] + 0.02
    # on the base, cross-fitting gives the out-of-fold scores at the labelled rows and the all-data probe elsewhere
    cf_base = F.cross_fit_prob(full, nested["fold_probes"], b_idx, Xb)
    np.testing.assert_array_equal(cf_base[b_idx], nested["oof"])
    rest = np.setdiff1d(np.arange(len(rows)), b_idx)
    assert len(rest) == N_XS - N_XS_J2
    np.testing.assert_array_equal(cf_base[rest], full.prob(Xb[rest]))
    # a regressed unit on the same prompts: the all-data probe scores its flipped items (base refuses, unit answers)
    # toward 'refusal'; the cross-fitted scores come from probes that never saw those prompts
    Xu = np.load(root / "exploratory" / BASE_KEY / "decision_probe" / "acts" / "u_reg_a.npz")["P@last"][:, 0, :]
    cf, ins = F.cross_fit_prob(full, nested["fold_probes"], b_idx, Xu), full.prob(Xu)
    h = np.array([b in H_SIZES for b, _ in rows])
    flipped = np.flatnonzero(h & (labels["base"] == 1) & (labels["u_reg_a"] == 0))
    assert len(flipped) == 44
    assert np.all(cf[flipped] != ins[flipped])
    assert ins[flipped].mean() - cf[flipped].mean() > 0.05
    np.testing.assert_array_equal(cf[rest], ins[rest])  # outside the base's labelled rows nothing changes
    # the written (b) and (c) are the cross-fitted ones; frozen_full_probe keeps the in-sample ones
    y_u = labels["u_reg_a"][b_idx]  # the unit's labelled rows are the base's here (J2 labels every H item)
    assert rec["b"]["u_reg_a"]["n"] == len(b_idx)
    assert rec["b"]["u_reg_a"]["auroc"] == pytest.approx(F.auroc(y_u, cf[b_idx]), abs=1e-12)
    assert rec["b"]["u_reg_a"]["auroc_H_only"] == pytest.approx(F.auroc(y_u[h[b_idx]], cf[b_idx][h[b_idx]]), abs=1e-12)
    assert rec["frozen_full_probe"]["b"]["u_reg_a"]["auroc"] == pytest.approx(F.auroc(y_u, ins[b_idx]), abs=1e-12)
    c, cff = rec["c"]["u_reg_a"], rec["frozen_full_probe"]["c"]["u_reg_a"]
    assert c["decision_score_B0"] == pytest.approx(float(cf[b0].mean()), abs=1e-12)
    assert c["regression_score"] == pytest.approx(rec["base_decision_score_B0"] - float(cf[b0].mean()), abs=1e-12)
    assert c["predicted_refusal_rate_B0"] == pytest.approx(float((cf[b0] > rec["threshold"]).mean()), abs=1e-12)
    assert cff["decision_score_B0"] == pytest.approx(float(ins[b0].mean()), abs=1e-12)
    assert cff["decision_score_B0"] > c["decision_score_B0"]
