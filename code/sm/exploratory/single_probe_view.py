"""EXPLORATORY single-probe view of R1 and R1-C. POST HOC on Qwen3.5-9B; on Qwen3.8-27B the same view is a registered
descriptive line. Decides nothing.

Definition: docs/HYPOTHESIS_LEDGER.md, "R1-C and R1 on Qwen3.8-27B: added descriptive line" (registered at 0c3a5e5,
tightened at b6408c7, both before any 27B capture existed). Motivation (post hoc, the R1-C protocol check on 9B,
0c3a5e5, qualification (2) and the flag beside it): R1's registered cross-fitted scoring pools the scores of five
outer-fold probes whose inner-CV C differs by up to four and a half decades (a flat CV surface), so one pooled AUROC
compares items scored by differently calibrated probes. The registered pooled numbers alone decide R1 and R1-C on both
bases; nothing here changes them.

For reader (A), R1's decision probe (decision_probe_fit v2: the base's refuse-vs-answer labels, one probe per judge
reader), and reader (B), R1-C's content probe (the same fit on prompt labels; one probe), at every captured hidden
state, both positions and both judge readers, the SAME fitted probes (decision_probe_fit's folds, C selection and
fitting, through content_readers.probe_reader, the seeds as coded) are read five ways; only the scoring differs:
  pooled            the registered cross-fit (decision_probe_fit.cross_fit_prob): a row whose prompt is one of the
                    probe's rows (the base's and every unit's) is scored by the outer-fold probe that held that prompt
                    out, and the folds' scores are pooled into one AUROC; other rows carry the all-data probe's score.
  per_fold          (i) held-out prompts only: in each outer fold, that fold's own probe scores only the items whose
                    prompts the fold held out (the base's and every unit's); a unit's per-fold (b) is the AUROC over
                    those items, its (b) the mean over the folds (a fold whose items hold one class of the unit's
                    labels is undefined and left out; per-fold values are reported). The item scores are the pooled
                    ones bit for bit (checked): only the AUROC's grouping differs. Rows that are not the probe's are
                    never scored.
  standardized      (ii) each fold probe's decision value (the logit) z-scored against its own held-out base scores
                    (mean and SD over the base rows the fold held out, whatever their labels), then pooled as in the
                    registered cross-fit; no unit label is used. Rows that are not the probe's: the all-data probe,
                    z-scored against its scores on all the probe's base rows (in-sample, flagged).
  all_data          (iii) the fit's frozen all-data probe (refit on all base rows at the CV-chosen C) read directly,
                    as the fit's frozen_full_probe block (in-sample on the base-labelled rows).
  standardized_prob sensitivity, not part of the line: (ii) on the probability scale.
  (iv) the inner-CV C chosen in each outer fold, and the all-data C, per cell.
For every variant: the base's (a); per-unit (b) on the H items; the count at >= 0.75 and the median over R1's defined
units; (c) through decision_probe_fit.c_auroc unchanged, on the variant's footing (per_fold: one per fold, and the
mean). The R1-C delta b(A) - max(b(B), b(C)) at the registered Setting D cells (last prompt token, chosen on the
registered pooled (a) and held fixed), under every variant, (C) the registered in-sample harm direction throughout,
with R1-C's paired stratified bootstrap (2000 draws, seed 0) recomputing every reader's statistic on the variant's
footing in each draw; content_readers.verdict's arithmetic beside it. Descriptive.
Gate (a failure stops the run and nothing is written): reader (A) pooled must reproduce probe_results.json
(content_readers.reproduce, tolerance 1e-9) and also the fit's per-fold C, per-fold and fold-mean (a), in-sample (a),
(b), and its frozen (c) block with its AUROC; when content_readers.json exists, (B)'s pooled records, (C)'s (a) and
(b), the headline cells, the pooled deltas with their intervals and the input hashes must reproduce it.
Inputs (read only; numbers and labels, no text; nothing under a restricted tree): as content_readers, plus
content_readers.json when present. Output: <results>/exploratory/<base>/decision_probe/single_probe_view.json
(numbers and hashes only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sm.analysis import stats
from sm.exploratory import content_readers as CR
from sm.exploratory import decision_probe_fit as F

LABEL = ("POST HOC on 9B: single-probe view of R1 and R1-C; on 27B the same view is a registered descriptive line "
         "(b6408c7)")
VERSION = "single-probe-view-v1-2026-10-20"
DEFINITION = {"doc": "docs/HYPOTHESIS_LEDGER.md", "section": "R1-C and R1 on Qwen3.8-27B: added descriptive line",
              "registered": "0c3a5e5", "tightened": "b6408c7",
              "motivation": "docs/HYPOTHESIS_LEDGER.md, 'R1-C on Qwen3.5-9B: result', qualification (2) and the flag "
                            "for an independent check (0c3a5e5)"}
STATUS = {
    "qwen3.5-9b": "POST HOC: the view was defined after the 9B pooled numbers and the protocol check's diagnosis were "
                  "known; descriptive; decides nothing; R1's and R1-C's registered verdicts stand",
    "qwen3.8-27b": "registered descriptive line (0c3a5e5, tightened at b6408c7, before any 27B capture existed); "
                   "decides nothing; the registered pooled numbers alone decide the 27B R1 and R1-C verdicts; the line "
                   "names the last-prompt-token cells, the first generated token is reported beyond it",
}
VARIANTS = ("pooled", "per_fold", "standardized", "all_data")
SENSITIVITY = ("standardized_prob",)
ALL_VARIANTS = VARIANTS + SENSITIVITY
SCALES = ("logit", "prob")
STANDARDIZED_SCALE = {"standardized": "logit", "standardized_prob": "prob"}
VARIANT_NOTES = {
    "pooled": "the registered cross-fit: each prompt's rows scored by the outer-fold probe that held it out, the folds' "
              "scores pooled into one AUROC (decision_probe_fit.cross_fit_prob)",
    "per_fold": "(i) held-out prompts only: each fold's probe scores only the items whose prompts it held out; per-unit "
                "(b) = the mean over the folds of the within-fold AUROC",
    "standardized": "(ii) each fold probe's logit z-scored against its own held-out base scores, then pooled as "
                    "registered; no unit label used",
    "all_data": "(iii) the fit's frozen all-data probe read directly (in-sample on base-labelled rows)",
    "standardized_prob": "sensitivity, not part of the line: (ii) on the probability scale",
}
JUDGES = CR.JUDGES
HEADLINE_POSITION = CR.HEADLINE_POSITION
B_METRIC = F.B_VERDICT_METRIC
INTERPRETATIONS = [
    "the fitted probes are R1's and R1-C's own: decision_probe_fit's folds, C selection and fitting through "
    "content_readers.probe_reader, seeds as coded ((A) seed + 1000 r + 100 p + l; (B) the WildGuard reader's, r = 0); "
    "only the scoring differs",
    "per_fold: each fold's probe scores every row it held out exactly as decision_probe_fit.cross_fit_prob does "
    "(checked bit for bit against the pooled scores); a unit's value is the mean of the within-fold AUROCs over the "
    "folds whose items hold both classes of its labels; the base's per-fold (a) is the fit's own per-fold AUROC and "
    "its mean the fit's auroc_fold_mean (checked)",
    "rows whose prompt is not one of the probe's rows (for (A): prompts whose base label is unusable under the judge "
    "reader; none on 9B) are never scored in per_fold; in pooled and standardized they carry the all-data probe's "
    "score",
    "standardized: z = (logit - m_f) / s_f, with m_f and s_f the mean and SD (ddof 0) of fold f's probe's logits on "
    "every base row the fold held out (any label); the all-data probe's m and s come from its logits on all the "
    "probe's base rows (in-sample, flagged); standardized_prob does the same on the predicted probability "
    "(sensitivity)",
    "all_data: the fit's frozen_full_probe reading; on base-labelled rows it is in-sample (decision_probe_fit's "
    "FROZEN_NOTE: a unit can inherit the base's memorised label on the same prompt)",
    "(c) per variant through decision_probe_fit.c_auroc unchanged, its records from decision_probe_fit.c_scores on the "
    "variant's footing: per_fold, one (c) per fold over the B0 items that fold held out (base score: the base's mean "
    "on the same items; threshold: R1's 5 percent rule on the fold's held-out answered base rows) and the mean of the "
    "per-fold AUROCs; standardized, the z-scores (threshold: R1's rule on the base's z-scored answered rows); "
    "all_data, the fit's frozen block (the registered out-of-fold threshold); the c_auroc seed is the cell's",
    "the R1-C delta under every variant: b(A) - max(b(B), b(C)) per defined unit at the registered Setting D cells "
    "(content_readers.headline_layer on the registered pooled (a), held fixed across variants); (C) is the registered "
    "in-sample harm-direction projection under every variant (it has no folds); interval: R1-C's paired stratified "
    "bootstrap (2000 draws, seed 0; the unit's H items resampled within each class of its labels) with every reader's "
    "statistic recomputed in each draw on the variant's footing (per_fold: the fold means); for pooled this is "
    "content_readers.paired_delta_bootstrap value for value (checked against content_readers.json); "
    "content_readers.verdict applies the registered arithmetic, which decides nothing here",
    "summaries over R1's defined units (content_readers.defined_units: R1's regressed set restricted to units whose "
    "usable H labels under the judge reader hold both classes; 9B: 7, 27B: 3); the count at >= 0.75 counts defined "
    "values; (b) reads the H items only, R1's verdict metric",
    "(B) is one probe for both judge readers; its records are read against each judge reader's labels",
]


class ReproductionError(CR.ReproductionError):
    """A registered number did not reproduce; nothing is written."""


def _sha_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_for(base: str) -> str:
    return STATUS.get(base, "descriptive; decides nothing (no registered status for this base)")

# --------------------------------------------------------------------------------------------- scoring (pure)


def _score(probe: Any, X: np.ndarray, scale: str) -> np.ndarray:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    return np.asarray(probe.decision(X) if scale == "logit" else probe.prob(X), dtype=np.float64)


def fold_index(r: Mapping[str, Any], n_rows: int) -> np.ndarray:
    """The outer fold of the probe's nested CV that holds out each row of P; -1 for rows that are not the probe's."""
    out = np.full(int(n_rows), -1, dtype=int)
    ridx = np.asarray(r["idx"], dtype=int)
    for f, (te, _) in enumerate(r["nested"]["fold_probes"]):
        out[ridx[np.asarray(te, dtype=int)]] = f
    return out


def held_out_scores(r: Mapping[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(i)'s scoring of the rows of X (one model's states on P): each outer fold's probe scores only the rows whose
    prompts that fold held out, with the same call decision_probe_fit.cross_fit_prob makes for them; rows that are not
    the probe's are never scored (score NaN, fold -1) and the all-data probe is never used. Returns (fold, score)."""
    X = np.asarray(X)
    fold = np.full(X.shape[0], -1, dtype=int)
    s = np.full(X.shape[0], np.nan)
    ridx = np.asarray(r["idx"], dtype=int)
    for f, (te, pf) in enumerate(r["nested"]["fold_probes"]):
        held = ridx[np.asarray(te, dtype=int)]
        s[held] = pf.prob(X[held])
        fold[held] = f
    return fold, s


def per_fold_auroc(y: Sequence, fold: Sequence[int], s: Sequence[float], n_folds: int) -> dict[str, Any]:
    """Within each fold, the AUROC of s for y over the items that fold scored (None when they hold one class); the mean
    over the defined folds. Items no fold scored (fold -1) are left out."""
    y = np.asarray(y).astype(int).ravel()
    fold = np.asarray(fold, dtype=int).ravel()
    s = np.asarray(s, dtype=float).ravel()
    vals: list[float | None] = []
    ns, npos = [], []
    for f in range(int(n_folds)):
        m = fold == f
        ns.append(int(m.sum()))
        npos.append(int(y[m].sum()))
        vals.append(F.auroc(y[m], s[m]) if m.any() else None)
    ok = [v for v in vals if v is not None]
    return {"mean": float(np.mean(ok)) if ok else None, "folds": vals, "n_defined_folds": len(ok), "n_per_fold": ns,
            "n_pos_per_fold": npos, "n_unscored": int((fold < 0).sum())}


def fold_standardization(r: Mapping[str, Any], Xb: np.ndarray, scale: str = "logit") -> dict[str, Any]:
    """(ii)'s location and spread, from the base's states only and no label: for each outer-fold probe, the mean and SD
    (ddof 0) of its scores on the base rows it held out; for the all-data probe (which scores rows that are not the
    probe's), of its scores on all the probe's base rows (in-sample: it trained on them)."""
    Xb = np.asarray(Xb)
    ridx = np.asarray(r["idx"], dtype=int)
    folds = []
    for te, pf in r["nested"]["fold_probes"]:
        v = _score(pf, Xb[ridx[np.asarray(te, dtype=int)]], scale)
        folds.append([float(v.mean()), float(v.std())])
    v = _score(r["probe"], Xb[ridx], scale)
    full = [float(v.mean()), float(v.std())]
    for mu, sd in folds + [full]:
        if not (math.isfinite(mu) and math.isfinite(sd) and sd > 0):
            raise ValueError(f"a probe's base scores have no spread on the {scale} scale: cannot standardize")
    return {"scale": scale, "folds": folds, "all_data": full,
            "all_data_footing": "in-sample (the all-data probe trained on all the probe's base rows)"}


def standardized_scores(r: Mapping[str, Any], X: np.ndarray, stdz: Mapping[str, Any]) -> np.ndarray:
    """(ii)'s score of every row of X: the probe that scores the row in the registered cross-fit (the outer-fold probe
    that held its prompt out; for rows that are not the probe's, the all-data probe), z-scored with that probe's
    location and spread from fold_standardization."""
    X = np.asarray(X)
    scale = stdz["scale"]
    mu, sd = stdz["all_data"]
    z = (_score(r["probe"], X, scale) - mu) / sd
    ridx = np.asarray(r["idx"], dtype=int)
    for (te, pf), (mu_f, sd_f) in zip(r["nested"]["fold_probes"], stdz["folds"]):
        held = ridx[np.asarray(te, dtype=int)]
        z[held] = (_score(pf, X[held], scale) - mu_f) / sd_f
    return z


def pooled_stat(y: Sequence, s: Sequence[float]) -> Callable[[np.ndarray], float]:
    """The AUROC over the drawn items (indices into y and s)."""
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=float).ravel()
    return lambda idx: float(stats.auroc(y[idx], s[idx]))


def per_fold_stat(y: Sequence, fold: Sequence[int], s: Sequence[float], n_folds: int) -> Callable[[np.ndarray], float]:
    """The mean within-fold AUROC over the drawn items (folds with one class in the draw are left out; NaN if none)."""
    y = np.asarray(y).astype(int).ravel()
    fold = np.asarray(fold, dtype=int).ravel()
    s = np.asarray(s, dtype=float).ravel()

    def stat(idx: np.ndarray) -> float:
        yy, ff, ss = y[idx], fold[idx], s[idx]
        vals = []
        for f in range(int(n_folds)):
            m = ff == f
            if m.any():
                v = F.auroc(yy[m], ss[m])
                if v is not None:
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    return stat


def paired_delta_bootstrap_stats(y: Sequence, stat_a: Callable[[np.ndarray], float],
                                 stat_others: Sequence[Callable[[np.ndarray], float]], n_boot: int = CR.N_BOOT,
                                 seed: int = CR.BOOT_SEED, alpha: float = CR.ALPHA) -> dict[str, Any]:
    """delta = stat_a - max_k stat_others[k] on the same items, with R1-C's paired percentile bootstrap: every draw
    resamples the items within each class of y (content_readers.paired_delta_bootstrap's 'stratified' draw, the same
    generator calls) and recomputes every statistic on that draw. With pooled_stat statistics it is that function value
    for value; a draw where a statistic is undefined is counted and left out."""
    y = np.asarray(y).astype(bool).ravel()
    full = np.arange(y.size)
    b_a = stat_a(full)
    b_o = [st(full) for st in stat_others]
    point = b_a - max(b_o)
    pos_idx, neg_idx = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, size=pos_idx.size, replace=True),
                              rng.choice(neg_idx, size=neg_idx.size, replace=True)])
        a = stat_a(idx)
        o = [st(idx) for st in stat_others]
        boot[i] = np.nan if (math.isnan(a) or any(math.isnan(v) for v in o)) else a - max(o)
    ok = boot[~np.isnan(boot)]
    lo, hi = (float(v) for v in np.quantile(ok, [alpha / 2.0, 1.0 - alpha / 2.0])) if ok.size else (None, None)
    return {"delta": point, "b_a": b_a, "b_others": b_o, "ci": [lo, hi], "n_boot": int(n_boot), "seed": int(seed),
            "scheme": "stratified", "alpha": alpha, "n_boot_undefined": int(np.isnan(boot).sum()),
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def _b_value(x: Any) -> float | None:
    return x.get("mean") if isinstance(x, Mapping) else x


def b_summary(values: Mapping[str, Any], units: Sequence[str]) -> dict[str, Any]:
    """R1's (b) summary over the given units: the count at >= 0.75 and the median of the defined values."""
    d = [v for v in (_b_value(values.get(u)) for u in units) if v is not None]
    return {"n_ge_075": int(sum(1 for v in d if v >= F.B_MIN)), "n_defined": len(d), "n_units": len(units),
            "median": float(statistics.median(d)) if d else None}

# --------------------------------------------------------------------------------------------- one cell, five readings


def view_cell(r: Mapping[str, Any], jd: Any, Xb: np.ndarray, ux: Mapping[str, np.ndarray], pooled: Mapping[str, Any],
              ctx: Mapping[str, Any], seed_c: int, defined: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One fitted probe reader ((A) or (B)) at one cell under one judge reader, read the five ways. pooled: the
    registered record ("rec", content_readers.evaluate's), the cross-fitted unit scores ("scores") and the base's
    out-of-fold scores on jd.b_idx ("base_oof"). Returns the record and the item scores (rows of P) the delta needs."""
    Xb = np.asarray(Xb)
    n = int(Xb.shape[0])
    nf = len(r["nested"]["fold_probes"])
    fidx = fold_index(r, n)
    full = r["probe"]
    rec_p = pooled["rec"]
    jb = np.asarray(jd.b_idx, dtype=int)
    stdz = {v: fold_standardization(r, Xb, sc) for v, sc in STANDARDIZED_SCALE.items()}

    # the base on every footing
    fb, s_base = held_out_scores(r, Xb)
    if not np.array_equal(fb, fidx):
        raise RuntimeError("held-out scoring and the fold index disagree")
    if not np.array_equal(s_base[jb], np.asarray(pooled["base_oof"], float)):
        raise RuntimeError("the base's held-out scores differ from the nested CV's out-of-fold scores")
    z_base = {v: standardized_scores(r, Xb, sz) for v, sz in stdz.items()}
    a = {"pooled": rec_p["a"]["auroc"], "pooled_H_only": rec_p["a"]["auroc_H_only"],
         "per_fold": per_fold_auroc(jd.y_b, fidx[jb], s_base[jb], nf),
         **{v: F.auroc(jd.y_b, z_base[v][jb]) for v in stdz},
         "all_data_in_sample": F.auroc(jd.y_b, full.decision(Xb[jb]))}

    # the units: item scores on every footing, (b) on the H items
    items: dict[str, dict[str, np.ndarray]] = {}
    b: dict[str, dict[str, Any]] = {v: {} for v in ALL_VARIANTS}
    for u, X in ux.items():
        f_u, s_ho = held_out_scores(r, X)
        s_pool = np.asarray(pooled["scores"][u], float)
        held = f_u >= 0
        if not np.array_equal(s_ho[held], s_pool[held]):
            raise RuntimeError(f"{u}: held-out scores differ from the registered cross-fit's")
        arr = {"pooled": s_pool, "per_fold": s_ho, "all_data": np.asarray(full.prob(X), float)}
        for v, sz in stdz.items():
            arr[v] = standardized_scores(r, X, sz)
        items[u] = arr
        if u in jd.urows:
            idx_u, y_u, h_u = jd.urows[u]
            rows_h, y_h = idx_u[h_u], y_u[h_u]
            for v in ALL_VARIANTS:
                b[v][u] = per_fold_auroc(y_h, fidx[rows_h], s_ho[rows_h], nf) if v == "per_fold" \
                    else F.auroc(y_h, arr[v][rows_h])
            if b["pooled"][u] != rec_p["b"][u][B_METRIC]:
                raise RuntimeError(f"{u}: pooled (b) differs from the registered record")

    # (c) through decision_probe_fit.c_auroc, unchanged, on every footing
    b0 = np.asarray(jd.b0, dtype=int)

    def cauc(recs: Mapping[str, dict]) -> dict[str, Any]:
        return F.c_auroc(recs, ctx["labelled"], ctx["e2_units"], ctx["reg"], ctx["d2"], ctx["canary"], ctx["n_boot"],
                         seed_c)

    def reg_scores(recs: Mapping[str, dict]) -> dict[str, float]:
        return {u: v["regression_score"] for u, v in recs.items()}

    c: dict[str, Any] = {"pooled": {"threshold": rec_p["threshold"], "base_decision_score_B0": rec_p["base_decision_score_B0"],
                                    "regression_score": reg_scores(rec_p["c"]), "c_auroc": rec_p["c_auroc"]}}
    per = []
    for f in range(nf):
        rows_f = b0[fidx[b0] == f]
        ans_f = jb[(np.asarray(jd.y_b) == 0) & (fidx[jb] == f)]
        if not rows_f.size or not ans_f.size:
            per.append({"fold": f, "n_B0": int(rows_f.size), "c_auroc": None})
            continue
        thr_f, _ = F.threshold_at_frr(s_base[ans_f])
        base_f = float(s_base[rows_f].mean())
        recs = {u: F.c_scores(items[u]["per_fold"], rows_f, base_f, thr_f) for u in items}
        per.append({"fold": f, "n_B0": int(rows_f.size), "threshold": thr_f, "base_decision_score_B0": base_f,
                    "regression_score": reg_scores(recs), "c_auroc": cauc(recs)})

    def fold_mean(grp: str) -> tuple[list, float | None]:
        vals = [(p["c_auroc"] or {}).get(grp, {}).get("auroc") if p["c_auroc"] else None for p in per]
        ok = [v for v in vals if v is not None]
        return vals, (float(np.mean(ok)) if ok else None)

    lab_f, lab_m = fold_mean("labelled")
    e2_f, e2_m = fold_mean("e2_units")
    c["per_fold"] = {"folds": per, "labelled_auroc_by_fold": lab_f, "labelled_auroc_mean": lab_m,
                     "e2_units_auroc_by_fold": e2_f, "e2_units_auroc_mean": e2_m}
    for v in stdz:
        zb = z_base[v]
        thr_v, _ = F.threshold_at_frr(zb[jb][np.asarray(jd.y_b) == 0])
        base_v = float(zb[b0].mean())
        recs = {u: F.c_scores(items[u][v], b0, base_v, thr_v) for u in items}
        c[v] = {"threshold": thr_v, "base_decision_score_B0": base_v, "regression_score": reg_scores(recs),
                "c_auroc": cauc(recs)}
    base_all = float(full.prob(Xb[b0]).mean())
    recs = {u: F.c_scores(items[u]["all_data"], b0, base_all, rec_p["threshold"]) for u in items}
    c["all_data"] = {"threshold": rec_p["threshold"], "threshold_footing": "the registered out-of-fold threshold, as the "
                                                                           "fit's frozen block",
                     "base_decision_score_B0": base_all, "c": recs, "c_auroc": cauc(recs)}

    summary = {}
    for v in ALL_VARIANTS:
        s = b_summary(b[v], defined)
        if v == "per_fold":
            s["c_labelled"], s["c_e2_units"] = lab_m, e2_m
        else:
            s["c_labelled"] = c[v]["c_auroc"]["labelled"].get("auroc")
            s["c_e2_units"] = c[v]["c_auroc"]["e2_units"].get("auroc")
        summary[v] = s
    folds_info = r["nested"]["folds"]
    Cs = [float(fi["C"]) for fi in folds_info]
    edges = (min(ctx["grid"]), max(ctx["grid"]))
    fit_rec = {"seed": int(r["seed"]), "n": int(len(r["y"])), "n_pos": int(np.sum(r["y"])), "C_all_data": float(r["C"]),
               "C_all_data_at_grid_edge": float(r["C"]) in edges, "fold_C": Cs,
               "fold_C_at_grid_edge": [C in edges for C in Cs],
               "fold_C_span_decades": float(math.log10(max(Cs) / min(Cs))) if Cs else None,
               "fold_n_test": [int(fi["n_test"]) for fi in folds_info],
               "fold_auroc_on_fit_labels": [fi["auroc"] for fi in folds_info]}
    rec = {"fit": fit_rec, "standardization": stdz, "a": a, "b": b, "summary": summary, "c": c}
    return rec, {"fold": fidx, "n_folds": nf, "items": items}

# --------------------------------------------------------------------------------------------- the gate


def check_fit(VA: Mapping, fit: Mapping[str, Any], layers: Sequence[int]) -> dict[str, Any]:
    """Beyond content_readers.reproduce: (A)'s per-fold C, per-fold (a) and their mean, in-sample (a), pooled and
    all-data (b) and the all-data (c) block (records and AUROC) against the fit's probe_results.json."""
    cmp = CR._Cmp(CR.REPRO_TOL)
    for rd in JUDGES:
        for pos in F.POSITIONS:
            for L in layers:
                w = f"{rd}/{pos}/L{L}"
                th = fit["readers"][rd]["positions"][pos]["layers"][str(L)]
                mine = VA[rd][pos][L]
                tf = th["a"]["folds"]
                cmp(f"{w}/a/n_folds", len(mine["fit"]["fold_C"]), len(tf))
                for i, fo in enumerate(tf):
                    cmp(f"{w}/a/folds/{i}/C", mine["fit"]["fold_C"][i] if i < len(mine["fit"]["fold_C"]) else None, fo["C"])
                    cmp(f"{w}/a/folds/{i}/auroc", (mine["a"]["per_fold"]["folds"] + [None] * len(tf))[i], fo["auroc"])
                    cmp(f"{w}/a/folds/{i}/n_test", (mine["fit"]["fold_n_test"] + [None] * len(tf))[i], fo["n_test"])
                cmp(f"{w}/a/auroc_fold_mean", mine["a"]["per_fold"]["mean"], th["a"]["auroc_fold_mean"])
                cmp(f"{w}/a/auroc_in_sample", mine["a"]["all_data_in_sample"], th["a"]["auroc_in_sample"])
                cmp(f"{w}/C", mine["fit"]["C_all_data"], th["C"])
                for u in sorted(set(mine["b"]["pooled"]) | set(th["b"])):
                    cmp(f"{w}/b/{u}", mine["b"]["pooled"].get(u), (th["b"].get(u) or {}).get(B_METRIC))
                tff = th["frozen_full_probe"]
                for u in sorted(set(mine["b"]["all_data"]) | set(tff["b"])):
                    cmp(f"{w}/frozen/b/{u}", mine["b"]["all_data"].get(u), (tff["b"].get(u) or {}).get(B_METRIC))
                ca = mine["c"]["all_data"]
                cmp(f"{w}/frozen/base_decision_score_B0", ca["base_decision_score_B0"], tff["base_decision_score_B0"])
                for u in sorted(set(ca["c"]) | set(tff["c"])):
                    for k in ("decision_score_B0", "regression_score", "predicted_refusal_rate_B0"):
                        cmp(f"{w}/frozen/c/{u}/{k}", (ca["c"].get(u) or {}).get(k), (tff["c"].get(u) or {}).get(k))
                for grp in ("labelled", "e2_units"):
                    m, t = ca["c_auroc"][grp], tff["c_auroc"][grp]
                    cmp(f"{w}/frozen/c_auroc/{grp}/units", m.get("units"), t.get("units"))
                    cmp(f"{w}/frozen/c_auroc/{grp}/auroc", m.get("auroc"), t.get("auroc"))
                    if t.get("ci") is not None or m.get("ci") is not None:
                        cmp(f"{w}/frozen/c_auroc/{grp}/ci", m.get("ci"), t.get("ci"))
                    for name, cv in (t.get("comparators") or {}).items():
                        mc = (m.get("comparators") or {}).get(name)
                        for k in ("auroc", "diff_probe_minus", "diff_ci"):
                            cmp(f"{w}/frozen/c_auroc/{grp}/{name}/{k}", (mc or {}).get(k), (cv or {}).get(k))
    return {**cmp.summary(), "what": "(A)'s per-fold C, per-fold (a) and auroc_fold_mean, in-sample (a), all-data C, "
                                     "pooled and all-data (b), all-data (c) records and (c) AUROC against "
                                     "probe_results.json"}


def check_content_readers(cr: Mapping[str, Any], inputs: Mapping[str, Any], headline: Mapping, regB: Mapping,
                          C_a: Mapping, C_b: Mapping, delta_pooled: Mapping, layers: Sequence[int]) -> dict[str, Any]:
    """(B)'s pooled records at every cell, (C)'s (a) at every last-prompt-token layer and (b) at its headline cell, the
    headline cells, the pooled deltas with their intervals and the input hashes against R1-C's content_readers.json."""
    cmp = CR._Cmp(CR.REPRO_TOL)
    ci_ = cr.get("inputs") or {}
    for k in ("item_labels", "bank_csv", "e2_json", "probe_results", "directions"):
        cmp(f"inputs/{k}", (inputs.get(k) or {}).get("sha256"), (ci_.get(k) or {}).get("sha256"))
    acts_cr = ci_.get("acts") or {}
    for u in sorted(set(inputs["acts"]) | set(acts_cr)):
        cmp(f"inputs/acts/{u}", inputs["acts"].get(u), acts_cr.get(u))
    for rd in JUDGES:
        for R in ("A", "B", "C"):
            cmp(f"headline/{rd}/{R}", headline[rd][R]["layer"], cr["headline_cells"][rd][R]["layer"])
        for pos in F.POSITIONS:
            for L in layers:
                w = f"{rd}/{pos}/L{L}/B"
                mine, th = regB[rd][pos][L], cr["cells"][rd][pos][str(L)]["B"]
                for k in ("C", "seed", "threshold", "base_decision_score_B0"):
                    cmp(f"{w}/{k}", mine.get(k), th.get(k))
                for k in ("auroc", "auroc_H_only"):
                    cmp(f"{w}/a/{k}", mine["a"].get(k), th["a"].get(k))
                if sorted(mine["b"]) != sorted(th["b"]):
                    cmp.fails.append({"where": f"{w}/b/units", "mine": sorted(mine["b"]), "fit": sorted(th["b"])})
                for u in sorted(set(mine["b"]) & set(th["b"])):
                    for k in CR.A_FIELDS:
                        cmp(f"{w}/b/{u}/{k}", mine["b"][u].get(k), th["b"][u].get(k))
                for grp in ("labelled", "e2_units"):
                    cmp(f"{w}/c_auroc/{grp}/auroc", mine["c_auroc"][grp].get("auroc"), th["c_auroc"][grp].get("auroc"))
        for L in layers:
            cmp(f"{rd}/last/L{L}/C/a", C_a[rd][L], cr["cells"][rd][HEADLINE_POSITION][str(L)]["C"]["a"]["auroc"])
        LC = headline[rd]["C"]["layer"]
        cb = cr["cells"][rd][HEADLINE_POSITION][str(LC)]["C"]["b"]
        for u in sorted(set(C_b[rd][LC]) | set(cb)):
            cmp(f"{rd}/last/L{LC}/C/b/{u}", C_b[rd][LC].get(u), (cb.get(u) or {}).get(B_METRIC))
        per, reg = delta_pooled[rd]["per_unit"], cr["comparison"][rd]["per_unit"]
        cmp(f"{rd}/comparison/units", sorted(per), sorted(reg))
        for u in sorted(set(per) & set(reg)):
            for k in ("b_A", "b_B", "b_C", "delta", "ci"):
                cmp(f"{rd}/comparison/{u}/{k}", per[u][k], reg[u][k])
    return {"available": True, **cmp.summary(),
            "what": "(B)'s pooled records ((a), every per-unit (b) field, C, seed, threshold, base (c) score, (c) "
                    "AUROC) at every cell, (C)'s (a) and headline (b), the headline cells, the pooled deltas and their "
                    "intervals, and the input hashes, against content_readers.json"}

# --------------------------------------------------------------------------------------------- main computation


def _paths(results: Path, base: str, over: Mapping[str, Any]) -> dict[str, Path]:
    safe = CR.safe_name(base)
    x = results / "exploratory" / safe
    root = Path(over["root"]) if over.get("root") else x / "decision_probe"
    p = {"root": root, "acts_dir": root / "acts", "item_labels": x / "cpu" / "item_labels.jsonl",
         "judge_dir": results / "judge" / safe, "bank_csv": results / "tables" / "subsets" / safe / f"bank_{safe}.csv",
         "e2_json": x / "cpu" / "e2.json", "probe_results": root / "probe_results.json",
         "directions": root / "directions.npz", "content_readers": root / "content_readers.json",
         "out": root / "single_probe_view.json"}
    for k in ("acts_dir", "item_labels", "judge_dir", "bank_csv", "e2_json", "probe_results", "directions",
              "content_readers", "out"):
        if over.get(k):
            p[k] = Path(over[k])
    p["capture_json"] = Path(over["capture_json"]) if over.get("capture_json") else p["acts_dir"] / f"{F.BASE}.json"
    CR._guard(results, "results")
    for k, v in p.items():
        CR._guard(v, k)
    return p


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(f"[single-probe-view] {msg}", flush=True)


def _hash_or_none(path: Path) -> dict[str, str] | None:
    return {"path": str(path), "sha256": _sha_file(path)} if path.is_file() else None


def compute(results: Path | str, base: str, *, n_boot: int = CR.N_BOOT, seed: int = F.SEED,
            grid: Sequence[float] = F.C_GRID, rule_key: str | None = None, expect_defined: Sequence[str] | None = None,
            n_layers: int | None = None, reproduce_only: bool = False, quiet: bool = False,
            **paths: Any) -> dict[str, Any]:
    """Everything the view reports, as a dict (nothing is written). Raises ReproductionError when a registered number
    does not reproduce."""
    t0 = time.time()
    results = Path(results)
    p = _paths(results, base, paths)
    for k in ("item_labels", "bank_csv", "probe_results"):
        if not p[k].is_file():
            raise FileNotFoundError(f"{k}: {p[k]} missing")
    if not (p["acts_dir"] / f"{F.BASE}.npz").is_file():
        raise FileNotFoundError(f"no base capture under {p['acts_dir']}")
    item_labels = F.read_jsonl(p["item_labels"])
    CR._check_item_labels(item_labels)
    reg = F.regression_table(p["bank_csv"])
    e2 = json.loads(p["e2_json"].read_text()) if p["e2_json"].is_file() else None
    fit = json.loads(p["probe_results"].read_text())
    cr_out = json.loads(p["content_readers"].read_text()) if p["content_readers"].is_file() else None
    inputs: dict[str, Any] = {
        "item_labels": _hash_or_none(p["item_labels"]), "bank_csv": _hash_or_none(p["bank_csv"]),
        "e2_json": _hash_or_none(p["e2_json"]) if e2 is not None else None,
        "probe_results": _hash_or_none(p["probe_results"]), "directions": _hash_or_none(p["directions"]),
        "content_readers": _hash_or_none(p["content_readers"]) if cr_out is not None else None,
        "capture_json": None, "acts": {}, "xstest_safe_labels": {}}

    rows, layers, base_acts = F.load_acts(p["acts_dir"] / f"{F.BASE}.npz")
    inputs["acts"][F.BASE] = _sha_file(p["acts_dir"] / f"{F.BASE}.npz")
    if n_layers is None:
        if not p["capture_json"].is_file():
            raise FileNotFoundError(f"{p['capture_json']} missing: pass n_layers")
        cap = json.loads(p["capture_json"].read_text())
        inputs["capture_json"] = _hash_or_none(p["capture_json"])
        n_layers = int(cap["n_layers"])
        if [int(x) for x in cap.get("layers", layers)] != layers:
            raise ValueError("the capture record's layers differ from the capture's")
    deep = CR.deep_layers(layers, n_layers)
    if not deep:
        raise ValueError(f"no captured layer at or beyond 2n/3 (n = {n_layers}, layers {layers})")
    index = {k: i for i, k in enumerate(rows)}
    harmless_keys = [k for k in rows if k[0] == F.HARMLESS]
    judge_dir = p["judge_dir"]
    xs_base = judge_dir / F.BASE / "labels" / f"{F.XS}.J2.jsonl"
    xs_keep = {str(r["item_id"]) for r in F.read_jsonl(xs_base)} if xs_base.is_file() else set()
    units = sorted(q.stem for q in p["acts_dir"].glob("*.npz") if q.stem != F.BASE and not q.stem.endswith(".tmp"))
    labelled_runs = sorted({str(r["run_id"]) for r in item_labels} - {F.BASE})
    labelled = [u for u in labelled_runs if reg.get(u, {}).get("confirmed") is not None]
    e2_units = list(e2["units"]) if e2 is not None else []
    for u in [F.BASE] + labelled_runs:
        for rd in JUDGES:
            q = judge_dir / u / "labels" / f"{F.XS}.{rd}.jsonl"
            inputs["xstest_safe_labels"][f"{u}.{rd}"] = _sha_file(q) if q.is_file() else None
    comp_units = sorted(set(labelled) | set(e2_units))
    canary = {rd: F.registered_canary(item_labels, comp_units, rd) for rd in ("J1", "J2")}
    d2 = {u: reg.get(u, {}).get("D2") for u in comp_units}
    J = {rd: CR.judge_labels(item_labels, judge_dir, rows, index, harmless_keys, xs_keep, units, rd) for rd in JUDGES}
    regressed = CR.regressed_units(e2, labelled, reg)
    defined = {rd: CR.defined_units(regressed, J[rd]) for rd in JUDGES}
    undefined = {rd: [u for u in regressed if u not in defined[rd]] for rd in JUDGES}
    if expect_defined is not None and sorted(expect_defined) != sorted(defined[CR.PRIMARY_JUDGE]):
        raise ValueError(f"defined units {defined[CR.PRIMARY_JUDGE]} differ from the expected {sorted(expect_defined)}")
    rule_key = rule_key or base
    rule = CR.RULES.get(rule_key)
    _log(quiet, f"{base}: n = {n_layers}, layers {layers}; {len(units)} captured units; defined "
                f"{defined[CR.PRIMARY_JUDGE]}; R1-C rule {rule_key if rule else 'none'}; content_readers.json "
                f"{'present' if cr_out is not None else 'absent'}")

    unit_acts: dict[str, dict[str, np.ndarray]] = {}
    for u in units:
        r_u, l_u, a_u = F.load_acts(p["acts_dir"] / f"{u}.npz")
        if r_u != rows or l_u != layers:
            raise ValueError(f"{u}: rows or layers differ from the base capture")
        unit_acts[u] = a_u
        inputs["acts"][u] = _sha_file(p["acts_dir"] / f"{u}.npz")
    settings_used = {"C_grid": list(grid), "n_folds": F.N_FOLDS, "seed": seed, "n_boot": n_boot, "frr": F.FRR,
                     "layers": layers, "positions": list(F.POSITIONS)}
    ctx: dict[str, Any] = {"labelled": labelled, "e2_units": e2_units, "reg": reg, "d2": d2, "canary": canary,
                           "n_boot": n_boot, "inputs": inputs, "units": units, "e2": e2, "defined": defined,
                           "grid": tuple(float(x) for x in grid), "settings_used": settings_used}

    def cell_seed(ri: int, pi: int, li: int) -> int:
        return seed + 1000 * ri + 100 * pi + li  # decision_probe_fit.run_fit

    def cells_of(pos: str, li: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return base_acts[pos][:, li, :], {u: unit_acts[u][pos][:, li, :] for u in units}

    # ---- pass 1: reader (A), R1's fit exactly as content_readers rebuilds it, then the registered gate
    A: dict[str, dict[str, dict[int, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    for rd in JUDGES:
        jd = J[rd]
        for pi, pos in enumerate(F.POSITIONS):
            for li, L in enumerate(layers):
                s = cell_seed(jd.ri, pi, li)
                Xb, ux = cells_of(pos, li)
                r = CR.probe_reader(Xb, jd.b_idx, jd.y_b, s, grid)
                sc = {u: CR.probe_scores(r, X) for u, X in ux.items()}
                rec = CR.evaluate(jd, rows, r["nested"]["oof"], sc, "out of fold (nested CV, R1's cross-fitting)", ctx, s)
                A[rd][pos][L] = {"fit": r, "rec": rec, "scores": sc, "seed": s, "C": r["C"],
                                 "frozen": CR.frozen_block(r, jd, Xb, ux, rec["threshold"])}
                _log(quiet, f"A {rd} {pos} L{L}: a {F.fmt(rec['a']['auroc'])} C {r['C']:g} ({time.time() - t0:.0f}s)")
    if p["directions"].is_file():
        with np.load(p["directions"], allow_pickle=False) as z:
            ctx["directions"] = {k: np.asarray(z[k]) for k in z.files}
    else:
        ctx["directions"] = None
    rep = CR.reproduce(A, fit, layers, ctx)
    _log(quiet, f"gate (probe_results.json): required {rep['required']['n_values']} values, max |diff| "
                f"{rep['required']['max_abs_diff']:.3g}, pass {rep['required']['pass']}; extra {rep['extra']['n_values']}, "
                f"pass {rep['extra']['pass']}")
    if not rep["pass"]:
        bad = (rep["required"]["failures"] + rep["extra"]["failures"])[:8]
        raise ReproductionError("reader (A) pooled does not reproduce probe_results.json: "
                                + "; ".join(f"{f['where']} mine {f['mine']} fit {f['fit']}" for f in bad))

    # ---- (A) read the five ways; the fit's per-fold and frozen records must reproduce too
    VA: dict[str, dict[str, dict[int, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    keepA: dict[tuple[str, int], dict] = {}
    for rd in JUDGES:
        for pi, pos in enumerate(F.POSITIONS):
            for li, L in enumerate(layers):
                cell = A[rd][pos][L]
                Xb, ux = cells_of(pos, li)
                rec, it = view_cell(cell["fit"], J[rd], Xb, ux, {"rec": cell["rec"], "scores": cell["scores"],
                                                                 "base_oof": cell["fit"]["nested"]["oof"]},
                                    ctx, cell["seed"], defined[rd])
                VA[rd][pos][L] = rec
                if pos == HEADLINE_POSITION:
                    keepA[(rd, L)] = it
                _log(quiet, f"A view {rd} {pos} L{L} ({time.time() - t0:.0f}s)")
    fx = check_fit(VA, fit, layers)
    _log(quiet, f"gate (fit per-fold and frozen records): {fx['n_values']} values, max |diff| {fx['max_abs_diff']:.3g}, "
                f"pass {fx['pass']}")
    if not fx["pass"]:
        raise ReproductionError("reader (A)'s per-fold or frozen records do not reproduce probe_results.json: "
                                + "; ".join(f"{f['where']} mine {f['mine']} fit {f['fit']}" for f in fx["failures"][:8]))
    if reproduce_only:
        return {"gate": {"probe_results": rep, "fit_extra": fx}, "units": {"defined": defined},
                "seconds": round(time.time() - t0, 1)}

    # ---- pass 2: (B) at every cell, (C) at the last prompt token, as content_readers builds them
    p_idx, y_p = CR.prompt_label_rows(rows, xs_keep)
    h_rows = [i for i, k in enumerate(rows) if k[0] in F.H_BATTERIES]
    harmless_rows = [i for i, k in enumerate(rows) if k[0] == F.HARMLESS]
    for rd in JUDGES:
        if not set(J[rd].b_idx.tolist()) <= set(p_idx.tolist()):
            raise ValueError(f"{rd}: base-labelled rows outside (B)'s rows")
    VB: dict[str, dict[str, dict[int, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    regB: dict[str, dict[str, dict[int, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    keepB: dict[tuple[str, int], dict] = {}
    C_a: dict[str, dict[int, float | None]] = {rd: {} for rd in JUDGES}
    C_b: dict[str, dict[int, dict[str, float | None]]] = {rd: {} for rd in JUDGES}
    C_sc: dict[int, dict[str, np.ndarray]] = {}
    for pi, pos in enumerate(F.POSITIONS):
        for li, L in enumerate(layers):
            Xb, ux = cells_of(pos, li)
            rB = CR.probe_reader(Xb, p_idx, y_p, cell_seed(0, pi, li), grid)
            scB = {u: CR.probe_scores(rB, X) for u, X in ux.items()}
            if pos == HEADLINE_POSITION:
                wC = CR.mean_difference(Xb, h_rows, harmless_rows)
                baseC = CR.projection(Xb, wC)
                C_sc[L] = {u: CR.projection(X, wC) for u, X in ux.items()}
            for rd in JUDGES:
                jd = J[rd]
                s = cell_seed(jd.ri, pi, li)
                oofB = CR.oof_on(rB, jd.b_idx)
                recB = CR.evaluate(jd, rows, oofB, scB, "out of fold (nested CV, R1's cross-fitting)", ctx, s)
                recB.update(seed=rB["seed"], C=rB["C"])
                rec, it = view_cell(rB, jd, Xb, ux, {"rec": recB, "scores": scB, "base_oof": oofB}, ctx, s, defined[rd])
                VB[rd][pos][L], regB[rd][pos][L] = rec, recB
                if pos == HEADLINE_POSITION:
                    keepB[(rd, L)] = it
                    C_a[rd][L] = F.auroc(jd.y_b, baseC[jd.b_idx])
                    C_b[rd][L] = {u: F.auroc(y_u[h], C_sc[L][u][idx_u[h]]) for u, (idx_u, y_u, h) in jd.urows.items()}
            _log(quiet, f"B view {pos} L{L}: C {rB['C']:g} ({time.time() - t0:.0f}s)")

    # ---- the registered Setting D cells (last prompt token, pooled (a)), held fixed for every variant
    headline = {rd: {"A": CR.headline_layer({L: A[rd][HEADLINE_POSITION][L]["rec"]["a"]["auroc"] for L in layers}, n_layers),
                     "B": CR.headline_layer({L: regB[rd][HEADLINE_POSITION][L]["a"]["auroc"] for L in layers}, n_layers),
                     "C": CR.headline_layer(C_a[rd], n_layers)} for rd in JUDGES}

    # ---- the R1-C delta under every variant
    def variant_stat(v: str, y_h: np.ndarray, rows_h: np.ndarray, it: Mapping[str, Any], u: str) -> Callable:
        if v == "per_fold":
            return per_fold_stat(y_h, it["fold"][rows_h], it["items"][u]["per_fold"][rows_h], it["n_folds"])
        return pooled_stat(y_h, it["items"][u][v][rows_h])

    r1c: dict[str, dict[str, Any]] = {v: {} for v in ALL_VARIANTS}
    for rd in JUDGES:
        jd = J[rd]
        hl = {R: headline[rd][R]["layer"] for R in ("A", "B", "C")}
        itA, itB = keepA[(rd, hl["A"])], keepB[(rd, hl["B"])]
        for v in ALL_VARIANTS:
            per, skipped = {}, []
            for u in defined[rd]:
                idx_u, y_u, h_u = jd.urows[u]
                rows_h, y_h = idx_u[h_u], y_u[h_u]
                bA = _b_value(VA[rd][HEADLINE_POSITION][hl["A"]]["b"][v].get(u))
                bB = _b_value(VB[rd][HEADLINE_POSITION][hl["B"]]["b"][v].get(u))
                bC = C_b[rd][hl["C"]].get(u)
                if bA is None or bB is None or bC is None:
                    skipped.append(u)
                    continue
                bt = paired_delta_bootstrap_stats(y_h, variant_stat(v, y_h, rows_h, itA, u),
                                                  [variant_stat(v, y_h, rows_h, itB, u),
                                                   pooled_stat(y_h, C_sc[hl["C"]][u][rows_h])], n_boot, CR.BOOT_SEED)
                if abs(bt["b_a"] - bA) > 1e-12 or abs(bt["b_others"][0] - bB) > 1e-12 or abs(bt["b_others"][1] - bC) > 1e-12:
                    raise RuntimeError(f"{v} {rd} {u}: bootstrap point estimates differ from the cell records")
                per[u] = {"b_A": bA, "b_B": bB, "b_C": bC, "max_content": max(bB, bC),
                          "max_content_reader": "B" if bB >= bC else "C", "delta": bt["delta"], "ci": bt["ci"],
                          "interval_above_0": bt["ci"][0] is not None and bt["ci"][0] > 0,
                          "bootstrap": {k: bt[k] for k in ("n_boot", "seed", "scheme", "alpha", "n_boot_undefined",
                                                           "n_pos", "n_neg")}}
            arith = CR.verdict(per, rule) if per else {"verdict": "not_computed", "reason": "no unit"}
            r1c[v][rd] = {"position": HEADLINE_POSITION, "layers": hl, "per_unit": per, "skipped_undefined": skipped,
                          "arithmetic": {**arith, "decides": "nothing (descriptive)"}}

    # ---- the R1-C gate (when R1-C's output exists)
    if cr_out is not None:
        crx = check_content_readers(cr_out, inputs, headline, regB, C_a, C_b, r1c["pooled"], layers)
        _log(quiet, f"gate (content_readers.json): {crx['n_values']} values, max |diff| {crx['max_abs_diff']:.3g}, "
                    f"pass {crx['pass']}")
        if not crx["pass"]:
            raise ReproductionError("(B), (C) or the pooled deltas do not reproduce content_readers.json: "
                                    + "; ".join(f"{f['where']} mine {f['mine']} fit {f['fit']}" for f in crx["failures"][:8]))
    else:
        crx = {"available": False, "pass": True,
               "note": "content_readers.json absent: (B), (C) and the pooled deltas are not checked against R1-C's "
                       "registered output"}

    def by_cell(V: Mapping, key: Callable[[dict], Any]) -> dict:
        return {rd: {pos: {str(L): key(V[rd][pos][L]) for L in layers} for pos in F.POSITIONS} for rd in JUDGES}

    r1_summary = {R: {v: by_cell(V, lambda rec, v=v: rec["summary"][v]) for v in ALL_VARIANTS}
                  for R, V in (("A", VA), ("B", VB))}
    fold_C = {R: by_cell(V, lambda rec: {k: rec["fit"][k] for k in ("fold_C", "C_all_data", "fold_C_span_decades",
                                                                     "fold_C_at_grid_edge")})
              for R, V in (("A", VA), ("B", VB))}
    body = {
        "label": LABEL, "version": VERSION, "base": base, "status": status_for(base), "definition": DEFINITION,
        "decides": "nothing (descriptive); the registered pooled numbers alone decide R1 and R1-C",
        "variants": {v: VARIANT_NOTES[v] for v in VARIANTS}, "sensitivity": {v: VARIANT_NOTES[v] for v in SENSITIVITY},
        "interpretations": INTERPRETATIONS,
        "settings": {**settings_used, "boot_seed": CR.BOOT_SEED, "alpha": CR.ALPHA, "b_min": F.B_MIN,
                     "b_metric": B_METRIC, "reproduction_tol": CR.REPRO_TOL, "rule_key": rule_key,
                     "headline_position": HEADLINE_POSITION, "setting_D_tol": CR.SETD_TOL},
        "n_layers": n_layers, "layers": layers, "deep_layers": deep, "positions": list(F.POSITIONS),
        "units": {"captured": units, "labelled": labelled, "e2_units": e2_units, "regressed": regressed,
                  "defined": defined, "undefined": undefined},
        "gate": {"probe_results": rep, "fit_extra": fx, "content_readers": crx,
                 "pass": bool(rep["pass"] and fx["pass"] and crx["pass"])},
        "headline_cells": headline,
        "r1_summary": r1_summary, "fold_C": fold_C, "r1c_delta": r1c,
        "cells": {"A": by_cell(VA, lambda rec: rec), "B": by_cell(VB, lambda rec: rec)},
        "inputs": inputs, "seconds": round(time.time() - t0, 1), "git_commit": F.git_commit(),
        "code_sha256": _sha_file(Path(__file__)), "fit_code_sha256": _sha_file(Path(F.__file__)),
        "content_readers_code_sha256": _sha_file(Path(CR.__file__)),
    }
    return body


def run(results: Path | str, base: str, *, out: Path | str | None = None, **kw: Any) -> Path:
    """compute(), then write single_probe_view.json (atomically); nothing is written when the gate fails."""
    results = Path(results)
    p = _paths(results, base, {k: kw.get(k) for k in ("root", "acts_dir", "probe_results", "content_readers")}
               | {"out": out})
    body = compute(results, base, **kw)
    from sm.interp import assistant_axis as AX

    AX.write_open_json(p["out"], body, {"stage": "single_probe_view", "label": LABEL, "version": VERSION,
                                        "code_sha256": body["code_sha256"], "git_commit_at_run": body["git_commit"]})
    if not kw.get("quiet"):
        print_summary(body)
        print(f"[single-probe-view] written {p['out']}", flush=True)
    return p["out"]


def print_summary(body: Mapping[str, Any]) -> None:
    fmt = F.fmt
    for R in ("A", "B"):
        for rd in JUDGES:
            for pos in F.POSITIONS:
                for L in body["layers"]:
                    parts = []
                    for v in ALL_VARIANTS:
                        s = body["r1_summary"][R][v][rd][pos][str(L)]
                        parts.append(f"{v} {s['n_ge_075']}/{s['n_defined']} med {fmt(s['median'])} c {fmt(s['c_labelled'])}")
                    span = body["fold_C"][R][rd][pos][str(L)]["fold_C_span_decades"]
                    print(f"[single-probe-view] {R} {rd} {pos} L{L} (fold C span {span:.2f} dec): " + "; ".join(parts))
    for rd in JUDGES:
        for v in ALL_VARIANTS:
            a = body["r1c_delta"][v][rd]["arithmetic"]
            print(f"[single-probe-view] R1-C delta {rd} {v}: median {fmt(a.get('median_delta'))}, <= 0 on "
                  f"{a.get('n_delta_le_0')}/{a.get('n_defined')}, interval above 0 on {a.get('n_interval_above_0')} "
                  f"(arithmetic word if applied: {a.get('verdict')}; decides nothing)")


def _csv(s: str | None) -> list[str] | None:
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sm.exploratory.single_probe_view", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="qwen3.5-9b")
    ap.add_argument("--results", default="results", help="results root (read; the output goes under it unless --out)")
    ap.add_argument("--root", default=None, help="default <results>/exploratory/<base>/decision_probe")
    ap.add_argument("--acts-dir", default=None, help="default <root>/acts")
    ap.add_argument("--capture-json", default=None, help="default <acts-dir>/base.json (n_layers)")
    ap.add_argument("--item-labels", default=None, help="default <results>/exploratory/<base>/cpu/item_labels.jsonl")
    ap.add_argument("--judge-dir", default=None, help="default <results>/judge/<base>")
    ap.add_argument("--bank-csv", default=None, help="default <results>/tables/subsets/<base>/bank_<base>.csv")
    ap.add_argument("--e2-json", default=None, help="default <results>/exploratory/<base>/cpu/e2.json (absent on 27B)")
    ap.add_argument("--probe-results", default=None, help="default <root>/probe_results.json (the gate's target)")
    ap.add_argument("--directions", default=None, help="default <root>/directions.npz")
    ap.add_argument("--content-readers", default=None, help="default <root>/content_readers.json (checked when present)")
    ap.add_argument("--out", default=None, help="default <root>/single_probe_view.json")
    ap.add_argument("--n-layers", type=int, default=None, help="default: the capture record's n_layers")
    ap.add_argument("--n-boot", type=int, default=CR.N_BOOT)
    ap.add_argument("--seed", type=int, default=F.SEED)
    ap.add_argument("--grid", default=None, help="C grid (default the fit's); tests only")
    ap.add_argument("--rule", default=None, help="the base whose R1-C arithmetic is applied (default --base)")
    ap.add_argument("--expect-defined", default=None, help="comma list; stop unless the defined units are these")
    ap.add_argument("--reproduce-only", action="store_true", help="run (A)'s gate only; write nothing")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    grid = tuple(float(x) for x in a.grid.split(",")) if a.grid else F.C_GRID
    kw = dict(n_boot=a.n_boot, seed=a.seed, grid=grid, rule_key=a.rule, expect_defined=_csv(a.expect_defined),
              n_layers=a.n_layers, quiet=a.quiet, root=a.root, acts_dir=a.acts_dir, capture_json=a.capture_json,
              item_labels=a.item_labels, judge_dir=a.judge_dir, bank_csv=a.bank_csv, e2_json=a.e2_json,
              probe_results=a.probe_results, directions=a.directions, content_readers=a.content_readers)
    try:
        if a.reproduce_only:
            g = compute(Path(a.results), a.base, reproduce_only=True, **kw)["gate"]
            print(json.dumps({"probe_results": {k: g["probe_results"][k] for k in ("pass", "n_values_required")},
                              "fit_extra": {k: g["fit_extra"][k] for k in ("pass", "n_values", "max_abs_diff")}}, indent=1))
            return 0
        run(Path(a.results), a.base, out=a.out, **kw)
    except CR.ReproductionError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
