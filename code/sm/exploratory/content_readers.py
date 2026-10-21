"""EXPLORATORY R1-C: matched content readers at the decision position.

Registration: docs/HYPOTHESIS_LEDGER.md, section "R1-C, matched content readers at the decision position", committed
69bfdc0 (2026-10-20 13:22:58 -0700) before any content reader was computed (heading time corrected at 0c2ae11). It
decides no registered hypothesis.

Why. R1's headline contrast changes the labels and the read position at once (the concept monitors were read at the
card's template cell, the decision probe at the last prompt token in late layers). R1-C holds the cell fixed and
changes only the labels (B, C) or the estimator (D). Four readers, each fitted on the BASE only, frozen, and read on
every unit's residual at the same position and hidden state:
  (A) R1's decision probe, sm.exploratory.decision_probe_fit v2 itself (its functions are called here; the file is not
      edited): base refuse-vs-answer labels per judge reader, standardized L2 logistic regression, C by 5-fold CV,
      nested cross-fitting, the seeds as coded (seed + 1000 r + 100 p + l), the all-data ("frozen full") probe.
  (B) a content probe: the same fit with only the labels swapped to prompt labels (y = 1 for the H items, 0 for the
      XSTest-safe items of P and the harmless prompts). One probe, read against both judge readers' labels; its seeds
      are the WildGuard reader's (r = 0), so against WildGuard it differs from (A) only through its labels.
  (C) the harm direction: the base mean over the H items minus the base mean over the harmless prompts; score = the
      projection on that direction; no behaviour label, no fit.
  (D) a decision direction: the base mean over the base-refused items of P minus over the base-answered ones (per judge
      reader); score = projection. It separates the label from the estimator.
Threshold (R1's rule): the score above which 5 percent of the base's answered items of P fall (per judge reader;
out-of-fold scores for A and B, in-sample projections for C and D); predict refusal when score > threshold.
Measures per reader, judge reader, position and captured hidden state: (a) base AUROC on the base's own decisions over
P (out of fold for A and B; in-sample for C and D, flagged); (b) per unit, AUROC on the unit's own decisions over the H
items (all of P beside it); (c) R1's regression-ranking AUROC (decision_probe_fit.c_auroc on the per-unit scores);
recall and false-alarm rate at the frozen threshold; cosines between the four raw-space weight vectors.
Headline cell per reader (and judge reader): Setting D's rule at the last prompt token, the deepest captured hidden
state l with 3 l >= 2 n whose (a) is >= that reader's best (a) over all captured layers minus 0.005 (inclusive), else
the deep layer with the highest (a) (ties to the deeper, flagged); chosen on the base only. Upper bound for the content
readers (B, C): the deep layer that maximises the reader's median (b) over the defined units (chosen by the units;
descriptive).
Comparison: per defined unit, delta = b(A) - max(b(B), b(C)), each at its own headline cell, with a 95 percent paired
bootstrap interval over the unit's H items (2000 draws, seed 0, reader scores fixed; the draw is the codebase's paired
AUROC bootstrap, stratified by the unit's own label; an i.i.d. item draw is reported beside it as a sensitivity line).
Rules (RULES; WildGuard decides, J1 alongside without a decision): see the registration.
Reproduction gate: before any content reader is computed, reader (A) must reproduce the fit's probe_results.json at
every cell, (a) and every per-unit (b) field (tolerance 1e-9), and also C, the threshold, (c), the (c) AUROC, the
frozen-full block, the saved directions and the input hashes; otherwise the run stops and writes nothing.
Inputs (read only; numbers and labels, no text; nothing under a restricted tree): acts/<unit>.npz and acts/base.json
(capture), the text-free item-label table, the XSTest-safe judge labels, the bank table, e2.json when present, the
fit's probe_results.json and directions.npz. Output: <results>/exploratory/<base>/decision_probe/content_readers.json
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sm.analysis import stats
from sm.exploratory import decision_probe_fit as F

LABEL = ("EXPLORATORY R1-C matched content readers at the decision position (docs/HYPOTHESIS_LEDGER.md R1-C, "
         "registered at commit 69bfdc0 before computing); not a measured stage; decides no registered hypothesis")
VERSION = "content-readers-v1-2026-10-20"
REGISTRATION = {"doc": "docs/HYPOTHESIS_LEDGER.md",
                "section": "R1-C, matched content readers at the decision position",
                "commit": "69bfdc0", "commit_full": "69bfdc09133b437f80476beb47872d8e69b35f4a",
                "committed": "2026-10-20 13:22:58 -0700", "heading_time_corrected_at": "0c2ae11",
                "status": "exploratory; decides no registered hypothesis"}
READERS = ("A", "B", "C", "D")
READER_NAMES = {"A": "R1 decision probe (decision_probe_fit v2: base refuse-vs-answer labels, per judge reader)",
                "B": "content probe (the same fit, prompt labels: H items 1, XSTest-safe and harmless 0; one probe)",
                "C": "harm direction (base mean of the H items minus base mean of the harmless prompts; projection)",
                "D": "decision direction (base mean of base-refused minus base-answered items of P; per judge reader)"}
CONTENT = ("B", "C")
PROBE_READERS = ("A", "B")
JUDGES = F.READERS  # ("J2", "J1"): the fit's order, which sets its seeds
PRIMARY_JUDGE = "J2"
HEADLINE_POSITION = "last"
SETD_TOL = 0.005
SETD_EPS = 1e-12  # the inclusive boundary under floating point, as paper/scripts/probe27b.py
N_BOOT = 2000
BOOT_SEED = 0
ALPHA = 0.05
REPRO_TOL = 1e-9
BOOT_SCHEMES = ("stratified", "iid")
ITEM_LABEL_FIELDS = frozenset({"run_id", "battery", "item_id", "sample_idx", "judge", "label", "parse_ok",
                               "completion_sha256"})
RULES: dict[str, dict[str, Any]] = {
    "qwen3.5-9b": {"base": "qwen3.5-9b", "judge": PRIMARY_JUDGE, "n_units": 7, "holds_min_interval_above_0": 5,
                   "holds_requires_delta_positive": True, "falsify_median_below": 0.02, "falsify_min_delta_le_0": 3,
                   "text": "holds ('the decision labels carry the gain'): delta > 0 with the 95 percent interval above "
                           "0 on at least 5 of the 7 defined units; falsified ('monitor the decision position'): median "
                           "delta over the 7 units below 0.02, or delta <= 0 on 3 or more units; anything else = "
                           "neither; WildGuard decides, J1 alongside"},
    "qwen3.8-27b": {"base": "qwen3.8-27b", "judge": PRIMARY_JUDGE, "n_units": 3, "holds_min_interval_above_0": 3,
                    "holds_requires_delta_positive": False, "falsify_median_below": 0.02, "falsify_min_delta_le_0": 2,
                    "text": "out of sample: holds when the interval is above 0 on all 3 defined units; falsified when "
                            "median delta < 0.02 or delta <= 0 on 2 or more; else neither; WildGuard decides, J1 "
                            "alongside"},
}
PRECEDENCE_NOTE = ("the registration does not order its two conditions; when both are met (possible only through the "
                   "median), the falsifier is reported (a prediction whose falsifier is met is not reported as "
                   "holding) and both_conditions_met records it")
INTERPRETATIONS = [
    "(B) is one probe for both judge readers; its seeds are the WildGuard reader's (seed + 100 p + l, r = 0), so against "
    "WildGuard it differs from (A) only through its labels and rows; against J1 its seeds are WildGuard's",
    "(B)'s rows are every H item of P (y = 1), the XSTest-safe ids WildGuard labelled on the base (R1's 25) and the 128 "
    "harmless prompts (y = 0), whatever the base's label usability; (B)'s folds are stratified on its own labels, as "
    "the fit stratifies on the labels it is given",
    "(C) and (D) score by the projection on the unit vector of the mean difference (AUROC and the 5 percent threshold "
    "do not depend on the scale); their (a), thresholds and base (c) score are in-sample on the base (flagged)",
    "(c) for every reader: the unit's decision score is the mean of that reader's score over B0 (the base-refused H "
    "items under the judge reader), the base score the base's mean on B0 on the reader's footing (out of fold for A "
    "and B, in-sample for C and D); then decision_probe_fit.c_auroc, unchanged",
    "defined units: R1's regressed set (E2's confirmed units when e2.json exists, as the fit's verdict block; else "
    "every confirmed labelled unit, as on 27B) restricted to units whose usable H labels under the judge reader hold "
    "both classes",
    "the paired bootstrap resamples the unit's H items within each class of the unit's own label (the draw of "
    "sm.analysis.stats.paired_bootstrap_auroc_diff, class counts fixed, so every draw is defined); max(b(B), b(C)) is "
    "recomputed in every draw; percentile interval over the 2000 draws; an i.i.d. item draw (undefined draws counted) "
    "is reported beside it and decides nothing",
    "Setting D's rule as implemented in paper/scripts/probe27b.py: deep layers are those with 3 l >= 2 n; eligible "
    "when (a) >= best - 0.005 - 1e-12; the deepest eligible, else the deep layer with the highest (a), ties to the deeper",
    "the headline cell is chosen per reader AND judge reader ((B) and (C) have one score each, but their (a) is read "
    "against each judge reader's base labels); the content-reader upper bound ties to the deeper layer",
    "recall and false-alarm rate are reported over P (R1's b_scores) and over the H items only",
    "the first generated token is reported at every hidden state; the cells chosen there by the same rule and their "
    "delta are descriptive only (the registration's headline cell is the last prompt token)",
    PRECEDENCE_NOTE,
]


class ReproductionError(RuntimeError):
    """Reader (A) did not reproduce the fit's probe_results.json; nothing is written."""


def _sha_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(base: str) -> str:
    from sm.interp import assistant_axis as AX

    return AX.safe_name(base)


def _guard(path: Path, what: str) -> Path:
    if "restricted" in Path(os.path.normpath(str(path))).parts:
        raise ValueError(f"{what}: {path} lies under a restricted tree; R1-C opens nothing there")
    return path

# --------------------------------------------------------------------------------------------- pure rules


def deep_layers(layers: Sequence[int], n_layers: int) -> list[int]:
    """The captured hidden states at or beyond 2n/3 (3 l >= 2 n): 9B {24, 28, 31, 32}, 27B {48, 56, 63, 64}."""
    return [int(L) for L in sorted(layers) if 3 * int(L) >= 2 * int(n_layers)]


def headline_layer(a: Mapping[int, float | None], n_layers: int, tol: float = SETD_TOL) -> dict[str, Any]:
    """Setting D's rule on one reader's base (a) by layer (last prompt token): the deepest deep layer whose (a) is within
    tol of the best (a) over all captured layers (inclusive), else the deep layer with the highest (a), ties to the
    deeper, flagged as the fallback."""
    vals = {int(L): float(v) for L, v in a.items() if v is not None}
    if not vals:
        raise ValueError("no defined (a) at any layer")
    best = max(vals.values())
    best_layer = max((L for L in vals if vals[L] == best))
    deep = deep_layers(list(a), n_layers)
    ok = [L for L in deep if L in vals and vals[L] >= best - tol - SETD_EPS]
    pool = [L for L in deep if L in vals]
    if not pool:
        raise ValueError("no deep layer has a defined (a)")
    sel = max(ok) if ok else max(pool, key=lambda L: (vals[L], L))
    return {"layer": sel, "fallback": not ok, "a": vals[sel], "best_a_all_layers": best, "best_layer": best_layer,
            "eligible": ok, "deep_layers": deep, "tol": tol}


def upper_bound_layer(b: Mapping[int, Mapping[str, float | None]], deep: Sequence[int],
                      units: Sequence[str]) -> dict[str, Any]:
    """The deep layer that maximises the median (b) over the units (ties to the deeper)."""
    med = {}
    for L in deep:
        v = [b[L][u] for u in units if b[L].get(u) is not None]
        med[int(L)] = float(statistics.median(v)) if v else None
    pool = [L for L in med if med[L] is not None]
    if not pool:
        return {"layer": None, "median_b": None, "median_by_layer": {str(L): med[L] for L in med}, "b": {}}
    sel = max(pool, key=lambda L: (med[L], L))
    return {"layer": sel, "median_b": med[sel], "median_by_layer": {str(L): med[L] for L in med},
            "b": {u: b[sel].get(u) for u in units}}


def verdict(per_unit: Mapping[str, Mapping[str, Any]], rule: Mapping[str, Any] | None) -> dict[str, Any]:
    """The registered arithmetic on per_unit[u] = {"delta": d, "ci": [lo, hi]}: the count of units whose interval lies
    above 0 (lo > 0, and d > 0 where the rule says so), the median delta, the count with delta <= 0, and the word."""
    units = list(per_unit)
    d = {u: float(per_unit[u]["delta"]) for u in units}
    lo = {u: per_unit[u]["ci"][0] for u in units}
    need_pos = True if rule is None else bool(rule.get("holds_requires_delta_positive", True))
    above = [u for u in units if lo[u] is not None and lo[u] > 0 and (d[u] > 0 or not need_pos)]
    le0 = [u for u in units if d[u] <= 0]
    med = float(statistics.median(d.values())) if units else None
    out: dict[str, Any] = {"n_defined": len(units), "units": units, "n_interval_above_0": len(above),
                           "units_interval_above_0": above, "median_delta": med, "n_delta_le_0": len(le0),
                           "units_delta_le_0": le0, "deltas": d}
    if rule is None:
        out.update(verdict="not_computed", reason="no registered rule for this base")
        return out
    out["rule"] = dict(rule)
    if len(units) != int(rule["n_units"]):
        out.update(verdict="not_computed",
                   reason=f"{len(units)} defined units; the registration's rule is written for {rule['n_units']}")
        return out
    holds = len(above) >= int(rule["holds_min_interval_above_0"])
    by_median = med < float(rule["falsify_median_below"])
    by_count = len(le0) >= int(rule["falsify_min_delta_le_0"])
    falsified = by_median or by_count
    out.update(holds_condition=holds, falsified_by_median=by_median, falsified_by_count=by_count,
               falsified_condition=falsified, both_conditions_met=holds and falsified,
               verdict="falsified" if falsified else ("holds" if holds else "neither"), precedence=PRECEDENCE_NOTE)
    return out

# --------------------------------------------------------------------------------------------- readers


def probe_reader(Xb: np.ndarray, idx: np.ndarray, y: np.ndarray, seed: int,
                 grid: Sequence[float] = F.C_GRID) -> dict[str, Any]:
    """R1's probe exactly as decision_probe_fit.run_fit builds it, on the base rows idx with labels y: the nested CV
    (out-of-fold scores and the outer-fold probes), C by 5-fold CV, the all-data probe at that C. Readers A and B both
    come from here; they differ only in (idx, y) and the seed."""
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(y, dtype=int)
    nested = F.nested_cv(Xb[idx], y, grid, F.N_FOLDS, seed)
    C, ctable = F.choose_C(Xb[idx], y, grid, F.N_FOLDS, seed)
    probe = F.fit_probe(Xb[idx], y, C)
    return {"idx": idx, "y": y, "seed": int(seed), "nested": nested, "C": C, "C_cv": ctable, "probe": probe}


def probe_scores(r: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """Cross-fitted predicted probabilities of X's rows (decision_probe_fit.cross_fit_prob)."""
    return F.cross_fit_prob(r["probe"], r["nested"]["fold_probes"], r["idx"], X)


def oof_on(r: Mapping[str, Any], rows_idx: np.ndarray) -> np.ndarray:
    """The nested CV's out-of-fold scores at the given row indices of P (each must be one of the probe's rows)."""
    pos = {int(i): j for j, i in enumerate(r["idx"])}
    return r["nested"]["oof"][[pos[int(i)] for i in rows_idx]]


def prompt_label_rows(rows: Sequence[tuple[str, str]], xs_keep: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """(B)'s rows and labels in row order: every H item of P (y = 1), the XSTest-safe items whose ids WildGuard labelled
    on the base (R1's XSTest-safe part of P) and the harmless prompts (y = 0)."""
    idx, y = [], []
    for i, (b, item) in enumerate(rows):
        if b in F.H_BATTERIES:
            idx.append(i)
            y.append(1)
        elif b == F.HARMLESS or (b == F.XS and item in xs_keep):
            idx.append(i)
            y.append(0)
    return np.array(idx, dtype=int), np.array(y, dtype=int)


def mean_difference(X: np.ndarray, pos_rows: Sequence[int], neg_rows: Sequence[int]) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return X[np.asarray(pos_rows, int)].mean(0) - X[np.asarray(neg_rows, int)].mean(0)


def projection(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(w))
    if not n > 0:
        raise ValueError("zero direction")
    return np.asarray(X, np.float64) @ (np.asarray(w, np.float64) / n)


def paired_delta_bootstrap(y: Sequence, s_a: Sequence, s_others: Sequence[Sequence], n_boot: int = N_BOOT,
                           seed: int = BOOT_SEED, alpha: float = ALPHA, scheme: str = "stratified") -> dict[str, Any]:
    """delta = AUROC(s_a) - max_k AUROC(s_others[k]) on the same items, with a paired percentile bootstrap: every draw
    resamples the items once and recomputes every reader's AUROC on that draw (scores fixed). scheme "stratified"
    draws within each class of y (the draw of sm.analysis.stats.paired_bootstrap_auroc_diff: class counts fixed);
    "iid" draws items with replacement (a draw with one class is undefined and counted)."""
    if scheme not in BOOT_SCHEMES:
        raise ValueError(f"scheme must be one of {BOOT_SCHEMES}")
    y = np.asarray(y).astype(bool).ravel()
    a = np.asarray(s_a, float).ravel()
    others = [np.asarray(s, float).ravel() for s in s_others]
    b_a = float(stats.auroc(y, a))
    b_o = [float(stats.auroc(y, s)) for s in others]
    point = b_a - max(b_o)
    pos_idx, neg_idx = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        if scheme == "stratified":
            idx = np.concatenate([rng.choice(pos_idx, size=pos_idx.size, replace=True),
                                  rng.choice(neg_idx, size=neg_idx.size, replace=True)])
        else:
            idx = rng.integers(0, y.size, size=y.size)
        yy = y[idx]
        if yy.all() or not yy.any():
            boot[i] = np.nan
            continue
        boot[i] = stats.auroc(yy, a[idx]) - max(stats.auroc(yy, s[idx]) for s in others)
    ok = boot[~np.isnan(boot)]
    lo, hi = (float(v) for v in np.quantile(ok, [alpha / 2.0, 1.0 - alpha / 2.0])) if ok.size else (None, None)
    return {"delta": point, "b_a": b_a, "b_others": b_o, "ci": [lo, hi], "n_boot": int(n_boot), "seed": int(seed),
            "scheme": scheme, "alpha": alpha, "n_boot_undefined": int(np.isnan(boot).sum()),
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}

# --------------------------------------------------------------------------------------------- labels


@dataclass
class Judge:
    rd: str
    ri: int
    blab: dict
    b_idx: np.ndarray
    y_b: np.ndarray
    h_b: np.ndarray
    b0_pos: np.ndarray
    b0: np.ndarray
    ulabs: dict
    urows: dict  # unit -> (idx, y, h), the fit's per-unit rows of P


def judge_labels(item_labels: Sequence[Mapping], judge_dir: Path, rows: Sequence[tuple[str, str]],
                 index: Mapping[tuple[str, str], int], harmless_keys: Sequence[tuple[str, str]], xs_keep: set[str],
                 units: Sequence[str], rd: str) -> Judge:
    """The base's and the units' labels under one judge reader, built line for line as decision_probe_fit.run_fit."""
    blab = F.unit_labels(item_labels, judge_dir, F.BASE, rd, xs_keep, harmless_keys)
    bkeys = [k for k in rows if k in blab]
    b_idx = np.array([index[k] for k in bkeys], dtype=int)
    y_b = np.array([blab[k] for k in bkeys], int)
    h_b = np.array([k[0] in F.H_BATTERIES for k in bkeys], dtype=bool)
    b0_pos = np.flatnonzero(h_b & (y_b == 1))
    ulabs = {u: F.unit_labels(item_labels, judge_dir, u, rd, xs_keep, harmless_keys) for u in units}
    urows = {}
    for u, lab in ulabs.items():
        if lab:
            kk = [k for k in rows if k in lab]
            urows[u] = (np.array([index[k] for k in kk]), np.array([lab[k] for k in kk], int),
                        np.array([k[0] in F.H_BATTERIES for k in kk], dtype=bool))
    return Judge(rd, F.READERS.index(rd), blab, b_idx, y_b, h_b, b0_pos, b_idx[b0_pos], ulabs, urows)


def regressed_units(e2: Mapping | None, labelled: Sequence[str], reg: Mapping[str, dict]) -> list[str]:
    """R1's regressed set: E2's confirmed units when E2's file exists (the fit's verdict block), else every confirmed
    labelled unit (27B: no E2 file, every slice unit regressed)."""
    if e2 is not None:
        return [u for u in e2["units"] if reg.get(u, {}).get("confirmed")]
    return [u for u in labelled if reg.get(u, {}).get("confirmed")]


def defined_units(regressed: Sequence[str], jd: Judge) -> list[str]:
    out = []
    for u in regressed:
        if u in jd.urows:
            _, y, h = jd.urows[u]
            if y[h].size and y[h].any() and not y[h].all():
                out.append(u)
    return out

# --------------------------------------------------------------------------------------------- measures


def evaluate(jd: Judge, rows: Sequence[tuple[str, str]], s_base: np.ndarray, unit_scores: Mapping[str, np.ndarray],
             footing: str, ctx: Mapping[str, Any], seed_c: int) -> dict[str, Any]:
    """R1's measures for one reader at one cell under one judge reader, through the fit's own functions. s_base: the
    base's scores on jd.b_idx in order (out of fold for A and B, in-sample for C and D); unit_scores: each captured
    unit's scores on every row of P."""
    y_b, h_b = jd.y_b, jd.h_b
    thr, frr_real = F.threshold_at_frr(s_base[y_b == 0])
    base_score = float(s_base[jd.b0_pos].mean())
    rec: dict[str, Any] = {
        "a": {"auroc": F.auroc(y_b, s_base), "auroc_H_only": F.auroc(y_b[h_b], s_base[h_b]), "n": int(len(y_b)),
              "n_refusal": int(y_b.sum()), "n_H": int(h_b.sum()), "n_H_refusal": int(y_b[h_b].sum()),
              "footing": footing},
        "threshold": thr, "base_false_alarm_rate": frr_real,
        "base_at_threshold": F.thresholded(y_b.astype(bool), s_base, thr),
        "base_decision_score_B0": base_score, "b": {}, "c": {}}
    for u, p in unit_scores.items():
        c_rec = F.c_scores(p, jd.b0, base_score, thr)
        lab = jd.ulabs.get(u) or {}
        if lab:
            idx, y_u, h = jd.urows[u]
            on_b0 = [lab.get(rows[i]) for i in jd.b0]
            c_rec["judged_refusal_rate_B0"] = F._f(np.mean([v for v in on_b0 if v is not None])) \
                if any(v is not None for v in on_b0) else None
            th_h = F.thresholded(y_u[h].astype(bool), p[idx][h], thr)
            rec["b"][u] = {**F.b_scores(y_u, p[idx], h, thr), "n": int(len(y_u)), "n_refusal": int(y_u.sum()),
                           "n_H": int(h.sum()), "n_H_refusal": int(y_u[h].sum()),
                           "recall_H": th_h["tpr_refusals"], "false_alarm_H": th_h["false_refusal_rate_answered"]}
        rec["c"][u] = c_rec
    rec["c_auroc"] = F.c_auroc(rec["c"], ctx["labelled"], ctx["e2_units"], ctx["reg"], ctx["d2"], ctx["canary"],
                               ctx["n_boot"], seed_c)
    rr = [(rec["c"][u]["decision_score_B0"], rec["c"][u].get("judged_refusal_rate_B0")) for u in ctx["labelled"]
          if u in rec["c"] and rec["c"][u].get("judged_refusal_rate_B0") is not None]
    rec["c_spearman_score_vs_judged_rate_B0"] = F.spearman([a for a, _ in rr], [b for _, b in rr])
    return rec


def frozen_block(r: Mapping[str, Any], jd: Judge, Xb: np.ndarray, unit_X: Mapping[str, np.ndarray],
                 thr: float) -> dict[str, Any]:
    """Secondary, as the fit's frozen_full_probe: the all-data probe read directly (in-sample on base-labelled rows,
    the out-of-fold threshold applied); no rule reads it."""
    probe = r["probe"]
    out: dict[str, Any] = {"note": F.FROZEN_NOTE, "a_in_sample": F.auroc(jd.y_b, probe.decision(Xb[jd.b_idx])), "b": {}}
    for u, Xu in unit_X.items():
        if u in jd.urows:
            idx, y_u, h = jd.urows[u]
            bs = F.b_scores(y_u, probe.prob(Xu)[idx], h, thr)
            out["b"][u] = {"auroc": bs["auroc"], "auroc_H_only": bs["auroc_H_only"]}
    return out


def cosines(w: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    keys = list(READERS)
    return {f"{a}_{b}": F.cosine(w[a], w[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}

# --------------------------------------------------------------------------------------------- reproduction

A_FIELDS = ("auroc", "auroc_H_only", "tpr_refusals", "false_refusal_rate_answered", "predicted_refusal_rate",
            "judged_refusal_rate", "accuracy", "n", "n_refusal", "n_H", "n_H_refusal")
C_FIELDS = ("decision_score_B0", "regression_score", "predicted_refusal_rate_B0", "judged_refusal_rate_B0")


class _Cmp:
    def __init__(self, tol: float):
        self.tol, self.n, self.max_diff, self.fails = tol, 0, 0.0, []

    def __call__(self, where: str, mine: Any, theirs: Any) -> None:
        self.n += 1
        if mine is None or theirs is None or isinstance(mine, (str, bool)) or isinstance(theirs, (str, bool)):
            if mine != theirs:
                self.fails.append({"where": where, "mine": mine, "fit": theirs})
            return
        if isinstance(mine, (list, tuple)) or isinstance(theirs, (list, tuple)):
            if not isinstance(mine, (list, tuple)) or not isinstance(theirs, (list, tuple)) or len(mine) != len(theirs):
                self.fails.append({"where": where, "mine": mine, "fit": theirs})
                return
            for i, (m, t) in enumerate(zip(mine, theirs)):
                self(f"{where}[{i}]", m, t)
            return
        d = abs(float(mine) - float(theirs))
        if not math.isfinite(d) or d > self.tol:
            self.fails.append({"where": where, "mine": mine, "fit": theirs, "abs_diff": d})
        self.max_diff = max(self.max_diff, d) if math.isfinite(d) else self.max_diff

    def summary(self) -> dict[str, Any]:
        return {"pass": not self.fails, "n_values": self.n, "max_abs_diff": self.max_diff, "tol": self.tol,
                "n_fail": len(self.fails), "failures": self.fails[:40]}


def reproduce(A: Mapping, fit: Mapping[str, Any], layers: Sequence[int], ctx: Mapping[str, Any],
              tol: float = REPRO_TOL) -> dict[str, Any]:
    """Reader (A) against the fit's probe_results.json: REQUIRED, (a) (pooled and H-only) and every field of every
    per-unit (b) at every (judge reader, position, layer); EXTRA, C, the threshold and its realised rate, the base (c)
    score, every per-unit (c) record, the (c) AUROC with its interval and comparators, the frozen-full block, the
    saved directions, the fit's settings and input hashes, and R1's defined units."""
    req, ext = _Cmp(tol), _Cmp(tol)
    for rd in JUDGES:
        for pos in F.POSITIONS:
            for L in layers:
                w = f"{rd}/{pos}/L{L}"
                mine, th = A[rd][pos][L]["rec"], fit["readers"][rd]["positions"][pos]["layers"][str(L)]
                req(f"{w}/a/auroc_heldout", mine["a"]["auroc"], th["a"]["auroc_heldout"])
                req(f"{w}/a/auroc_heldout_H_only", mine["a"]["auroc_H_only"], th["a"]["auroc_heldout_H_only"])
                if sorted(mine["b"]) != sorted(th["b"]):
                    req.fails.append({"where": f"{w}/b/units", "mine": sorted(mine["b"]), "fit": sorted(th["b"])})
                for u in sorted(set(mine["b"]) & set(th["b"])):
                    for k in A_FIELDS:
                        req(f"{w}/b/{u}/{k}", mine["b"][u].get(k), th["b"][u].get(k))
                fr = A[rd][pos][L]
                ext(f"{w}/C", fr["C"], th["C"])
                ext(f"{w}/seed", fr["seed"], th["seed"])
                ext(f"{w}/a/n", mine["a"]["n"], th["a"]["n"])
                ext(f"{w}/a/n_H_refusal", mine["a"]["n_H_refusal"], th["a"]["n_H_refusal"])
                ext(f"{w}/threshold", mine["threshold"], th["threshold"])
                ext(f"{w}/base_oof_false_refusal_rate", mine["base_false_alarm_rate"], th["base_oof_false_refusal_rate"])
                for k, v in th["base_oof_at_threshold"].items():
                    ext(f"{w}/base_oof_at_threshold/{k}", mine["base_at_threshold"].get(k), v)
                ext(f"{w}/base_decision_score_B0", mine["base_decision_score_B0"], th["base_decision_score_B0"])
                for u in sorted(set(mine["c"]) | set(th["c"])):
                    for k in C_FIELDS:
                        ext(f"{w}/c/{u}/{k}", (mine["c"].get(u) or {}).get(k), (th["c"].get(u) or {}).get(k))
                for grp in ("labelled", "e2_units"):
                    m, t = mine["c_auroc"][grp], th["c_auroc"][grp]
                    ext(f"{w}/c_auroc/{grp}/units", m.get("units"), t.get("units"))
                    ext(f"{w}/c_auroc/{grp}/auroc", m.get("auroc"), t.get("auroc"))
                    if t.get("ci") is not None or m.get("ci") is not None:
                        ext(f"{w}/c_auroc/{grp}/ci", m.get("ci"), t.get("ci"))
                    for name, cv in (t.get("comparators") or {}).items():
                        mc = (m.get("comparators") or {}).get(name)
                        for k in ("auroc", "diff_probe_minus", "diff_ci"):
                            ext(f"{w}/c_auroc/{grp}/{name}/{k}", (mc or {}).get(k), (cv or {}).get(k))
                ext(f"{w}/c_spearman", mine["c_spearman_score_vs_judged_rate_B0"], th["c_spearman_score_vs_judged_rate_B0"])
                ff, tf = fr["frozen"], th["frozen_full_probe"]
                ext(f"{w}/frozen/a_in_sample", ff["a_in_sample"], th["a"]["auroc_in_sample"])
                for u in sorted(set(ff["b"]) | set(tf["b"])):
                    for k in ("auroc", "auroc_H_only"):
                        ext(f"{w}/frozen/b/{u}/{k}", (ff["b"].get(u) or {}).get(k), (tf["b"].get(u) or {}).get(k))
    s, st = ctx["settings_used"], fit["settings"]
    for k in ("C_grid", "n_folds", "seed", "n_boot", "frr", "layers", "positions"):
        ext(f"settings/{k}", s[k], st[k])
    ext("inputs/item_labels", ctx["inputs"]["item_labels"]["sha256"], fit["inputs"]["item_labels"]["sha256"])
    ext("inputs/bank_csv", ctx["inputs"]["bank_csv"]["sha256"], fit["inputs"]["bank_csv"]["sha256"])
    ext("inputs/e2_json", (ctx["inputs"]["e2_json"] or {}).get("sha256"), (fit["inputs"]["e2_json"] or {}).get("sha256"))
    for u, h in fit["inputs"]["acts"].items():
        ext(f"inputs/acts/{u}", ctx["inputs"]["acts"].get(u), h)
    for k, h in fit["inputs"]["xstest_safe_labels"].items():
        ext(f"inputs/xstest_safe_labels/{k}", ctx["inputs"]["xstest_safe_labels"].get(k), h)
    ext("units/captured", sorted(ctx["units"]), sorted(fit["units"]["captured"]))
    ext("units/labelled", list(ctx["labelled"]), list(fit["units"]["labelled"]))
    if ctx["e2"] is not None:
        for rd in JUDGES:
            vb = fit["readers"][rd]["positions"][HEADLINE_POSITION]["b"]
            ext(f"{rd}/R1_defined_units", ctx["defined"][rd], [u for u in vb["regressed_units"] if u not in vb["undefined"]])
    dirs = ctx.get("directions")
    d_max = None
    if dirs is not None:
        ext("directions/sha256", ctx["inputs"]["directions"]["sha256"], fit["directions"]["sha256"])
        d_max = 0.0
        for rd in JUDGES:
            for pos in F.POSITIONS:
                for L in layers:
                    key = f"{rd}__{pos}__L{L}__{F.BASE}"
                    w_, b_ = A[rd][pos][L]["fit"]["probe"].raw_direction()
                    mine = np.append(w_, b_).astype(np.float32)
                    saved = np.asarray(dirs[key])
                    dd = float(np.max(np.abs(mine.astype(np.float64) - saved.astype(np.float64))))
                    d_max = max(d_max, dd)
                    ext(f"directions/{key}", dd, 0.0)
    out = {"required": {**req.summary(), "what": "(a) pooled and H-only, and every field of every per-unit (b), at "
                                                 "every judge reader, position and layer"},
           "extra": {**ext.summary(), "directions_max_abs_diff": d_max,
                     "what": "C, seed, threshold and realised rate, base (c) score, per-unit (c), the (c) AUROC with its "
                             "interval and comparators, Spearman, frozen-full (a) and (b), the saved raw directions "
                             "(float32), settings, input hashes, R1's defined units"},
           "fit_code_sha256_now": _sha_file(Path(F.__file__)), "fit_code_sha256_in_results": fit.get("code_sha256"),
           "fit_git_commit": fit.get("git_commit")}
    out["fit_code_unchanged"] = out["fit_code_sha256_now"] == out["fit_code_sha256_in_results"]
    out["n_values_required"] = req.n
    out["pass"] = bool(req.summary()["pass"] and ext.summary()["pass"])
    return out

# --------------------------------------------------------------------------------------------- main computation


def _paths(results: Path, base: str, over: Mapping[str, Any]) -> dict[str, Path]:
    safe = safe_name(base)
    x = results / "exploratory" / safe
    root = Path(over["root"]) if over.get("root") else x / "decision_probe"
    p = {"root": root, "acts_dir": root / "acts", "item_labels": x / "cpu" / "item_labels.jsonl",
         "judge_dir": results / "judge" / safe, "bank_csv": results / "tables" / "subsets" / safe / f"bank_{safe}.csv",
         "e2_json": x / "cpu" / "e2.json", "probe_results": root / "probe_results.json",
         "directions": root / "directions.npz", "out": root / "content_readers.json"}
    for k in ("acts_dir", "item_labels", "judge_dir", "bank_csv", "e2_json", "probe_results", "directions", "out"):
        if over.get(k):
            p[k] = Path(over[k])
    p["capture_json"] = Path(over["capture_json"]) if over.get("capture_json") else p["acts_dir"] / f"{F.BASE}.json"
    _guard(results, "results")
    for k, v in p.items():
        _guard(v, k)
    return p


def _check_item_labels(item_labels: Sequence[Mapping]) -> None:
    extra = set().union(*(set(r) for r in item_labels)) - ITEM_LABEL_FIELDS if item_labels else set()
    if extra:
        raise ValueError(f"item-label table carries fields beyond the text-free set: {sorted(extra)}")


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(f"[content-readers] {msg}", flush=True)


def compute(results: Path | str, base: str, *, n_boot: int = N_BOOT, seed: int = F.SEED,
            grid: Sequence[float] = F.C_GRID, rule_key: str | None = None, expect_defined: Sequence[str] | None = None,
            n_layers: int | None = None, reproduce_only: bool = False, quiet: bool = False,
            **paths: Any) -> dict[str, Any]:
    """Everything R1-C reports, as a dict (nothing is written). Raises ReproductionError when (A) does not reproduce
    the fit, before any content reader is computed."""
    t0 = time.time()
    results = Path(results)
    p = _paths(results, base, paths)
    for k in ("item_labels", "bank_csv", "probe_results"):
        if not p[k].is_file():
            raise FileNotFoundError(f"{k}: {p[k]} missing")
    if not (p["acts_dir"] / f"{F.BASE}.npz").is_file():
        raise FileNotFoundError(f"no base capture under {p['acts_dir']}")
    item_labels = F.read_jsonl(p["item_labels"])
    _check_item_labels(item_labels)
    reg = F.regression_table(p["bank_csv"])
    e2 = json.loads(p["e2_json"].read_text()) if p["e2_json"].is_file() else None
    fit = json.loads(p["probe_results"].read_text())
    inputs: dict[str, Any] = {
        "item_labels": {"path": str(p["item_labels"]), "sha256": _sha_file(p["item_labels"])},
        "bank_csv": {"path": str(p["bank_csv"]), "sha256": _sha_file(p["bank_csv"])},
        "e2_json": {"path": str(p["e2_json"]), "sha256": _sha_file(p["e2_json"])} if e2 is not None else None,
        "probe_results": {"path": str(p["probe_results"]), "sha256": _sha_file(p["probe_results"])},
        "directions": {"path": str(p["directions"]), "sha256": _sha_file(p["directions"])} if p["directions"].is_file() else None,
        "capture_json": None, "acts": {}, "xstest_safe_labels": {}}

    rows, layers, base_acts = F.load_acts(p["acts_dir"] / f"{F.BASE}.npz")
    inputs["acts"][F.BASE] = _sha_file(p["acts_dir"] / f"{F.BASE}.npz")
    if n_layers is None:
        if not p["capture_json"].is_file():
            raise FileNotFoundError(f"{p['capture_json']} missing: pass n_layers")
        cap = json.loads(p["capture_json"].read_text())
        inputs["capture_json"] = {"path": str(p["capture_json"]), "sha256": _sha_file(p["capture_json"])}
        n_layers = int(cap["n_layers"])
        if [int(x) for x in cap.get("layers", layers)] != layers:
            raise ValueError("the capture record's layers differ from the capture's")
    deep = deep_layers(layers, n_layers)
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
    J = {rd: judge_labels(item_labels, judge_dir, rows, index, harmless_keys, xs_keep, units, rd) for rd in JUDGES}
    regressed = regressed_units(e2, labelled, reg)
    defined = {rd: defined_units(regressed, J[rd]) for rd in JUDGES}
    undefined = {rd: [u for u in regressed if u not in defined[rd]] for rd in JUDGES}
    if expect_defined is not None and sorted(expect_defined) != sorted(defined[PRIMARY_JUDGE]):
        raise ValueError(f"defined units {defined[PRIMARY_JUDGE]} differ from the expected {sorted(expect_defined)}")
    rule_key = rule_key or base
    rule = RULES.get(rule_key)
    _log(quiet, f"{base}: n = {n_layers}, layers {layers}, deep {deep}; {len(units)} captured units; defined "
                f"{defined[PRIMARY_JUDGE]}; rule {rule_key if rule else 'none'}")

    unit_acts: dict[str, dict[str, np.ndarray]] = {}
    for u in units:
        r_u, l_u, a_u = F.load_acts(p["acts_dir"] / f"{u}.npz")
        if r_u != rows or l_u != layers:
            raise ValueError(f"{u}: rows or layers differ from the base capture")
        unit_acts[u] = a_u
        inputs["acts"][u] = _sha_file(p["acts_dir"] / f"{u}.npz")
    ctx: dict[str, Any] = {"labelled": labelled, "e2_units": e2_units, "reg": reg, "d2": d2, "canary": canary,
                           "n_boot": n_boot, "inputs": inputs, "units": units, "e2": e2, "defined": defined,
                           "settings_used": {"C_grid": list(grid), "n_folds": F.N_FOLDS, "seed": seed, "n_boot": n_boot,
                                             "frr": F.FRR, "layers": layers, "positions": list(F.POSITIONS)}}

    def cell_seed(ri: int, pi: int, li: int) -> int:
        return seed + 1000 * ri + 100 * pi + li  # decision_probe_fit.run_fit

    # ---- pass 1: reader (A), R1's fit, then the reproduction gate (before any content reader)
    A: dict[str, dict[str, dict[int, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    for rd in JUDGES:
        jd = J[rd]
        for pi, pos in enumerate(F.POSITIONS):
            for li, L in enumerate(layers):
                s = cell_seed(jd.ri, pi, li)
                Xb = base_acts[pos][:, li, :]
                r = probe_reader(Xb, jd.b_idx, jd.y_b, s, grid)
                ux = {u: unit_acts[u][pos][:, li, :] for u in units}
                sc = {u: probe_scores(r, X) for u, X in ux.items()}
                rec = evaluate(jd, rows, r["nested"]["oof"], sc, "out of fold (nested CV, R1's cross-fitting)", ctx, s)
                A[rd][pos][L] = {"fit": r, "rec": rec, "scores": sc, "seed": s, "C": r["C"],
                                 "frozen": frozen_block(r, jd, Xb, ux, rec["threshold"])}
                _log(quiet, f"A {rd} {pos} L{L}: a {F.fmt(rec['a']['auroc'])} C {r['C']:g} ({time.time() - t0:.0f}s)")
    if p["directions"].is_file():
        with np.load(p["directions"], allow_pickle=False) as z:
            ctx["directions"] = {k: np.asarray(z[k]) for k in z.files}
    else:
        ctx["directions"] = None
    rep = reproduce(A, fit, layers, ctx)
    _log(quiet, f"reproduction: required {rep['required']['n_values']} values, max |diff| "
                f"{rep['required']['max_abs_diff']:.3g}, pass {rep['required']['pass']}; extra {rep['extra']['n_values']} "
                f"values, pass {rep['extra']['pass']}")
    if not rep["pass"]:
        bad = (rep["required"]["failures"] + rep["extra"]["failures"])[:8]
        raise ReproductionError("reader (A) does not reproduce probe_results.json: "
                                + "; ".join(f"{f['where']} mine {f['mine']} fit {f['fit']}" for f in bad))
    if reproduce_only:
        return {"reproduction": rep, "units": {"defined": defined}, "seconds": round(time.time() - t0, 1)}

    # ---- pass 2: the content readers (B, C) and the decision direction (D), at the same cells
    p_idx, y_p = prompt_label_rows(rows, xs_keep)
    h_rows = [i for i, k in enumerate(rows) if k[0] in F.H_BATTERIES]
    harmless_rows = [i for i, k in enumerate(rows) if k[0] == F.HARMLESS]
    for rd in JUDGES:
        if not set(J[rd].b_idx.tolist()) <= set(p_idx.tolist()):
            raise ValueError(f"{rd}: base-labelled rows outside (B)'s rows")
    cells: dict[str, dict[str, dict[str, dict]]] = {rd: {pos: {} for pos in F.POSITIONS} for rd in JUDGES}
    S: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}  # (judge, pos, layer, reader) -> unit scores
    for pi, pos in enumerate(F.POSITIONS):
        for li, L in enumerate(layers):
            Xb = base_acts[pos][:, li, :]
            ux = {u: unit_acts[u][pos][:, li, :] for u in units}
            rB = probe_reader(Xb, p_idx, y_p, cell_seed(0, pi, li), grid)
            scB = {u: probe_scores(rB, X) for u, X in ux.items()}
            wB = rB["probe"].raw_direction()[0]
            wC = mean_difference(Xb, h_rows, harmless_rows)
            baseC = projection(Xb, wC)
            scC = {u: projection(X, wC) for u, X in ux.items()}
            for rd in JUDGES:
                jd = J[rd]
                s = cell_seed(jd.ri, pi, li)
                Ar = A[rd][pos][L]
                wD = mean_difference(Xb, jd.b_idx[jd.y_b == 1], jd.b_idx[jd.y_b == 0])
                baseD = projection(Xb, wD)
                scD = {u: projection(X, wD) for u, X in ux.items()}
                recA = dict(Ar["rec"], seed=Ar["seed"], C=Ar["C"], C_at_grid_edge=Ar["C"] in (min(grid), max(grid)),
                            C_cv=Ar["fit"]["C_cv"], frozen_full_probe=Ar["frozen"],
                            fit={"labels": "decision", "n": int(len(jd.y_b)), "n_pos": int(jd.y_b.sum())})
                recB = evaluate(jd, rows, oof_on(rB, jd.b_idx), scB, "out of fold (nested CV, R1's cross-fitting)", ctx, s)
                recB.update(seed=rB["seed"], C=rB["C"], C_at_grid_edge=rB["C"] in (min(grid), max(grid)),
                            C_cv=rB["C_cv"], frozen_full_probe=frozen_block(rB, jd, Xb, ux, recB["threshold"]),
                            fit={"labels": "prompt", "n": int(len(y_p)), "n_pos": int(y_p.sum()),
                                 "auroc_heldout_prompt_labels": rB["nested"]["auroc_pooled_oof"]})
                recC = evaluate(jd, rows, baseC[jd.b_idx], scC, "in-sample (base mean difference; flagged)", ctx, s)
                recD = evaluate(jd, rows, baseD[jd.b_idx], scD, "in-sample (base mean difference; flagged)", ctx, s)
                wA = Ar["fit"]["probe"].raw_direction()[0]
                cells[rd][pos][str(L)] = {"A": recA, "B": recB, "C": recC, "D": recD,
                                          "cosines": cosines({"A": wA, "B": wB, "C": wC, "D": wD})}
                for R, sc in (("A", Ar["scores"]), ("B", scB), ("C", scC), ("D", scD)):
                    S[(rd, pos, L, R)] = sc
            _log(quiet, f"B, C, D {pos} L{L} ({time.time() - t0:.0f}s)")

    # ---- cells, comparison, rule
    def b_of(rd: str, pos: str, L: int, R: str, u: str) -> float | None:
        return cells[rd][pos][str(L)][R]["b"].get(u, {}).get(F.B_VERDICT_METRIC)

    def h_scores(rd: str, pos: str, L: int, R: str, u: str) -> tuple[np.ndarray, np.ndarray]:
        idx, y_u, h = J[rd].urows[u]
        return y_u[h], S[(rd, pos, L, R)][u][idx][h]

    def compare_at(rd: str, pos: str, cell: Mapping[str, int], what: str) -> dict[str, Any]:
        per = {}
        for u in defined[rd]:
            y, sA = h_scores(rd, pos, cell["A"], "A", u)
            _, sB = h_scores(rd, pos, cell["B"], "B", u)
            _, sC = h_scores(rd, pos, cell["C"], "C", u)
            bt = paired_delta_bootstrap(y, sA, [sB, sC], n_boot, BOOT_SEED)
            iid = paired_delta_bootstrap(y, sA, [sB, sC], n_boot, BOOT_SEED, scheme="iid")
            bA, bB, bC = (b_of(rd, pos, cell[R], R, u) for R in ("A", "B", "C"))
            if abs(bt["b_a"] - bA) > 1e-12 or abs(bt["b_others"][0] - bB) > 1e-12 or abs(bt["b_others"][1] - bC) > 1e-12:
                raise RuntimeError(f"{what} {rd} {u}: bootstrap point estimates differ from the cell records")
            per[u] = {"b_A": bA, "b_B": bB, "b_C": bC, "b_D": b_of(rd, pos, cell["D"], "D", u) if "D" in cell else None,
                      "max_content": max(bB, bC), "max_content_reader": "B" if bB >= bC else "C",
                      "delta": bt["delta"], "ci": bt["ci"], "layers": dict(cell),
                      "bootstrap": {k: bt[k] for k in ("n_boot", "seed", "scheme", "alpha", "n_boot_undefined", "n_pos", "n_neg")},
                      "iid_sensitivity": {"ci": iid["ci"], "n_boot_undefined": iid["n_boot_undefined"],
                                          "interval_above_0": iid["ci"][0] is not None and iid["ci"][0] > 0}}
        return per

    headline = {rd: {R: headline_layer({L: cells[rd][HEADLINE_POSITION][str(L)][R]["a"]["auroc"] for L in layers},
                                       n_layers) for R in READERS} for rd in JUDGES}
    comparison, upper, first_tok, est_DA, op_points, c_rank, cos_sum = {}, {}, {}, {}, {}, {}, {}
    for rd in JUDGES:
        hl = {R: headline[rd][R]["layer"] for R in READERS}
        per = compare_at(rd, HEADLINE_POSITION, hl, "headline")
        arith = verdict(per, rule)
        comparison[rd] = {"position": HEADLINE_POSITION, "layers": hl, "per_unit": per, "arithmetic": arith,
                          "iid_sensitivity": verdict({u: {"delta": v["delta"], "ci": v["iid_sensitivity"]["ci"]}
                                                      for u, v in per.items()}, rule)}
        # the content-reader upper bound (chosen by the units; descriptive)
        ub = {R: upper_bound_layer({L: {u: b_of(rd, HEADLINE_POSITION, L, R, u) for u in defined[rd]} for L in deep},
                                   deep, defined[rd]) for R in CONTENT}
        ub_cell = {"A": hl["A"], "B": ub["B"]["layer"], "C": ub["C"]["layer"]}
        ub_per = compare_at(rd, HEADLINE_POSITION, ub_cell, "upper bound")
        upper[rd] = {"label": "UPPER BOUND for the content readers: each at the deep layer maximising its median (b) "
                              "over the defined units (chosen by the units); descriptive, decides nothing",
                     "B": ub["B"], "C": ub["C"], "A_layer": hl["A"], "per_unit": ub_per,
                     "arithmetic": {**verdict(ub_per, rule), "decides": "nothing (descriptive)"}}
        # (D) against (A): the estimator, at each one's headline cell
        est_DA[rd] = {}
        for u in defined[rd]:
            y, sA = h_scores(rd, HEADLINE_POSITION, hl["A"], "A", u)
            _, sD = h_scores(rd, HEADLINE_POSITION, hl["D"], "D", u)
            dd = stats.paired_bootstrap_auroc_diff(y, sA, sD, n_boot=n_boot, seed=BOOT_SEED, secondary=False)
            est_DA[rd][u] = {"b_A": dd["auroc_a"], "b_D": dd["auroc_b"], "diff_A_minus_D": dd["diff"],
                             "ci": [dd["ci_low"], dd["ci_high"]], "layers": {"A": hl["A"], "D": hl["D"]}}
        # the first generated token: every hidden state in `cells`; the same rule's cells and their delta, descriptive
        g_cells = {R: headline_layer({L: cells[rd]["gen1"][str(L)][R]["a"]["auroc"] for L in layers}, n_layers)
                   for R in READERS}
        g_per = compare_at(rd, "gen1", {R: g_cells[R]["layer"] for R in READERS}, "gen1")
        first_tok[rd] = {"decides": "nothing (descriptive)",
                         "note": "the registration reports the first generated token at every hidden state; the cells "
                                 "below apply Setting D's rule to each reader's (a) at that position (not registered)",
                         "cells": g_cells, "per_unit": g_per,
                         "arithmetic": {**verdict(g_per, rule), "decides": "nothing (descriptive)"},
                         "b_by_layer": {R: {str(L): {u: b_of(rd, "gen1", L, R, u) for u in defined[rd]} for L in layers}
                                        for R in READERS},
                         "a_by_layer": {R: {str(L): cells[rd]["gen1"][str(L)][R]["a"]["auroc"] for L in layers}
                                        for R in READERS}}
        op_points[rd], c_rank[rd] = {}, {}
        for R in READERS:
            cr = cells[rd][HEADLINE_POSITION][str(hl[R])][R]
            op_points[rd][R] = {
                "layer": hl[R], "threshold": cr["threshold"], "base_false_alarm_rate": cr["base_false_alarm_rate"],
                "base_recall": cr["base_at_threshold"]["tpr_refusals"],
                "per_unit": {u: {"recall_P": cr["b"][u]["tpr_refusals"],
                                 "false_alarm_P": cr["b"][u]["false_refusal_rate_answered"],
                                 "recall_H": cr["b"][u]["recall_H"], "false_alarm_H": cr["b"][u]["false_alarm_H"],
                                 "predicted_refusal_rate_P": cr["b"][u]["predicted_refusal_rate"],
                                 "judged_refusal_rate_P": cr["b"][u]["judged_refusal_rate"]} for u in defined[rd]}}
            c_rank[rd][R] = {"layer": hl[R], "labelled": cr["c_auroc"]["labelled"], "e2_units": cr["c_auroc"]["e2_units"],
                             "by_layer_last": {str(L): cells[rd][HEADLINE_POSITION][str(L)][R]["c_auroc"]["labelled"].get("auroc")
                                               for L in layers},
                             "by_layer_gen1": {str(L): cells[rd]["gen1"][str(L)][R]["c_auroc"]["labelled"].get("auroc")
                                               for L in layers}}
        cos_sum[rd] = {"at_A_headline_layer": {"layer": hl["A"], **cells[rd][HEADLINE_POSITION][str(hl["A"])]["cosines"]},
                       "by_layer_last": {str(L): cells[rd][HEADLINE_POSITION][str(L)]["cosines"] for L in layers},
                       "by_layer_gen1": {str(L): cells[rd]["gen1"][str(L)]["cosines"] for L in layers}}

    prim = comparison[PRIMARY_JUDGE]["arithmetic"]
    v_block = {"judge": PRIMARY_JUDGE, "judge_name": F.READER_NAMES[PRIMARY_JUDGE], **prim,
               "rule": prim.get("rule", dict(rule) if rule else None),
               "iid_sensitivity_verdict": comparison[PRIMARY_JUDGE]["iid_sensitivity"]["verdict"],
               "prediction": "the decision labels carry the gain (delta = b(A) - max(b(B), b(C)) > 0)",
               "falsifier_reading": "monitor the decision position: the labels add nothing over content at the same cell"}
    j1 = comparison["J1"]["arithmetic"]
    j1_block = {"judge": "J1", "judge_name": F.READER_NAMES["J1"], "decision": "none (reported alongside)",
                **{k: v for k, v in j1.items() if k != "verdict"}, "rule_word_if_applied": j1.get("verdict")}
    body = {
        "label": LABEL, "version": VERSION, "registration": REGISTRATION, "base": base,
        "readers": READER_NAMES, "content_readers": list(CONTENT),
        "judges": {rd: F.READER_NAMES[rd] for rd in JUDGES}, "primary_judge": PRIMARY_JUDGE,
        "settings": {**ctx["settings_used"], "boot_seed": BOOT_SEED, "n_boot_delta": n_boot, "alpha": ALPHA,
                     "bootstrap": "paired, stratified by the unit's label (stats.paired_bootstrap_auroc_diff's draw); "
                                  "i.i.d. item draw beside it", "setting_D_tol": SETD_TOL,
                     "b_metric": F.B_VERDICT_METRIC, "headline_position": HEADLINE_POSITION,
                     "reproduction_tol": REPRO_TOL, "rule_key": rule_key},
        "interpretations": INTERPRETATIONS, "n_layers": n_layers, "layers": layers, "deep_layers": deep,
        "positions": list(F.POSITIONS),
        "units": {"captured": units, "labelled": labelled, "e2_units": e2_units, "regressed": regressed,
                  "defined": defined, "undefined": undefined,
                  "confirmed": {u: reg.get(u, {}).get("confirmed") for u in sorted(set(units) | set(labelled))},
                  "b_prompt_rows": {"n": int(len(y_p)), "n_H": int(y_p.sum()), "n_xstest_safe": len(xs_keep & {k[1] for k in rows if k[0] == F.XS}),
                                    "n_harmless": len(harmless_rows)}},
        "reproduction": rep,
        "headline_cells": headline,
        "verdict": v_block, "J1_alongside": j1_block,
        "comparison": comparison, "upper_bound": upper, "estimator_D_vs_A": est_DA, "first_generated_token": first_tok,
        "operating_points": op_points, "c_regression_ranking": c_rank, "cosines": cos_sum,
        "cells": cells, "inputs": inputs, "seconds": round(time.time() - t0, 1),
        "git_commit": F.git_commit(), "code_sha256": _sha_file(Path(__file__)), "fit_code_sha256": _sha_file(Path(F.__file__)),
    }
    return body


def run(results: Path | str, base: str, *, out: Path | str | None = None, **kw: Any) -> Path:
    """compute(), then write content_readers.json (atomically); nothing is written when the reproduction gate fails."""
    results = Path(results)
    p = _paths(results, base, {k: kw.get(k) for k in ("root", "acts_dir", "probe_results")} | {"out": out})
    body = compute(results, base, **kw)
    from sm.interp import assistant_axis as AX

    AX.write_open_json(p["out"], body, {"stage": "content_readers", "label": LABEL, "version": VERSION})
    if not kw.get("quiet"):
        print_summary(body)
        print(f"[content-readers] written {p['out']}", flush=True)
    return p["out"]


def print_summary(body: Mapping[str, Any]) -> None:
    fmt = F.fmt
    for rd in JUDGES:
        hc = body["headline_cells"][rd]
        print(f"[content-readers] {rd} headline cells (last prompt token): "
              + ", ".join(f"{R} L{hc[R]['layer']}{' (fallback)' if hc[R]['fallback'] else ''} a {fmt(hc[R]['a'])}"
                          for R in READERS))
        for u, v in body["comparison"][rd]["per_unit"].items():
            print(f"[content-readers] {rd} {u}: b A {fmt(v['b_A'])} B {fmt(v['b_B'])} C {fmt(v['b_C'])} D "
                  f"{fmt(v['b_D'])}; delta {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
        a = body["comparison"][rd]["arithmetic"]
        print(f"[content-readers] {rd}: interval above 0 on {a['n_interval_above_0']}/{a['n_defined']}, median delta "
              f"{a['median_delta']:+.3f}, delta <= 0 on {a['n_delta_le_0']} -> {a.get('verdict')}"
              f"{'' if rd == PRIMARY_JUDGE else ' (alongside, no decision)'}")


def _csv(s: str | None) -> list[str] | None:
    return None if s is None else [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sm.exploratory.content_readers", description=__doc__,
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
    ap.add_argument("--probe-results", default=None, help="default <root>/probe_results.json (the reproduction target)")
    ap.add_argument("--directions", default=None, help="default <root>/directions.npz")
    ap.add_argument("--out", default=None, help="default <root>/content_readers.json")
    ap.add_argument("--n-layers", type=int, default=None, help="default: the capture record's n_layers")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=F.SEED)
    ap.add_argument("--grid", default=None, help="C grid (default the fit's); tests only")
    ap.add_argument("--rule", default=None, help="the base whose registered rule applies (default --base)")
    ap.add_argument("--expect-defined", default=None, help="comma list; stop unless the defined units are these")
    ap.add_argument("--reproduce-only", action="store_true", help="run the reproduction gate only; write nothing")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    grid = tuple(float(x) for x in a.grid.split(",")) if a.grid else F.C_GRID
    kw = dict(n_boot=a.n_boot, seed=a.seed, grid=grid, rule_key=a.rule, expect_defined=_csv(a.expect_defined),
              n_layers=a.n_layers, quiet=a.quiet, root=a.root, acts_dir=a.acts_dir, capture_json=a.capture_json,
              item_labels=a.item_labels, judge_dir=a.judge_dir, bank_csv=a.bank_csv, e2_json=a.e2_json,
              probe_results=a.probe_results, directions=a.directions)
    try:
        if a.reproduce_only:
            rep = compute(Path(a.results), a.base, reproduce_only=True, **kw)["reproduction"]
            print(json.dumps({k: rep[k] for k in ("pass", "n_values_required", "fit_code_unchanged")}
                             | {"required": {k: rep["required"][k] for k in ("pass", "n_values", "max_abs_diff", "n_fail")},
                                "extra": {k: rep["extra"][k] for k in ("pass", "n_values", "max_abs_diff", "n_fail",
                                                                       "directions_max_abs_diff")}}, indent=1))
            return 0
        run(Path(a.results), a.base, out=a.out, **kw)
    except ReproductionError as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
