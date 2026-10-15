"""Statistics used by the pre-registered analysis (prereg Section 7).

All functions are pure: they take numpy-compatible arrays and return floats or
plain dicts.

Bases (v0.3, fidelity review M13 and N5). The bank pools about 15 derivatives of each of four bases, so units are
not independent across bases. Every bootstrap below takes an optional ``clusters`` array (the base of each unit).
With two or more distinct bases the primary resampling (``scheme="within_base"``, the default) is a base-stratified
bootstrap: every base is kept in every draw and its units are resampled within it (within each (base, class) cell
for the stratified AUROC routines), because the four bases are a fixed design factor, not a sample of bases. The
two-stage cluster bootstrap of M13 (bases drawn with replacement, then units within each drawn base) is computed as
well and reported under ``secondary`` (a sensitivity line: four clusters give only 35 distinct cluster multisets,
and a draw without a regressed unit has no AUROC). ``scheme="cluster"`` swaps the two. Paired routines use the same
draw for both scores. With no clusters, or a single one, the original i.i.d. resampling runs unchanged (same draws
for the same seed) and there is no secondary. Every result records ``bootstrap`` ("iid", "within_base_stratified"
or "cluster_two_stage"), ``n_clusters`` and the undefined-draw counts ``n_boot``, ``n_boot_undefined`` and
``undefined_share`` (NaN draws are never dropped silently: intervals are taken over the defined draws and say so in
``ci_from_defined_draws_only``, and ``bootstrap_p`` counts an undefined draw against the alternative). Bootstrap
routines take an explicit seed and never touch global random state. Bootstrap distributions are returned under the key "boot" so the
caller can derive one-sided p-values at any null value; strip that key before
serialising.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata

Z95 = 1.959963984540054
DEFAULT_N_BOOT = 2000


def _as_arrays(labels: Sequence, scores: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(bool).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"labels and scores differ in length: {y.shape} vs {s.shape}")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN; drop missing units before calling")
    return y, s


def auroc(labels: Sequence, scores: Sequence) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half).

    Returns NaN when either class is empty. Higher scores are taken to indicate
    the positive class.
    """
    y, s = _as_arrays(labels, scores)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def recall_at_fpr(labels: Sequence, scores: Sequence, fpr: float = 0.01) -> float:
    """True-positive rate at the strictest threshold whose false-positive rate is <= fpr.

    A unit is flagged when its score is strictly greater than the threshold.
    With n negatives, at most floor(fpr * n) negatives may be flagged, so the
    threshold is the (floor(fpr * n) + 1)-th highest negative score. Ties at the
    threshold are not flagged, which keeps the realised FPR at or below fpr.
    """
    y, s = _as_arrays(labels, scores)
    pos = s[y]
    neg = np.sort(s[~y])[::-1]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    k = int(math.floor(fpr * neg.size + 1e-12))
    threshold = neg[k] if k < neg.size else -np.inf
    return float(np.mean(pos > threshold))


def threshold_at_fpr(negative_scores: Sequence, fpr: float = 0.01) -> float:
    """Threshold used by recall_at_fpr, computed from negative scores alone."""
    neg = np.sort(np.asarray(negative_scores, dtype=float).ravel())[::-1]
    if neg.size == 0:
        return float("nan")
    k = int(math.floor(fpr * neg.size + 1e-12))
    return float(neg[k]) if k < neg.size else float("-inf")


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k / n (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def _percentile_ci(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile interval of the defined draws. Undefined (NaN) draws are never dropped silently: every caller
    reports their number with ``_undefined`` and ``bootstrap_p`` counts them against the alternative."""
    b = boot[~np.isnan(boot)]
    if b.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(b, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def _undefined(boot: np.ndarray) -> dict:
    """Counts of undefined bootstrap draws (an AUROC draw with one class, a constant Spearman draw)."""
    b = np.asarray(boot, dtype=float)
    n_undef = int(np.isnan(b).sum())
    return {"n_boot": int(b.size), "n_boot_undefined": n_undef,
            "undefined_share": float(n_undef / b.size) if b.size else float("nan"),
            "ci_from_defined_draws_only": bool(n_undef > 0)}


def bootstrap_p(boot: Sequence[float], null: float, alternative: str) -> float:
    """One-sided bootstrap p-value with the +1 correction.

    alternative "greater": evidence that the statistic exceeds `null`;
    p = (1 + #{boot <= null} + #{undefined}) / (B + 1), B counting every draw. "less" mirrors it. An undefined
    (NaN) draw counts as a draw that does not support the alternative (FIDELITY_REVIEW N5: never dropped, which
    would condition the p-value on the draws that worked); with no defined draw the p-value is NaN.
    """
    b = np.asarray(boot, dtype=float)
    n_undef = int(np.isnan(b).sum())
    d = b[~np.isnan(b)]
    if d.size == 0:
        return float("nan")
    if alternative == "greater":
        count = int(np.sum(d <= null))
    elif alternative == "less":
        count = int(np.sum(d >= null))
    else:
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")
    return float((1 + count + n_undef) / (b.size + 1))


def _stratified_resample(rng: np.random.Generator, pos_idx: np.ndarray, neg_idx: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [rng.choice(pos_idx, size=pos_idx.size, replace=True), rng.choice(neg_idx, size=neg_idx.size, replace=True)]
    )


def _cluster_groups(clusters: Sequence | None, n: int) -> list[np.ndarray] | None:
    """Unit indices per cluster, or None when there are fewer than two clusters (i.i.d. resampling)."""
    if clusters is None:
        return None
    c = np.asarray(clusters, dtype=object).ravel()
    if c.size != n:
        raise ValueError(f"clusters has {c.size} entries for {n} units")
    keys = sorted({str(x) for x in c})
    if len(keys) < 2:
        return None
    cs = np.array([str(x) for x in c], dtype=object)
    return [np.flatnonzero(cs == k) for k in keys]


def _within(rng: np.random.Generator, idx: np.ndarray, y: np.ndarray | None, parts: list[np.ndarray]) -> None:
    if y is None:
        parts.append(rng.choice(idx, size=idx.size, replace=True))
        return
    for cls in (True, False):
        sub = idx[y[idx] == cls]
        if sub.size:
            parts.append(rng.choice(sub, size=sub.size, replace=True))


def cluster_resample(rng: np.random.Generator, groups: Sequence[np.ndarray], y: np.ndarray | None = None) -> np.ndarray:
    """Two-stage cluster bootstrap draw: clusters with replacement, then units within each drawn cluster.

    With ``y`` the within-cluster draw is stratified by class (each drawn cluster keeps its class counts).
    """
    picks = rng.integers(0, len(groups), size=len(groups))
    parts: list[np.ndarray] = []
    for g in picks:
        _within(rng, groups[int(g)], y, parts)
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def within_base_resample(rng: np.random.Generator, groups: Sequence[np.ndarray], y: np.ndarray | None = None) -> np.ndarray:
    """Base-stratified bootstrap draw (FIDELITY_REVIEW N5): every base is kept, units are resampled within it.

    The four bases are a fixed design factor, not a sample of bases, so the primary interval resamples only the
    units inside each base. With ``y`` the draw is stratified by class within each base (every (base, class) cell
    keeps its count, so an AUROC draw is defined whenever the pooled point estimate is).
    """
    parts: list[np.ndarray] = []
    for idx in groups:
        _within(rng, idx, y, parts)
    return np.concatenate(parts) if parts else np.array([], dtype=int)


PRIMARY_SCHEME = "within_base"
SCHEMES = ("within_base", "cluster")


def _scheme_name(scheme: str, groups: list[np.ndarray] | None) -> str:
    if not groups:
        return "iid"
    return {"within_base": "within_base_stratified", "cluster": "cluster_two_stage"}[scheme]


def _boot_meta(groups: list[np.ndarray] | None, scheme: str = PRIMARY_SCHEME) -> dict:
    return {"bootstrap": _scheme_name(scheme, groups), "n_clusters": len(groups) if groups else 1}


def _check_scheme(scheme: str) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"bootstrap scheme must be one of {SCHEMES}, got {scheme!r}")


def _draw(rng: np.random.Generator, groups: list[np.ndarray], scheme: str, y: np.ndarray | None = None) -> np.ndarray:
    return within_base_resample(rng, groups, y) if scheme == "within_base" else cluster_resample(rng, groups, y)


def _secondary(groups: list[np.ndarray] | None, scheme: str, secondary: bool, run: Callable[[str, int], np.ndarray],
               seed: int, alpha: float) -> dict | None:
    """The other scheme's interval (with clusters only): the four-base cluster bootstrap when the primary is the
    within-base one (FIDELITY_REVIEW N5: a sensitivity line, 4 clusters give 35 distinct cluster multisets)."""
    if not groups or not secondary:
        return None
    other = "cluster" if scheme == "within_base" else "within_base"
    boot = run(other, seed + 7919)
    lo, hi = _percentile_ci(boot, alpha)
    return {"bootstrap": _scheme_name(other, groups), "role": "secondary (sensitivity)", "ci_low": lo, "ci_high": hi,
            **_undefined(boot), "boot": boot}


def bootstrap_auroc_ci(
    labels: Sequence,
    scores: Sequence,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence | None = None,
    scheme: str = PRIMARY_SCHEME,
    secondary: bool = True,
) -> dict:
    """AUROC with a stratified percentile bootstrap CI (class counts held fixed). With ``clusters`` (two or more
    bases) the primary draw is ``scheme`` ("within_base": units resampled within each base and class, every base
    kept; "cluster": the two-stage cluster bootstrap) and the other scheme is reported under ``secondary``."""
    _check_scheme(scheme)
    y, s = _as_arrays(labels, scores)
    point = auroc(y, s)
    groups = _cluster_groups(clusters, y.size)
    out = {"auroc": point, "n": int(y.size), "n_pos": int(y.sum()), "n_neg": int((~y).sum()), **_boot_meta(groups, scheme)}
    if math.isnan(point):
        out.update(ci_low=float("nan"), ci_high=float("nan"), boot=np.array([]), **_undefined(np.array([])))
        return out
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)

    def run(sch: str, sd: int) -> np.ndarray:
        rng = np.random.default_rng(sd)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            idx = _draw(rng, groups, sch, y) if groups else _stratified_resample(rng, pos_idx, neg_idx)
            boot[b] = auroc(y[idx], s[idx])
        return boot

    boot = run(scheme, seed)
    lo, hi = _percentile_ci(boot, alpha)
    out.update(ci_low=lo, ci_high=hi, boot=boot, **_undefined(boot))
    sec = _secondary(groups, scheme, secondary, run, seed, alpha)
    if sec is not None:
        out["secondary"] = sec
    return out


def paired_bootstrap_auroc_diff(
    labels: Sequence,
    scores_a: Sequence,
    scores_b: Sequence,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence | None = None,
    scheme: str = PRIMARY_SCHEME,
    secondary: bool = True,
) -> dict:
    """Paired stratified bootstrap of AUROC(a) - AUROC(b) on the same units (schemes as ``bootstrap_auroc_ci``)."""
    _check_scheme(scheme)
    y, a = _as_arrays(labels, scores_a)
    _, b_scores = _as_arrays(labels, scores_b)
    auc_a = auroc(y, a)
    auc_b = auroc(y, b_scores)
    groups = _cluster_groups(clusters, y.size)
    out = {
        **_boot_meta(groups, scheme),
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "diff": auc_a - auc_b,
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
    }
    if math.isnan(auc_a) or math.isnan(auc_b):
        out.update(ci_low=float("nan"), ci_high=float("nan"), boot=np.array([]), **_undefined(np.array([])))
        return out
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)

    def run(sch: str, sd: int) -> np.ndarray:
        rng = np.random.default_rng(sd)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = _draw(rng, groups, sch, y) if groups else _stratified_resample(rng, pos_idx, neg_idx)
            boot[i] = auroc(y[idx], a[idx]) - auroc(y[idx], b_scores[idx])
        return boot

    boot = run(scheme, seed)
    lo, hi = _percentile_ci(boot, alpha)
    out.update(ci_low=lo, ci_high=hi, boot=boot, **_undefined(boot))
    sec = _secondary(groups, scheme, secondary, run, seed, alpha)
    if sec is not None:
        out["secondary"] = sec
    return out


def bootstrap_stat_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.median,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence | None = None,
    scheme: str = PRIMARY_SCHEME,
    secondary: bool = True,
) -> dict:
    """Percentile bootstrap CI for a one-sample statistic (median by default); schemes as ``bootstrap_auroc_ci``.
    NaN values (missing units) are dropped before resampling and counted as ``n_missing``."""
    _check_scheme(scheme)
    v = np.asarray(values, dtype=float).ravel()
    keep = ~np.isnan(v)
    c = None if clusters is None else np.asarray(clusters, dtype=object).ravel()
    if c is not None and c.size != v.size:
        raise ValueError(f"clusters has {c.size} entries for {v.size} values")
    n_missing = int((~keep).sum())
    v = v[keep]
    c = None if c is None else c[keep]
    groups = _cluster_groups(c, v.size)
    if v.size == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_missing": n_missing,
                "boot": np.array([]), **_boot_meta(None), **_undefined(np.array([]))}

    def run(sch: str, sd: int) -> np.ndarray:
        rng = np.random.default_rng(sd)
        if groups:
            return np.array([float(stat(v[_draw(rng, groups, sch)])) for _ in range(n_boot)])
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        return np.array([float(stat(v[row])) for row in idx])

    boot = run(scheme, seed)
    lo, hi = _percentile_ci(boot, alpha)
    out = {"estimate": float(stat(v)), "ci_low": lo, "ci_high": hi, "n": int(v.size), "n_missing": n_missing, "boot": boot,
           **_boot_meta(groups, scheme), **_undefined(boot)}
    sec = _secondary(groups, scheme, secondary, run, seed, alpha)
    if sec is not None:
        out["secondary"] = sec
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties); NaN if either side is constant."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size != ya.size:
        raise ValueError("x and y differ in length")
    if xa.size < 2:
        return float("nan")
    rx = rankdata(xa)
    ry = rankdata(ya)
    sx = rx.std()
    sy = ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence | None = None,
    scheme: str = PRIMARY_SCHEME,
    secondary: bool = True,
) -> dict:
    """Spearman rho with a pairs-bootstrap percentile CI (schemes as ``bootstrap_auroc_ci``) and the asymptotic p."""
    _check_scheme(scheme)
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    keep = ~(np.isnan(xa) | np.isnan(ya))
    c = None if clusters is None else np.asarray(clusters, dtype=object).ravel()
    if c is not None and c.size != keep.size:
        raise ValueError(f"clusters has {c.size} entries for {keep.size} pairs")
    xa, ya = xa[keep], ya[keep]
    c = None if c is None else c[keep]
    rho = spearman(xa, ya)
    n = int(xa.size)
    groups = _cluster_groups(c, n)
    out: dict = {"rho": rho, "n": n, **_boot_meta(groups, scheme)}
    if math.isnan(rho):
        out.update(p_two_sided=float("nan"), ci_low=float("nan"), ci_high=float("nan"), boot=np.array([]), **_undefined(np.array([])))
        return out
    if n > 2 and abs(rho) < 1.0:
        from scipy.stats import t as t_dist

        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
        p = float(2.0 * t_dist.sf(abs(t_stat), df=n - 2))
    else:
        p = 0.0 if n > 2 else float("nan")
    def run(sch: str, sd: int) -> np.ndarray:
        rng = np.random.default_rng(sd)
        if groups:
            return np.array([spearman(xa[row], ya[row]) for row in (_draw(rng, groups, sch) for _ in range(n_boot))])
        idx = rng.integers(0, n, size=(n_boot, n))
        return np.array([spearman(xa[row], ya[row]) for row in idx])

    boot = run(scheme, seed)
    lo, hi = _percentile_ci(boot, alpha)
    out.update(p_two_sided=p, ci_low=lo, ci_high=hi, boot=boot, **_undefined(boot))
    sec = _secondary(groups, scheme, secondary, run, seed, alpha)
    if sec is not None:
        out["secondary"] = sec
    return out


def holm(pvalues: Sequence[float | None], alpha: float = 0.05, family_size: int | None = None) -> list[dict]:
    """Holm step-down adjustment.

    None or NaN entries get no adjusted p-value. By default the family size m is the number of
    present p-values; with ``family_size`` m is fixed (prereg Section 7: six primary tests), which
    is the same as counting every untestable hypothesis as p = 1.

    Returns one dict per input: {"p": raw, "p_adj": adjusted or None, "reject": bool or None}.
    """
    present = [(i, float(p)) for i, p in enumerate(pvalues) if p is not None and not math.isnan(float(p))]
    m = len(present)
    if family_size is not None:
        if family_size < m:
            raise ValueError(f"family_size {family_size} is smaller than the {m} p-values given")
        m = int(family_size)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (i, p) in enumerate(sorted(present, key=lambda t: t[1])):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    out = []
    for i, p in enumerate(pvalues):
        if i in adjusted:
            out.append({"p": float(p), "p_adj": adjusted[i], "reject": bool(adjusted[i] <= alpha)})
        else:
            out.append({"p": None if p is None else float(p), "p_adj": None, "reject": None})
    return out


def strip_boot(d: dict) -> dict:
    """Copy of a result dict without the bootstrap distributions (the primary one and the secondary one)."""
    out = {k: v for k, v in d.items() if k != "boot"}
    if isinstance(out.get("secondary"), dict):
        out["secondary"] = {k: v for k, v in out["secondary"].items() if k != "boot"}
    return out


def secondary_p(d: dict, null: float, alternative: str) -> float | None:
    """One-sided bootstrap p-value of the secondary (cluster) scheme of a result, None when it has none."""
    sec = d.get("secondary")
    if not isinstance(sec, dict) or "boot" not in sec:
        return None
    return bootstrap_p(sec["boot"], null, alternative)
