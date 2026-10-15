"""Direction extraction and controls on residual-stream activations.

All functions take activations as numpy arrays of shape (n, hidden) and are
pure (no I/O) except the save/load helpers at the bottom. Randomness is always
drawn from an explicit ``numpy.random.Generator`` built from a ``seed``.

Conventions
-----------
* ``pos`` is the positive class (harmful prompts for the refusal direction,
  trait-eliciting responses for persona vectors); ``neg`` is the contrast class.
* Directions point from ``neg`` to ``pos``: projections of ``pos`` are larger.
* AUROC values use ``pos`` as the positive label.

Direction file format (docs/HARNESS_CONTRACTS.md): ``<name>.npy`` float32 of
shape (hidden,), unit norm, plus a ``<name>.json`` sidecar with keys base,
revision, layer, position, n_pos, n_neg, prompt_set_sha256, vector_sha256,
created. ``vector_sha256`` is the SHA-256 of the little-endian float32 bytes
of the vector (``np.asarray(v, "<f4").tobytes()``), not of the .npy file, so it
is independent of the numpy header version.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MAX_SUBSPACE_K = 5
SIDECAR_REQUIRED = (
    "base",
    "revision",
    "layer",
    "position",
    "n_pos",
    "n_neg",
    "prompt_set_sha256",
    "vector_sha256",
    "created",
)


# ---------------------------------------------------------------------------
# Basic linear algebra
# ---------------------------------------------------------------------------


def _as2d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, hidden), got shape {a.shape}")
    return a


def _check_pair(pos: np.ndarray, neg: np.ndarray, min_n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    p = _as2d(pos, "pos")
    n = _as2d(neg, "neg")
    if p.shape[1] != n.shape[1]:
        raise ValueError(f"hidden size mismatch: pos {p.shape[1]} vs neg {n.shape[1]}")
    if p.shape[0] < min_n or n.shape[0] < min_n:
        raise ValueError(f"need at least {min_n} rows per class, got pos={p.shape[0]} neg={n.shape[0]}")
    return p, n


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return v / ||v||; raises on a zero vector."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(a))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("cannot normalise a zero or non-finite vector")
    return a / norm


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity of two 1-D vectors (0.0 if either is zero)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx < eps or ny < eps:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def diff_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Raw difference of class means, mean(pos) - mean(neg), float64 (hidden,)."""
    p, n = _check_pair(pos, neg)
    return p.mean(axis=0) - n.mean(axis=0)


def dom_direction(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit difference-of-means direction and the norm of the raw difference."""
    d = diff_of_means(pos, neg)
    norm = float(np.linalg.norm(d))
    return unit(d), norm


def random_unit_vectors(n: int, hidden: int, seed: int) -> np.ndarray:
    """n i.i.d. uniformly random unit vectors, shape (n, hidden)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, hidden))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Separation statistics along a direction
# ---------------------------------------------------------------------------


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Mann-Whitney AUROC, P(score_pos > score_neg) + 0.5 P(tie)."""
    sp = np.asarray(scores_pos, dtype=np.float64).reshape(-1)
    sn = np.asarray(scores_neg, dtype=np.float64).reshape(-1)
    if sp.size == 0 or sn.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([sp, sn]))
    r_pos = float(ranks[: sp.size].sum())
    u = r_pos - sp.size * (sp.size + 1) / 2.0
    return float(u / (sp.size * sn.size))


def projection_stats(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> dict[str, float]:
    """Projections of both classes on the unit direction: means, stds, gap, d-prime, AUROC."""
    p, n = _check_pair(pos, neg)
    u = unit(direction)
    if u.shape[0] != p.shape[1]:
        raise ValueError("direction and activations differ in hidden size")
    sp, sn = p @ u, n @ u
    var_p = float(sp.var(ddof=1)) if sp.size > 1 else 0.0
    var_n = float(sn.var(ddof=1)) if sn.size > 1 else 0.0
    pooled = float(np.sqrt(0.5 * (var_p + var_n)))
    gap = float(sp.mean() - sn.mean())
    return {
        "mean_pos": float(sp.mean()),
        "mean_neg": float(sn.mean()),
        "std_pos": float(np.sqrt(var_p)),
        "std_neg": float(np.sqrt(var_n)),
        "gap": gap,
        "dprime": float(gap / pooled) if pooled > 0 else float("inf") if gap > 0 else 0.0,
        "auroc": auroc(sp, sn),
    }


def projection_auroc(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    return projection_stats(pos, neg, direction)["auroc"]


def dprime(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    return projection_stats(pos, neg, direction)["dprime"]


# ---------------------------------------------------------------------------
# Stability and null controls
# ---------------------------------------------------------------------------


def split_half_cosine(pos: np.ndarray, neg: np.ndarray, k: int = 10, seed: int = 0) -> dict[str, Any]:
    """Cosine between DoM directions fit on two disjoint random halves, over k splits.

    Each split permutes pos and neg independently and halves each class, so both
    halves contain floor(n/2) rows of every class.
    """
    p, n = _check_pair(pos, neg, min_n=2)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(k)):
        ip = rng.permutation(p.shape[0])
        ineg = rng.permutation(n.shape[0])
        hp, hn = p.shape[0] // 2, n.shape[0] // 2
        d1 = diff_of_means(p[ip[:hp]], n[ineg[:hn]])
        d2 = diff_of_means(p[ip[hp : 2 * hp]], n[ineg[hn : 2 * hn]])
        vals.append(cosine(d1, d2))
    arr = np.asarray(vals)
    return {
        "k": int(k),
        "values": [float(v) for v in arr],
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
    }


def shuffled_label_null(
    pos: np.ndarray,
    neg: np.ndarray,
    n_shuffles: int = 10,
    k_splits: int = 10,
    seed: int = 0,
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Label-permutation null for the DoM direction.

    For each shuffle the pooled rows are relabelled at random (class sizes kept),
    then (a) the split-half cosine of the shuffled DoM is computed (mean over
    k_splits) and (b) the cosine of the shuffled DoM with ``reference`` (default:
    the true DoM direction). The prereg validity rule compares the true
    split-half cosine with ``split_half_max``.
    """
    p, n = _check_pair(pos, neg, min_n=2)
    ref = unit(reference) if reference is not None else dom_direction(p, n)[0]
    pooled = np.concatenate([p, n], axis=0)
    n_pos = p.shape[0]
    rng = np.random.default_rng(seed)
    sh_means: list[float] = []
    cos_true: list[float] = []
    for i in range(int(n_shuffles)):
        perm = rng.permutation(pooled.shape[0])
        sp, sn = pooled[perm[:n_pos]], pooled[perm[n_pos:]]
        sh = split_half_cosine(sp, sn, k=k_splits, seed=int(rng.integers(0, 2**31 - 1)))
        sh_means.append(sh["mean"])
        cos_true.append(cosine(diff_of_means(sp, sn), ref))
    return {
        "n_shuffles": int(n_shuffles),
        "split_half": sh_means,
        "split_half_max": float(np.max(sh_means)),
        "split_half_mean": float(np.mean(sh_means)),
        "cos_to_reference": cos_true,
        "abs_cos_to_reference_max": float(np.max(np.abs(cos_true))),
    }


def random_direction_baseline(
    pos: np.ndarray,
    neg: np.ndarray,
    n_dirs: int = 10,
    seed: int = 0,
    reference: np.ndarray | None = None,
) -> dict[str, Any]:
    """Projection AUROC and d-prime of n_dirs random unit directions.

    Because a random direction has arbitrary sign, ``auroc_oriented`` reports
    max(AUROC, 1 - AUROC). If ``reference`` is given, |cos| to it is reported.
    """
    p, n = _check_pair(pos, neg, min_n=1)
    dirs = random_unit_vectors(int(n_dirs), p.shape[1], seed)
    aurocs, oriented, dps, coss = [], [], [], []
    for d in dirs:
        st = projection_stats(p, n, d)
        aurocs.append(st["auroc"])
        oriented.append(max(st["auroc"], 1.0 - st["auroc"]))
        dps.append(abs(st["dprime"]) if np.isfinite(st["dprime"]) else 0.0)
        if reference is not None:
            coss.append(abs(cosine(d, reference)))
    out: dict[str, Any] = {
        "n_dirs": int(n_dirs),
        "auroc": aurocs,
        "auroc_oriented": oriented,
        "auroc_oriented_max": float(np.max(oriented)),
        "auroc_oriented_mean": float(np.mean(oriented)),
        "abs_dprime": dps,
        "abs_dprime_max": float(np.max(dps)),
    }
    if reference is not None:
        out["abs_cos_to_reference"] = coss
        out["abs_cos_to_reference_max"] = float(np.max(coss))
    return out


# ---------------------------------------------------------------------------
# Logistic probe
# ---------------------------------------------------------------------------


@dataclass
class LinearProbe:
    """Standardised logistic probe: score(x) = coef . ((x - mean) / scale) + intercept."""

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    C: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def decision_function(self, X: np.ndarray, shift: np.ndarray | None = None) -> np.ndarray:
        """Probe logits. ``shift`` (hidden,) is subtracted from X first.

        Label-free realignment (prereg Section 4) is
        ``decision_function(X_deriv, shift=mu_deriv_harmless - mu_base_harmless)``.
        """
        x = np.asarray(X, dtype=np.float64)
        if shift is not None:
            x = x - np.asarray(shift, dtype=np.float64).reshape(1, -1)
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def raw_direction(self) -> np.ndarray:
        """Unit weight vector in the original (unstandardised) activation space."""
        return unit(self.coef / self.scale)


def _make_probe_pipeline(C: float, seed: int, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
    )


def _xy(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, n = _check_pair(pos, neg)
    X = np.concatenate([p, n], axis=0)
    y = np.concatenate([np.ones(p.shape[0], dtype=int), np.zeros(n.shape[0], dtype=int)])
    return X, y


def cv_probe_auroc(
    X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 0, C: float = 1.0, max_iter: int = 2000
) -> dict[str, Any]:
    """Out-of-fold AUROC of the standardised logistic probe (StratifiedKFold)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    n_min = int(min(np.sum(y == 1), np.sum(y == 0)))
    folds = max(2, min(int(cv), n_min))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(y.shape[0], dtype=np.float64)
    fold_aucs: list[float] = []
    for tr, te in skf.split(X, y):
        pipe = _make_probe_pipeline(C, seed, max_iter)
        pipe.fit(X[tr], y[tr])
        s = pipe.decision_function(X[te])
        oof[te] = s
        if len(np.unique(y[te])) == 2:
            fold_aucs.append(float(roc_auc_score(y[te], s)))
    return {
        "folds": folds,
        "auroc_cv": float(roc_auc_score(y, oof)),
        "auroc_folds": fold_aucs,
    }


def fit_logistic_probe(
    pos: np.ndarray,
    neg: np.ndarray,
    cv: int = 5,
    seed: int = 0,
    C: float = 1.0,
    max_iter: int = 2000,
) -> tuple[LinearProbe, dict[str, Any]]:
    """Fit the probe on all rows and report its cross-validated AUROC.

    Returns (probe fitted on all data, {"folds", "auroc_cv", "auroc_folds"}).
    """
    X, y = _xy(pos, neg)
    cvres = cv_probe_auroc(X, y, cv=cv, seed=seed, C=C, max_iter=max_iter)
    pipe = _make_probe_pipeline(C, seed, max_iter)
    pipe.fit(X, y)
    scaler, lr = pipe[0], pipe[-1]
    probe = LinearProbe(
        coef=np.asarray(lr.coef_[0], dtype=np.float64),
        intercept=float(lr.intercept_[0]),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        C=float(C),
    )
    return probe, cvres


def probe_auroc(probe: LinearProbe, pos: np.ndarray, neg: np.ndarray, shift: np.ndarray | None = None) -> float:
    """AUROC of a fitted probe on held-out activations."""
    p, n = _check_pair(pos, neg)
    return auroc(probe.decision_function(p, shift), probe.decision_function(n, shift))


def probe_shuffled_null(
    pos: np.ndarray,
    neg: np.ndarray,
    n_shuffles: int = 10,
    cv: int = 5,
    seed: int = 0,
    C: float = 1.0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Cross-validated probe AUROC under permuted labels (expected near 0.5)."""
    X, y = _xy(pos, neg)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_shuffles)):
        ys = rng.permutation(y)
        vals.append(cv_probe_auroc(X, ys, cv=cv, seed=int(rng.integers(0, 2**31 - 1)), C=C, max_iter=max_iter)["auroc_cv"])
    return {
        "n_shuffles": int(n_shuffles),
        "auroc_cv": vals,
        "auroc_cv_max": float(np.max(vals)),
        "auroc_cv_mean": float(np.mean(vals)),
    }


# ---------------------------------------------------------------------------
# Refusal subspace
# ---------------------------------------------------------------------------


def refusal_subspace(
    pos: np.ndarray, neg: np.ndarray, k: int = MAX_SUBSPACE_K, paired: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-k principal directions of the per-prompt contrast residuals.

    Contrast residual for harmful prompt i: c_i = pos_i - mean(neg) (or
    pos_i - neg_i when ``paired`` and the sets are row-aligned). The basis is
    the top-k right singular vectors of the UNCENTRED contrast matrix, so the
    first component tracks the DoM direction when the mean shift dominates and
    later components capture prompt-dependent refusal variation. Each component
    is sign-fixed so that the mean contrast projects positively.

    Returns (basis of shape (k, hidden) with orthonormal rows, info dict).
    """
    p, n = _check_pair(pos, neg, min_n=1)
    if k < 1 or k > MAX_SUBSPACE_K:
        raise ValueError(f"k must be in [1, {MAX_SUBSPACE_K}], got {k}")
    if paired:
        if p.shape[0] != n.shape[0]:
            raise ValueError("paired=True needs equal-size, row-aligned pos and neg")
        C = p - n
    else:
        C = p - n.mean(axis=0, keepdims=True)
    k_eff = int(min(k, C.shape[0], C.shape[1]))
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    basis = vt[:k_eff].copy()
    mean_c = C.mean(axis=0)
    for i in range(k_eff):
        if float(basis[i] @ mean_c) < 0:
            basis[i] = -basis[i]
    energy = s**2
    dom = unit(mean_c) if np.linalg.norm(mean_c) > 0 else None
    info = {
        "k_requested": int(k),
        "k": k_eff,
        "paired": bool(paired),
        "singular_values": [float(x) for x in s[:k_eff]],
        "energy_fraction": [float(x) for x in (energy[:k_eff] / energy.sum())] if energy.sum() > 0 else [],
        "cos_pc1_dom": float(cosine(basis[0], dom)) if dom is not None else 0.0,
    }
    return basis, info


def subspace_projection_norm(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Norm of each row of X projected onto the row space of an orthonormal basis."""
    x = _as2d(X, "X")
    b = _as2d(basis, "basis")
    return np.linalg.norm(x @ b.T, axis=1)


# ---------------------------------------------------------------------------
# Hashing and file I/O (direction file format from the contracts)
# ---------------------------------------------------------------------------


def vector_sha256(v: np.ndarray) -> str:
    """SHA-256 of the little-endian float32 bytes of v (C order)."""
    a = np.ascontiguousarray(np.asarray(v, dtype="<f4"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def prompt_set_sha256(pos_prompts: Sequence[str], neg_prompts: Sequence[str] | None = None) -> str:
    """SHA-256 of the canonical JSON of the prompt lists used to fit a direction."""
    payload = {"pos": list(pos_prompts), "neg": list(neg_prompts) if neg_prompts is not None else []}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _stem(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.suffix in (".npy", ".json", ".npz"):
        p = p.with_suffix("")
    return p


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _npy_bytes(arr: np.ndarray) -> bytes:
    import io as _io

    buf = _io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_direction(
    path: str | os.PathLike,
    vector: np.ndarray,
    *,
    base: str,
    revision: str | None,
    layer: int,
    n_pos: int,
    n_neg: int,
    prompt_set_sha256: str,
    position: str = "last",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<stem>.npy`` (float32, unit norm) and ``<stem>.json``; return the sidecar.

    The vector is normalised before saving. ``extra`` keys are added to the
    sidecar (for example kind, trait, raw_norm); they cannot override the
    required keys.
    """
    stem = _stem(path)
    v = unit(vector).astype("<f4")
    sidecar: dict[str, Any] = dict(extra or {})
    sidecar.update(
        {
            "base": base,
            "revision": revision,
            "layer": int(layer),
            "position": position,
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "prompt_set_sha256": prompt_set_sha256,
            "vector_sha256": vector_sha256(v),
            "created": _now(),
        }
    )
    _atomic_write_bytes(stem.with_suffix(".npy"), _npy_bytes(v))
    _atomic_write_bytes(stem.with_suffix(".json"), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"))
    return sidecar


def load_direction(path: str | os.PathLike, verify: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a direction file pair; with ``verify`` check keys, shape, norm and hash."""
    stem = _stem(path)
    v = np.load(stem.with_suffix(".npy"), allow_pickle=False)
    sidecar = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if verify:
        missing = [k for k in SIDECAR_REQUIRED if k not in sidecar]
        if missing:
            raise ValueError(f"{stem}.json missing keys {missing}")
        if v.ndim != 1:
            raise ValueError(f"{stem}.npy must be 1-D, got shape {v.shape}")
        if abs(float(np.linalg.norm(v.astype(np.float64))) - 1.0) > 1e-4:
            raise ValueError(f"{stem}.npy is not unit norm")
        if vector_sha256(v) != sidecar["vector_sha256"]:
            raise ValueError(f"{stem}: vector_sha256 mismatch")
    return v.astype(np.float32), sidecar


def save_subspace(
    path: str | os.PathLike,
    basis: np.ndarray,
    *,
    base: str,
    revision: str | None,
    layer: int,
    n_pos: int,
    n_neg: int,
    prompt_set_sha256: str,
    position: str = "last",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Subspace variant of the direction format: ``.npy`` float32 (k, hidden), orthonormal rows."""
    stem = _stem(path)
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or not (1 <= b.shape[0] <= MAX_SUBSPACE_K):
        raise ValueError(f"basis must be (k<= {MAX_SUBSPACE_K}, hidden), got {b.shape}")
    q = b.astype("<f4")
    sidecar: dict[str, Any] = dict(extra or {})
    sidecar.update(
        {
            "base": base,
            "revision": revision,
            "layer": int(layer),
            "position": position,
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "k": int(b.shape[0]),
            "prompt_set_sha256": prompt_set_sha256,
            "vector_sha256": vector_sha256(q),
            "created": _now(),
        }
    )
    _atomic_write_bytes(stem.with_suffix(".npy"), _npy_bytes(q))
    _atomic_write_bytes(stem.with_suffix(".json"), json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"))
    return sidecar


def load_subspace(path: str | os.PathLike, verify: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    stem = _stem(path)
    b = np.load(stem.with_suffix(".npy"), allow_pickle=False)
    sidecar = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if verify:
        if b.ndim != 2:
            raise ValueError(f"{stem}.npy must be 2-D")
        if vector_sha256(b) != sidecar.get("vector_sha256"):
            raise ValueError(f"{stem}: vector_sha256 mismatch")
    return b.astype(np.float32), sidecar


def save_probe(path: str | os.PathLike, probe: LinearProbe, sidecar: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``<stem>.npz`` (coef, intercept, mean, scale, C plus any array extras) and ``<stem>.json``."""
    import io as _io

    stem = _stem(path)
    arrays: dict[str, np.ndarray] = {
        "coef": np.asarray(probe.coef, dtype=np.float64),
        "intercept": np.asarray([probe.intercept], dtype=np.float64),
        "mean": np.asarray(probe.mean, dtype=np.float64),
        "scale": np.asarray(probe.scale, dtype=np.float64),
        "C": np.asarray([probe.C], dtype=np.float64),
    }
    for k, v in probe.extra.items():
        if isinstance(v, np.ndarray):
            arrays[f"extra__{k}"] = v
    buf = _io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    meta = dict(sidecar or {})
    meta["npz_sha256"] = hashlib.sha256(data).hexdigest()
    meta["created"] = _now()
    meta["format"] = "score = coef . ((x - mean) / scale) + intercept"
    _atomic_write_bytes(stem.with_suffix(".npz"), data)
    _atomic_write_bytes(stem.with_suffix(".json"), json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))
    return meta


def load_probe(path: str | os.PathLike) -> tuple[LinearProbe, dict[str, Any]]:
    stem = _stem(path)
    z = np.load(stem.with_suffix(".npz"), allow_pickle=False)
    extra = {k[len("extra__") :]: z[k] for k in z.files if k.startswith("extra__")}
    probe = LinearProbe(
        coef=z["coef"],
        intercept=float(z["intercept"][0]),
        mean=z["mean"],
        scale=z["scale"],
        C=float(z["C"][0]),
        extra=extra,
    )
    side_path = stem.with_suffix(".json")
    sidecar = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}
    return probe, sidecar
