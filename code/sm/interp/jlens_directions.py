"""EXPLORATORY R5 (docs/PROGRAM_2026-10-13.md, line B5): J-lens readouts of arbitrary direction vectors at one sm layer.

Reads, through one or more fitted lenses (format ``sm-jlens-v1``, sm.interp.jlens), every direction named on the
command line and reports its top 20 and bottom 20 vocabulary entries, annotated with the registered token sets of
docs/JLENS_READOUT_PLAN.md section 3.3, beside the controls of that plan's section 4. Descriptive; decides nothing.

Directions:
  offset/<unit>@<pos>  the decision offset d_L: mean over the harmful rows of the unit minus the same mean for the base,
                       at sm layer L and position pos (default ``last``, the last prompt token), from a mechanism set's
                       residual cache (``<acts-dir>/<unit>.npz``: ``layers`` [n_l] and ``harmful@<pos>`` /
                       ``harmless@<pos>`` [n, n_l, d] float16, written by sm.interp.mechanism.save_acts).
  harm/base@<pos>      the base's harm direction: mean harmful minus mean harmless rows of the base at L and pos. It is
                       the positive control (plan section 4: "reads as refusal" when at least one ``refusal_vocab`` token
                       is in its top 20), in place of the card's r, which lives at L* = 13.
  file directions      ``--vector NAME=PATH[:KEY][@ROW]`` (.npy, or .npz with KEY) and ``--vector-glob PREFIX=GLOB``
                       (each matching file holds one vector; name PREFIX/<stem>): e.g. the R1 probe directions. They must
                       be in the raw residual basis at layer L (a standardised probe's coefficients divided by the
                       feature scale), oriented so that + is the probe's positive class.

Controls (plan section 4, per lens): 100 seeded random unit vectors (default tokens: in the top 20 of at least 10),
the base's split-half drift of the harmful rows at the primary position (sampling null). Two or more lenses: the
stability block of plan section 6 between the first two (Jacobian cosine, relative Frobenius distance, top-20 overlaps;
rule cos >= 0.90 and the positive-control direction's top-20 overlap >= 0.70, with the base harm direction at L in
place of r), and the half-vs-half noise floor from each lens's four shard lenses when the official code imports.

Index conventions: sm layer L = ``hidden_states[L]`` = output of 1-based block L = jlens layer L - 1 (the lens file's
key); the ``layers`` array of a residual cache lists the hidden-state indices of its second axis, and the column is
looked up by value, never assumed to equal L.

Outputs: one JSON of numbers, vocabulary entries and stamps (``results/exploratory/<base>/jlens/readouts_L<L>.json``);
no prompt or generated text is read or written.
"""

from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sm.interp import jlens as J

VERSION = "jlens-directions-v1-2026-10-13"
LABEL = ("EXPLORATORY R5 lens at the decision layers (docs/PROGRAM_2026-10-13.md); J-lens readouts of direction "
         "vectors; descriptive, decides no registered hypothesis")
PROGRAM = "docs/PROGRAM_2026-10-13.md"
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[3]
_CODE_FILES = ("code/sm/interp/jlens_directions.py", "code/sm/interp/jlens.py")


# ----------------------------------------------------------------------------------------- provenance


def _sha_file(path: str | os.PathLike) -> str | None:
    p = Path(path)
    return J._sha256(p) if p.is_file() else None


_FILES_AT_IMPORT = {f: _sha_file(REPO_ROOT / f) for f in _CODE_FILES}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(_HERE.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def stamps(config: dict) -> dict[str, Any]:
    cfg = J._json_safe(config)
    return {"git_commit": git_commit(), "code_commit_env": os.environ.get("SM_CODE_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "file_sha256": dict(_FILES_AT_IMPORT), "config": cfg,
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}


# ----------------------------------------------------------------------------------------- inputs


def layer_column(layers: Sequence[int], layer: int, where: str = "") -> int:
    """Column of hidden-state index ``layer`` in a cache whose second axis holds ``layers``."""
    ls = [int(x) for x in np.asarray(layers).ravel()]
    if int(layer) not in ls:
        raise ValueError(f"{where or 'cache'}: hidden-state index {layer} not captured (layers {ls})")
    return ls.index(int(layer))


def load_rows(path: str | os.PathLike, key: str, layer: int) -> np.ndarray:
    """[n, d] float64 rows of ``key`` at hidden-state index ``layer`` from a mechanism residual cache."""
    with np.load(path) as z:
        if "layers" not in z.files:
            raise ValueError(f"{path}: no 'layers' array (not a sm.interp.mechanism residual cache)")
        if key not in z.files:
            raise KeyError(f"{path}: no array {key!r} (has {sorted(z.files)})")
        X = z[key]
        if X.ndim != 3:
            raise ValueError(f"{path} {key}: expected [n, n_layers, d], got {X.shape}")
        return X[:, layer_column(z["layers"], layer, f"{path}"), :].astype(np.float64)


def parse_vector_spec(spec: str) -> tuple[str, str, str | None, int | None]:
    """NAME=PATH[:KEY][@ROW] -> (name, path, key, row)."""
    name, sep, rest = spec.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"--vector wants NAME=PATH[:KEY][@ROW], got {spec!r}")
    row = None
    if "@" in rest:
        head, _, tail = rest.rpartition("@")
        if tail.isdigit():
            rest, row = head, int(tail)
    key = None
    if ".npz:" in rest:
        rest, _, key = rest.partition(".npz:")
        rest += ".npz"
    return name, rest, (key or None), row


def load_vector(path: str | os.PathLike, key: str | None = None, row: int | None = None) -> np.ndarray:
    """One float64 [d] vector from .npy ([d], or [k, d] with ``row``) or .npz (``key``, or its only array)."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as z:
            if key is None:
                if len(z.files) != 1:
                    raise ValueError(f"{p}: several arrays {sorted(z.files)}; name one as PATH:KEY")
                key = z.files[0]
            if key not in z.files:
                raise KeyError(f"{p}: no array {key!r} (has {sorted(z.files)})")
            a = np.asarray(z[key])
    else:
        a = np.load(p, allow_pickle=False)
    a = np.asarray(a, np.float64)
    if a.ndim == 2:
        if row is None:
            if a.shape[0] != 1:
                raise ValueError(f"{p}: shape {a.shape}; pick a row with @ROW")
            row = 0
        a = a[int(row)]
    elif a.ndim != 1:
        raise ValueError(f"{p}: expected one vector, got shape {a.shape}")
    elif row not in (None, 0):
        raise ValueError(f"{p}: a [d] vector has no row {row}")
    return a


def file_directions(specs: Sequence[str] = (), globs: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """{name: {"kind": "file", "vector", "source"}} from --vector and --vector-glob specs."""
    out: dict[str, dict[str, Any]] = {}

    def add(name: str, path: str, key: str | None, row: int | None) -> None:
        if name in out:
            raise ValueError(f"direction name {name!r} given twice")
        out[name] = {"kind": "file", "vector": load_vector(path, key, row),
                     "source": {"path": str(path), "sha256": _sha_file(path), "key": key, "row": row}}

    for s in specs:
        add(*parse_vector_spec(s))
    for g in globs:
        prefix, sep, pattern = g.partition("=")
        if not sep or not prefix or not pattern:
            raise ValueError(f"--vector-glob wants PREFIX=GLOB, got {g!r}")
        hits = sorted(p for p in _glob.glob(pattern) if p.endswith((".npy", ".npz")))
        if not hits:
            raise FileNotFoundError(f"--vector-glob {g!r} matched no .npy/.npz file")
        for p in hits:
            add(f"{prefix}/{Path(p).stem}", p, None, None)
    return out


def cache_directions(acts_dir: str | os.PathLike, layer: int, positions: Sequence[str] = ("last",),
                     units: Sequence[str] | None = None, n_flips: int = J.N_FLIPS,
                     seed: int = 0) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Offsets (unit minus base, harmful rows) and the base harm direction at ``layer`` from a residual-cache directory.

    Returns ({name: record}, {"checks": IC-A record, "files": ..., "null": {pos: split-half drift}})."""
    d = Path(acts_dir)
    base_p = d / "base.npz"
    if not base_p.is_file():
        raise FileNotFoundError(f"no base.npz under {d}")
    found = sorted(p.stem for p in d.glob("*.npz") if p.stem != "base" and not p.stem.endswith(".tmp"))
    us = list(found) if not units else [u for u in units if u != "base"]
    missing = [u for u in us if u not in found]
    if missing:
        raise FileNotFoundError(f"no residual cache for {missing} under {d}")
    with np.load(base_p) as z:
        base_layers = [int(x) for x in z["layers"]]
    dirs: dict[str, dict[str, Any]] = {}
    fails: list[str] = []
    files = {"base": {"path": str(base_p), "sha256": _sha_file(base_p), "layers": base_layers}}
    nulls: dict[str, np.ndarray] = {}
    base_rows: dict[str, dict[str, np.ndarray]] = {}
    for pos in positions:
        hb = load_rows(base_p, f"harmful@{pos}", layer)
        hn = load_rows(base_p, f"harmless@{pos}", layer)
        if not (np.isfinite(hb).all() and np.isfinite(hn).all()):
            fails.append(f"base@{pos}: non-finite rows")
        base_rows[pos] = {"harmful": hb}
        mean_b = hb.mean(0)
        harm = mean_b - hn.mean(0)
        dirs[f"harm/base@{pos}"] = {"kind": "harm", "position": pos, "vector": harm, "n_harmful": hb.shape[0],
                                    "n_harmless": hn.shape[0], "source": {"path": str(base_p), "rule": "mean harmful - mean harmless (base)"}}
        nulls[pos] = J.split_half_drift(hb, seed=seed)
    for u in us:
        p = d / f"{u}.npz"
        with np.load(p) as z:
            ul = [int(x) for x in z["layers"]]
        files[u] = {"path": str(p), "sha256": _sha_file(p), "layers": ul}
        if ul != base_layers:
            fails.append(f"{u}: layers {ul} differ from the base's {base_layers}")
        for pos in positions:
            hu = load_rows(p, f"harmful@{pos}", layer)
            hb = base_rows[pos]["harmful"]
            if hu.shape != hb.shape:
                fails.append(f"{u}@{pos}: harmful rows {hu.shape} vs the base's {hb.shape}")
                continue
            if not np.isfinite(hu).all():
                fails.append(f"{u}@{pos}: non-finite rows")
            v = J.drift_vector(hu, hb)
            mean_b = hb.mean(0)
            dirs[f"offset/{u}@{pos}"] = {
                "kind": "offset", "unit": u, "position": pos, "vector": v, "n_items": hu.shape[0],
                "defined": not J.is_zero_drift(v, mean_b), "rel_norm": float(np.linalg.norm(v) / np.linalg.norm(mean_b)),
                "sign_flip_ratio": J.sign_flip_ratio(hu, hb, n_flips=n_flips, seed=seed),
                "_means": (hu.mean(0), mean_b),
                "source": {"path": str(p), "rule": "mean harmful(unit) - mean harmful(base), same items in file order"}}
    check = {"status": "fail" if fails else "pass", "failures": fails,
             "rule": "unit and base caches list the same hidden-state indices, the same harmful row count per position, "
                     "and finite rows at the layer"}
    return dirs, {"check": check, "files": files, "null": nulls}


def lens_record(name: str, path: str | os.PathLike, layer: int) -> dict[str, Any]:
    d = J._lens_dir(path)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
    return {"name": name, "dir": str(d), "lens_sha256": _sha_file(d / "lens.pt"),
            "meta_merged_sha256": (meta.get("merged") or {}).get("lens_sha256"),
            "unembed_sha256": _sha_file(d / "unembed.pt"), "vocab_sha256": _sha_file(d / "vocab.json"),
            "sm_layers": meta.get("sm_layers"), "prompts_file": meta.get("prompts_file"),
            "prompts_sha256": meta.get("prompts_sha256"), "n_prompts": (meta.get("merged") or {}).get("n_prompts"),
            "official_commit_found": (meta.get("official") or {}).get("commit_found"), "fit": meta.get("fit"),
            "layer_requested": int(layer)}


def lens_checks(recs: dict[str, dict], layer: int) -> dict[str, Any]:
    """IC-L: every lens holds sm layer ``layer`` per its meta, lens.pt is the merged file the meta records, and all
    lenses share one vocabulary and one unembedding (one model)."""
    bad: list[str] = []
    for n, r in recs.items():
        if r["sm_layers"] is not None and int(layer) not in [int(x) for x in r["sm_layers"]]:
            bad.append(f"{n}: meta sm_layers {r['sm_layers']} lack {layer}")
        if r["meta_merged_sha256"] is not None and r["lens_sha256"] != r["meta_merged_sha256"]:
            bad.append(f"{n}: lens.pt differs from meta merged.lens_sha256")
    for field in ("vocab_sha256", "unembed_sha256"):
        vals = {r[field] for r in recs.values()}
        if len(vals) > 1:
            bad.append(f"lenses differ in {field}")
    return {"status": "fail" if bad else "pass", "failures": bad}


# ----------------------------------------------------------------------------------------- readouts


def _cos(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na == 0.0 or nb == 0.0 else float(a @ b / (na * nb))


def build(lenses: dict[str, Any], directions: dict[str, dict[str, Any]], layer: int, *, positions: Sequence[str],
          nulls: dict[str, np.ndarray] | None = None, token_sets: dict[str, list[int]] | None = None,
          check_registered: bool = True, k: int = J.TOP_K, seed: int = 0, n_random: int = J.N_RANDOM,
          shard_dirs: dict[str, str | os.PathLike] | None = None, failed_checks: Sequence[str] = ()) -> dict[str, Any]:
    """Readouts of ``directions`` through each lens at sm layer ``layer`` with the plan's controls, stability and
    noise floor. ``directions[name]["vector"]`` is a [d] vector; offsets may carry ``_means`` (unit, base) for the
    reference variant. Returns the JSON body (without inputs and stamps)."""
    lenses = {n: (v if isinstance(v, J.LoadedLens) else J.load_lens(v)) for n, v in lenses.items()}
    names = list(lenses)
    primary = names[0]
    lay = int(layer)
    d_model = next(iter(lenses.values())).d_model
    for n, Lz in lenses.items():
        Lz.pick_layer(lay)
        if Lz.d_model != d_model:
            raise ValueError(f"lens {n}: d_model {Lz.d_model} != {d_model}")
    for dn, rec in directions.items():
        v = np.asarray(rec["vector"], np.float64)
        if v.shape != (d_model,):
            raise ValueError(f"direction {dn}: shape {v.shape}, expected ({d_model},)")
        if not np.isfinite(v).all():
            raise ValueError(f"direction {dn}: non-finite entries")
    pos0 = positions[0] if positions else None
    harm_key = f"harm/base@{pos0}" if pos0 is not None else None
    if harm_key not in directions:
        harm_key = None
    sets_by_lens = {n: (token_sets if token_sets is not None else J.registered_token_sets(Lz.vocab, check=check_registered))
                    for n, Lz in lenses.items()}
    rand = J.random_unit_vectors(n_random, d_model, seed=seed)
    order = [dn for dn in directions]
    defined = {}
    for dn in order:
        rec = directions[dn]
        v = np.asarray(rec["vector"], np.float64)
        defined[dn] = bool(rec.get("defined", float(np.linalg.norm(v)) > 0.0))

    per_dir: dict[str, Any] = {}
    for dn in order:
        rec = directions[dn]
        v = np.asarray(rec["vector"], np.float64)
        pos = rec.get("position")
        harm_same = directions.get(f"harm/base@{pos}") if pos is not None else None
        ref = harm_same if harm_same is not None else (directions.get(harm_key) if harm_key else None)
        out = {k_: rec[k_] for k_ in ("kind", "unit", "position", "n_items", "n_harmful", "n_harmless", "rel_norm",
                                       "sign_flip_ratio", "source") if k_ in rec}
        out.update({"defined": defined[dn], "norm": float(np.linalg.norm(v)),
                    "cos_with_base_harm": (None if ref is None or ref is rec else _cos(v, np.asarray(ref["vector"], np.float64))),
                    "lenses": {}})
        per_dir[dn] = out

    readout_ids: dict[str, dict[str, list[int]]] = {}
    controls: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for name, Lz in lenses.items():
        sets = sets_by_lens[name]
        rows, tags = [], []
        for dn in order:
            if defined[dn]:
                rows.append(np.asarray(directions[dn]["vector"], np.float64))
                tags.append(("dir", dn))
                means = directions[dn].get("_means")
                if means is not None:
                    rows.extend([np.asarray(means[0]), np.asarray(means[1])])
                    tags.extend([("mean_u", dn), ("mean_b", dn)])
        null_v = (nulls or {}).get(pos0) if pos0 is not None else None
        if null_v is not None:
            rows.append(np.asarray(null_v, np.float64))
            tags.append(("null", pos0))
        for i in range(rand.shape[0]):
            rows.append(rand[i])
            tags.append(("random", str(i)))
        logits = Lz.logits(np.stack(rows).astype(np.float32), lay)
        at = {t: i for i, t in enumerate(tags)}
        rand_top = [J._ids(J.topk_records(Lz, logits[at[("random", str(i))]], k)) for i in range(rand.shape[0])]
        default = J.default_token_counts(rand_top, J.DEFAULT_TOKEN_MIN)
        null_rec = J.topk_records(Lz, logits[at[("null", pos0)]], k) if ("null", pos0) in at else None
        null_ids = set(J._ids(null_rec)) if null_rec is not None else set()
        ids_here: dict[str, list[int]] = {}
        for dn in order:
            if not defined[dn]:
                per_dir[dn]["lenses"][name] = {"defined": False}
                continue
            row = logits[at[("dir", dn)]]
            top = J.topk_records(Lz, row, k)
            bottom = J.topk_records(Lz, row, k, largest=False)
            ids_here[dn] = J._ids(top)
            entry = {"top": top, "bottom": bottom, **J._annotate(top, sets, set(default), null_ids),
                     "bottom_refusal_vocab": [{"rank": r["rank"], "token_id": r["token_id"], "token": r["token"]}
                                              for r in bottom if r["token_id"] in set(sets["refusal_vocab"])],
                     "bottom_compliance_onset": [{"rank": r["rank"], "token_id": r["token_id"], "token": r["token"]}
                                                 for r in bottom if r["token_id"] in set(sets["compliance_onset"])]}
            if ("mean_u", dn) in at:
                entry["reference_top"] = J.topk_records(Lz, logits[at[("mean_u", dn)]] - logits[at[("mean_b", dn)]], k)
            per_dir[dn]["lenses"][name] = entry
        ids_here["_random"] = rand_top  # type: ignore[assignment]
        readout_ids[name] = ids_here
        pc = None
        if harm_key is not None and defined.get(harm_key):
            top = per_dir[harm_key]["lenses"][name]["top"]
            ref_ids = set(sets["refusal_vocab"])
            pc = {"direction": harm_key, "top": top, "refusal_vocab": [x for x in top if x["token_id"] in ref_ids],
                  "reads_as_refusal": any(x["token_id"] in ref_ids for x in top),
                  "rule": "at least one refusal_vocab token in the top 20 (docs/JLENS_READOUT_PLAN.md section 4, with the "
                          "base harm direction at this layer in place of r)"}
        controls[name] = {
            "positive_control": pc if pc is not None else {"available": False, "reason": "no base harm direction"},
            "default_tokens": [{"token_id": i, "token": Lz.token(i), "count": c} for i, c in default.items()],
            "default_token_rule": f"in the top {k} of at least {J.DEFAULT_TOKEN_MIN} of {n_random} seeded random unit vectors",
            "sampling_null_split_half": ({"position": pos0, "top": null_rec, "norm": float(np.linalg.norm(null_v)),
                                          **J._annotate(null_rec, sets, set(default), set())}
                                         if null_rec is not None else {"available": False}),
        }
        offs = [dn for dn in order if directions[dn].get("kind") == "offset" and directions[dn].get("position") == pos0]
        offs_def = [dn for dn in offs if defined[dn]]
        with_c = [dn for dn in offs_def if per_dir[dn]["lenses"][name]["compliance_onset"]]
        with_c_f = [dn for dn in offs_def if any(not (x["default_token"] or x["in_sampling_null"])
                                                 for x in per_dir[dn]["lenses"][name]["compliance_onset"])]
        others = [dn for dn in order if directions[dn].get("kind") in ("file",) and defined[dn]]
        summary[name] = {
            "position": pos0,
            "offsets": {"n": len(offs), "n_defined": len(offs_def), "n_with_compliance_onset": len(with_c),
                        "with_compliance_onset": with_c, "undefined": [dn for dn in offs if not defined[dn]],
                        "n_with_compliance_onset_excluding_default_and_null": len(with_c_f),
                        "n_with_refusal_vocab": sum(1 for dn in offs_def if per_dir[dn]["lenses"][name]["refusal_vocab"])},
            "file_directions": {dn: {"refusal_vocab_top": len(per_dir[dn]["lenses"][name]["refusal_vocab"]),
                                     "compliance_onset_top": len(per_dir[dn]["lenses"][name]["compliance_onset"]),
                                     "refusal_vocab_bottom": len(per_dir[dn]["lenses"][name]["bottom_refusal_vocab"]),
                                     "compliance_onset_bottom": len(per_dir[dn]["lenses"][name]["bottom_compliance_onset"])}
                                for dn in others},
        }

    # ---------------- stability (first two lenses; plan section 6 with the base harm direction in place of r)
    stability: dict[str, Any] = {"available": False, "reason": "one lens given"}
    if len(names) >= 2:
        na, nb = names[0], names[1]
        Ja, Jb = lenses[na].J[lay].numpy(), lenses[nb].J[lay].numpy()
        ia, ib = readout_ids[na], readout_ids[nb]
        rand_ov = [J.topk_overlap(a, b) for a, b in zip(ia["_random"], ib["_random"])]
        dir_ov = {dn: J.topk_overlap(ia[dn], ib[dn]) for dn in ib if dn != "_random" and dn in ia}
        h_ov = dir_ov.get(harm_key) if harm_key else None
        stability = {"available": True, "lenses": [na, nb], "sm_layer": lay,
                     "jacobian_cosine": J.jacobian_cosine(Ja, Jb), "relative_frobenius": J.relative_frobenius(Ja, Jb),
                     "positive_control_direction": harm_key, "positive_control_top20_overlap": h_ov,
                     "positive_control_top1_agreement": (bool(ia[harm_key][0] == ib[harm_key][0]) if h_ov is not None else None),
                     "direction_top20_overlap": dir_ov,
                     "random_top20_overlap": {"mean": float(np.mean(rand_ov)), "min": float(np.min(rand_ov)),
                                              "p05": float(np.quantile(rand_ov, 0.05)), "n": len(rand_ov)},
                     "rule": f"stable iff jacobian_cosine >= {J.STABLE_COS_MIN} and the positive-control direction's "
                             f"top-20 overlap >= {J.STABLE_OVERLAP_MIN} (docs/JLENS_READOUT_PLAN.md section 6)"}
        stability["verdict"] = (J.stability_verdict(stability["jacobian_cosine"], h_ov) if h_ov is not None
                                else "not evaluable: no positive-control direction")
    floors: dict[str, Any] = {}
    for name, sd in (shard_dirs or {}).items():
        if name not in lenses:
            continue
        try:
            halves = J._official_half_lenses(Path(sd), lay)
        except ImportError as exc:
            floors[name] = {"available": False, "reason": f"official jacobian-lens code not importable: {exc}"}
            continue
        if halves is None:
            floors[name] = {"available": False, "reason": f"four shard lenses not all under {sd}"}
            continue
        HA, HB = (J._with_J(lenses[name], h, lay) for h in halves)
        probe = [np.asarray(directions[harm_key]["vector"], np.float64)] if harm_key and defined.get(harm_key) else []
        stack = np.stack(probe + [rand[i] for i in range(rand.shape[0])]).astype(np.float32)
        la, lb = HA.logits(stack, lay), HB.logits(stack, lay)
        ta = [J._ids(J.topk_records(HA, la[i], k)) for i in range(stack.shape[0])]
        tb = [J._ids(J.topk_records(HB, lb[i], k)) for i in range(stack.shape[0])]
        o = len(probe)
        floors[name] = {"available": True, "halves": "shards 0+1 vs 2+3 (official JacobianLens.merge)",
                        "jacobian_cosine": J.jacobian_cosine(halves[0].numpy(), halves[1].numpy()),
                        "relative_frobenius": J.relative_frobenius(halves[0].numpy(), halves[1].numpy()),
                        "positive_control_top20_overlap": J.topk_overlap(ta[0], tb[0]) if o else None,
                        "positive_control_top1_agreement": bool(ta[0][0] == tb[0][0]) if o else None,
                        "random_top20_overlap_mean": float(np.mean([J.topk_overlap(a, b) for a, b in zip(ta[o:], tb[o:])]))}
    stability["noise_floor_half_vs_half"] = floors

    # ---------------- reading (descriptive; the program's two predictions, primary lens)
    s0, c0 = summary[primary], controls[primary]
    pc0 = c0["positive_control"]
    preds = {
        "lens": primary, "position": pos0,
        "positive_control_harm_reads_as_refusal": pc0.get("reads_as_refusal") if pc0.get("available", True) else None,
        "offsets_not_compliance_onset": {
            "holds": (s0["offsets"]["n_with_compliance_onset"] == 0) if s0["offsets"]["n_defined"] else None,
            "n_offsets_defined": s0["offsets"]["n_defined"], "n_with_compliance_onset": s0["offsets"]["n_with_compliance_onset"],
            "n_with_compliance_onset_excluding_default_and_null": s0["offsets"]["n_with_compliance_onset_excluding_default_and_null"],
            "rule": "holds when no defined offset has a compliance_onset token in its top 20 (raw count; the count after "
                    "removing default tokens and sampling-null tokens is reported beside it)"},
        "note": "R5 is descriptive (docs/PROGRAM_2026-10-13.md); these readings decide nothing",
    }
    if failed_checks:
        msg = f"not read: instrument check {', '.join(failed_checks)} failed"
        verdicts = {"readouts": msg, "predictions": msg, "stability": msg}
    else:
        verdicts = {"readouts": "descriptive (no decision rule)", "predictions": preds,
                    "stability": stability.get("verdict", "not evaluable: one lens")}
    for rec in directions.values():
        rec.pop("_means", None)
    return {"label": LABEL, "version": VERSION, "sm_layer": lay, "jlens_layer": J.sm_to_jlens_layer(lay), "k": k,
            "positions": list(positions), "primary_position": pos0, "lenses": names, "primary_lens": primary,
            "directions": per_dir, "controls": controls, "summary": summary, "stability": stability,
            "failed_checks": list(failed_checks), "verdicts": verdicts}


def run(base: str, layer: int, lenses: dict[str, str], out: str | os.PathLike, *, acts_dir: str | None = None,
        units: Sequence[str] | None = None, positions: Sequence[str] = ("last",), vectors: Sequence[str] = (),
        vector_globs: Sequence[str] = (), k: int = J.TOP_K, seed: int = 0, floors: bool = True,
        check_registered: bool = True, token_sets: dict[str, list[int]] | None = None) -> Path:
    """Load the inputs, run the instrument checks, ``build`` and write ``out`` with inputs and stamps."""
    t0 = time.time()
    lay = int(layer)
    lens_recs = {n: lens_record(n, p, lay) for n, p in lenses.items()}
    ics: dict[str, Any] = {"IC_L_lenses": lens_checks(lens_recs, lay)}
    directions: dict[str, dict[str, Any]] = {}
    nulls: dict[str, np.ndarray] = {}
    cache_files: dict[str, Any] = {}
    if acts_dir is not None:
        cd, info = cache_directions(acts_dir, lay, positions=positions, units=units, seed=seed)
        directions.update(cd)
        nulls, cache_files = info["null"], info["files"]
        ics["IC_A_caches"] = info["check"]
    else:
        ics["IC_A_caches"] = {"status": "skipped", "reason": "no --acts-dir"}
    fd = file_directions(vectors, vector_globs)
    clash = sorted(set(fd) & set(directions))
    if clash:
        raise ValueError(f"file direction names clash with cache directions: {clash}")
    directions.update(fd)
    if not directions:
        raise ValueError("no directions: give --acts-dir and/or --vector / --vector-glob")
    failed = sorted(kk for kk, v in ics.items() if v.get("status") == "fail")
    shard_dirs = {n: J._lens_dir(p) / "shards" for n, p in lenses.items()} if floors else None
    body = build(lenses, directions, lay, positions=positions if acts_dir is not None else [],
                 nulls=nulls, token_sets=token_sets, check_registered=check_registered, k=k, seed=seed,
                 shard_dirs=shard_dirs, failed_checks=failed)
    body["instrument_checks"] = ics
    body["inputs"] = {"base": base, "acts_dir": acts_dir, "cache_files": cache_files, "lenses": lens_recs,
                      "token_sets": ({"override": True} if token_sets is not None else
                                     {"rule": J.TOKEN_SET_RULE, "ids": J.TOKEN_SET_IDS, "sha256": J.TOKEN_SET_SHA256,
                                      "checked_against_registered": bool(check_registered)})}
    program = REPO_ROOT / PROGRAM
    body["program"] = {"path": PROGRAM, "sha256": _sha_file(program), "line": "R5 (B5)"}
    config = {"base": base, "layer": lay, "lenses": dict(lenses), "acts_dir": acts_dir, "units": list(units) if units else None,
              "positions": list(positions), "vectors": list(vectors), "vector_globs": list(vector_globs), "k": k,
              "seed": seed, "floors": floors, "check_registered": check_registered}
    body["stamps"] = stamps(config)
    body["meta"] = {**J._runtime_meta(), "seconds": round(time.time() - t0, 1)}
    return J._write_json_atomic(Path(out), body)


# ----------------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R5: J-lens readouts of direction vectors at one sm layer (CPU).")
    ap.add_argument("--base", default="qwen3.5-9b")
    ap.add_argument("--layer", type=int, required=True, help="sm layer = hidden_states index (lens key = layer - 1)")
    ap.add_argument("--lens", action="append", required=True,
                    help="NAME=DIR, repeatable; the first is primary, the first two enter the stability block")
    ap.add_argument("--acts-dir", default=None, help="mechanism residual caches (base.npz and <unit>.npz)")
    ap.add_argument("--units", default=None, help="comma list of units in --acts-dir (default: every cache but base)")
    ap.add_argument("--positions", default="last", help="comma list; the first is primary (default last)")
    ap.add_argument("--vector", action="append", default=[], help="NAME=PATH[:KEY][@ROW], repeatable")
    ap.add_argument("--vector-glob", action="append", default=[], help="PREFIX=GLOB, repeatable (one vector per file)")
    ap.add_argument("--out", default=None, help="default results/exploratory/<base>/jlens/readouts_L<layer>.json")
    ap.add_argument("--k", type=int, default=J.TOP_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-floors", action="store_true", help="skip the half-vs-half noise floor")
    ap.add_argument("--allow-unregistered", action="store_true", help="skip the registered token-set check (tests only)")
    a = ap.parse_args(argv)
    lenses: dict[str, str] = {}
    for it in a.lens:
        n, sep, p = it.partition("=")
        if not sep or not n or not p:
            raise SystemExit(f"--lens wants NAME=DIR, got {it!r}")
        if n in lenses:
            raise SystemExit(f"--lens name {n!r} given twice")
        lenses[n] = p
    sb = J._safe_name(a.base)
    out = a.out or f"results/exploratory/{sb}/jlens/readouts_L{int(a.layer)}.json"
    path = run(a.base, a.layer, lenses, out, acts_dir=a.acts_dir, units=a.units.split(",") if a.units else None,
               positions=[p for p in a.positions.split(",") if p], vectors=a.vector, vector_globs=a.vector_glob,
               k=a.k, seed=a.seed, floors=not a.no_floors, check_registered=not a.allow_unregistered)
    body = json.loads(Path(path).read_text())
    print(json.dumps({"out": str(path), "failed_checks": body.get("failed_checks"),
                      "stability": (body.get("verdicts") or {}).get("stability"),
                      "predictions": (body.get("verdicts") or {}).get("predictions")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
