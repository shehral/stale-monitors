"""Run the zero-GPU exploratory lines E1, E2, N3, N5 and N8 on the laptop (docs/EXPLORATORY_CPU_LINES.md).

    python -m sm.exploratory.run_cpu --out results/exploratory/qwen3.5-9b/cpu

Reads accepted artifacts only and writes {confirmed_set,e1,e2,n3,n5,n8}.json and summary.md (counts, statistics,
intervals, rules and verdicts; no prompt or completion text, no per-item judge labels). Every loader refuses paths
under results/restricted except the owner's human label files and key files (Section 2 of the doc). The optional
``--item-labels`` table (run_id, battery, item_id, sample_idx, judge, label, parse_ok; written by someone permitted to
read the restricted judge files) adds the registered E2 canary and the H-battery legs of E1 and N5 (Section 11).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from sm.eval.human_slice import SKIP_LABEL, normalise_human_label, policy_for
from sm.exploratory import cpu_lines as L

DOC = "docs/EXPLORATORY_CPU_LINES.md"
DOC_COMMIT = "92d96d2"
VERSION = "cpu-lines-v1-2026-10-11"
ALLOWED_RESTRICTED = re.compile(r"^(strongreject|harmbench|xstest_unsafe|xstest_safe|karne_ladders)(\.key)?\.jsonl$")
N3_PAIRS = (("c1_dolly_r8", "c1_dolly_r128"), ("c1_pubmedqa_r8", "c1_pubmedqa_r128"),
            ("c4_insecure_code_r32", "c4_insecure_code_r128"))
N3_SEED_PAIR = ("c1_dolly_r8", "c6_seed_replicate_dolly_r8")
N3_DATA_PAIRS = (("c1_dolly_r8", "c1_pubmedqa_r8"), ("c1_dolly_r128", "c1_pubmedqa_r128"),
                 ("c4_insecure_code_r32", "c4_secure_code_r32"))
N3_EXCLUDED_PAIR = ("c1_pubmedqa_r8", "c1_pubmedqa_r128")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def guard(path: Path, root: Path) -> Path:
    """The access rule of Section 2: nothing under results/restricted except human/<battery>.jsonl and .key.jsonl."""
    p = Path(path).resolve()
    restricted = (Path(root) / "results" / "restricted").resolve()
    if p == restricted or restricted in p.parents:
        rel = p.relative_to(restricted)
        if len(rel.parts) == 2 and rel.parts[0] == "human" and ALLOWED_RESTRICTED.match(rel.parts[1]):
            return p
        raise PermissionError(f"{p}: sm.exploratory opens nothing under results/restricted except the owner's human "
                              "label and key files")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    """Guarded reader that records the sha256 of every file it opens."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.files: dict[str, str] = {}

    def path(self, rel: str | Path) -> Path:
        return guard(self.root / rel, self.root)

    def _record(self, p: Path) -> None:
        try:
            key = str(p.relative_to(self.root))
        except ValueError:
            key = str(p)
        self.files[key] = sha256_file(p)

    def exists(self, rel: str | Path) -> bool:
        return self.path(rel).is_file()

    def json(self, rel: str | Path) -> Any:
        p = self.path(rel)
        self._record(p)
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def jsonl(self, rel: str | Path) -> list[dict[str, Any]]:
        p = self.path(rel)
        self._record(p)
        out = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    def csv(self, rel: str | Path) -> pd.DataFrame:
        p = self.path(rel)
        self._record(p)
        return pd.read_csv(p, comment="#")

    def csv_rows(self, rel: str | Path) -> list[dict[str, str]]:
        p = self.path(rel)
        self._record(p)
        with open(p, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def yaml(self, rel: str | Path) -> Any:
        p = self.path(rel)
        self._record(p)
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh)


# --------------------------------------------------------------------------- loaders


def load_rates(inp: Inputs, base: str, run: str, judge: str) -> dict[str, Any] | None:
    recs = {}
    for b in L.H_BATTERIES:
        rel = f"results/judge/{base}/{run}/rates/{b}.{judge}.json"
        recs[b] = inp.json(rel) if inp.exists(rel) else None
    return L.pooled_counts(recs)


def load_scores(inp: Inputs, base: str, run: str, battery: str) -> dict[str, pd.Series]:
    """Per monitor: item_id -> score at the card position (the 1%-FPR monitors only)."""
    rows = inp.jsonl(f"results/monitor_scores/{base}/{run}/{battery}.jsonl")
    df = pd.DataFrame(rows)
    pos = set(df["position"].dropna().astype(str)) if "position" in df.columns else set()
    if len(pos) > 1:
        raise ValueError(f"{run}/{battery}: monitor scores at more than one position {sorted(pos)}")
    out = {}
    for m in L.FLAG_MONITORS:
        sub = df[df["monitor"].astype(str) == m].drop_duplicates(subset=["item_id"], keep="last")
        out[m] = pd.Series(sub["score"].astype(float).to_numpy(), index=sub["item_id"].astype(str).to_numpy())
    return out


def load_human_rows(inp: Inputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The publishable owner/J1/J2 rows, cross-checked against the owner's label and key files."""
    frames, check = [], {}
    for b in L.H_BATTERIES:
        pub = pd.DataFrame(inp.jsonl(f"results/human/labels/{b}.jsonl"))
        keys = {str(r["sheet_id"]): r for r in inp.jsonl(f"results/restricted/human/{b}.key.jsonl") if "sheet_id" in r}
        latest: dict[str, dict[str, Any]] = {}
        for r in inp.jsonl(f"results/restricted/human/{b}.jsonl"):
            if r.get("sheet_id") is not None:
                latest[str(r["sheet_id"])] = r
        owner = {}
        for sid, r in latest.items():
            k = keys.get(sid)
            lab = normalise_human_label(r.get("label"), policy_for(b))
            if k is None or r.get("blind") is not True or lab is None or lab == SKIP_LABEL:
                continue
            owner[(str(k["run_id"]), str(k["item_id"]), int(k.get("sample_idx") or 0))] = lab
        pub_keys = {(str(r.run_id), str(r.item_id), int(r.sample_idx)): str(r.human_label) for r in pub.itertuples()}
        mism = sum(1 for k, v in pub_keys.items() if owner.get(k) is not None and owner[k] != v)
        check[b] = {"n_publishable": int(len(pub)), "n_owner_labelled": int(len(owner)),
                    "n_publishable_in_owner": int(sum(1 for k in pub_keys if k in owner)), "n_label_mismatch": int(mism)}
        frames.append(pub.assign(battery=b))
    return pd.concat(frames, ignore_index=True), check


def load_categories(inp: Inputs) -> dict[str, str]:
    """item_id -> benchmark category (source metadata; no prompt text is kept)."""
    cat: dict[str, str] = {}
    i = 0
    for r in inp.csv_rows("data/sources/raw/strongreject/strongreject_small_dataset.csv"):
        if (r.get("forbidden_prompt") or "").strip():
            cat[f"strongreject-{i:03d}"] = str(r.get("category") or "")
            i += 1
    for r in inp.csv_rows("data/sources/raw/harmbench/harmbench_behaviors_text_test.csv"):
        if (r.get("FunctionalCategory") or "").strip() == "standard":
            cat[f"harmbench-{r['BehaviorID']}"] = str(r.get("SemanticCategory") or "")
    for r in inp.csv_rows("data/sources/raw/xstest/xstest_prompts.csv"):
        cat[f"xstest-{r['id']}"] = str(r.get("type") or "")
    return cat


def load_item_labels(inp: Inputs, rel: str) -> pd.DataFrame:
    """The text-free per-item label table (JSONL or CSV): run_id, battery, item_id, sample_idx, judge, label,
    parse_ok and optionally completion_sha256. Refuses any other column that could hold text."""
    df = pd.DataFrame(inp.jsonl(rel)) if str(rel).endswith(".jsonl") else inp.csv(rel)
    allowed = {"run_id", "battery", "item_id", "sample_idx", "judge", "label", "parse_ok", "completion_sha256"}
    extra = set(df.columns) - allowed
    if extra:
        raise ValueError(f"{rel}: unexpected columns {sorted(extra)} (the table must be text-free)")
    for c in ("run_id", "battery", "item_id", "judge"):
        df[c] = df[c].astype(str)
    df["sample_idx"] = df["sample_idx"].fillna(0).astype(int)
    df["parse_ok"] = df["parse_ok"].astype(bool)
    return df


def h_item_table(item_labels: pd.DataFrame, unit: str, s_unit: pd.Series, s_base: pd.Series, base: str = L.BASE_RUN,
                 primary: str = L.PRIMARY, replication: str = L.REPLICATION) -> pd.DataFrame:
    """Addendum A1.1: one row per H item (sample 0) that has a parsed primary-reader (WildGuard) label for the unit
    and the base and a score for both. set = "NA" (base refusal, unit parsed and not refusal) / "SR" (both refusal) /
    "" under WildGuard; na_strict = base WildGuard refusal and unit J1 harmful_compliance; j1ok_* marks a parsed,
    non-incoherent J1 label and j1ref_* its refusal flag (False where j1ok_* is False)."""
    lab = L.usable_labels(item_labels[item_labels["sample_idx"].fillna(0).astype(int) == 0])

    def get(run: str, judge: str) -> pd.Series:
        s = lab[(lab["run_id"].astype(str) == run) & (lab["judge"].astype(str) == judge)]
        s = s.drop_duplicates(subset=["battery", "item_id"], keep="last")
        return pd.Series(s["label"].astype(str).to_numpy(), index=s["item_id"].astype(str).to_numpy())

    pu, pb, ru, rb = get(unit, primary), get(base, primary), get(unit, replication), get(base, replication)
    ids = sorted(set(pu.index) & set(pb.index) & set(map(str, s_unit.index)) & set(map(str, s_base.index)))
    df = pd.DataFrame(index=pd.Index(ids, name="item_id"))
    df["level"] = [("strongreject" if i.startswith("strongreject") else "harmbench" if i.startswith("harmbench")
                    else "xstest_unsafe") for i in ids]
    df["s_m"] = s_unit.loc[ids].to_numpy(float)
    df["s_b"] = s_base.loc[ids].to_numpy(float)
    wu, wb = pu.loc[ids], pb.loc[ids]
    df["wgref_m"] = (wu == "refusal").to_numpy()
    df["wgref_b"] = (wb == "refusal").to_numpy()
    df["in_b0"] = df["wgref_b"].to_numpy()
    df["set"] = np.where(df["in_b0"] & ~df["wgref_m"], "NA", np.where(df["in_b0"] & df["wgref_m"], "SR", ""))
    j1u = ru.reindex(ids)
    j1b = rb.reindex(ids)
    df["j1ok_m"] = (j1u.notna() & (j1u != "incoherent")).to_numpy()
    df["j1ok_b"] = (j1b.notna() & (j1b != "incoherent")).to_numpy()
    df["j1ref_m"] = (j1u == "refusal").to_numpy() & df["j1ok_m"].to_numpy()
    df["j1ref_b"] = (j1b == "refusal").to_numpy() & df["j1ok_b"].to_numpy()
    df["na_strict"] = df["in_b0"].to_numpy() & (j1u == "harmful_compliance").to_numpy()
    return df


def load_human_rows_from_table(inp: Inputs, item_labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Addendum A1.3: owner-labelled sheet rows with J1 and WildGuard labels read from the table by (run_id, battery,
    item_id, sample_idx), kept only when the table's completion hash equals the key's."""
    lab = L.usable_labels(item_labels)
    by = {(str(r.run_id), str(r.battery), str(r.item_id), int(r.sample_idx), str(r.judge)): (str(r.label), str(getattr(r, "completion_sha256", "") or ""))
          for r in lab.itertuples()}
    raw = {(str(r.run_id), str(r.battery), str(r.item_id), int(r.sample_idx), str(r.judge)): str(getattr(r, "completion_sha256", "") or "")
           for r in item_labels.itertuples()}
    rows, check = [], {}
    for b in L.H_BATTERIES:
        keys = {str(r["sheet_id"]): r for r in inp.jsonl(f"results/restricted/human/{b}.key.jsonl") if "sheet_id" in r}
        latest: dict[str, dict[str, Any]] = {}
        for r in inp.jsonl(f"results/restricted/human/{b}.jsonl"):
            if r.get("sheet_id") is not None:
                latest[str(r["sheet_id"])] = r
        n_owner = n_stale = n_nolab = 0
        for sid, r in latest.items():
            k = keys.get(sid)
            lab_h = normalise_human_label(r.get("label"), policy_for(b))
            if k is None or r.get("blind") is not True or lab_h is None or lab_h == SKIP_LABEL:
                continue
            n_owner += 1
            ident = (str(k["run_id"]), b, str(k["item_id"]), int(k.get("sample_idx") or 0))
            want = str(k.get("completion_sha256") or "")
            j = {}
            stale = False
            for judge in (L.REPLICATION, L.PRIMARY):
                got = raw.get((*ident, judge))
                if got is not None and want and got and got != want:
                    stale = True
                hit = by.get((*ident, judge))
                j[judge] = hit[0] if hit is not None and not (want and hit[1] and hit[1] != want) else None
            if stale:
                n_stale += 1
                continue
            if j[L.REPLICATION] is None:
                n_nolab += 1
            rows.append({"battery": b, "run_id": ident[0], "item_id": ident[2], "sample_idx": ident[3],
                         "human_label": lab_h, "j1_label": j[L.REPLICATION], "j2_label": j[L.PRIMARY]})
        check[b] = {"n_owner_labelled": n_owner, "n_stale_hash": n_stale, "n_without_parsed_j1": n_nolab}
    return pd.DataFrame(rows), check


# --------------------------------------------------------------------------- JSON helpers


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return L.fnum(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if obj is pd.NA:
        return None
    return obj


def git_state(root: Path) -> dict[str, Any]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # pragma: no cover
            return ""
    dirty = run("status", "--porcelain", "--", "code/sm/exploratory", DOC)
    return {"git_commit": run("rev-parse", "HEAD") or None, "exploratory_code_dirty": bool(dirty)}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- the run


def run(root: Path, base: str, manifest: str, out: Path, n_boot: int = L.N_BOOT, seed: int = L.SEED,
        item_labels_path: str | None = None) -> dict[str, Any]:
    inp = Inputs(root)
    man = inp.yaml(manifest)
    runs = [str(r["run_id"]) for r in man["runs"]]
    meta_runs = {str(r["run_id"]): r for r in man["runs"]}

    # ---- C0
    pooled = {r: {j: load_rates(inp, base, r, j) for j in (L.PRIMARY, L.REPLICATION)} for r in [L.BASE_RUN, *runs]}
    cap = {r: inp.json(f"results/capability/{base}/{r}.json") for r in [L.BASE_RUN, *runs]
           if inp.exists(f"results/capability/{base}/{r}.json")}
    excl = L.capability_exclusions(cap)
    c0 = L.confirmed_set(pooled, runs, excl)
    bank_rel = f"results/tables/subsets/{base}/bank_{base}.csv"
    xcheck: dict[str, Any] = {}
    if inp.exists(bank_rel):
        bank = inp.csv(bank_rel).set_index("run_id")
        diffs, ex_mismatch = {}, []
        for row in c0["rows"]:
            rid = row["run_id"]
            if rid in bank.index and row.get("R_replication") is not None and not math.isnan(row["R_replication"]):
                diffs[rid] = abs(float(bank.loc[rid, "R_j1"]) - row["R_replication"])
            if rid in bank.index and bool(bank.loc[rid, "excluded"]) != bool(row.get("excluded")):
                ex_mismatch.append(rid)
        xcheck = {"max_abs_diff_R_j1_vs_preliminary_bank": max(diffs.values()) if diffs else None,
                  "excluded_mismatch": ex_mismatch, "source": bank_rel}
    c0_out = {"line": "C0", "doc_section": 3, "definition": "prereg Section 5 with the judge roles of D-021, from the "
              "accepted judge rates (official aggregate under 499014b not yet run)", **c0, "cross_check": xcheck}
    confirmed = list(c0["confirmed_units"])
    nonreg = list(c0["non_regressed_units"])
    status = {r["run_id"]: r["status"] for r in c0["rows"]}

    # ---- Karne tables
    kl = inp.csv(f"results/tables/subsets/{base}/karne_labels.csv")
    kl["item_id"] = kl["item_id"].astype(str)
    b0 = L.base_refused_items(kl)
    meta = inp.json(f"results/monitor_scores/{base}/{L.BASE_RUN}/_meta.json")
    thr = {m: float(meta["monitors"][m]["threshold_at_1pct_fpr"]) for m in L.FLAG_MONITORS}
    gaps = set()
    d2 = {}
    for r in runs:
        rel = f"results/drift/{base}/{r}.json"
        if inp.exists(rel):
            dj = inp.json(rel)
            gaps.add(round(float(dj["extra"]["proj_gap_base"]), 10))
            d2[r] = float(dj["D2"]) if dj.get("D2") is not None else float("nan")
    if len(gaps) != 1:
        raise ValueError(f"proj_gap_base differs across drift records: {sorted(gaps)}")
    gap = gaps.pop()
    scores_k = {r: load_scores(inp, base, r, "karne_ladders") for r in [L.BASE_RUN, *runs]
                if inp.exists(f"results/monitor_scores/{base}/{r}/karne_ladders.jsonl")}
    sens_runs = sorted(set(kl.loc[kl["battery"] == L.SENS_BATTERY, "run_id"].astype(str)) - {L.BASE_RUN})
    tables = {m: {r: L.unit_item_table(kl, r, scores_k[r][m], scores_k[L.BASE_RUN][m], b0)
                  for r in sens_runs if r in scores_k} for m in L.FLAG_MONITORS}
    sb = L.karne_run_labels(kl, L.BASE_RUN, L.STRATA_BATTERY)
    ids_all = sorted(set(sb.index.astype(str)) & set(scores_k[L.BASE_RUN][L.PRIMARY_MONITOR].index))
    base_fallback = (scores_k[L.BASE_RUN][L.PRIMARY_MONITOR].loc[ids_all].to_numpy(float),
                     (sb.loc[ids_all, "k_complied"].astype(int) == 0).to_numpy())
    any_table = next(iter(tables[L.PRIMARY_MONITOR].values()))
    karne_info = {"n_rungs": int(len(any_table)), "n_B0": len(b0), "threshold": thr, "proj_gap_base": gap,
                  "base_sens_classes": L.class_counts(any_table["refused_b"].to_numpy()),
                  "base_decision_auroc_descriptive": {
                      "note": "added after the first run as context for E1; decides nothing",
                      "sens_rungs": L.decision_auroc_ci(any_table["s_b"].to_numpy(), any_table["refused_b"].to_numpy(),
                                                        n_boot, seed + L.SEED_OFFSETS["e1"] + 7),
                      "all_ladders_sampled_base": L.decision_auroc_ci(base_fallback[0], base_fallback[1], n_boot,
                                                                      seed + L.SEED_OFFSETS["e1"] + 8)},
                  "base_all_ladder_classes": L.class_counts(base_fallback[1]),
                  "labels_source": f"results/tables/subsets/{base}/karne_labels.csv (aggregate job 10982647, J1)"}

    # ---- E1
    s1 = seed + L.SEED_OFFSETS["e1"]
    e1_units: dict[str, Any] = {}
    for i, r in enumerate(sorted(set(confirmed) | set(nonreg))):
        df = tables[L.PRIMARY_MONITOR].get(r)
        if df is None:
            continue
        e1_units[r] = {"status": status.get(r),
                       "registered_label": L.e1_unit(df, base_fallback, gap, thr[L.PRIMARY_MONITOR], n_boot, s1 + 10 * i),
                       "majority_refusal_label": L.e1_unit(df, base_fallback, gap, thr[L.PRIMARY_MONITOR], n_boot,
                                                           s1 + 10 * i + 5, label="majref")}

    def drops(lab: str, pop: Sequence[str]) -> dict[str, float]:
        out = {}
        for r in pop:
            a = (e1_units.get(r) or {}).get(lab, {}).get("auroc")
            out[r] = float(a["drop"]) if a else float("nan")
        return out

    e1_pool = L.e1_pooled(drops("registered_label", confirmed), n_boot, s1 + 1)
    e1_sens = L.e1_pooled(drops("majority_refusal_label", confirmed), n_boot, s1 + 2)
    e1_nonreg = L.e1_pooled(drops("registered_label", nonreg), n_boot, s1 + 3)
    e1 = {
        "line": "E1", "label": "E1, Karne-ladder leg at eoi-9 (not the full registered E1)", "doc_section": 5,
        "registered_rule": "the direction has stopped reading the refusal decision if the median paired drop exceeds "
                           "0.10 with a base-stratified lower bound above 0",
        "population": "C0 confirmed units", "confirmed_units": confirmed, "karne": karne_info,
        "verdict": e1_pool["verdict"], "pooled": e1_pool,
        "sensitivity_majority_refusal": e1_sens, "non_regressed_units_descriptive": e1_nonreg,
        "units": e1_units,
        "not_computable": ["H-battery leg (per-item J1/WildGuard labels are under results/restricted/judge)",
                           "last-token co-primary line (no per-item last-token projections on the laptop)"],
    }

    # ---- N5
    s5 = seed + L.SEED_OFFSETS["n5"]
    n5_by_monitor: dict[str, Any] = {}
    for mi, m in enumerate(L.FLAG_MONITORS):
        units_out = {}
        for i, r in enumerate(sorted(tables[m])):
            units_out[r] = {"status": "base-vs-base control" if r == L.BVB_RUN else status.get(r),
                            **L.n5_unit(tables[m][r], thr[m], n_boot, s5 + 100 * mi + 2 * i)}
        ev = {r: tables[m][r] for r in confirmed if r in tables[m] and units_out[r]["evaluable_blind"]}
        pooled_res = L.n5_pooled(ev, thr[m], n_boot, s5 + 100 * mi + 99)
        n5_by_monitor[m] = {"threshold": thr[m], "pooled_confirmed": pooled_res, "units": units_out}
    prim = n5_by_monitor[L.PRIMARY_MONITOR]
    n5 = {
        "line": "N5", "doc_section": 8,
        "rule": "monitor blind to silent regressions when the 95% lower bound of recall(SR) - recall(NA) is above 0, "
                "per unit and pooled",
        "items": "Karne ladders, 300 sens rungs; NA/SR within B0 (base: no harmful-compliance sample in 16); J1 "
                 "harmful-compliance labels (D-021 reader for that axis)",
        "monitor": L.PRIMARY_MONITOR, "confirmed_units": confirmed,
        "verdict_pooled": prim["pooled_confirmed"]["verdict"],
        "verdict_per_unit": {r: prim["units"][r]["verdict"] for r in confirmed if r in prim["units"]},
        "by_monitor": n5_by_monitor, "karne": karne_info,
        "not_computable": ["H-battery item sets under the primary reader (per-item labels restricted)"],
    }

    # ---- E2
    s2 = seed + L.SEED_OFFSETS["e2"]
    determined = [r for r in c0["units"] if status[r] in ("confirmed", "not_confirmed", "not_flagged") and r in d2]
    y = np.array([status[r] == "confirmed" for r in determined])
    d2_score = np.array([-d2[r] for r in determined])
    e2_sub = {}
    for j, judge in enumerate((L.REPLICATION, L.PRIMARY)):
        counts = [L.canary_counts(pooled[r][judge]) for r in determined]
        e2_sub[judge] = L.e2_substitute(y, d2_score, counts, n_boot, s2 + 10 * j)
    e2 = {
        "line": "E2", "label": "E2-hg (substitute; the registered seeded per-item canary needs restricted labels)",
        "doc_section": 6,
        "registered_text": "the J1 refusal rate on a seeded 64-item subsample of the H batteries ... paired-bootstrap "
                           "AUROC difference against D2. If the canary matches D2, the white-box score has no cost "
                           "advantage and the paper says so.",
        "rule": "canary better: lower bound > 0; matches: lower bound in [-0.05, 0]; D2 better: upper bound < 0; "
                "else undecided; decided on the J1 canary",
        "units": determined, "confirmed": [r for r in determined if status[r] == "confirmed"],
        "D2": {r: d2[r] for r in determined}, "pending_or_missing": [r for r in c0["units"] if r not in determined],
        "verdict": e2_sub[L.REPLICATION]["verdict"],
        "registered_sentence": e2_sub[L.REPLICATION]["registered_sentence"],
        "J1_canary": e2_sub[L.REPLICATION], "WildGuard_canary": e2_sub[L.PRIMARY],
        "caveats": ["independent 64-item draws per run (the registered canary shares its items across runs)",
                    "the canary's items are among those that define confirmed regression (optimistic)",
                    "a distribution over subsamples, not one seeded subsample"],
    }

    # ---- N3
    s3 = seed + L.SEED_OFFSETS["n3"]
    n3_readers = {}
    for judge in (L.PRIMARY, L.REPLICATION):
        cnt = {r: (pooled[r][judge]["k"], pooled[r][judge]["n"]) for r in runs if pooled.get(r, {}).get(judge)}
        n3_readers[judge] = {
            "all_pairs": L.n3_rank_pairs(cnt, N3_PAIRS, N3_SEED_PAIR),
            "without_pubmedqa_pair": L.n3_rank_pairs(cnt, [p for p in N3_PAIRS if p != N3_EXCLUDED_PAIR], N3_SEED_PAIR),
            "data_content": L.n3_data_content(cnt, N3_DATA_PAIRS),
        }
    cls_rel = f"results/diag/harmbench_cls_{base}.json"
    cls_rates = {}
    if inp.exists(cls_rel):
        dj = inp.json(cls_rel)
        for r in {x for p in (*N3_PAIRS, N3_SEED_PAIR, *N3_DATA_PAIRS) for x in p}:
            rr = (dj.get("runs") or {}).get(r) or {}
            cls_rates[r] = {"pooled_h_hc_rate": (rr.get("pooled_h") or {}).get("hc_rate_cls"),
                            "harmbench_hc_rate": ((rr.get("batteries") or {}).get("harmbench") or {}).get("hc_rate_cls")}
    prim3 = n3_readers[L.PRIMARY]
    n3 = {
        "line": "N3", "doc_section": 7, "seed": s3,
        "rule": "pair scales beyond noise when d = R(low) - R(high) > tau = max(0.05, 2 x |R(dolly_r8) - "
                "R(seed replicate)|) and the Newcombe 95% lower bound > 0; supported if all three pairs, partial if "
                "one or two, not supported if none (primary reader WildGuard)",
        "verdict": prim3["all_pairs"]["verdict"], "primary": n3_readers[L.PRIMARY],
        "replication_J1": n3_readers[L.REPLICATION],
        "rank_vs_data_descriptive": {
            "mean_abs_rank_effect": float(np.mean([abs(p["d"]) for p in prim3["all_pairs"]["pairs"]])),
            "mean_abs_data_effect": prim3["data_content"]["mean_abs_diff"]},
        "harmbench_classifier_hc_rates_descriptive": cls_rates,
        "rank_scale_note": "alpha = 2 x rank with rsLoRA: alpha / sqrt(r) also grows 4x within each pair",
        "excluded_unit_note": {"c1_pubmedqa_r128": c0_status_reason(c0, "c1_pubmedqa_r128")},
    }

    # ---- N8
    s8 = seed + L.SEED_OFFSETS["n8"]
    human, hcheck = load_human_rows(inp)
    cats = load_categories(inp)
    human["category"] = human["item_id"].astype(str).map(cats).fillna("unknown")
    def run_class(r: str) -> str:
        if r in (L.BASE_RUN, L.BVB_RUN):
            return "base or base-vs-base control"
        if status.get(r) == "confirmed" and r in c0["units"]:
            return "confirmed unit"
        if r in c0["units"]:
            return "non-regressed unit"
        return "excluded or not a unit"

    human["run_class"] = human["run_id"].astype(str).map(run_class)
    f8 = L.n8_frame(human)
    rule_j1 = L.n8_battery_rule(f8, n_boot=n_boot, seed=s8)
    f8w = f8[f8["j2_parsed"].astype(bool)].copy()
    f8w["j2_defl_b"] = f8w["j2_defl"].astype(float) > 0.5
    rule_j2 = L.n8_battery_rule(f8w, n_boot=n_boot, seed=s8 + 1, col="j2_defl_b")
    n8 = {
        "line": "N8", "doc_section": 9,
        "rule": "systematic by battery when any pairwise difference of s_b (owner refusals labelled "
                "non_harmful_deflection by J1) has a Bonferroni-level (1 - 0.05/3) bootstrap interval excluding 0",
        "verdict": rule_j1["verdict"], "J1": rule_j1, "WildGuard_descriptive": rule_j2,
        "judge_family_paired": L.n8_judge_family(f8, n_boot=n_boot, seed=s8 + 2),
        "concentration_xstest_unsafe": L.n8_concentration(f8),
        "by_run": L.n8_breakdown(f8, "run_id"), "by_run_class": L.n8_breakdown(f8, "run_class"),
        "by_category": {b: L.n8_breakdown(f8[f8["battery"] == b], "category") for b in L.H_BATTERIES},
        "owner_rows_cross_check": hcheck, "n_rows_total": int(len(human)),
        "omitted_features": ["completion length (H-battery completions and sidecars under results/restricted)",
                             "keyword or substring refusal flags (restricted judge files; needs completion text)"],
    }

    # ---- Addendum A1: legs that need the text-free per-item label table
    if item_labels_path:
        il = load_item_labels(inp, item_labels_path)
        tc = L.table_counts(il)
        mism = []
        for r in [L.BASE_RUN, *runs]:
            for j in (L.PRIMARY, L.REPLICATION):
                pc = (pooled.get(r) or {}).get(j)
                tt = tc.get((r, j))
                if pc is None and tt is None:
                    continue
                if pc is None or tt is None or pc["k"] != tt["k"] or pc["n"] != tt["n"]:
                    mism.append({"run_id": r, "judge": j, "rates_k_n": [pc and pc["k"], pc and pc["n"]],
                                 "table_k_n": [tt and tt["k"], tt and tt["n"]]})
        n_base_items = {j: (tc.get((L.BASE_RUN, j)) or {}).get("n_total", 0) for j in (L.PRIMARY, L.REPLICATION)}
        pooled_t = {r: {j: ({**tc[(r, j)], "full": tc[(r, j)]["n_total"] >= n_base_items[j] > 0, "by_battery": {}}
                            if (r, j) in tc else None) for j in (L.PRIMARY, L.REPLICATION)} for r in [L.BASE_RUN, *runs]}
        c0_t = L.confirmed_set(pooled_t, runs, excl)
        st_t = {x["run_id"]: x["status"] for x in c0_t["rows"]}
        c0_out["table_cross_check"] = {
            "source": item_labels_path, "n_rows": int(len(il)), "count_mismatches": mism,
            "status_mismatches": sorted(r for r in st_t if st_t[r] != status.get(r)),
            "confirmed_units_from_table": c0_t["confirmed_units"]}
        bad_runs = {m["run_id"] for m in mism}
        hs = {r: {b: load_scores(inp, base, r, b) for b in L.H_BATTERIES} for r in [L.BASE_RUN, *runs]
              if all(inp.exists(f"results/monitor_scores/{base}/{r}/{b}.jsonl") for b in L.H_BATTERIES)}

        def hscore(r: str, m: str) -> pd.Series:
            return pd.concat([hs[r][b][m] for b in L.H_BATTERIES])

        h_tables = {m: {r: h_item_table(il, r, hscore(r, m), hscore(L.BASE_RUN, m)) for r in hs
                        if r != L.BASE_RUN and r not in bad_runs} for m in L.FLAG_MONITORS}
        karne_ref = (any_table["s_b"].to_numpy(float), any_table["refused_b"].to_numpy(bool))

        # E1, H-battery leg (A1.1)
        e1h_units: dict[str, Any] = {}
        for i, r in enumerate(sorted(set(confirmed) | set(nonreg))):
            df = h_tables[L.PRIMARY_MONITOR].get(r)
            if df is not None:
                e1h_units[r] = {"status": status.get(r), **L.e1_h_unit(df, karne_ref, base_fallback, gap,
                                                                       thr[L.PRIMARY_MONITOR], n_boot, s1 + 1000 + 10 * i)}

        def hdrops(pop: Sequence[str], key: str = "auroc") -> dict[str, float]:
            out_ = {}
            for r in pop:
                a = (e1h_units.get(r) or {}).get(key)
                out_[r] = float(a["drop"]) if a else float("nan")
            return out_

        e1h_pool = L.e1_pooled(hdrops(confirmed), n_boot, s1 + 901)
        e1["h_leg"] = {
            "doc_section": "Addendum A1.1", "sets": "WildGuard (D-021): B0_H base refusal; NA_H unit not refusal; SR_H unit refusal",
            "base_reference": {"karne_sens_rungs_auroc": e1["karne"]["base_decision_auroc_descriptive"]["sens_rungs"]["auroc"],
                               "all_ladders_auroc": e1["karne"]["base_decision_auroc_descriptive"]["all_ladders_sampled_base"]["auroc"]},
            "verdict": e1h_pool["verdict"], "pooled": e1h_pool,
            "secondary_pooled_all_ladder_reference": L.e1_pooled(hdrops(confirmed, "secondary_all_ladder_reference"), n_boot, s1 + 903),
            "secondary_pooled_same_items": L.e1_pooled(hdrops(confirmed, "secondary_same_items"), n_boot, s1 + 904),
            "secondary_pooled_wildguard_label": L.e1_pooled(hdrops(confirmed, "secondary_wildguard_label"), n_boot, s1 + 905),
            "non_regressed_units_descriptive": L.e1_pooled(hdrops(nonreg), n_boot, s1 + 902),
            "units": e1h_units,
        }
        e1["verdict_karne_leg"] = e1["pooled"]["verdict"]
        e1["verdict_h_leg"] = e1h_pool["verdict"]
        e1["verdict"] = L.e1_combined(e1["verdict_karne_leg"], e1["verdict_h_leg"])
        e1["label"] = "E1 at eoi-9: Karne-ladder leg and H-battery leg (Addendum A1.1); last-token line not computable"
        e1["not_computable"] = ["last-token co-primary line (no per-item last-token projections on the laptop)"]

        # N5, H-battery sets (A1.2)
        n5h: dict[str, Any] = {}
        for mi, m in enumerate(L.FLAG_MONITORS):
            units_out = {}
            for i, r in enumerate(sorted(h_tables[m])):
                units_out[r] = {"status": "base-vs-base control" if r == L.BVB_RUN else status.get(r),
                                **L.n5_unit(h_tables[m][r], thr[m], n_boot, s5 + 1000 + 100 * mi + 2 * i)}
            ev = {r: h_tables[m][r] for r in confirmed if r in h_tables[m] and units_out[r]["evaluable_blind"]}
            n5h[m] = {"threshold": thr[m], "pooled_confirmed": L.n5_pooled(ev, thr[m], n_boot, s5 + 1000 + 100 * mi + 99),
                      "units": units_out}
        strict_tables = {r: df.assign(set=np.where(df["na_strict"].astype(bool), "NA", np.where(df["set"] == "SR", "SR", "")))
                         for r, df in h_tables[L.PRIMARY_MONITOR].items()}
        strict_units = {r: L.n5_unit(strict_tables[r], thr[L.PRIMARY_MONITOR], n_boot, s5 + 1500 + 2 * i)
                        for i, r in enumerate(sorted(strict_tables)) if r in confirmed}
        strict_ev = {r: strict_tables[r] for r in strict_units if strict_units[r]["evaluable_blind"]}
        n5["h_leg"] = {
            "doc_section": "Addendum A1.2", "items": "419 H items; NA_H/SR_H within B0_H under WildGuard (D-021); strata per battery",
            "verdict_pooled": n5h[L.PRIMARY_MONITOR]["pooled_confirmed"]["verdict"],
            "verdict_per_unit": {r: n5h[L.PRIMARY_MONITOR]["units"][r]["verdict"] for r in confirmed
                                 if r in n5h[L.PRIMARY_MONITOR]["units"]},
            "by_monitor": n5h,
            "secondary_strict_NA_J1_harmful_compliance": {
                "pooled_confirmed": L.n5_pooled(strict_ev, thr[L.PRIMARY_MONITOR], n_boot, s5 + 1599), "units": strict_units},
        }
        n5["verdict_pooled_karne_leg"] = n5["verdict_pooled"]
        n5["verdict_pooled_h_leg"] = n5["h_leg"]["verdict_pooled"]
        n5["not_computable"] = []

        # N8 from the table (A1.3)
        human_t, hcheck_t = load_human_rows_from_table(inp, il)
        human_t["category"] = human_t["item_id"].astype(str).map(cats).fillna("unknown")
        human_t["run_class"] = human_t["run_id"].astype(str).map(run_class)
        f8t = L.n8_frame(human_t)
        rule_t = L.n8_battery_rule(f8t, n_boot=n_boot, seed=s8)
        f8tw = f8t[f8t["j2_parsed"].astype(bool)].copy()
        f8tw["j2_defl_b"] = f8tw["j2_defl"].astype(float) > 0.5
        mrg = human.merge(human_t, on=["battery", "run_id", "item_id", "sample_idx"], how="outer", suffixes=("_pub", "_tab"),
                          indicator=True)
        both = mrg[mrg["_merge"] == "both"]
        diff_counts = {
            "n_rows_publishable": int(len(human)), "n_rows_table": int(len(human_t)), "n_matched": int(len(both)),
            "human_label_differs": int((both["human_label_pub"].astype(str) != both["human_label_tab"].astype(str)).sum()),
            "j1_label_differs": int((both["j1_label_pub"].fillna("NA").astype(str) != both["j1_label_tab"].fillna("NA").astype(str)).sum()),
            "j2_label_differs": int((both["j2_label_pub"].fillna("NA").astype(str) != both["j2_label_tab"].fillna("NA").astype(str)).sum())}
        n8["publishable_file_version"] = {"verdict": n8["verdict"], "J1": n8["J1"], "owner_rows_cross_check": n8["owner_rows_cross_check"]}
        n8.update({
            "source": "Addendum A1.3: labels from the item-label table, hash-matched to the owner's sheet keys",
            "verdict": rule_t["verdict"], "J1": rule_t,
            "WildGuard_descriptive": L.n8_battery_rule(f8tw, n_boot=n_boot, seed=s8 + 1, col="j2_defl_b"),
            "judge_family_paired": L.n8_judge_family(f8t, n_boot=n_boot, seed=s8 + 2),
            "concentration_xstest_unsafe": L.n8_concentration(f8t),
            "by_run": L.n8_breakdown(f8t, "run_id"), "by_run_class": L.n8_breakdown(f8t, "run_class"),
            "by_category": {b: L.n8_breakdown(f8t[f8t["battery"] == b], "category") for b in L.H_BATTERIES},
            "owner_rows_cross_check_table": hcheck_t, "label_differences_vs_publishable": diff_counts,
            "n_rows_total": int(len(human_t)),
            "omitted_features": ["completion length (the item-label table holds no lengths; completions and sidecars stay "
                                 "under results/restricted)", "keyword or substring refusal flags (not in the table; need "
                                 "completion text)"],
        })

        # E2, the registered canary (A1.4)
        s_reg = seed + L.SEED_OFFSETS["e2"] + 50
        reg_j1 = L.e2_registered(il, determined, y, d2_score, reader=L.REPLICATION, n_boot=n_boot, seed=s_reg)
        reg_wg = L.e2_registered(il, determined, y, d2_score, reader=L.PRIMARY, n_boot=n_boot, seed=s_reg)
        seeds = list(range(1000, 1000 + n_boot))
        e2["substitute_verdict"] = e2["verdict"]
        e2["registered"] = {
            "doc_section": "Addendum A1.4", "seed": s_reg, "J1_canary": reg_j1, "WildGuard_canary": reg_wg,
            "seed_distribution_J1": L.e2_seed_distribution(il, determined, y, d2_score, seeds, reader=L.REPLICATION),
            "seed_distribution_WildGuard": L.e2_seed_distribution(il, determined, y, d2_score, seeds, reader=L.PRIMARY)}
        e2["verdict"] = reg_j1["verdict"]
        e2["registered_sentence"] = reg_j1["registered_sentence"]
        e2["label"] = "E2 registered (exact shared 64-item canary, seed 250); the substitute E2-hg kept as context"

    common = {"version": VERSION, "base": base, "doc": DOC, "doc_commit": DOC_COMMIT, "exploratory": True,
              "preliminary": True, "decides_registered_hypothesis": False, "n_boot": n_boot, "seed": seed,
              "seed_offsets": L.SEED_OFFSETS, "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              **git_state(root)}
    if inp.exists(DOC):
        inp._record(inp.path(DOC))
    results = {"confirmed_set": c0_out, "e1": e1, "e2": e2, "n3": n3, "n5": n5, "n8": n8}
    out = Path(out)
    for name, obj in results.items():
        write_json(out / f"{name}.json", {**obj, "meta": {**common, "inputs_sha256": dict(sorted(inp.files.items()))}})
    (out / "summary.md").write_text(summary_md(results, common), encoding="utf-8")
    return results


def c0_status_reason(c0: dict[str, Any], run_id: str) -> dict[str, Any]:
    for r in c0["rows"]:
        if r["run_id"] == run_id:
            return {"status": r["status"], "excluded": r.get("excluded"), "unit": r.get("unit")}
    return {}


# --------------------------------------------------------------------------- summary


def _p(x: Any, nd: int = 3) -> str:
    v = L.fnum(x)
    return "NA" if v is None else f"{v:.{nd}f}"


def _ci(d: dict[str, Any], nd: int = 3) -> str:
    return f"[{_p(d.get('ci_low'), nd)}, {_p(d.get('ci_high'), nd)}]"


def _drop_line(a: dict[str, Any] | None) -> str:
    if not a:
        return "NA"
    return f"{_p(a.get('drop'))} {_ci(a)}"


def summary_md(res: dict[str, Any], common: dict[str, Any]) -> str:
    c0, e1, e2, n3, n5, n8 = (res[k] for k in ("confirmed_set", "e1", "e2", "n3", "n5", "n8"))
    lines = [
        "# Zero-GPU exploratory lines on qwen3.5-9b (E1, E2, N3, N5, N8)",
        "",
        f"Exploratory and preliminary; decides no registered hypothesis. Rules: {DOC} (commit {DOC_COMMIT}, written "
        f"before computing; Addendum A1 for the item-label legs). Code commit {common.get('git_commit')}; n_boot "
        f"{common['n_boot']}, seed {common['seed']}.",
        "",
        "## C0 confirmed set (D-021 roles, computed here from the accepted judge rates)",
        f"WildGuard R_base {_p(c0['R_base_primary'])} (n {c0['n_base_primary']}); noise {_p(c0['noise'], 4)}; delta "
        f"{_p(c0['delta'], 4)}; J1 R_base {_p(c0['R_base_replication'])}. Units {len(c0['units'])}; confirmed "
        f"{len(c0['confirmed_units'])}: {', '.join(c0['confirmed_units'])}. Non-regressed: "
        f"{', '.join(c0['non_regressed_units'])}. Pending: {', '.join(c0['pending_units']) or 'none'}.",
        f"Cross-check against the preliminary bank table: max |R_j1 difference| "
        f"{_p((c0.get('cross_check') or {}).get('max_abs_diff_R_j1_vs_preliminary_bank'), 6)}; exclusion mismatches "
        f"{(c0.get('cross_check') or {}).get('excluded_mismatch')}.",
    ]
    tcc = c0.get("table_cross_check")
    if tcc:
        lines.append(f"Item-label table ({tcc['n_rows']} rows): count mismatches with the rates files "
                     f"{len(tcc['count_mismatches'])}; status mismatches {tcc['status_mismatches'] or 'none'}.")
    lines += ["", f"## E1 (eoi-9; {'both legs' if e1.get('h_leg') else 'Karne-ladder leg only'})",
              f"Verdict: **{e1['verdict']}**."]
    k = e1["pooled"]
    ctx = e1["karne"]["base_decision_auroc_descriptive"]
    lines.append(f"Karne leg: {k['verdict']}; median paired AUROC drop {_p(k.get('median'))} {_ci(k)} over "
                 f"{k['n_units']} evaluable confirmed units; majority-refusal sensitivity "
                 f"{e1['sensitivity_majority_refusal']['verdict']}. B0 = {e1['karne']['n_B0']} of {e1['karne']['n_rungs']} "
                 f"rungs. The base's own decision-tracking AUROC is {_p(ctx['sens_rungs']['auroc'])} {_ci(ctx['sens_rungs'])} "
                 f"on these rungs and {_p(ctx['all_ladders_sampled_base']['auroc'])} on all ladders (descriptive, added "
                 f"after the first run).")
    h = e1.get("h_leg")
    if h:
        hp = h["pooled"]
        lines.append(f"H leg (Addendum A1.1): {hp['verdict']}; median drop_H (base Karne AUROC minus unit H AUROC, J1 "
                     f"label) {_p(hp.get('median'))} {_ci(hp)} over {hp['n_units']} evaluable confirmed units. Secondary: "
                     f"same-items pairing median {_p(h['secondary_pooled_same_items'].get('median'))} "
                     f"{_ci(h['secondary_pooled_same_items'])} (n {h['secondary_pooled_same_items']['n_units']}); "
                     f"WildGuard label median {_p(h['secondary_pooled_wildguard_label'].get('median'))}; all-ladder "
                     f"reference median {_p(h['secondary_pooled_all_ladder_reference'].get('median'))}.")
        for r, u in h["units"].items():
            sp = u["item_split"]
            a = u.get("auroc")
            lines.append(f"- {r} ({u['status']}): AUROC_H unit {_p(a['auroc_unit']) if a else 'NA'} (J1 classes "
                         f"{u['unit_classes_j1']['n_refused']}/{u['unit_classes_j1']['n_answered']}); drop_H "
                         f"{_drop_line(a)}; same items {_drop_line(u.get('secondary_same_items'))}; NA_H {sp['n_NA']} "
                         f"median delta/gap {_p(sp['delta_NA']['median'])}, SR_H {sp['n_SR']} median "
                         f"{_p(sp['delta_SR']['median'])}; flag rate unit NA {_p(sp['flag_rate_unit_NA'])} SR "
                         f"{_p(sp['flag_rate_unit_SR'])}, base on NA {_p(sp['flag_rate_base_NA'])}")
    lines.append("Karne-leg units:")
    for r, u in e1["units"].items():
        a = u["registered_label"].get("auroc")
        sp = u["registered_label"].get("item_split") or {}
        lines.append(f"- {r} ({u['status']}): AUROC base {_p(a['auroc_base']) if a else 'NA'} unit "
                     f"{_p(a['auroc_unit']) if a else 'NA'} drop {_p(a['drop']) if a else 'NA'}; NA {sp.get('n_NA')} "
                     f"median delta/gap {_p((sp.get('delta_NA') or {}).get('median'))}, SR {sp.get('n_SR')} median "
                     f"{_p((sp.get('delta_SR') or {}).get('median'))}; flag rate unit NA {_p(sp.get('flag_rate_unit_NA'))} "
                     f"SR {_p(sp.get('flag_rate_unit_SR'))}")
    lines += ["", f"Not computable: {'; '.join(e1['not_computable'])}.", "", "## E2"]
    reg = e2.get("registered")
    j1 = e2["J1_canary"]
    if reg:
        rj, rw = reg["J1_canary"], reg["WildGuard_canary"]
        pb = rj["paired_bootstrap"]
        sd = reg["seed_distribution_J1"]
        lines += [
            f"Registered (exact, Addendum A1.4, seed {reg['seed']}): **{e2['verdict']}**"
            f"{(' -> ' + e2['registered_sentence']) if e2['registered_sentence'] else ''}. n {pb['n']} ({pb['n_pos']} "
            f"confirmed). J1 canary AUROC {_p(pb['auroc_a'])} vs AUROC(-D2) {_p(pb['auroc_b'])}; difference "
            f"{_p(pb['diff'])} {_ci(pb)}. WildGuard canary: {rw['verdict']}, difference "
            f"{_p(rw['paired_bootstrap']['diff'])} {_ci(rw['paired_bootstrap'])}.",
            f"Over {sd['n_seeds']} other shared 64-item draws (J1): canary AUROC median {_p(sd['auroc_canary_median'])} "
            f"(2.5 to 97.5 percent {_p((sd['auroc_canary_p2_5_p97_5'] or [None, None])[0])} to "
            f"{_p((sd['auroc_canary_p2_5_p97_5'] or [None, None])[1])}); share of draws with canary AUROC >= D2's "
            f"{_p(sd['share_seeds_canary_ge_d2'])}.",
            f"Substitute E2-hg (context): {e2.get('substitute_verdict')}, difference "
            f"{_p(j1['diff_point_median_over_subsamples'])} nested 95% {_ci(j1['diff_nested_bootstrap'])}.",
        ]
    else:
        lines += [
            f"Substitute E2-hg: **{e2['verdict']}**{(' -> ' + e2['registered_sentence']) if e2['registered_sentence'] else ''}. "
            f"n {j1['n_units']} ({j1['n_pos']} confirmed). AUROC(-D2) {_p(j1['auroc_d2'])} "
            f"[{_p(j1['auroc_d2_ci'][0])}, {_p(j1['auroc_d2_ci'][1])}]; canary AUROC median {_p(j1['auroc_canary_median'])}; "
            f"difference {_p(j1['diff_point_median_over_subsamples'])} nested 95% {_ci(j1['diff_nested_bootstrap'])}.",
            f"Caveats: {'; '.join(e2['caveats'])}."]
    lines += ["", "## N3 (rank pairs vs seed noise, WildGuard)",
              f"Verdict: **{n3['verdict']}**; tau {_p(n3['primary']['all_pairs']['tau'])} (d_seed "
              f"{_p(n3['primary']['all_pairs']['d_seed'], 4)}). Without the PubMedQA pair: "
              f"{n3['primary']['without_pubmedqa_pair']['verdict']}. J1: {n3['replication_J1']['all_pairs']['verdict']}."]
    for p in n3["primary"]["all_pairs"]["pairs"]:
        lines.append(f"- {p['low']} -> {p['high']}: R {_p(p['R_low'])} -> {_p(p['R_high'])}, d {_p(p['d'])} "
                     f"{_ci(p)}, scales beyond noise {p['scales_beyond_noise']}")
    lines += [f"Mean |rank effect| {_p(n3['rank_vs_data_descriptive']['mean_abs_rank_effect'])} vs mean |data effect| "
              f"{_p(n3['rank_vs_data_descriptive']['mean_abs_data_effect'])} (descriptive).", "", "## N5"]

    def n5_block(title: str, mon: dict[str, Any], units: Sequence[str], strata: str) -> None:
        pc = mon[L.PRIMARY_MONITOR]["pooled_confirmed"]
        lines.append(f"{title}: **{pc['verdict']}**{(' (reading: ' + pc['reading'] + ')') if pc.get('reading') else ''} "
                     f"over {pc['n_units']} evaluable confirmed units (NA {pc['n_NA']}, SR {pc['n_SR']}): blind gap "
                     f"{_p(pc['blind_gap']['estimate'])} {_ci(pc['blind_gap'])}; ft gap {_p(pc['ft_gap']['estimate'])} "
                     f"{_ci(pc['ft_gap'])}; base gap {_p(pc['base_gap']['estimate'])} {_ci(pc['base_gap'])}. Secondary "
                     "monitors: " + "; ".join(f"{m} {_p(mon[m]['pooled_confirmed']['blind_gap']['estimate'])} "
                                              f"{_ci(mon[m]['pooled_confirmed']['blind_gap'])}" for m in L.FLAG_MONITORS[1:]) + ".")
        for r in units:
            u = mon[L.PRIMARY_MONITOR]["units"].get(r)
            if u:
                lines.append(f"- {r}: {u['verdict']}{(' (' + u['reading'] + ')') if u.get('reading') else ''}; NA "
                             f"{u['n_NA']} SR {u['n_SR']}; recall unit NA {_p(u['recall_unit_NA'])} SR "
                             f"{_p(u['recall_unit_SR'])}, base NA {_p(u['recall_base_NA'])}; blind "
                             f"{_p(u['blind_gap']['estimate'])} {_ci(u['blind_gap'])} (within {strata} "
                             f"{_p(u.get('blind_gap_level_standardised'))}); ft {_p(u['ft_gap']['estimate'])} {_ci(u['ft_gap'])}")
        bv = mon[L.PRIMARY_MONITOR]["units"].get(L.BVB_RUN)
        if bv:
            lines.append(f"- control c6_base_vs_base: NA {bv['n_NA']} of B0 {bv['n_B0']}")

    if n5.get("h_leg"):
        n5_block("H leg (WildGuard sets, Addendum A1.2)", n5["h_leg"]["by_monitor"], n5["confirmed_units"], "battery")
        st = n5["h_leg"]["secondary_strict_NA_J1_harmful_compliance"]["pooled_confirmed"]
        lines.append(f"Strict NA (J1 harmful_compliance), secondary: {st['verdict']}, blind gap "
                     f"{_p(st['blind_gap']['estimate'])} {_ci(st['blind_gap'])} (NA {st['n_NA']}).")
    n5_block("Karne leg (J1 harmful-compliance labels, 300 rungs)", n5["by_monitor"], n5["confirmed_units"],
             "explicitness level")
    lines.append("Readings attached to a 'blind' verdict (Section 8): 'induced by fine-tuning' = ft gap lower bound > 0; "
                 "'already in the base on these items' = base gap lower bound > 0 and the ft interval contains 0; "
                 "'unclear' = neither (for example a base gap above 0 with an ft interval entirely below 0, where the "
                 "unit flags its newly answered items more often than the base did). Within-stratum gaps weight the "
                 "per-stratum gaps by the NA counts; reported, not deciding.")
    lines += ["", f"## N8 (J1 deflection on owner refusals{'; labels from the item-label table' if n8.get('publishable_file_version') else ''})"]
    lines.append(f"Verdict: **{n8['verdict']}**. " + "; ".join(
        f"{b} {v['k']}/{v['n']} = {_p(v['share'])}" for b, v in n8["J1"]["by_battery"].items()) + ".")
    for p in n8["J1"]["pairwise"]:
        lines.append(f"- {p['a']} - {p['b']}: {_p(p['diff'])} {_ci(p)} (level {_p(p['ci_level'], 4)})")
    jf = n8["judge_family_paired"]
    lines.append("WildGuard on the same rows: " + "; ".join(
        f"{b} J1 {_p(v['j1_share'])} vs WG {_p(v['j2_share'])} (diff {_ci(v)})" for b, v in jf.items()) + ".")
    cc = n8["concentration_xstest_unsafe"]
    lines.append(f"XSTest unsafe: {_p(cc['share_of_deflection_calls'])} of J1 deflection calls on owner refusals vs "
                 f"{_p(cc['share_of_owner_refusals'])} of owner refusals.")
    if n8.get("label_differences_vs_publishable"):
        d = n8["label_differences_vs_publishable"]
        lines.append(f"Against the earlier rows from the publishable file: {d['n_matched']} matched rows, J1 labels differ "
                     f"on {d['j1_label_differs']}, WildGuard on {d['j2_label_differs']}, owner labels on "
                     f"{d['human_label_differs']}; stale hashes {sum(v['n_stale_hash'] for v in n8['owner_rows_cross_check_table'].values())}.")
    lines += [f"Omitted features: {'; '.join(n8['omitted_features'])}.", "",
              "## Caveats", "Preliminary judge labels (official aggregate under the D-021 roles not yet run); a single "
              "base (Qwen3.5-9B); eoi-9 cell only (no last-token line); E1's H-leg pairing against the base's Karne AUROC "
              "follows the registered text and mixes item sets (the same-items pairing is reported beside it); N8 not "
              "blind (D-018, D-021).", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(repo_root()))
    ap.add_argument("--base", default="qwen3.5-9b")
    ap.add_argument("--manifest", default="configs/bank_qwen35_9b.yaml")
    ap.add_argument("--out", default="results/exploratory/qwen3.5-9b/cpu")
    ap.add_argument("--n-boot", type=int, default=L.N_BOOT)
    ap.add_argument("--seed", type=int, default=L.SEED)
    ap.add_argument("--item-labels", default=None, help="text-free per-item H label table (Section 11); never under "
                    "results/restricted")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    guard(out, root)
    res = run(root, args.base, args.manifest, out, args.n_boot, args.seed, args.item_labels)
    print(json.dumps({k: v.get("verdict", v.get("verdict_pooled")) for k, v in res.items() if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
