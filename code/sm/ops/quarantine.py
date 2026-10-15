"""Quarantine judged-harmful completions out of open battery files (deviation D-007).

C7c (``sm.accept.c7c``) scans the open battery files under ``results/batteries/`` for whole copies of every
completion J1 labelled harmful_compliance (misaligned for em_*, trait_expressed for persona_evil) under a
restricted battery (rubric v0.4.3, D-005). A derivative can give the same text to a benign prompt (an XSTest-safe
or IFEval item), so that text then sits in an open file. Policy: such rows move to restricted storage; the scan is
not relaxed.

``python -m sm.ops.quarantine --base KEY [--dry-run] [--root DIR]``

1. Recompute the C7c whole-completion needles exactly as c7c does, with its own helpers (``leak_label``,
   ``judge_label_path``, ``last_labels``, ``read_jsonl``, ``collapse_ws``; every restricted battery of every base,
   as the scan is global).
2. Every row of every open battery file ``results/batteries/KEY/<run>/<battery>.jsonl`` whose JSON line hits a
   needle under the C7c open-battery rule (the JSON-escaped needle in the raw line, or the needle inside a
   normalised view of the line from ``leak_views``) is affected; a row already marked ``redacted: true`` is skipped.
3. Per affected file: the original lines are appended to
   ``results/restricted/batteries/KEY/<run>/<battery>.quarantine.jsonl`` (dirs 0700, file 0600, append-only, one
   line per (item_id, sample_idx), never duplicated); the open file is rewritten atomically with each affected row's
   ``completion`` replaced by ``[REDACTED: restricted; sha256=<sha256 of the raw original completion>]`` and
   ``redacted: true`` added (every other field, every other line byte for byte, and the row order kept); the sidecar
   ``<battery>.meta.json`` gets ``output_sha256`` = SHA-256 of the new bytes and a ``redaction`` block::

       {"n_rows", "rows": [{"item_id", "sample_idx"}], "output_sha256_before", "output_sha256_after",
        "quarantine_file", "deviation": "D-007", "tool": "sm.ops.quarantine", "timestamp"}

   A second pass on the same file keeps the first ``output_sha256_before`` (the hash the producer and every
   downstream input map recorded), accumulates ``rows`` and moves the previous block under ``history``.
4. The rewritten text is re-scanned with the C7c rule before anything is written; a residual hit (a needle in a
   field other than ``completion``, or across rows) refuses that file.

The report prints file paths, row counts and hashes only: never completion text or item ids. Exit 0 when every
file is clean or was quarantined, 1 when a file was refused, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sm import ops as OPS
from sm.accept import (
    Ctx,
    Missing,
    collapse_ws,
    judge_label_path,
    last_labels,
    leak_label,
    leak_views,
    now_iso,
    read_jsonl,
    sha256_bytes,
    sha256_text,
)

DEVIATION = "D-007"
TOOL = "sm.ops.quarantine"
QUARANTINE_SUFFIX = ".quarantine.jsonl"


def redaction_marker(raw: str) -> str:
    """The marker carries the sha256 of the RAW original completion, which is what the row's completion_sha256 holds
    (BT-1 compares the two); the needle match itself is done on the collapsed text."""
    return f"[REDACTED: restricted; sha256={sha256_text(raw)}]"


def full_needles(ctx: Ctx) -> set[str]:
    """The C7c open-battery needles: every collapsed completion of >= 80 characters that J1 labelled with the
    battery's leak label (sm.accept.leak_label) in a restricted battery file of any base (as sm.accept.c7c)."""
    rb = ctx.p("results/restricted/batteries")
    out: set[str] = set()
    if not rb.is_dir():
        return out
    for jl in rb.rglob("*.jsonl"):
        b = jl.name[: -len(".jsonl")]
        want = leak_label(ctx, b)
        if want is None:
            continue
        base, run = jl.parent.parent.name, jl.parent.name
        lp = judge_label_path(ctx, base, run, b, "J1")
        if lp is None:
            continue
        labels = last_labels(lp)
        for r in read_jsonl(jl):
            key = (str(r.get("item_id")), int(r.get("sample_idx") or 0))
            lab = labels.get(key)
            if lab and lab.get("label") == want:
                c = collapse_ws(str(r.get("completion") or ""))
                if len(c) >= 80:
                    out.add(c)
    return out


class Scanner:
    """The c7c open-battery test (the JSON-escaped needle in the raw text, or the needle in a leak_views view)."""

    def __init__(self, needles: set[str]) -> None:
        self.needles = needles
        self.escaped = {json.dumps(n)[1:-1] for n in needles}

    def hits(self, path: Path, txt: str) -> bool:
        if not self.needles:
            return False
        return any(e in txt for e in self.escaped) or any(n in v for v in leak_views(path, txt) for n in self.needles)


def _split_lines(data: bytes) -> list[str]:
    """Rows as text lines, split on "\\n" only (D-001: U+0085 inside a completion is not a row boundary)."""
    return data.decode("utf-8").split("\n")


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _restricted_dir(root: Path, rel_parts: list[str]) -> Path:
    """results/restricted/<rel_parts...>, every level from results/restricted down created or set to 0700."""
    cur = root / "results" / "restricted"
    chain = [cur]
    for part in rel_parts:
        cur = cur / part
        chain.append(cur)
    for d in chain:
        d.mkdir(exist_ok=True)
        os.chmod(d, 0o700)
    return cur


def _row_key(r: dict) -> tuple[str, int]:
    return str(r.get("item_id")), int(r.get("sample_idx") or 0)


@dataclass
class FilePlan:
    jsonl: Path
    sidecar: Path
    quarantine: Path
    sha_before: str = ""
    sha_after: str = ""
    new_bytes: bytes = b""
    new_rows: list[tuple[int, dict, str]] = field(default_factory=list)  # (line index, row, original line)
    already: int = 0
    error: str | None = None


def plan_file(root: Path, base: str, jl: Path, scan: Scanner) -> FilePlan | None:
    """The quarantine plan of one open battery file; None when the file holds no hit and no redacted row."""
    run, b = jl.parent.name, jl.name[: -len(".jsonl")]
    sidecar = jl.with_name(f"{b}.meta.json")
    qf = root / "results" / "restricted" / "batteries" / base / run / f"{b}{QUARANTINE_SUFFIX}"
    plan = FilePlan(jsonl=jl, sidecar=sidecar, quarantine=qf)
    data = jl.read_bytes()
    plan.sha_before = sha256_bytes(data)
    lines = _split_lines(data)
    out_lines = list(lines)
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        row = json.loads(ln)
        if row.get("redacted") is True:
            plan.already += 1
            continue
        if not scan.hits(jl, ln):
            continue
        orig = collapse_ws(str(row.get("completion") or ""))
        new = dict(row)
        new["completion"] = redaction_marker(str(row.get("completion") or ""))
        new["redacted"] = True
        out_lines[i] = json.dumps(new, ensure_ascii=False)
        plan.new_rows.append((i, row, ln))
    whole_hit = scan.hits(jl, data.decode("utf-8"))
    if not plan.new_rows and not plan.already and not whole_hit:
        return None
    plan.new_bytes = "\n".join(out_lines).encode("utf-8")
    plan.sha_after = sha256_bytes(plan.new_bytes)
    if scan.hits(jl, plan.new_bytes.decode("utf-8")):
        plan.error = "a needle remains after redacting the completion field (another field, or across rows)"
        return plan
    if not sidecar.is_file():
        plan.error = "sidecar missing"
        return plan
    sc = json.loads(sidecar.read_text(encoding="utf-8"))
    red = sc.get("redaction") if isinstance(sc.get("redaction"), dict) else None
    recorded = red.get("output_sha256_after") if red else sc.get("output_sha256")
    if recorded != plan.sha_before:
        plan.error = ("file hash differs from the sidecar's " +
                      ("redaction.output_sha256_after" if red else "output_sha256") + " (hash chain broken; not touched)")
        return plan
    if not plan.new_rows and plan.already and red is None:
        plan.error = "rows are marked redacted but the sidecar has no redaction block"
    return plan


def apply_plan(root: Path, base: str, plan: FilePlan) -> None:
    run, b = plan.jsonl.parent.name, plan.jsonl.name[: -len(".jsonl")]
    # 1. restricted copy first (append-only; a crash after this step is repaired by re-running: keys dedupe).
    _restricted_dir(root, ["batteries", base, run])
    have: set[tuple[str, int]] = set()
    if plan.quarantine.is_file():
        have = {_row_key(r) for r in read_jsonl(plan.quarantine)}
    fd = os.open(plan.quarantine, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        for _, row, ln in plan.new_rows:
            k = _row_key(row)
            if k in have:
                continue
            fh.write(ln + "\n")
            have.add(k)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(plan.quarantine, 0o600)
    # 2. the open file, atomically, in its own mode.
    mode = plan.jsonl.stat().st_mode & 0o777
    _atomic_write(plan.jsonl, plan.new_bytes, mode)
    # 3. the sidecar: output_sha256 is the new file, the redaction block records the chain.
    sc = json.loads(plan.sidecar.read_text(encoding="utf-8"))
    prev = sc.get("redaction") if isinstance(sc.get("redaction"), dict) else None
    rows = list(prev.get("rows") or []) if prev else []
    rows += [{"item_id": str(row.get("item_id")), "sample_idx": int(row.get("sample_idx") or 0)} for _, row, _ in plan.new_rows]
    block: dict[str, Any] = {
        "n_rows": len(rows),
        "rows": rows,
        "output_sha256_before": prev["output_sha256_before"] if prev else sc.get("output_sha256"),
        "output_sha256_after": plan.sha_after,
        "quarantine_file": str(plan.quarantine.relative_to(root)),
        "deviation": DEVIATION,
        "tool": TOOL,
        "timestamp": now_iso(),
    }
    if prev:
        block["history"] = list(prev.get("history") or []) + [{k: v for k, v in prev.items() if k != "history"}]
    sc["output_sha256"] = plan.sha_after
    sc["redaction"] = block
    smode = plan.sidecar.stat().st_mode & 0o777
    _atomic_write(plan.sidecar, (json.dumps(sc, indent=2, sort_keys=True) + "\n").encode("utf-8"), smode)


def run(root: Path, base: str, dry_run: bool = False, out: Any = None) -> int:
    out = out if out is not None else sys.stdout
    open_root = root / "results" / "batteries" / base
    if not open_root.is_dir():
        print(f"no open battery directory {open_root.relative_to(root)}", file=out)
        return 2
    ctx = Ctx(root, "battery", base, None, "quarantine")
    try:
        needles = full_needles(ctx)
    except Missing as e:
        print(f"cannot recompute the C7c needles: missing {e}", file=out)
        return 2
    scan = Scanner(needles)
    files = sorted(p for p in open_root.glob("*/*.jsonl") if p.is_file())
    print(f"{TOOL} ({DEVIATION}){' DRY RUN' if dry_run else ''}: base {base}, {len(needles)} flagged completions, "
          f"{len(files)} open battery files", file=out)
    rc = changed = total = 0
    for jl in files:
        plan = plan_file(root, base, jl, scan)
        if plan is None:
            continue
        rel = jl.relative_to(root)
        if plan.error:
            print(f"REFUSED {rel}: {plan.error} ({len(plan.new_rows)} rows hit, {plan.already} already redacted)", file=out)
            rc = 1
            continue
        if not plan.new_rows:
            print(f"ok      {rel}: 0 new rows, {plan.already} already redacted; sha256 {plan.sha_before}", file=out)
            continue
        if not dry_run:
            apply_plan(root, base, plan)
        changed += 1
        total += len(plan.new_rows)
        print(f"{'would ' if dry_run else ''}quarantine {rel}: {len(plan.new_rows)} rows ({plan.already} already redacted); "
              f"sha256 before {plan.sha_before} after {plan.sha_after}; -> {plan.quarantine.relative_to(root)}", file=out)
    print(f"{'would change' if dry_run else 'changed'} {changed} files, {total} rows; exit {rc}", file=out)
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=TOOL, description="Quarantine judged-harmful completions out of open battery files (D-007).")
    ap.add_argument("--base", required=True, help="base key, e.g. qwen3.5-9b (results/batteries/<base>/)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    ap.add_argument("--root", default=None, help="repository root (default: this checkout)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else OPS.REPO_ROOT
    return run(root, args.base, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
