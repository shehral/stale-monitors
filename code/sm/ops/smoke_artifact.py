"""Machine-readable smoke results for G1 (prereg/ACCEPTANCE.md OPEN-8, checks SM-1 to SM-3 and C3).

Three ways to write one artifact, all with the same schema (``SCHEMA``):

- **cluster** (``jobs/smoke.sbatch`` through ``jobs/common.sh sm_smoke_artifact``): reads the checks the job
  recorded (``<smoke_out>/checks.tsv``, one ``<name>\\t<rc>`` per line, and ``skipped.txt``) and the pytest
  summary line from ``<smoke_out>/offline.log``; writes ``results/smoke/<base>/<jobid>.json``
  (``results/smoke/_nobase/<jobid>.json`` for a smoke without BASE or BANK, which covers no base).
- **log** (``--from-log logs/smoke-<jobid>.out``): rebuilds the same artifact from a finished smoke job's
  ``.out`` log (the summary ``PASS``/``FAIL``/``SKIP`` lines, the ``frozen: base= pin= commit=`` line, the
  pytest summary and the ``LEDGER`` line). This is how smokes that ran before this module existed (commit
  5acbc0a, job 10893609) get an artifact; ``source_log_sha256`` ties it to the log.
- **split cluster smoke** (the default since chain 3; job 10899610 timed out at gpu-short's 1 h cap with the GPU
  idle while pytest ran on NFS): ``jobs/smoke_cpu.sbatch`` (partition short) runs pytest and the eleven smoke
  CLIs and writes the CPU half ``results/smoke/<base>/cpu/<cpu_jobid>.json`` (``--checks ... --half cpu``;
  kind ``cpu_half``, never read by acceptance). ``jobs/smoke.sbatch`` (gpu-short) runs only the GPU checks:
  at its start ``--find-cpu-half`` locates the CPU half of the same git commit (``SMOKE_CPU_JOB``, else the
  newest passing one; exit 5 when there is none), and at its end ``--checks ... --cpu-half <path>`` merges
  the two halves (``merge_halves``) into ``results/smoke/<base>/<gpu_jobid>.json``: the same schema as a
  single-job smoke (checks of both halves, git_commit, code_tree, pin, pytest_summary) plus cpu_job_id,
  gpu_job_id and the CPU half's pytest settings, so SM-1 to SM-4 read it unchanged.
- **local** (``--local``): the laptop half of G1. Runs pytest on ``tests/`` (unless ``--skip-pytest``) and
  every smoke CLI (``SMOKE_CLIS``, each ``python -m <module> --smoke --out <tmp>/<name> --force``) on CPU
  with ``SM_OFFLINE=1`` and ``SM_DEVICE=cpu``, then writes ``results/smoke/local/<git_sha>.json``
  (``<git_sha>.dirty.json`` when ``code/ configs/ jobs/ tests/`` differ from that commit, so a dirty run
  never stands in for a clean one).

Artifact fields: schema, kind, base, slurm_job_id, checks {name: rc}, skipped [names], failures, passed,
git_commit, code_tree (``git rev-parse <commit>:code``, the tree C3 compares), code_dirty, pin (the base
revision, 40 hex, or null), judges {J1, J2: {model, revision}} when the job served them, pytest_summary,
pytest_rc, cli_modules, hostname, timestamp, plus source_log / source_log_sha256 for kind "log".
Merged (split) artifacts add cpu_job_id, gpu_job_id, cpu_half (path, sha256, hostname, timestamp),
pytest_workers (xdist worker count, 1 = serial), pytest_markexpr (``not slow``: the subprocess-heavy tests the
laptop half runs) and cpu_tmp_root / cpu_tmp_kind (node-local or scratch fallback).
``passed`` is true when there is at least one check, every rc is 0 and nothing was skipped (SM-2: no FAIL
or SKIP line); a log artifact also needs ``failures=0`` and a LEDGER rc of 0, and a local one a clean tree.
Metadata only: no completion, judge or prompt text.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from sm import ops as OPS

SCHEMA = "sm.smoke_artifact/1"
# The eleven CLIs jobs/smoke.sbatch runs with --smoke (ACCEPTANCE.md SM-1), in the job's order.
SMOKE_CLIS: tuple[str, ...] = (
    "sm.data.build_prompt_sets",
    "sm.data.build_bank_datasets",
    "sm.train.lora_sft",
    "sm.eval.battery",
    "sm.eval.judge",
    "sm.eval.capability",
    "sm.interp.base_card",
    "sm.interp.causal",
    "sm.interp.drift",
    "sm.interp.monitor_scores",
    "sm.analysis.aggregate",
)
NOBASE_DIR = "_nobase"
LOCAL_DIR = "local"
CPU_HALF_DIR = "cpu"
CPU_HALF_KIND = "cpu_half"
# Exit code of the GPU smoke when no CPU half of its git commit exists (jobs/common.sh: upstream output missing).
EXIT_NO_CPU_HALF = 5
DIRTY_PATHS = ("code", "configs", "jobs", "tests")

_PYTEST_SUMMARY = re.compile(
    r"^=*\s*(?:\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)(?:, )?)+.* in [0-9.]+s(?: \([0-9:]+\))?\s*=*$"
)
_RESULT_LINE = re.compile(r"^\s*(PASS|FAIL|SKIP)\s{2}(.*?)\s*$")
_FAIL_RC = re.compile(r"^(.*) \(rc=(-?\d+)\)$")
_FROZEN = re.compile(r"frozen: base=(\S+) pin=(\S+) commit=(\S+)")
_ENV_JOB = re.compile(r"\benv: host=(\S+) job=(\S+)")
_LEDGER = re.compile(r"^LEDGER job=(\S*) .*\brc=(-?\d+)")
_DONE = re.compile(r"=== sm-smoke done: failures=(\d+) ===")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pytest_summary(text: str) -> str | None:
    """The last pytest summary line in ``text`` ("425 passed, 2 skipped in 31.20s"), '=' padding removed."""
    found = None
    for line in text.splitlines():
        s = line.strip()
        if _PYTEST_SUMMARY.match(s):
            found = s.strip("= ").strip()
    return found


def read_checks(tsv: str | Path) -> dict[str, int]:
    """{name: rc} from ``<name>\\t<rc>`` lines (the last line per name wins)."""
    out: dict[str, int] = {}
    p = Path(tsv)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        name, rc = line.rsplit("\t", 1)
        try:
            out[name] = int(rc.strip())
        except ValueError:
            out[name] = 1
    return out


def _read_lines(path: str | Path | None) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _judges_from_env() -> dict[str, Any]:
    out = {}
    for j in ("J1", "J2"):
        m, r = os.environ.get(f"SM_{j}_MODEL"), os.environ.get(f"SM_{j}_REVISION")
        if m or r:
            out[j] = {"model": m, "revision": r}
    return out


def build_artifact(
    kind: str,
    checks: dict[str, int],
    skipped: Sequence[str] = (),
    base: str | None = None,
    slurm_job_id: str | None = None,
    pin: str | None = None,
    git_commit: str | None = None,
    pytest_text: str | None = None,
    pytest_rc: int | None = None,
    cli_modules: Sequence[str] = SMOKE_CLIS,
    extra: dict[str, Any] | None = None,
    repo: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the artifact dict (pure apart from git lookups of the commit's code tree)."""
    commit = git_commit if git_commit is not None else OPS.git_commit(repo)
    failures = sum(1 for rc in checks.values() if rc != 0)
    art: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "base": base or None,
        "slurm_job_id": slurm_job_id,
        "checks": dict(checks),
        "skipped": list(skipped),
        "failures": failures,
        # ACCEPTANCE.md SM-2: no FAIL and no SKIP line.
        "passed": bool(checks) and failures == 0 and not list(skipped),
        "git_commit": commit,
        "code_tree": OPS.git_object(commit, "code", repo) if commit else None,
        "pin": pin if pin else None,
        "pytest_summary": pytest_summary(pytest_text) if pytest_text else None,
        "pytest_rc": pytest_rc if pytest_rc is not None else checks.get("pytest"),
        "cli_modules": list(cli_modules),
        "hostname": socket.gethostname(),
        "timestamp": _utc_now(),
    }
    art.update(extra or {})
    return art


def artifact_path(out_root: str | Path, art: dict[str, Any]) -> Path:
    """results/smoke/<base>/<jobid>.json (cluster, log); results/smoke/local/<git_sha>[.dirty].json (local)."""
    root = Path(out_root)
    if art["kind"] == CPU_HALF_KIND:
        sub = art.get("base") or NOBASE_DIR
        return root / _safe(sub) / CPU_HALF_DIR / f"{art.get('slurm_job_id') or 'nojob'}.json"
    if art["kind"] == "local":
        sha = art.get("git_commit") or "nogit"
        return root / LOCAL_DIR / (f"{sha}.dirty.json" if art.get("code_dirty") else f"{sha}.json")
    sub = art.get("base") or NOBASE_DIR
    return root / _safe(sub) / f"{art.get('slurm_job_id') or 'nojob'}.json"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def write_artifact(out_root: str | Path, art: dict[str, Any]) -> Path:
    path = artifact_path(out_root, art)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(art, indent=2, sort_keys=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# --------------------------------------------------------------------------- cluster job


def from_job(
    checks_tsv: str | Path,
    skipped_txt: str | Path | None,
    offline_log: str | Path | None,
    job: str | None,
    base: str | None,
    pin: str | None,
    smoke_out: str | Path | None = None,
    repo: str | Path | None = None,
    kind: str = "cluster",
    cpu_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The artifact of a running smoke job (called at its end by jobs/common.sh sm_smoke_artifact, or with
    ``kind="cpu_half"`` by sm_smoke_cpu_half at the end of jobs/smoke_cpu.sbatch)."""
    text = Path(offline_log).read_text(encoding="utf-8", errors="replace") if offline_log and Path(offline_log).is_file() else None
    commit = OPS.git_commit(repo)
    clean = OPS.paths_clean(["code", "configs", "jobs"], repo)
    extra: dict[str, Any] = {"code_dirty": None if clean is None else (not clean), "smoke_out": str(smoke_out) if smoke_out else None}
    if kind == CPU_HALF_KIND:
        extra.update(cpu_extra or {})
    else:
        extra["judges"] = _judges_from_env()
    return build_artifact(
        kind, read_checks(checks_tsv), _read_lines(skipped_txt), base=base, slurm_job_id=job, pin=pin,
        git_commit=commit, pytest_text=text, repo=repo, extra=extra,
    )


# --------------------------------------------------------------------------- split smoke: CPU half + GPU half


class CpuHalfError(Exception):
    """No usable CPU half for the GPU smoke (exit EXIT_NO_CPU_HALF)."""


def load_cpu_half(path: str | Path, git_commit: str | None, base: str | None) -> dict[str, Any]:
    """Read a CPU half and check it belongs to this GPU smoke: kind cpu_half, same git commit (and so the same
    code tree), same base, passed. Raises CpuHalfError otherwise."""
    p = Path(path)
    if not p.is_file():
        raise CpuHalfError(f"no CPU half at {p}")
    try:
        cpu = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CpuHalfError(f"CPU half {p} unreadable: {exc}") from exc
    if cpu.get("kind") != CPU_HALF_KIND or cpu.get("schema") != SCHEMA:
        raise CpuHalfError(f"{p} is not a CPU half (kind={cpu.get('kind')!r}, schema={cpu.get('schema')!r})")
    if not git_commit or cpu.get("git_commit") != git_commit:
        raise CpuHalfError(f"CPU half {p} is at commit {cpu.get('git_commit')}, this checkout is at {git_commit}: re-run jobs/smoke_cpu.sbatch")
    if (cpu.get("base") or None) != (base or None):
        raise CpuHalfError(f"CPU half {p} covers base {cpu.get('base')!r}, not {base!r}")
    if not cpu.get("passed"):
        raise CpuHalfError(f"CPU half {p} did not pass (failures={cpu.get('failures')}, skipped={cpu.get('skipped')})")
    return cpu


def find_cpu_half(out_root: str | Path, base: str | None, git_commit: str | None, cpu_job: str | None = None) -> Path:
    """The CPU half the GPU smoke merges: ``<root>/<base>/cpu/<cpu_job>.json`` when the job id is given
    (submit_bank.sh exports SMOKE_CPU_JOB), else the newest passing half of this git commit. Raises CpuHalfError."""
    d = Path(out_root) / _safe(base or NOBASE_DIR) / CPU_HALF_DIR
    if cpu_job:
        p = d / f"{cpu_job}.json"
        load_cpu_half(p, git_commit, base)
        return p
    cands = sorted(d.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True) if d.is_dir() else []
    reasons = []
    for p in cands:
        try:
            load_cpu_half(p, git_commit, base)
            return p
        except CpuHalfError as exc:
            reasons.append(str(exc))
    raise CpuHalfError(f"no passing CPU half for commit {git_commit} in {d}" + (f" ({len(reasons)} other halves: {reasons[0]})" if reasons else ""))


def merge_halves(cpu: dict[str, Any], gpu: dict[str, Any], cpu_path: str | Path | None = None) -> dict[str, Any]:
    """One smoke artifact from the CPU half and the GPU job's own artifact, in the single-job format acceptance
    reads (SM-1 to SM-4): checks of both halves (CPU first), skipped of both, git_commit/code_tree/pin of the GPU
    job, pytest_summary/pytest_rc of the CPU half, plus cpu_job_id and gpu_job_id. Raises CpuHalfError when the
    halves disagree on commit or base."""
    if cpu.get("git_commit") != gpu.get("git_commit") or not gpu.get("git_commit"):
        raise CpuHalfError(f"CPU half commit {cpu.get('git_commit')} != GPU job commit {gpu.get('git_commit')}")
    if (cpu.get("base") or None) != (gpu.get("base") or None):
        raise CpuHalfError(f"CPU half base {cpu.get('base')!r} != GPU job base {gpu.get('base')!r}")
    if cpu.get("pin") and gpu.get("pin") and cpu["pin"] != gpu["pin"]:
        raise CpuHalfError(f"CPU half pin {cpu['pin']} != GPU job pin {gpu['pin']}")
    checks = {**cpu.get("checks", {}), **gpu.get("checks", {})}
    skipped = [*cpu.get("skipped", []), *gpu.get("skipped", [])]
    failures = sum(1 for rc in checks.values() if rc != 0)
    art = dict(gpu)
    art.update({
        "kind": "cluster",
        "checks": checks,
        "skipped": skipped,
        "failures": failures,
        "passed": bool(checks) and failures == 0 and not skipped,
        "pytest_summary": cpu.get("pytest_summary"),
        "pytest_rc": cpu.get("pytest_rc"),
        "cli_modules": cpu.get("cli_modules", list(SMOKE_CLIS)),
        "code_dirty": bool(cpu.get("code_dirty") or gpu.get("code_dirty")) if (cpu.get("code_dirty") is not None or gpu.get("code_dirty") is not None) else None,
        "split": True,
        "cpu_job_id": cpu.get("slurm_job_id"),
        "gpu_job_id": gpu.get("slurm_job_id"),
        "cpu_half": {
            "path": str(cpu_path) if cpu_path else None,
            "sha256": _sha256_bytes(Path(cpu_path).read_bytes()) if cpu_path and Path(cpu_path).is_file() else None,
            "hostname": cpu.get("hostname"),
            "timestamp": cpu.get("timestamp"),
            "smoke_out": cpu.get("smoke_out"),
        },
        "pytest_workers": cpu.get("pytest_workers"),
        "pytest_markexpr": cpu.get("pytest_markexpr"),
        "cpu_tmp_root": cpu.get("tmp_root"),
        "cpu_tmp_kind": cpu.get("tmp_kind"),
    })
    return art


# --------------------------------------------------------------------------- finished job log


def parse_log(text: str) -> dict[str, Any]:
    """Fields of a jobs/smoke.sbatch .out log: checks, skipped, frozen line, job id, ledger rc, failures."""
    checks: dict[str, int] = {}
    skipped: list[str] = []
    in_summary = False
    frozen = None
    job = host = None
    ledger_rc = ledger_job = None
    failures_line = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if "=== smoke summary" in line:
            in_summary, checks, skipped = True, {}, []
            continue
        m = _FROZEN.search(line)
        if m:
            frozen = {"base": m.group(1), "pin": m.group(2), "commit": m.group(3)}
        m = _ENV_JOB.search(line)
        if m and job is None:
            host, job = m.group(1), m.group(2)
        m = _LEDGER.match(line.strip())
        if m:
            ledger_job, ledger_rc = m.group(1), int(m.group(2))
        m = _DONE.search(line)
        if m:
            failures_line = int(m.group(1))
            in_summary = False
            continue
        if in_summary:
            m = _RESULT_LINE.match(line)
            if not m:
                continue
            tag, rest = m.group(1), m.group(2)
            if tag == "SKIP":
                skipped.append(rest)
            elif tag == "PASS":
                checks[rest] = 0
            else:
                f = _FAIL_RC.match(rest)
                checks[f.group(1) if f else rest] = int(f.group(2)) if f else 1
    return {
        "checks": checks, "skipped": skipped, "frozen": frozen, "job": job, "host": host,
        "ledger_job": ledger_job, "ledger_rc": ledger_rc, "failures_line": failures_line,
    }


_BASE_FROM_CHECK = re.compile(r"^(?:revision pin|real-base load) \(([^,)]+)")


def from_log(log_path: str | Path, repo: str | Path | None = None) -> dict[str, Any]:
    """Rebuild the artifact of a finished smoke job from its .out log (deterministic in the log bytes,
    apart from hostname and timestamp, which describe the rebuild and are kept under rebuilt_*)."""
    p = Path(log_path)
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    info = parse_log(text)
    job = info["ledger_job"] or info["job"]
    if not job:
        m = re.search(r"smoke-(\d+)\.out$", p.name)
        job = m.group(1) if m else None
    frozen = info["frozen"] or {}
    base = frozen.get("base")
    if not base:
        for name in info["checks"]:
            m = _BASE_FROM_CHECK.match(name)
            if m:
                base = m.group(1).strip()
                break
    commit = frozen.get("commit") if frozen.get("commit") and _HEX40.match(frozen.get("commit", "")) else None
    pin = frozen.get("pin") if frozen.get("pin") and _HEX40.match(frozen.get("pin", "")) else None
    art = build_artifact(
        "log", info["checks"], info["skipped"], base=base, slurm_job_id=job, pin=pin, git_commit=commit,
        pytest_text=text, repo=repo,
        extra={
            "code_dirty": None,
            "judges": {},
            "source_log": str(p),
            "source_log_sha256": _sha256_bytes(data),
            "job_hostname": info["host"],
            "job_failures_line": info["failures_line"],
            "job_rc": info["ledger_rc"],
            "complete_log": info["ledger_rc"] is not None and info["failures_line"] is not None,
        },
    )
    art["rebuilt_hostname"] = art.pop("hostname")
    art["rebuilt_at"] = art.pop("timestamp")
    # The job's own verdict also needs its closing lines: "failures=0" and a LEDGER line with rc 0.
    art["passed"] = bool(art["passed"] and info["failures_line"] == 0 and info["ledger_rc"] == 0)
    return art


# --------------------------------------------------------------------------- laptop (G1 local half)

Runner = Callable[[list[str], dict[str, str], int], tuple[int, str]]


def _subprocess_runner(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, env=env, cwd=str(OPS.REPO_ROOT), capture_output=True, text=True, timeout=timeout, check=False)
        return res.returncode, (res.stdout or "") + (res.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, out + f"\n(timeout after {timeout}s)"


def run_local(
    out_root: str | Path,
    modules: Sequence[str] = SMOKE_CLIS,
    skip_pytest: bool = False,
    pytest_args: Sequence[str] = ("tests",),
    timeout: int = 900,
    runner: Runner | None = None,
    scratch: str | Path | None = None,
    log: Callable[[str], None] = print,
    write: bool = True,
) -> tuple[dict[str, Any], Path | None]:
    """Run the laptop smoke (pytest + the smoke CLIs on CPU) and write results/smoke/local/<git_sha>.json.

    ``runner(cmd, env, timeout) -> (rc, output)`` is injectable for tests. Device policy
    (docs/HARNESS_CONTRACTS.md): SM_DEVICE=cpu, SM_OFFLINE=1, one pytest process.
    """
    run = runner or _subprocess_runner
    env = dict(os.environ)
    env.update({"SM_OFFLINE": "1", "SM_DEVICE": "cpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    env.pop("SM_TINY_MODEL", None)
    code_dir = str(OPS.REPO_ROOT / "code")
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    commit = OPS.git_commit()
    clean = OPS.paths_clean(list(DIRTY_PATHS))
    checks: dict[str, int] = {}
    skipped: list[str] = []
    pytest_text = None
    t0 = time.time()
    tmp_ctx = tempfile.TemporaryDirectory(prefix="sm_smoke_local_") if scratch is None else None
    work = Path(scratch) if scratch is not None else Path(tmp_ctx.name)  # type: ignore[union-attr]
    try:
        if skip_pytest:
            skipped.append("pytest (--skip-pytest)")
        else:
            rc, out = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_args], env, timeout)
            pytest_text = out
            checks["pytest"] = rc
            log(f"{'PASS' if rc == 0 else 'FAIL'}  pytest ({pytest_summary(out) or 'no summary line'})")
        for mod in modules:
            rc, out = run([sys.executable, "-m", mod, "--smoke", "--out", str(work / mod.rsplit(".", 1)[-1]), "--force"], env, timeout)
            checks[mod] = rc
            log(f"{'PASS' if rc == 0 else 'FAIL'}  {mod}" + ("" if rc == 0 else f" (rc={rc})"))
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    missing = [m for m in SMOKE_CLIS if m not in modules]
    skipped.extend(f"{m} (not selected)" for m in missing)
    art = build_artifact(
        "local", checks, skipped, git_commit=commit, pytest_text=pytest_text, cli_modules=list(modules),
        extra={"code_dirty": None if clean is None else (not clean), "judges": {}, "elapsed_s": round(time.time() - t0, 1),
               "device": "cpu", "python": sys.executable},
    )
    # A local G1 pass needs every smoke CLI and pytest, at a clean commit.
    art["passed"] = bool(art["passed"] and art.get("code_dirty") is False)
    path = write_artifact(out_root, art) if write else None
    return art, path


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sm.ops.smoke_artifact", description="Write the machine-readable smoke artifact (G1; ACCEPTANCE.md OPEN-8).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="laptop G1: run pytest and the smoke CLIs on CPU, write results/smoke/local/<git_sha>.json")
    mode.add_argument("--checks", help="cluster: checks.tsv written by jobs/smoke.sbatch (<name>\\t<rc> lines)")
    mode.add_argument("--from-log", help="rebuild from a finished smoke job's .out log (logs/smoke-<jobid>.out)")
    mode.add_argument("--find-cpu-half", action="store_true",
                      help="GPU smoke start: print the path of the CPU half of this git commit (--cpu-job, else the newest passing one); exit 5 when none")
    ap.add_argument("--half", choices=("cpu",), default=None, help="cluster: write the CPU half (jobs/smoke_cpu.sbatch) to <out>/<base>/cpu/<jobid>.json")
    ap.add_argument("--cpu-half", default=None, help="cluster GPU smoke: merge this CPU half into the artifact (exit 5 when missing or at another commit)")
    ap.add_argument("--cpu-job", default=None, help="--find-cpu-half: the CPU smoke job id (SMOKE_CPU_JOB)")
    ap.add_argument("--checks-out", default=None, help="--find-cpu-half: also write the CPU half's checks here (<name>\\t<rc> lines)")
    ap.add_argument("--pytest-workers", type=int, default=None, help="cpu half: pytest-xdist workers (1 = serial)")
    ap.add_argument("--pytest-markexpr", default=None, help="cpu half: the -m expression pytest ran with (for example 'not slow')")
    ap.add_argument("--tmp-root", default=None, help="cpu half: the temp root pytest and the CLIs used")
    ap.add_argument("--tmp-kind", default=None, help="cpu half: node-local or scratch (fallback)")
    ap.add_argument("--out", default=str(OPS.REPO_ROOT / "results" / "smoke"), help="artifact root (default <repo>/results/smoke)")
    ap.add_argument("--skipped", default=None, help="cluster: skipped.txt (one skipped part per line)")
    ap.add_argument("--offline-log", default=None, help="cluster: offline.log (pytest output, for pytest_summary)")
    ap.add_argument("--smoke-out", default=None, help="cluster: the job's smoke output directory (recorded)")
    ap.add_argument("--job", default=None, help="cluster: raw SLURM_JOB_ID")
    ap.add_argument("--base", default=None, help="cluster: registry key of the base the smoke covered")
    ap.add_argument("--pin", default=None, help="cluster: the base revision the smoke checked (40 hex)")
    ap.add_argument("--modules", nargs="+", default=None, help="local: subset of the smoke CLIs (the artifact then does not pass)")
    ap.add_argument("--skip-pytest", action="store_true", help="local: do not run pytest (the artifact then does not pass)")
    ap.add_argument("--timeout", type=int, default=900, help="local: seconds per command")
    ap.add_argument("--print", dest="print_only", action="store_true", help="print the artifact, write nothing")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.local:
        modules = list(args.modules) if args.modules else list(SMOKE_CLIS)
        unknown = [m for m in modules if m not in SMOKE_CLIS]
        if unknown:
            print(f"error: not smoke CLIs: {unknown} (choose from {list(SMOKE_CLIS)})", file=sys.stderr)
            return 2
        art, path = run_local(args.out, modules, skip_pytest=args.skip_pytest, timeout=args.timeout, write=not args.print_only)
    elif args.find_cpu_half:
        try:
            p = find_cpu_half(args.out, args.base, OPS.git_commit(), args.cpu_job)
        except CpuHalfError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NO_CPU_HALF
        if args.checks_out:
            cpu = json.loads(p.read_text(encoding="utf-8"))
            Path(args.checks_out).write_text("".join(f"{k}\t{v}\n" for k, v in cpu["checks"].items()), encoding="utf-8")
        print(p)
        return 0
    elif args.from_log:
        if not Path(args.from_log).is_file():
            print(f"error: no log at {args.from_log}", file=sys.stderr)
            return 2
        art = from_log(args.from_log)
        path = None if args.print_only else write_artifact(args.out, art)
    else:
        if args.half == "cpu":
            cpu_extra = {"pytest_workers": args.pytest_workers, "pytest_markexpr": args.pytest_markexpr,
                         "tmp_root": args.tmp_root, "tmp_kind": args.tmp_kind}
            art = from_job(args.checks, args.skipped, args.offline_log, args.job, args.base, args.pin, args.smoke_out,
                           kind=CPU_HALF_KIND, cpu_extra=cpu_extra)
        else:
            art = from_job(args.checks, args.skipped, args.offline_log, args.job, args.base, args.pin, args.smoke_out)
        if not art["checks"]:
            print(f"error: no checks recorded in {args.checks}", file=sys.stderr)
            return 2
        if args.cpu_half:
            try:
                cpu = load_cpu_half(args.cpu_half, art["git_commit"], art.get("base"))
                art = merge_halves(cpu, art, args.cpu_half)
            except CpuHalfError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_NO_CPU_HALF
        path = None if args.print_only else write_artifact(args.out, art)
    if args.print_only:
        print(json.dumps(art, indent=2))
    else:
        print(f"smoke artifact: {path} (passed={art['passed']}, failures={art['failures']}, git_commit={art['git_commit']})")
    if args.local:
        return 0 if art["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
