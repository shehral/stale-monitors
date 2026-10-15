"""submit_bank.sh harness fixes (2026-10-08): QOS-sized array chunks, de-duplicated sbatch retries, and the
submit-time vLLM refusal. sbatch and squeue are fakes on PATH; nothing touches SLURM, a GPU or the network.

Run: .venv/bin/python -m pytest -q tests/test_submit_bank_qos.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
PIN = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
BANK = "configs/bank_qwen35_9b.yaml"

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="needs bash")

FAKE_SBATCH = r"""#!/bin/bash
# fake sbatch: logs its arguments, hands out increasing ids, optionally records array tasks as queued.
st="$FAKE_STATE"
echo "$*" >> "$st/calls"
if [ -f "$st/fail_accept_once" ]; then          # accepted by SLURM, but the client saw an error
  name="$(cat "$st/fail_accept_once")"; rm -f "$st/fail_accept_once"
  id=$(( $(cat "$st/next") )); echo $(( id + 1 )) > "$st/next"
  mkdir -p "$st/names"; echo "$id" >> "$st/names/$name"
  echo "sbatch: error: Socket timed out on send/recv operation" >&2; exit 1
fi
if [ -f "$st/fail_reject_once" ]; then rm -f "$st/fail_reject_once"; echo "sbatch: error: transient" >&2; exit 1; fi
id=$(( $(cat "$st/next") )); echo $(( id + 1 )) > "$st/next"
if [ -f "$st/record" ]; then
  arr="$(printf '%s\n' "$@" | sed -n 's/^--array=\([^%]*\).*/\1/p')"
  n=1; [ -n "$arr" ] && n=$(echo "$arr" | tr ',' '\n' | grep -c .)
  for _ in $(seq "$n"); do echo "$id" >> "$st/queued"; done
fi
echo "$id"
"""

FAKE_SQUEUE = r"""#!/bin/bash
# fake squeue: -p -> the queued file (one line per task); -n NAME -> ids recorded under that name; else nothing.
st="$FAKE_STATE"
prev=""
for a in "$@"; do
  if [ "$prev" = "-p" ]; then cat "$st/queued" 2>/dev/null; exit 0; fi
  if [ "$prev" = "-n" ]; then cat "$st/names/$a" 2>/dev/null; exit 0; fi
  prev="$a"
done
exit 0
"""


def _setup(tmp_path: Path, queued: int = 0, record: bool = False) -> dict:
    hf_home = tmp_path / "hf"
    snap = hf_home / "hub" / ("models--" + PIN[0].replace("/", "--")) / "snapshots" / PIN[1]
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_text("x")
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"soft_cap_h200_hours_per_day": 24.0, "by_day": {}}))
    streak = tmp_path / "streak.json"
    streak.write_text(json.dumps({"auto_rollback_triggered": False}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    (state / "next").write_text("1000\n")
    (state / "queued").write_text("".join(f"{900 + i}\n" for i in range(queued)))
    if record:
        (state / "record").write_text("1")
    env = {k: v for k, v in os.environ.items() if k not in ("SM_SBATCH", "SM_SQUEUE", "BATTERY_BACKEND", "BATTERY_EXTRA_ARGS",
                                                          "SM_QOS_SUBMIT_LIMIT", "SM_QOS_PARTITIONS")}
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_STATE": str(state), "SM_PROJ": str(ROOT),
        "PYTHON": sys.executable, "SM_HF_HOME": str(hf_home), "SM_LOGDIR": str(tmp_path / "logs"),
        "SM_BUDGET_LEDGER": str(ledger), "SM_STREAK_LEDGER": str(streak), "USER": "tester",
        "SM_SBATCH_RETRY_SLEEP": "0",
    })
    return env


def _run(tmp_path: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(JOBS / "submit_bank.sh"), "--bank", BANK, *args],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)


def _calls(env: dict) -> list[str]:
    f = Path(env["FAKE_STATE"]) / "calls"
    return f.read_text().splitlines() if f.exists() else []


def _arr(line: str) -> list[str]:
    m = re.search(r"--array=([^%\s]+)", line)
    return m.group(1).split(",") if m else []


def test_dry_run_slices_gpu_arrays_to_the_submit_limit(tmp_path):
    env = _setup(tmp_path)
    r = _run(tmp_path, env, "--dry-run")
    assert r.returncode == 0, r.stderr + r.stdout
    lines = [x for x in r.stdout.splitlines() if x.startswith("DRY: ")]
    ids = {i: 900001 + i for i in range(len(lines))}
    by_script: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        by_script.setdefault(re.search(r"jobs/(\S+)$", line).group(1), []).append(i)
    lora = by_script["lora.sbatch"]
    assert len(lora) == 2, r.stdout                                    # 14 tasks (13 + the E8 control) -> 8 + 6
    assert [len(_arr(lines[i])) for i in lora] == [8, 6]
    assert all(lines[i].rstrip().split("jobs/")[0].count("%4") == 1 for i in lora)
    # chunk 2 waits on chunk 1; the organism stage waits on every lora chunk
    assert f"afterany:{ids[lora[0]]}" in lines[lora[1]]
    organism = next(i for i in by_script["base_card.sbatch"] if "ORGANISM=1" in lines[i])
    assert f"afterany:{ids[lora[0]]}:{ids[lora[1]]}," in lines[organism]
    # every array on the gpu partitions has at most 8 tasks; capability (short partition) is not cut
    for s in ("lora.sbatch", "battery.sbatch", "drift.sbatch", "monitor_scores.sbatch"):
        assert all(len(_arr(lines[i])) <= 8 for i in by_script[s]), s
    assert len(by_script["capability.sbatch"]) == 1 and len(_arr(lines[by_script["capability.sbatch"][0]])) == 17
    # the union of the chunks is the stage's index list, in order
    orig = re.search(r"estimate drift\s+(\d+) x", r.stdout).group(1)
    drift_idx = sum((_arr(lines[i]) for i in by_script["drift.sbatch"]), [])
    assert len(drift_idx) == int(orig) and len(set(drift_idx)) == len(drift_idx)
    # the next stage after the last drift chunk lists every drift chunk id
    ms_first = by_script["monitor_scores.sbatch"][0]
    assert "afterany:" + ":".join(str(ids[i]) for i in by_script["drift.sbatch"]) + "," in lines[ms_first]
    # the summary and the ledger line carry every chunk id
    assert re.search(r"lora_c1=\d+ lora_c2=\d+", r.stdout)
    ledger = re.search(r"sm_ledger_add (\S+)", r.stdout).group(1).split(",")
    assert len(ledger) == len(lines)


def test_live_chunks_chain_and_downstream_waits_on_all(tmp_path):
    env = _setup(tmp_path, queued=0, record=False)
    r = _run(tmp_path, env, "--stages", "lora,organism", "--no-smoke", "--no-auto-chain")
    assert r.returncode == 0, r.stderr + r.stdout
    calls = _calls(env)
    assert len(calls) == 3 and "lora.sbatch" in calls[0] and "lora.sbatch" in calls[1]
    assert [len(_arr(c)) for c in calls[:2]] == [8, 6]                 # 14 lora tasks (13 + the E8 control)
    assert "--dependency" not in calls[0] and "--dependency=afterany:1000 " in calls[1]
    assert "--dependency=afterany:1000:1001 " in calls[2] and "ORGANISM=1" in calls[2]
    assert "submitted: lora_c1=1000 lora_c2=1001 organism=1002" in r.stdout


def test_live_chunk_size_uses_the_queue_and_stops_when_full(tmp_path):
    env = _setup(tmp_path, queued=3, record=True)
    r = _run(tmp_path, env, "--stages", "lora,organism", "--no-smoke", "--no-auto-chain")
    assert r.returncode == 4, r.stderr + r.stdout
    calls = _calls(env)
    assert len(calls) == 1 and len(_arr(calls[0])) == 5               # 8 - 3 queued
    assert "QOS STOP" in r.stderr and "were NOT submitted" in r.stderr
    assert "ABORTED (rc=4) after queueing: lora_c1=1000" in r.stderr
    # a smaller limit from the environment is honoured
    env2 = _setup(tmp_path / "b", queued=0, record=False)
    env2["SM_QOS_SUBMIT_LIMIT"] = "4"
    r2 = _run(tmp_path / "b", env2, "--stages", "lora", "--no-smoke", "--no-auto-chain")
    assert r2.returncode == 0, r2.stderr
    assert [len(_arr(c)) for c in _calls(env2)] == [4, 4, 4, 2]


def test_retry_reuses_a_job_slurm_accepted_despite_the_error(tmp_path):
    env = _setup(tmp_path)
    state = Path(env["FAKE_STATE"])
    (state / "names").mkdir()
    (state / "names" / "sm-lora").write_text("777\n")                 # an older sm-lora job: never taken
    (state / "fail_accept_once").write_text("sm-lora")
    env["SM_QOS_SUBMIT_LIMIT"] = "100"
    r = _run(tmp_path, env, "--stages", "lora", "--no-smoke", "--no-auto-chain")
    assert r.returncode == 0, r.stderr + r.stdout
    assert len(_calls(env)) == 1                                       # no second submission
    assert "SLURM accepted job 1000 (sm-lora); reusing it" in r.stderr
    assert "submitted: lora=1000" in r.stdout


def test_retry_resubmits_when_nothing_was_accepted(tmp_path):
    env = _setup(tmp_path)
    (Path(env["FAKE_STATE"]) / "fail_reject_once").write_text("1")
    env["SM_QOS_SUBMIT_LIMIT"] = "100"
    r = _run(tmp_path, env, "--stages", "lora", "--no-smoke", "--no-auto-chain")
    assert r.returncode == 0, r.stderr + r.stdout
    assert len(_calls(env)) == 2 and "reusing" not in r.stderr
    assert "submitted: lora=1000" in r.stdout


def test_vllm_battery_backend_is_refused_at_submit_time(tmp_path):
    env = _setup(tmp_path)
    env["BATTERY_BACKEND"] = "vllm"
    r = _run(tmp_path, env, "--dry-run", "--stages", "battery", "--no-smoke")
    assert r.returncode == 2 and "REFUSED: BATTERY_BACKEND=vllm" in r.stderr, r.stderr
    assert "DRY:" not in r.stdout
    env["BATTERY_EXTRA_ARGS"] = "--allow-vllm-measured"
    r = _run(tmp_path, env, "--dry-run", "--stages", "battery", "--no-smoke")
    assert r.returncode == 0 and "DRY:" in r.stdout, r.stderr
    env.pop("BATTERY_EXTRA_ARGS")
    r = _run(tmp_path, env, "--dry-run", "--stages", "lora", "--no-smoke")     # no battery stage queued
    assert r.returncode == 0, r.stderr
