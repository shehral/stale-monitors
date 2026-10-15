#!/bin/bash
# Smoke gate, CPU half (partition short): pytest on the tiny random model and every CLI end to end with --smoke,
# fully offline (SM_OFFLINE=1, SM_DEVICE=cpu). No GPU, no model download, no judge.
# Why a separate job: the single smoke on gpu-short ran pytest first on the GPU node's CPUs; at commit 0239981
# (chain 3, job 10899610) pytest ran 55+ minutes with the GPU idle and hit the 1 h cap (IdleBot would also cancel
# a GPU job idle for an hour). The GPU half, jobs/smoke.sbatch, now runs only the GPU checks and merges this
# half's artifact (same git commit) into results/smoke/<base>/<gpu_jobid>.json.
#   0. guards, when BASE or BANK is set: the stage config's revision equals the configs/bases.yaml pin
#      (sm_check_config_rev) and the code is frozen at a clean git commit (sm_require_frozen); exit 2 otherwise.
#   1. pytest -m "not slow" (the slow tests spawn fresh interpreters that re-import torch from the NFS env; the
#      laptop half, python -m sm.ops.smoke_artifact --local, runs them), with pytest-xdist -n $SLURM_CPUS_PER_TASK
#      when the env has it, else serial. Every temp directory is node-local (sm_node_tmp:
#      ${SLURM_TMPDIR:-/tmp}/sm-pytest-<jobid>; pytest --basetemp=<root>/pytest, TMPDIR=<root>/tmp, the CLI
#      outputs under <root>/cli); the /scratch fallback is used, and logged, only when that cannot be created.
#   2. the eleven smoke CLIs (sm.ops.smoke_artifact.SMOKE_CLIS), each `python -m <mod> --smoke --out ... --force`.
#   3. the CPU half artifact results/smoke/<base>/cpu/<jobid>.json (jobs/common.sh sm_smoke_cpu_half), with the
#      checks, git commit and code tree, pin, pytest summary, worker count, -m expression and temp root.
# Exit 0 only when every part passes (jobs/submit_bank.sh chains smoke.sbatch on this job with afterok).
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/smoke_cpu.sbatch
#     or: sbatch --export=ALL,BASE=<registry key> jobs/smoke_cpu.sbatch   (or neither: _nobase/cpu/)
# Optional env: SKIP_PYTEST=1, SKIP_CLI=1 (the half then does not pass), SM_PYTEST_MARKEXPR (default "not slow";
# empty runs everything), ALLOW_UNFROZEN_CODE=1 (logged), SM_PROJ.
#SBATCH --job-name=sm-smoke-cpu
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/smoke_cpu-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/smoke_cpu-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
sm_log "=== sm-smoke-cpu start on $(hostname) ==="
sm_setup_cpu_env 0
[ -x "$PYTHON" ] || sm_die 2 "env python not found at $PYTHON"

# --- 0: pinned revision and frozen code (as the measured stages) ----------------
BASE_KEY=""
PIN=""
if [ -n "${BASE:-}" ] || { [ -n "${BANK:-}" ] && [ -f "$BANK" ]; }; then
  sm_resolve_base_key
  CARD_CFG="${BASE_CARD_CONFIG:-configs/base_card_${BASE_KEY}.yaml}"
  [ -f "$CARD_CFG" ] || sm_die 2 "$CARD_CFG not found"
  PIN="$(sm_py config-rev "$CARD_CFG" "$BASE_KEY")" || sm_die 2 "the stage config revision for $BASE_KEY is missing or differs from configs/bases.yaml"
  sm_log "revision: $CARD_CFG pins $PIN (matches configs/bases.yaml)"
  sm_require_frozen "$BASE_KEY"
else
  sm_log "no BASE or BANK: the CPU half goes to results/smoke/_nobase/cpu/ and covers no base"
fi

# Small bookkeeping files (logs, checks) stay on scratch; everything with many small files is node-local.
SMOKE_OUT="$SM_SCRATCH/smoke/cpu-${SLURM_JOB_ID:-local}"
mkdir -p "$SMOKE_OUT"
sm_node_tmp sm-pytest
export TMPDIR="$SM_NODE_TMP/tmp"
mkdir -p "$TMPDIR" "$SM_NODE_TMP/cli"
PYTEST_BASETEMP="$SM_NODE_TMP/pytest"
SM_PYTEST_MARKEXPR="${SM_PYTEST_MARKEXPR-not slow}"
SM_PYTEST_WORKERS=1
declare -a XDIST=()
if "$PYTHON" -c "import xdist" >/dev/null 2>&1; then
  SM_PYTEST_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
  XDIST=(-n "$SM_PYTEST_WORKERS")
fi
export SM_PYTEST_MARKEXPR SM_PYTEST_WORKERS
declare -a MARK=()
[ -n "$SM_PYTEST_MARKEXPR" ] && MARK=(-m "$SM_PYTEST_MARKEXPR")
sm_log "pytest: workers=$SM_PYTEST_WORKERS ($([ "${#XDIST[@]}" -gt 0 ] && echo xdist || echo serial)) markexpr='${SM_PYTEST_MARKEXPR}' basetemp=$PYTEST_BASETEMP TMPDIR=$TMPDIR"

declare -a RESULTS=()
fails=0
: >"$SMOKE_OUT/checks.tsv"
: >"$SMOKE_OUT/skipped.txt"
record() {  # record <name> <rc>; also <name>\t<rc> into checks.tsv for the smoke artifact
  if [ "$2" -eq 0 ]; then RESULTS+=("PASS  $1"); else RESULTS+=("FAIL  $1 (rc=$2)"); fails=$((fails + 1)); fi
  printf '%s\t%s\n' "$1" "$2" >>"$SMOKE_OUT/checks.tsv"
}
skip() {  # skip <what>
  RESULTS+=("SKIP  $1")
  printf '%s\n' "$1" >>"$SMOKE_OUT/skipped.txt"
}

# --- 1 + 2: tiny model, fully offline -----------------------------------------
(
  export SM_OFFLINE=1 SM_DEVICE=cpu
  unset SM_TINY_MODEL
  rc=0
  if [ -z "${SKIP_PYTEST:-}" ]; then
    t0=$(date +%s)
    rc=0; "$PYTHON" -m pytest -q -p no:cacheprovider "${XDIST[@]}" "${MARK[@]}" --basetemp="$PYTEST_BASETEMP" \
      --durations=25 tests || rc=$?
    echo "pytest wall: $(( $(date +%s) - t0 )) s"
    echo "SMOKE_RESULT pytest $rc"
  fi
  if [ -z "${SKIP_CLI:-}" ]; then
    for mod in sm.data.build_prompt_sets sm.data.build_bank_datasets sm.train.lora_sft sm.eval.battery \
               sm.eval.judge sm.eval.capability sm.interp.base_card sm.interp.causal sm.interp.drift \
               sm.interp.monitor_scores sm.analysis.aggregate; do
      out="$SM_NODE_TMP/cli/${mod##*.}"
      rc=0; timeout 600 "$PYTHON" -m "$mod" --smoke --out "$out" --force || rc=$?
      echo "SMOKE_RESULT $mod $rc"
    done
  fi
) 2>&1 | tee "$SMOKE_OUT/offline.log"
while read -r _tag name rc; do record "$name" "$rc"; done < <(grep '^SMOKE_RESULT ' "$SMOKE_OUT/offline.log" || true)
[ -n "${SKIP_PYTEST:-}" ] && skip "pytest (SKIP_PYTEST=1)"
[ -n "${SKIP_CLI:-}" ] && skip "smoke CLIs (SKIP_CLI=1)"

# --- 3: the CPU half artifact ----------------------------------------------------------------------
rc=0; sm_smoke_cpu_half "$SMOKE_OUT" "$BASE_KEY" "$PIN" || rc=$?
if [ "$rc" -ne 0 ]; then RESULTS+=("FAIL  smoke CPU half results/smoke (rc=$rc)"); fails=$((fails + 1)); fi

# The node-local root is this job's own directory (never the scratch fallback, which the purge handles).
if [ "$SM_NODE_TMP_KIND" = "node-local" ]; then
  case "$SM_NODE_TMP" in */sm-pytest-"${SLURM_JOB_ID:-$$}"-??????) rm -rf -- "$SM_NODE_TMP" || true ;; esac
fi

sm_log "=== smoke CPU half summary ($SMOKE_OUT; tmp $SM_NODE_TMP_KIND) ==="
printf '  %s\n' "${RESULTS[@]}"
sm_log "=== sm-smoke-cpu done: failures=$fails ==="
[ "$fails" -eq 0 ] || exit 1
exit 0
