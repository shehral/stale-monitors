#!/bin/bash
# EXPLORATORY R1 decision probe, CPU fit (docs/PROGRAM_2026-10-13.md line R1; for the Qwen3.8-27B slice see
# docs/HYPOTHESIS_LEDGER.md "R1 on Qwen3.8-27B (2026-10-20)"): python -m sm.exploratory.decision_probe_fit --base <BASE>
# on the short partition (CPU only, never inside a GPU allocation; the code is the 9B fit, unchanged). Reads
# results/exploratory/<base>/decision_probe/acts/*.npz, the text-free item-label table
# results/exploratory/<base>/cpu/item_labels.jsonl, the XSTest-safe labels under results/judge/<base>/<unit>/labels and the
# subset bank table results/tables/subsets/<base>/bank_<base>.csv; writes results/exploratory/<base>/decision_probe/
# {probe_results.json, directions.npz}. Numbers only; no text is read or written. Runs from the side tree (its results,
# data and logs are symlinks into the main tree). Hand-submitted: ledger with sm_ledger_add <jobid> after it ends.
#
# Env: BASE (required), EXTRA (extra CLI flags, set in the submitting shell), SM_PROJ, SM_CODE_COMMIT
# Submit: sbatch --export=ALL,BASE=qwen3.8-27b,SM_PROJ=/projects/interpretability/shehral.m/monitors-diag3 jobs/decision_probe_fit.sbatch
# Exit codes: 2 setup (env, arguments), 5 inputs missing.
#SBATCH --job-name=sm-dprobe-fit
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/dprobe-fit-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/dprobe-fit-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag3}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
[ -n "${BASE:-}" ] || sm_die 2 "BASE is required"
sm_log "=== sm-dprobe-fit start on $(hostname) base=$BASE code=${SM_CODE_COMMIT:-unset} ==="
sm_setup_cpu_env 0
[ -x "$PYTHON" ] || sm_die 2 "env python not found at $PYTHON"
SAFE="$(sm_py safe-name "$BASE")"
for f in "results/exploratory/$SAFE/decision_probe/acts/base.npz" "results/exploratory/$SAFE/cpu/item_labels.jsonl" \
  "results/tables/subsets/$SAFE/bank_$SAFE.csv"; do
  [ -f "$f" ] || sm_die 5 "input $f missing"
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# EXTRA is split on whitespace on purpose (a list of CLI flags).
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.exploratory.decision_probe_fit --base "$BASE" ${EXTRA:-} || rc=$?
[ "$rc" -eq 0 ] || sm_die "$rc" "sm.exploratory.decision_probe_fit failed"
sm_log "=== sm-dprobe-fit done base=$BASE ==="
exit 0
