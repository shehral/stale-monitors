#!/bin/bash
# EXPLORATORY R1 decision probe, GPU capture (docs/PROGRAM_2026-10-13.md line R1, registered before any code):
# sm.exploratory.decision_probe on one unit of one base (default qwen3.5-9b). Renders P (the three H batteries, XSTest
# safe and the 128 test-harmless contrast prompts) as the battery stage does, loads the unit as the drift stage does,
# and saves fp16 residuals at the last prompt token and the first generated position for hidden states
# 13,16,20,24,28,31,32 to results/exploratory/<base>/decision_probe/acts/<unit>.npz (+ <unit>.json: stamps, hashes,
# checks). No text is written. Not a measured stage, so no frozen-code gate; outputs record the git commit,
# SM_CODE_COMMIT, the code file hashes, the adapter SHA-256 and a config hash. No judge server (labels come from the
# existing judge outputs; the fit is CPU-only: python -m sm.exploratory.decision_probe_fit on the short partition).
# Runs from the side tree (SM_PROJ default /projects/interpretability/shehral.m/monitors-diag; its results, data and
# logs are symlinks into the main tree).
#
# Env:
#   UNIT     "base" or a run id of results/drift/<base>/_configs (required)
#   EXTRA    extra CLI arguments, e.g. "--batch-size 8" (exported variable: --export splits its list on commas)
#   BASE (default qwen3.5-9b), FORCE=1 (recompute), SM_PROJ
# Submit (one H200; about 10 min wall per unit, model load included):
#   sbatch --export=ALL,UNIT=base,SM_CODE_COMMIT=<sha> jobs/decision_probe.sbatch
# Hand-submitted: ledger the GPU job with sm_ledger_add <jobid> after it ends (the EXIT trap prints the LEDGER line).
# Exit codes: 2 setup (env, GPU, compat libcuda, snapshot, arguments), 5 inputs missing (prompt files).
#SBATCH --job-name=sm-dprobe
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/dprobe-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/dprobe-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
[ -n "${UNIT:-}" ] || sm_die 2 "UNIT is required (base or a run id)"
sm_log "=== sm-dprobe start on $(hostname) unit=$UNIT code=${SM_CODE_COMMIT:-unset} ==="
sm_setup_env
sm_gpu_preflight
# NVIDIA forward-compat libcuda (cuda-compat, exported by jobs/common.sh at source time): fail before any model load
# if it is not first on the library path (the nodes run driver 570.86 and the env torch is a cu130 build).
if [ "$SM_USE_COMPAT" = "1" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in ":$SM_COMPAT_DIR:"*) ;; *) sm_die 2 "SM_COMPAT_DIR=$SM_COMPAT_DIR is not first on LD_LIBRARY_PATH" ;; esac
fi

BASE_KEY="${BASE:-qwen3.5-9b}"
SAFE="$(sm_py safe-name "$BASE_KEY")"
for f in data/prompts/batteries/strongreject.jsonl data/prompts/batteries/harmbench.jsonl \
  data/prompts/batteries/xstest_unsafe.jsonl data/prompts/batteries/xstest_safe.jsonl \
  data/prompts/contrast/test_harmless.jsonl; do
  [ -f "$f" ] || sm_die 5 "prompt file $f missing"
done
if [ "$UNIT" != "base" ]; then
  [ -f "results/drift/$SAFE/_configs/$UNIT.yaml" ] || sm_die 5 "run config results/drift/$SAFE/_configs/$UNIT.yaml missing"
fi
sm_preflight_snapshot "$BASE_KEY"

# EXTRA is split on whitespace on purpose (a list of CLI flags).
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.exploratory.decision_probe --base "$BASE_KEY" --unit "$UNIT" ${FORCE:+--force} ${EXTRA:-} || rc=$?
[ "$rc" -eq 0 ] || sm_die "$rc" "sm.exploratory.decision_probe failed"
sm_log "=== sm-dprobe done unit=$UNIT ==="
exit 0
