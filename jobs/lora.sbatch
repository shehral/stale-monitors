#!/bin/bash
# LoRA SFT, one array task per bank run (sm.train.lora_sft). Non-training runs
# (kind base | model | ablation) exit 0 at once; jobs/submit_bank.sh only puts
# training indices in --array. Idempotent: skips a run whose train_meta.json
# says status ok (set FORCE=1 to retrain).
#
# Submit (jobs/submit_bank.sh does this for you):
#   sbatch --array=<training indices>%4 --export=ALL,BANK=configs/bank_<base>.yaml jobs/lora.sbatch
#   (indices: source jobs/common.sh; sm_py indices "$BANK" lora)
#   Single run without an array: --export=ALL,BANK=...,TASK_INDEX=<manifest index>
# Guards: GPU pre-flight (exit 2); a passed V4 file for the base (exit 2, jobs/common.sh sm_require_v4);
#   the bank manifest's revision equals the configs/bases.yaml pin (lora_sft loads the base with it).
# Optional env: FORCE=1, LORA_EXTRA_ARGS="--max-steps 50", ALLOW_FAILED_V4=1 (descriptive-only), SM_PROJ.
#SBATCH --job-name=sm-lora
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/lora-%A_%a.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/lora-%A_%a.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-lora start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_load_task lora

if [ "$TASK_ACTIVE" != "1" ]; then
  sm_log "SKIP: run $TASK_RUN_ID (kind $TASK_KIND) is not a training run"
  exit 0
fi

OUT=results/adapters
META="$OUT/$BASE_KEY/$TASK_RUN_ID/train_meta.json"
if [ -z "${FORCE:-}" ] && sm_py train-ok "$META"; then
  sm_log "already done: $META"
  exit 0
fi

sm_require_v4 "$BASE_KEY"
sm_check_config_rev "$BANK" "$BASE_KEY"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.train.lora_sft --config "$BANK" --run-id "$TASK_RUN_ID" --out "$OUT" ${FORCE:+--force} ${LORA_EXTRA_ARGS:-} || rc=$?
sm_log "=== sm-lora done rc=$rc run=$TASK_RUN_ID ==="
exit "$rc"
