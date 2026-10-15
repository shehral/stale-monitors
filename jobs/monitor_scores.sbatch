#!/bin/bash
# Per-item monitor scores, one array task per run_id (sm.interp.monitor_scores): every monitor of the base
# card (refusal_direction, refusal_subspace, probe, persona_evil, persona_sycophancy, persona_hallucination)
# scored at the last prompt token on every item of the harmful, over-refusal and karne_ladders batteries,
# with the base card's 1%-FPR thresholds. Output results/monitor_scores/<base>/<run_id>/<battery>.jsonl
# (one row per item and monitor), the H5 input together with the J1 labels.
# Index map: 0 = the base (H5 needs the base gap), manifest run i at i+1 (`sm_py runs <bank> monitor_scores`);
# ablation runs are inactive (no battery outputs, so no compliance labels).
# Each task writes a per-run config results/monitor_scores/<base>/_configs/<run_id>.yaml from
# MONITOR_SCORES_TEMPLATE (default configs/monitor_scores_<base>.yaml, pinned revision checked) plus the
# derivative block (kind lora: the adapter, applied unmerged; kind model: the external model and its revision).
# Guards: GPU pre-flight (exit 2), a passed V4 file (exit 2), the base card (exit 5), a finished adapter (exit 5).
# Idempotent: the module skips finished battery files (FORCE=1 recomputes).
#
# Submit: sbatch --array=<indices>%4 --export=ALL,BANK=configs/bank_<base>.yaml jobs/monitor_scores.sbatch
# Optional env: MONITOR_SCORES_TEMPLATE, FORCE=1, ALLOW_FAILED_V4=1 (descriptive-only), SM_PROJ.
#SBATCH --job-name=sm-monitor-scores
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/monitor-scores-%A_%a.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/monitor-scores-%A_%a.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-monitor-scores start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_load_task monitor_scores
if [ "$TASK_ACTIVE" != "1" ]; then
  sm_log "SKIP: run $TASK_RUN_ID (kind $TASK_KIND) is inactive for the monitor_scores stage"
  exit 0
fi
sm_require_v4 "$BASE_KEY"

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/monitor_scores
CARD="results/base_cards/$SAFE.json"
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing (run jobs/base_card.sbatch first)"
TEMPLATE="${MONITOR_SCORES_TEMPLATE:-configs/monitor_scores_${BASE_KEY}.yaml}"
[ -f "$TEMPLATE" ] || sm_die 2 "monitor_scores template $TEMPLATE not found"
sm_check_config_rev "$TEMPLATE" "$BASE_KEY"
[ -f "${ITEMS_DIR:-data/prompts}/MANIFEST.json" ] || sm_die 2 "battery items missing: ${ITEMS_DIR:-data/prompts}/MANIFEST.json"

case "$TASK_KIND" in
  lora)
    ADAPTER="results/adapters/$BASE_KEY/$TASK_RUN_ID"
    sm_py train-ok "$ADAPTER/train_meta.json" || sm_die 5 "no finished adapter at $ADAPTER"
    ;;
  model)
    [ -n "$TASK_MODEL" ] || sm_die 2 "run $TASK_RUN_ID: kind model needs 'model' in the manifest"
    [ -n "$TASK_MODEL_REVISION" ] || [ "${ALLOW_UNPINNED:-0}" = "1" ] || sm_die 2 "run $TASK_RUN_ID: external model $TASK_MODEL has no model_revision pin"
    sm_preflight_snapshot "$TASK_MODEL" "$TASK_MODEL_REVISION"
    ;;
  base) ;;
  *) sm_die 2 "unexpected kind $TASK_KIND for the monitor_scores stage" ;;
esac

CFG="$OUT/$SAFE/_configs/$TASK_RUN_ID.yaml"
sm_py stage-config monitor_scores "$BANK" "$TASK_INDEX" "$CFG" "$TEMPLATE" >/dev/null || sm_die 2 "could not build $CFG"
sm_log "monitor_scores config: $CFG"

sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"
rc=0; sm_run "$PYTHON" -m sm.interp.monitor_scores --config "$CFG" --out "$OUT" ${FORCE:+--force} || rc=$?
sm_log "=== sm-monitor-scores done rc=$rc run=$TASK_RUN_ID ==="
exit "$rc"
