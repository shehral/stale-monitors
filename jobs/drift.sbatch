#!/bin/bash
# White-box drift, one array task per bank run (sm.interp.drift): D1-D3, monitor_auroc per monitor,
# realigned recall at the base 1%-FPR threshold, persona shifts, adapter projection.
# Each task writes a per-run config results/drift/<base>/_configs/<run_id>.yaml
# from the template DRIFT_TEMPLATE (default configs/drift_<base>.yaml: prompts,
# persona prompts, batch size, the pinned revision) plus the run's derivative block:
#   kind lora     -> {kind: lora, adapter_path: results/adapters/<base>/<run_id>} (applied unmerged)
#   kind base     -> {kind: base}
#   kind model    -> {kind: model, model: <run.model>, revision: <run.model_revision>}
#   kind ablation -> {kind: ablation, direction: <run.direction> or the card's r}
# A run's own `derivative:` and `drift:` mappings in the manifest override these.
# Needs a passed V4 file (exit 2), the base card (exit 5 when absent) and, for lora runs, a finished adapter.
# Idempotent: skips when results/drift/<base>/<run_id>.json exists (FORCE=1 recomputes).
#
# Submit: sbatch --array=<indices>%4 --export=ALL,BANK=configs/bank_<base>.yaml jobs/drift.sbatch
# Optional env: DRIFT_TEMPLATE, FORCE=1, ALLOW_FAILED_V4=1 (descriptive-only), SM_PROJ,
#   ALL_CELLS=1 (exploratory E6: also write results/drift/<base>/<run_id>_cells.json, per-cell staleness over the
#   base card's (position, layer) grid; an existing <run_id>.json is kept and only the cells are back-filled).
#SBATCH --job-name=sm-drift
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/drift-%A_%a.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/drift-%A_%a.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-drift start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_load_task drift
if [ "$TASK_ACTIVE" != "1" ]; then
  sm_log "SKIP: run $TASK_RUN_ID (kind $TASK_KIND) is inactive for the drift stage"
  exit 0
fi

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/drift
OUT_JSON="$OUT/$SAFE/$TASK_RUN_ID.json"
OUT_CELLS="$OUT/$SAFE/${TASK_RUN_ID}_cells.json"
CELLS_FLAG=""
if [ "${ALL_CELLS:-0}" = "1" ]; then
  CELLS_FLAG="--all-cells"
fi
if [ -f "$OUT_JSON" ] && [ -z "${FORCE:-}" ]; then
  if [ -z "$CELLS_FLAG" ] || [ -f "$OUT_CELLS" ]; then
    sm_log "already done: $OUT_JSON"
    exit 0
  fi
  sm_log "ALL_CELLS=1: $OUT_JSON exists and is kept; back-filling $OUT_CELLS (capture-only pass)"
fi

sm_require_v4 "$BASE_KEY"
CARD="results/base_cards/$SAFE.json"
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing (run jobs/base_card.sbatch first)"
TEMPLATE="${DRIFT_TEMPLATE:-configs/drift_${BASE_KEY}.yaml}"
[ -f "$TEMPLATE" ] || sm_die 2 "drift template $TEMPLATE not found"
sm_check_config_rev "$TEMPLATE" "$BASE_KEY"

case "$TASK_KIND" in
  lora)
    ADAPTER="results/adapters/$BASE_KEY/$TASK_RUN_ID"
    sm_py train-ok "$ADAPTER/train_meta.json" || sm_die 5 "no finished adapter at $ADAPTER"
    ;;
  model)
    [ -n "$TASK_MODEL" ] || sm_die 2 "run $TASK_RUN_ID: kind model needs 'model' in the manifest"
    [ -n "$TASK_MODEL_REVISION" ] || [ "${ALLOW_UNPINNED:-0}" = "1" ] || sm_die 2 "run $TASK_RUN_ID: external model $TASK_MODEL has no model_revision (external_revision) pin"
    sm_preflight_snapshot "$TASK_MODEL" "$TASK_MODEL_REVISION"
    ;;
  base|ablation) ;;
  *) sm_die 2 "unexpected kind $TASK_KIND" ;;
esac

CFG="$OUT/$SAFE/_configs/$TASK_RUN_ID.yaml"
sm_py stage-config drift "$BANK" "$TASK_INDEX" "$CFG" "$TEMPLATE" >/dev/null || sm_die 2 "could not build $CFG"
sm_log "drift config: $CFG"

sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"
rc=0; sm_run "$PYTHON" -m sm.interp.drift --config "$CFG" --out "$OUT" ${FORCE:+--force} $CELLS_FLAG || rc=$?
sm_log "=== sm-drift done rc=$rc run=$TASK_RUN_ID ==="
exit "$rc"
