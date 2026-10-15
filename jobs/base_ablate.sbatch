#!/bin/bash
# Base card, phase 3 of 3 (sm.interp.base_card --phase ablate): GPU with the J1 server. From the capture
# checkpoint (jobs/base_card.sbatch) and the stats checkpoint (jobs/base_stats.sbatch) under
# results/base_cards/<base>/_partial/, it runs the judged L* rule (ablation sweep on dev harmful prompts,
# coherence filter), writes the per-layer directions and probes, the refusal subspace, the persona vectors
# (generation + J1 trait scores), the frozen H2 declaration (declared_best_drift_score: D2), the monitor
# operating points (monitor_thresholds) and results/base_cards/<base>.json. The card is the one the
# single-process run (--phase all) writes from the same activations. The V4 causal check is the next stage
# (jobs/causal.sbatch).
# Acceptance: this is the job whose id goes to `python -m sm.accept --stage base_card --base <base> --job <id>`;
# its log is logs/base-card-ablate-<id>.out (accept finds it under the base-card prefix by the LEDGER job id)
# and carries the `frozen:` line and the LEDGER line.
# Idempotent: exits 0 "already done" when results/base_cards/<base>.json exists (FORCE=1 recomputes).
# Inputs are checked before the J1 server starts (about 11 min the first time) or the base loads:
# data/prompts/MANIFEST.json and both checkpoints must exist (exit 5); the module also exits 5 when the
# checkpoints do not match the configured prompt sets, pin, stats config or seed.
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/base_ablate.sbatch
#     or: sbatch --export=ALL,BASE=<registry key> jobs/base_ablate.sbatch
# Config: BASE_CARD_CONFIG (default configs/base_card_<base>.yaml, required; its revision must equal the
#         configs/bases.yaml pin).
# Judge server: J1 (vLLM, port J1_PORT, default 8000) at the configs/judge.yaml pin, on the same GPU,
#         started before the base loads (the judged L* rule and the persona trait scores call it).
#         START_J1=0 only for a config with no server judge. J1_GPU_UTIL (default 0.25).
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, pinned revision, frozen code (jobs/common.sh).
# Optional env: SEED (default 0, must match the other two phases), FORCE=1, SM_PROJ.
#SBATCH --job-name=sm-base-ablate
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/base-card-ablate-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/base-card-ablate-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-base-ablate start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/base_cards
CARD="$OUT/$SAFE.json"
CFG="${BASE_CARD_CONFIG:-configs/base_card_${BASE_KEY}.yaml}"
[ -f "$CFG" ] || sm_die 2 "base card config $CFG not found"
if [ -f "$CARD" ] && [ -z "${FORCE:-}" ]; then
  sm_log "already done: $CARD"
  exit 0
fi
sm_require_data data/prompts/MANIFEST.json
[ -f "$OUT/$SAFE/_partial/capture/manifest.json" ] || sm_die 5 "capture checkpoint $OUT/$SAFE/_partial/capture/manifest.json missing (run jobs/base_card.sbatch first)"
[ -f "$OUT/$SAFE/_partial/stats/stats.json" ] || sm_die 5 "stats checkpoint $OUT/$SAFE/_partial/stats/stats.json missing (run jobs/base_stats.sbatch first)"
sm_check_config_rev "$CFG" "$BASE_KEY"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

if [ "${START_J1:-1}" = "1" ]; then
  sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}"
fi

rc=0
sm_run "$PYTHON" -m sm.interp.base_card --config "$CFG" --base "$BASE_KEY" --out "$OUT" --seed "${SEED:-0}" \
  --phase ablate ${FORCE:+--force} || rc=$?
if [ "$rc" -eq 0 ]; then
  [ -f "$OUT/$SAFE/refusal.npy" ] || { sm_log "base card wrote no refusal direction at $OUT/$SAFE/refusal.npy"; rc=5; }
fi
sm_log "=== sm-base-ablate done rc=$rc base=$BASE_KEY ==="
exit "$rc"
