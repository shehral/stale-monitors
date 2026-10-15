#!/bin/bash
# V4 causal check (prereg Section 4) on the base card's refusal direction (sm.interp.causal): ablate r at
# every layer and position on held-out harmful prompts, add r at L* on harmless prompts, 10 random unit
# directions as controls, refusal judged by J1. It first gates on the base card's own validity checks
# (prereg Section 4: token-length SMD <= 0.1, split-half cosine >= 0.9 above the shuffled-null max, the L*
# coherence filter; jobs/common.sh sm_require_card_validity), so a base that fails them stops here (exit 2)
# instead of running the whole bank (ALLOW_FAILED_VALIDITY=1 continues, descriptive only).
# This is a hard gate: the job exits 2 unless
# results/base_cards/<base>_causal.json says passed (scored with J1, not --smoke, not the keyword
# fallback), and every later stage refuses to run without that file (jobs/common.sh sm_require_v4).
# jobs/submit_bank.sh also makes every later stage depend on this job with afterok, so a failed V4
# cancels the rest of the chain instead of holding GPUs only to refuse.
# Idempotent: sm.interp.causal exits 0 "already done" when the file exists; the verdict is re-checked here.
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/causal.sbatch
# Config: CAUSAL_CONFIG (default configs/causal_<base>.yaml, required; pinned revision checked).
# Judge server: J1 at the configs/judge.yaml pin (J1_PORT 8000, J1_GPU_UTIL 0.25).
# Optional env: SEED (default 0), FORCE=1 (recompute the check), ALLOW_FAILED_VALIDITY=1, SM_PROJ.
#SBATCH --job-name=sm-causal
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/causal-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/causal-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-causal start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/base_cards
V4="$(sm_v4_path "$BASE_KEY")"
CFG="${CAUSAL_CONFIG:-configs/causal_${BASE_KEY}.yaml}"
[ -f "$CFG" ] || sm_die 2 "causal config $CFG not found: the V4 check is a required validity gate"
DIRECTION="$OUT/$SAFE/refusal.npy"
[ -f "$OUT/$SAFE.json" ] || sm_die 5 "base card $OUT/$SAFE.json missing (run jobs/base_card.sbatch first)"
[ -f "$DIRECTION" ] || sm_die 5 "refusal direction $DIRECTION missing"
sm_require_card_validity "$BASE_KEY"

rc=0
if [ -f "$V4" ] && [ -z "${FORCE:-}" ]; then
  sm_log "already done: $V4 (verdict re-checked below)"
else
  sm_check_config_rev "$CFG" "$BASE_KEY"
  sm_require_frozen "$BASE_KEY"
  sm_require_data data/prompts/MANIFEST.json
  sm_preflight_snapshot "$BASE_KEY"
  sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}"
  sm_run "$PYTHON" -m sm.interp.causal --config "$CFG" --base "$BASE_KEY" --direction "$DIRECTION" --out "$OUT" --seed "${SEED:-0}" ${FORCE:+--force} || rc=$?
  sm_log "sm.interp.causal rc=$rc"
fi

if [ ! -f "$V4" ] && [ "$rc" -ne 0 ]; then
  sm_log "=== sm-causal done: sm.interp.causal failed (rc=$rc) without writing $V4 ==="
  exit "$rc"
fi
vrc=0
verdict="$(sm_py v4 "$V4")" || vrc=$?
sm_log "V4 verdict: $verdict"
if [ "$vrc" -ne 0 ]; then
  sm_log "=== sm-causal done: V4 NOT PASSED for $BASE_KEY; later stages will refuse (the base is not evaluable for the white-box hypotheses) ==="
  exit 2
fi
sm_log "=== sm-causal done: V4 passed for $BASE_KEY ==="
exit 0
