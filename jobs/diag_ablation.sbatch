#!/bin/bash
# Directional-ablation instrument check (docs/GEMMA_ABLATION_DIAGNOSIS.md; sm.interp.diag_ablation). DIAGNOSTIC ONLY:
# not a measured stage, no judge, nothing downstream reads it. On 32 dev harmless and 32 dev harmful prompts of the
# base card's prompt set it reports the mean next-token KL(base || ablated) of the card's direction (its chosen, or
# for not_evaluable its top, grid cell) ablated with the base card's own instrument (DirectionHooks, every layer and
# position), against 3 random unit directions of the same norm, the zero direction (must give KL 0), the card's
# direction ablated at every position except the first token (and first token only, no mid-block, its layer only,
# massive dimensions zeroed), plus the incoherent share of 64-token greedy generations by the pre-declared heuristic
# (sm.interp.coherence), hook-call counts, a residual-norm profile and the architecture facts (layer_scalar, softcap).
# Cost (estimate, not measured): model load plus about 11 conditions x (128 forward passes + 64 x 64-token
# generations); well under 0.5 H200-hours for 12B/31B. v1 on gemma-4-12b-it (job 10946276): 2 min 59 s.
# SUITE=v2 (doc section 7, reading rules pre-registered there): 46 conditions (v1's 11, 2 no-edit probes, layer
# subsets, single hooks, layer-wise masked and whitened projections, alpha 0.25/0.5, last-position-only) with a
# per-hook coefficient trace and 48-token generations; estimate about 10 minutes, at most 0.25 H200-hours.
# Idempotent: exits 0 "already done" when the output exists (FORCE=1 recomputes).
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/diag_ablation.sbatch
#     or: sbatch --export=ALL,BASE=<registry key> jobs/diag_ablation.sbatch
#     add SUITE=v2 to the export list for the v2 suite (default SUITE=v1).
# Inputs: results/base_cards/<base>.json, results/base_cards/<base>/refusal.{npy,json}, configs/base_card_<base>.yaml
# and its dev prompt files (exit 5 when missing). Output: results/diag/<base>_ablation.json (SUITE=v1) or
# results/diag/<base>_ablation_v2.json (SUITE=v2).
# Optional env: SUITE (v1 or v2), CELL (P,L grid cell from the stats checkpoint instead of refusal.npy), N (default 32),
# SEED, FORCE=1.
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, snapshot pre-flight; no frozen-code gate (diagnostic).
#SBATCH --job-name=sm-diag-ablation
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/diag-ablation-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/diag-ablation-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-diag-ablation start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
CARDS=results/base_cards
SUITE="${SUITE:-v1}"
case "$SUITE" in
  v1) OUT="results/diag/${SAFE}_ablation.json"; MAX_NEW=64 ;;
  v2) OUT="results/diag/${SAFE}_ablation_v2.json"; MAX_NEW=48 ;;
  *) sm_die 2 "SUITE must be v1 or v2, got '$SUITE'" ;;
esac
if [ -f "$OUT" ] && [ -z "${FORCE:-}" ]; then
  sm_log "already done: $OUT"
  exit 0
fi
[ -f "$CARDS/$SAFE.json" ] || sm_die 5 "base card $CARDS/$SAFE.json missing (run jobs/base_ablate.sbatch first)"
[ -f "$CARDS/$SAFE/refusal.npy" ] || [ -n "${CELL:-}" ] || sm_die 5 "direction $CARDS/$SAFE/refusal.npy missing"
[ -f "configs/base_card_${BASE_KEY}.yaml" ] || sm_die 5 "configs/base_card_${BASE_KEY}.yaml missing"
sm_preflight_snapshot "$BASE_KEY"
mkdir -p results/diag

rc=0
sm_run "$PYTHON" -m sm.interp.diag_ablation --suite "$SUITE" --base "$BASE_KEY" --out "$OUT" --card "$CARDS/$SAFE.json" \
  --config "configs/base_card_${BASE_KEY}.yaml" --n "${N:-32}" --max-new-tokens "$MAX_NEW" \
  ${CELL:+--cell="$CELL"} ${SEED:+--seed "$SEED"} ${FORCE:+--force} || rc=$?
sm_log "=== sm-diag-ablation done rc=$rc base=$BASE_KEY suite=$SUITE out=$OUT ==="
exit "$rc"
