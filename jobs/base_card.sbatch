#!/bin/bash
# Base card, phase 1 of 3: activation capture (sm.interp.base_card --phase capture). GPU, no judge server:
# last-prompt-token activations of the construction, dev and test contrast sets at every layer, written to
# results/base_cards/<base>/_partial/capture/ (<set>.npz float16 + manifest.json with the prompt-set SHA-256s,
# render meta, layers and pin). Minutes of GPU work; nothing CPU-heavy runs in this allocation (IdleBot cancels
# a GPU job whose GPU idles for an hour, which is how job 10896275 died in the old single-job layout).
# The base card is three chained jobs (jobs/submit_bank.sh stage base_card, afterok between them):
#   base_card.sbatch  (gpu, this file)  -> capture checkpoint
#   base_stats.sbatch (short, 32 CPUs)  -> per-layer statistics from the checkpoint (no model, no GPU)
#   base_ablate.sbatch (gpu + J1)       -> L* sweep, persona vectors, monitor thresholds, the card
# Acceptance (sm.accept --stage base_card) reads the base_ablate job's log (logs/base-card-ablate-<id>.out).
# Idempotent: exits 0 "already done" when results/base_cards/<base>.json exists, and the module skips a capture
# checkpoint that matches the configured prompt sets and pin (FORCE=1 recomputes both).
# Inputs are checked before the base loads: data/prompts/MANIFEST.json must exist (exit 5).
#
# Organism mode (H4), ORGANISM=1: no J1 server; for every trained condition-4 run of the bank (or the ids in
# ORGANISM_RUNS) with a finished adapter, `sm.interp.base_card --organism-adapter results/adapters/<base>/<rid>
# --run-id <rid>` writes the derivative's own refusal direction at the base card's L* under
# results/base_cards/<base>/organism/. jobs/submit_bank.sh chains it after lora and before drift, so the
# drift JSONs carry the organism blocks aggregate builds the H4 table from.
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/base_card.sbatch
#     or: sbatch --export=ALL,BASE=<registry key> jobs/base_card.sbatch
# Config: BASE_CARD_CONFIG (default configs/base_card_<base>.yaml, required; its revision must equal the
#         configs/bases.yaml pin).
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, pinned revision, frozen code (jobs/common.sh).
# Optional env: SEED (default 0), FORCE=1, ORGANISM=1 (+ ORGANISM_RUNS, ORGANISM_CONDITION), SM_PROJ.
#SBATCH --job-name=sm-base-card
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/base-card-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/base-card-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-base-card start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/base_cards
CARD="$OUT/$SAFE.json"
CFG="${BASE_CARD_CONFIG:-configs/base_card_${BASE_KEY}.yaml}"
[ -f "$CFG" ] || sm_die 2 "base card config $CFG not found"
sm_require_data data/prompts/MANIFEST.json

if [ "${ORGANISM:-0}" = "1" ]; then
  [ -f "$CARD" ] || sm_die 5 "organism mode needs the base card $CARD (its L* fixes the layer)"
  sm_require_v4 "$BASE_KEY"
  sm_check_config_rev "$CFG" "$BASE_KEY"
  sm_require_frozen "$BASE_KEY"
  sm_preflight_snapshot "$BASE_KEY"
  if [ -n "${ORGANISM_RUNS:-}" ]; then
    read -r -a ORUNS <<<"$ORGANISM_RUNS"
  else
    [ -n "${BANK:-}" ] && [ -f "$BANK" ] || sm_die 2 "organism mode needs BANK (or ORGANISM_RUNS)"
    mapfile -t ORUNS < <(sm_py organism-runs "$BANK" "${ORGANISM_CONDITION:-4}")
  fi
  [ "${#ORUNS[@]}" -gt 0 ] || sm_die 2 "organism mode: no condition-${ORGANISM_CONDITION:-4} runs in $BANK"
  rc=0
  for rid in "${ORUNS[@]}"; do
    ADAPTER="results/adapters/$BASE_KEY/$rid"
    if ! sm_py train-ok "$ADAPTER/train_meta.json" >/dev/null 2>&1; then
      sm_log "organism $rid: no finished adapter at $ADAPTER"
      rc=5
      continue
    fi
    orc=0
    sm_run "$PYTHON" -m sm.interp.base_card --config "$CFG" --base "$BASE_KEY" --out "$OUT" --seed "${SEED:-0}" \
      --organism-adapter "$ADAPTER" --run-id "$rid" ${FORCE:+--force} || orc=$?
    sm_log "organism $rid rc=$orc"
    [ "$orc" -eq 0 ] || rc="$orc"
  done
  sm_log "=== sm-base-card (organism) done rc=$rc base=$BASE_KEY runs=${ORUNS[*]} ==="
  exit "$rc"
fi

if [ -f "$CARD" ] && [ -z "${FORCE:-}" ]; then
  sm_log "already done: $CARD"
  exit 0
fi
sm_check_config_rev "$CFG" "$BASE_KEY"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

rc=0
sm_run "$PYTHON" -m sm.interp.base_card --config "$CFG" --base "$BASE_KEY" --out "$OUT" --seed "${SEED:-0}" \
  --phase capture ${FORCE:+--force} || rc=$?
if [ "$rc" -eq 0 ] && [ ! -f "$CARD" ]; then
  [ -f "$OUT/$SAFE/_partial/capture/manifest.json" ] || { sm_log "capture wrote no manifest at $OUT/$SAFE/_partial/capture/manifest.json"; rc=5; }
fi
sm_log "=== sm-base-card (capture) done rc=$rc base=$BASE_KEY next=jobs/base_stats.sbatch ==="
exit "$rc"
