#!/bin/bash
# Base card, phase 2 of 3: per-layer statistics (sm.interp.base_card --phase stats). CPU only, partition short:
# no model, no GPU, no judge. Reads the capture checkpoint results/base_cards/<base>/_partial/capture/ (written
# by jobs/base_card.sbatch) and computes, per layer, the difference-of-means direction, split-half cosines, the
# shuffled-label null, the random-direction baseline, the cross-validated logistic probe and its shuffled
# null; writes results/base_cards/<base>/_partial/stats/{stats.json,directions.npz,probes.npz}.
# Layers run in parallel (joblib, one worker per CPU of this allocation via SLURM_CPUS_PER_TASK, override with
# SM_CPUS); each layer keeps its own seed (seed + 1000 * layer) and one BLAS thread, so the numbers do not depend
# on the worker count. This phase ran for over an hour on 8 cores inside the GPU job 10896275, which IdleBot
# cancelled; here it holds no GPU.
# Idempotent: exits 0 "already done" when results/base_cards/<base>.json exists, and the module skips a stats
# checkpoint computed from the current capture, stats config and seed (a stale one is recomputed; FORCE=1
# recomputes). A missing capture checkpoint exits 5.
# Next: jobs/base_ablate.sbatch (submit_bank.sh chains it with afterok).
#
# Submit: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/base_stats.sbatch
#     or: sbatch --export=ALL,BASE=<registry key> jobs/base_stats.sbatch
# Config: BASE_CARD_CONFIG (default configs/base_card_<base>.yaml; its revision must equal the bases.yaml pin).
# Guards: pinned revision, frozen code (jobs/common.sh); no GPU pre-flight (CPU job).
# Optional env: SEED (default 0, must match the other two phases), FORCE=1, SM_CPUS, SM_PROJ.
#SBATCH --job-name=sm-base-stats
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/base-card-stats-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/base-card-stats-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
sm_log "=== sm-base-stats start on $(hostname) ==="
sm_setup_cpu_env 0
[ -x "$PYTHON" ] || sm_die 2 "env python not found at $PYTHON"
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
[ -f "$OUT/$SAFE/_partial/capture/manifest.json" ] || sm_die 5 "capture checkpoint $OUT/$SAFE/_partial/capture/manifest.json missing (run jobs/base_card.sbatch first)"
sm_require_data data/prompts/MANIFEST.json
sm_check_config_rev "$CFG" "$BASE_KEY"
sm_require_frozen "$BASE_KEY"
sm_log "cpus: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} SM_CPUS=${SM_CPUS:-unset}"

rc=0
sm_run "$PYTHON" -m sm.interp.base_card --config "$CFG" --base "$BASE_KEY" --out "$OUT" --seed "${SEED:-0}" \
  --phase stats ${FORCE:+--force} || rc=$?
if [ "$rc" -eq 0 ] && [ ! -f "$CARD" ]; then
  [ -f "$OUT/$SAFE/_partial/stats/stats.json" ] || { sm_log "stats wrote no $OUT/$SAFE/_partial/stats/stats.json"; rc=5; }
fi
sm_log "=== sm-base-stats done rc=$rc base=$BASE_KEY next=jobs/base_ablate.sbatch ==="
exit "$rc"
