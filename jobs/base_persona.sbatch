#!/bin/bash
# Base card, persona phase (sm.interp.base_card --phase persona; docs/PERSONA_FIDELITY_RERUN.md 4.6): rebuilds only the
# persona part of an existing base card with the persona grader the config names (judge.trait): new persona vectors,
# their layers and selection records, the persona_* files, the persona monitors' orientation and diagnostic
# thresholds. L*, the refusal direction, subspace, probe, validity, the H2 declaration and their thresholds are copied
# from the card unchanged, so no L* sweep runs and J1 is not needed for it.
# Grader server on the same GPU, started before the base loads: judge.trait = sm.interp.persona_grader:score_trait_chen
# -> the persona grader (jobs/common.sh sm_start_grader: google/gemma-4-31B-it at its configs/bases.yaml pin, served
# name PG, port PG_PORT 8010, PG_GPU_UTIL 0.55, PG_MAX_LEN 4096); an sm.eval.judge callable (the v0.3 generic policy,
# the sensitivity line) -> J1 (sm_start_judge J1). The grader's self-test runs before any generation (exit 2 on failure).
# Acceptance: `python -m sm.accept --stage base_card --base <base> --job <this id>`; the log is
# logs/base-card-persona-<id>.out (accept finds it under the base-card prefix by the LEDGER job id) and carries the
# `frozen:` and LEDGER lines. The rewritten card changes its SHA-256, so organism, drift and monitor_scores re-run.
# Inputs: the card results/base_cards/<base>.json and its capture checkpoint _partial/capture/ (exit 5 when missing);
# the module refuses (exit 5) when the seed, the capture checkpoint or a non-persona config value differs from the card.
# Idempotent: the module skips a card whose inputs are unchanged; an ACCEPTED card is frozen, so the re-run needs FORCE=1.
# Cost (estimate, docs/PERSONA_FIDELITY_RERUN.md section 6): about 2.5 H200-hours for a 9B base (grader start 0.15,
# 6,000 generations of up to 1,000 tokens, 12,000 trait + coherence scores, the layer choice).
#
# Submit: sbatch --export=ALL,BASE=<registry key>,FORCE=1 jobs/base_persona.sbatch
#     or: sbatch --export=ALL,BANK=configs/bank_<base>.yaml,FORCE=1 jobs/base_persona.sbatch
# Config: BASE_CARD_CONFIG (default configs/base_card_<base>.yaml; its revision must equal the configs/bases.yaml pin).
# Optional env: SEED (default 0, must equal the card's), FORCE=1, PG_KEY, PG_PORT, PG_GPU_UTIL, PG_MAX_LEN, J1_PORT,
#   J1_GPU_UTIL, SM_PG_CONCURRENCY (default 32), SM_PROJ.
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, pinned revision, frozen code (jobs/common.sh).
#SBATCH --job-name=sm-base-persona
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/base-card-persona-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/base-card-persona-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-base-persona start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/base_cards
CARD="$OUT/$SAFE.json"
CFG="${BASE_CARD_CONFIG:-configs/base_card_${BASE_KEY}.yaml}"
[ -f "$CFG" ] || sm_die 2 "base card config $CFG not found"
sm_require_data data/prompts/MANIFEST.json
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing: the persona phase rebuilds the persona part of an existing card"
[ -f "$OUT/$SAFE/_partial/capture/manifest.json" ] || sm_die 5 "capture checkpoint $OUT/$SAFE/_partial/capture/manifest.json missing (the card's persona monitors are oriented on it)"
sm_check_config_rev "$CFG" "$BASE_KEY"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

SCORER="$(sm_persona_scorer "$CFG")" || sm_die 2 "cannot read judge.trait from $CFG"
case "$SCORER" in
  chen) sm_start_grader "${PG_PORT:-8010}" "${PG_GPU_UTIL:-0.55}" --max-model-len "${PG_MAX_LEN:-4096}" ;;
  j1) sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}" ;;
  *) sm_die 2 "judge.trait in $CFG names neither the persona grader nor a J1 callable" ;;
esac

rc=0
sm_run "$PYTHON" -m sm.interp.base_card --config "$CFG" --base "$BASE_KEY" --out "$OUT" --seed "${SEED:-0}" \
  --phase persona ${FORCE:+--force} || rc=$?
sm_log "=== sm-base-persona done rc=$rc base=$BASE_KEY scorer=$SCORER ==="
exit "$rc"
