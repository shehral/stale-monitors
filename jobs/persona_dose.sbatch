#!/bin/bash
# Persona steering dose test (docs/FIDELITY_REVIEW.md R2; Chen et al. 2507.21509 steering; pre-registered re-run in
# docs/PERSONA_FIDELITY_RERUN.md 4.7 and section 7): sm.interp.persona_dose on an existing base card. For every persona
# trait with status ok it adds the card's persona vector at persona.<trait>.layer, at the response positions, with each
# coefficient in COEFS and once unsteered, samples at temperature 1 and grades every sample.
# GRADER (default chen):
#   chen  the persona grader (jobs/common.sh sm_start_grader: Chen et al.'s trait and coherence prompts, verbatim, on
#         google/gemma-4-31B-it at its configs/bases.yaml pin; sm.interp.persona_grader:score_trait_chen). Defaults are
#         the pre-registered ones: every eval question (20) with its half of the card's split, 10 samples, 1,000 new
#         tokens, coefficients -2.5 to 2.5 in steps of 0.5. Output results/base_cards/<base>_persona_dose_chen.json
#         (per coefficient: mean trait, mean coherence, coherence window; per sample: numbers and the completion's
#         SHA-256; the readout of section 7).
#   j1    J1 with the generic v0.3 persona policy (sm.eval.judge:score_trait; sm_start_judge J1) on the same settings
#         and seeds (the generic-grader sensitivity line); output <base>_persona_dose_j1generic.json. The v0.3 run of
#         job 10943675 (<base>_persona_dose.json) used QUESTIONS=select N_QUESTIONS=10 N_SAMPLES=5 MAX_NEW=128
#         COEFS=-4,-2,-1,1,2,4.
# Cost (estimate): chen about 2 H200-hours for a 9B base (6,600 generations of up to 1,000 tokens, 13,200 scores, the
# grader start); j1 about the same.
# Idempotent: exits 0 "already done" when the output exists (FORCE=1 recomputes).
#
# Submit: sbatch --export=ALL,BASE=<registry key>[,GRADER=chen|j1] jobs/persona_dose.sbatch
#     or: sbatch --export=ALL,BANK=configs/bank_<base>.yaml jobs/persona_dose.sbatch
# Inputs: results/base_cards/<base>.json and its persona_<trait>.{npy,json} (exit 5 when missing).
# Optional env: GRADER, COEFS, QUESTIONS (eval|select), N_QUESTIONS, N_SAMPLES, MAX_NEW, TRAITS (space-separated), SEED,
#   FORCE=1, PG_PORT, PG_GPU_UTIL, PG_MAX_LEN, J1_PORT, J1_GPU_UTIL, SM_PG_CONCURRENCY.
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, pinned revision, frozen code (jobs/common.sh).
#SBATCH --job-name=sm-persona-dose
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/persona-dose-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/persona-dose-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-persona-dose start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key

GRADER="${GRADER:-chen}"
case "$GRADER" in
  chen) JUDGE=sm.interp.persona_grader:score_trait_chen; TAG=chen ;;
  j1) JUDGE=sm.eval.judge:score_trait; TAG=j1generic ;;
  *) sm_die 2 "GRADER must be chen or j1, got $GRADER" ;;
esac
SAFE="$(sm_py safe-name "$BASE_KEY")"
OUT=results/base_cards
CARD="$OUT/$SAFE.json"
DOSE="$OUT/${SAFE}_persona_dose_${TAG}.json"
if [ -f "$DOSE" ] && [ -z "${FORCE:-}" ]; then
  sm_log "already done: $DOSE"
  exit 0
fi
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing (run jobs/base_ablate.sbatch or jobs/base_persona.sbatch first)"
ls "$OUT/$SAFE"/persona_*.npy >/dev/null 2>&1 || sm_die 5 "no persona vectors under $OUT/$SAFE (the card has no ok trait)"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

if [ "$GRADER" = "chen" ]; then
  sm_start_grader "${PG_PORT:-8010}" "${PG_GPU_UTIL:-0.55}" --max-model-len "${PG_MAX_LEN:-4096}"
else
  sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}"
fi

rc=0
# shellcheck disable=SC2086
sm_run "$PYTHON" -m sm.interp.persona_dose --base "$BASE_KEY" --card "$CARD" --out "$DOSE" --judge "$JUDGE" \
  --questions "${QUESTIONS:-eval}" --n-questions "${N_QUESTIONS:-20}" --n-samples "${N_SAMPLES:-10}" \
  --max-new-tokens "${MAX_NEW:-1000}" --coefs="${COEFS:--2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5}" \
  ${TRAITS:+--traits $TRAITS} ${SEED:+--seed "$SEED"} ${FORCE:+--force} || rc=$?
sm_log "=== sm-persona-dose done rc=$rc base=$BASE_KEY grader=$GRADER out=$DOSE ==="
exit "$rc"
