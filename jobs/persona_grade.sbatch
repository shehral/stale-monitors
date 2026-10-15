#!/bin/bash
# Persona battery grades with Chen et al.'s prompts (sm.interp.persona_grader grade-battery; docs/PERSONA_FIDELITY_RERUN.md
# 4.9): scores the stored persona_<trait> battery samples (Chen's eval questions, 10 samples, 512 tokens) of the base and
# every derivative of one base with the persona grader (jobs/common.sh sm_start_grader: google/gemma-4-31B-it at its
# configs/bases.yaml pin) and writes numbers and hashes only: per-sample rows under
# results/restricted/persona_grades/<base>/<run>/persona_<trait>.jsonl (0700/0600) and
# results/persona_grades/<base>/<run>/summary.json. PRODUCER ONLY: no acceptance stage, and code/sm/analysis does not
# read these rows until the analysis-change draft of docs/PERSONA_FIDELITY_RERUN.md section 8 is adopted (until then the H1 persona lines and H3's
# expression link keep J1's generic grades and say so).
# Cost (estimate): about 1 H200-hour for 17 units x 3 traits x 200 samples (two prompts each) plus the grader start.
# Idempotent: a trait whose grades file matches the battery file's SHA-256 and the prompt set is skipped (FORCE=1 re-scores).
#
# Submit: sbatch --export=ALL,BASE=<registry key>[,RUNS="base c1_dolly_r8 ..."] jobs/persona_grade.sbatch
# Optional env: RUNS (space-separated; default every run with persona battery outputs), TRAITS, FORCE=1, PG_KEY, PG_PORT,
#   PG_GPU_UTIL (default 0.85: no base model shares the GPU), PG_MAX_LEN, SM_PG_CONCURRENCY (default 32), SM_PROJ.
# Guards: GPU pre-flight (exit 2) and the GPU watchdog, frozen code (jobs/common.sh).
#SBATCH --job-name=sm-persona-grade
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/persona-grade-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/persona-grade-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-persona-grade start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_resolve_base_key
umask 077
sm_require_frozen "$BASE_KEY"
[ -d "results/restricted/batteries/$(sm_py safe-name "$BASE_KEY")" ] || sm_die 5 "no battery outputs under results/restricted/batteries for $BASE_KEY"

sm_start_grader "${PG_PORT:-8010}" "${PG_GPU_UTIL:-0.85}" --max-model-len "${PG_MAX_LEN:-4096}"

rc=0
# shellcheck disable=SC2086
sm_run "$PYTHON" -m sm.interp.persona_grader grade-battery --base "$BASE_KEY" --results-root results \
  ${RUNS:+--runs $RUNS} ${TRAITS:+--traits $TRAITS} ${FORCE:+--force} || rc=$?
sm_log "=== sm-persona-grade done rc=$rc base=$BASE_KEY ==="
exit "$rc"
