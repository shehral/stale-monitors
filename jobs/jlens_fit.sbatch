#!/bin/bash
# J-lens fit for hypothesis lines J1/J2 (docs/JLENS_PLAN.md): calls the official anthropics/jacobian-lens code
# (third_party/jacobian-lens @ 581d398613e5602a5af361e1c34d3a92ea82ba8e, Apache-2.0) through
# sm.interp.jlens.fit on one H200, at the base card's L* by default (sm convention: hidden_states[L*] =
# jlens layer L*-1). Output directory (format sm-jlens-v1): results/jlens/<base>/  (lens.pt, unembed.pt,
# vocab.json, meta.json, shards/). Never run on the laptop.
#
# Modes (MODE env):
#   fit    (default) one shard. As an array, task id = shard: sbatch --array=0-3 --export=ALL,BASE=qwen3.5-9b,N_SHARDS=4 jobs/jlens_fit.sbatch
#          Single job, all prompts: sbatch --export=ALL,BASE=qwen3.5-9b jobs/jlens_fit.sbatch   (needs < 8 h; see plan)
#   smoke  timing smoke on gpu-short: 8 prompts into results/jlens/<base>_smoke/ (FORCE implied):
#          sbatch -p gpu-short --time=00:30:00 --export=ALL,BASE=qwen3.5-9b,MODE=smoke jobs/jlens_fit.sbatch
#   merge  CPU only, after every shard: sbatch -p short --gres=none --mem=16G --time=00:30:00 \
#              --dependency=afterok:<array id> --export=ALL,BASE=qwen3.5-9b,MODE=merge jobs/jlens_fit.sbatch
# Env: BASE or BANK (base key), LAYERS (sm layers, space-separated; default the card's L_star), PROMPTS
#   (default data/jlens/wikitext103_1000_seed0.jsonl, built by sm.data.build_jlens_corpus), N_PROMPTS
#   (default all), N_SHARDS (default 1), SHARD (default SLURM_ARRAY_TASK_ID or 0), DTYPE (default bf16),
#   DIM_BATCH (default 32), MAX_SEQ_LEN (default 128), OUT (default results/jlens/<base>), SM_PROJ.
# Exit codes: 2 setup (env, pin, frozen code, official code missing), 5 inputs missing (card, prompts).
# Idempotent: a finished shard/lens file is skipped by the module.
#SBATCH --job-name=sm-jlens-fit
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/jlens-fit-%A_%a.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/jlens-fit-%A_%a.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
MODE="${MODE:-fit}"
sm_log "=== sm-jlens-fit start mode=$MODE on $(hostname) ==="
if [ "$MODE" = "merge" ]; then sm_setup_cpu_env; else sm_setup_env; sm_gpu_preflight; fi
sm_resolve_base_key

SAFE="$(sm_py safe-name "$BASE_KEY")"
JLENS_DIR="${SM_JLENS_DIR:-$SM_PROJ/third_party/jacobian-lens}"
[ -f "$JLENS_DIR/jlens/fitting.py" ] || sm_die 2 "official jacobian-lens code missing at $JLENS_DIR (clone 581d3986, docs/JLENS_PLAN.md)"
export SM_JLENS_DIR="$JLENS_DIR"
OUT="${OUT:-results/jlens/$SAFE}"

if [ "$MODE" = "merge" ]; then
  rc=0
  sm_run "$PYTHON" -m sm.interp.jlens merge --out "$OUT" || rc=$?
  sm_log "=== sm-jlens-fit merge done rc=$rc base=$BASE_KEY ==="
  exit "$rc"
fi

CARD="results/base_cards/$SAFE.json"
if [ -z "${LAYERS:-}" ]; then
  [ -f "$CARD" ] || sm_die 5 "base card $CARD missing (LAYERS not set)"
  LAYERS="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['L_star'])" "$CARD")"
fi
PROMPTS="${PROMPTS:-data/jlens/wikitext103_1000_seed0.jsonl}"
sm_require_data "$PROMPTS"
sm_require_frozen "$BASE_KEY"
sm_preflight_snapshot "$BASE_KEY"

N_SHARDS="${N_SHARDS:-1}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
NP_ARGS=()
[ -n "${N_PROMPTS:-}" ] && NP_ARGS=(--n-prompts "$N_PROMPTS")
if [ "$MODE" = "smoke" ]; then
  OUT="${OUT}_smoke"; NP_ARGS=(--n-prompts "${N_PROMPTS:-8}"); N_SHARDS=1; SHARD=0
  rm -f "$OUT/lens.pt"
fi
sm_log "fit: base=$BASE_KEY layers=$LAYERS prompts=$PROMPTS shard=$SHARD/$N_SHARDS out=$OUT dtype=${DTYPE:-bf16} dim_batch=${DIM_BATCH:-32}"

rc=0
# shellcheck disable=SC2086
sm_run "$PYTHON" -m sm.interp.jlens fit --base "$BASE_KEY" --layer $LAYERS --prompts "$PROMPTS" --out "$OUT" \
  "${NP_ARGS[@]}" --shard "$SHARD" --n-shards "$N_SHARDS" --dtype "${DTYPE:-bf16}" \
  --dim-batch "${DIM_BATCH:-32}" --max-seq-len "${MAX_SEQ_LEN:-128}" || rc=$?
sm_log "=== sm-jlens-fit done rc=$rc base=$BASE_KEY shard=$SHARD/$N_SHARDS ==="
exit "$rc"
