#!/bin/bash
# E14 HarmBench classifier harmful-compliance line (docs/HARMBENCH_CLS_LINE.md; sm.eval.harmbench_cls). EXPLORATORY
# sensitivity line, pre-registered before any request: not a measured stage, no acceptance stage, nothing downstream reads
# it, and it changes no registered decision rule. It scores every qwen3.5-9b run's H items (strongreject, harmbench,
# xstest_unsafe from results/restricted/batteries) and the owner's HarmBench sheet (results/restricted/human/harmbench.
# {sheet,key}.jsonl, copied from the laptop for this job) with cais/HarmBench-Llama-2-13b-cls at its pinned revision:
# the model card's prompt, generations clipped to 512 classifier tokens, BOS, one greedy token, "yes" = harmful compliance.
# It compares the classifier with the current J1 labels per run, per battery and pooled (StrongREJECT and XSTest unsafe
# flagged out of distribution); the owner comparison runs on the laptop (python -m sm.eval.harmbench_cls --score-human).
# Run from the SIDE tree: SM_PROJ=/projects/interpretability/shehral.m/monitors-diag (its results/, data/ and logs/ are
# symlinks into the main tree). Offline flags and the forward-compatibility libcuda path come from jobs/common.sh.
# Outputs: results/diag/harmbench_cls_<base>.json (counts, rates, kappas, hashes; no text); raw rows under
# results/restricted/diag/harmbench_cls/<base>/<run>/<battery>.jsonl and .../human/harmbench.jsonl, and the labels-only
# verdict file .../human/harmbench.verdicts.jsonl (dirs 0700, files 0600; cluster only).
# Needs the snapshot (jobs/harmbench_cls_download.sbatch). Pre-flight before the server starts: the registered hashes,
# the pin, tokenizer.model, the model-card text and the shards (sm.eval.harmbench_cls --preflight; exit 2 on a mismatch).
# Cost (estimate, not measured): one vLLM start of a 13B bf16 model (3 to 6 min) and about 7,333 one-token requests:
# about 0.2 to 0.3 H200-hours. Idempotent: exits 0 "already done" when the summary is complete for the same runs; raw rows
# already written are reused (FORCE=1 re-scores everything). Hand-submitted: ledger it with sm_ledger_add <jobid> afterwards.
#
# Submit (from the side tree, after the download job):
#   cd /projects/interpretability/shehral.m/monitors-diag && \
#     sbatch --dependency=afterok:<download job> --export=ALL,SM_PROJ=/projects/interpretability/shehral.m/monitors-diag jobs/harmbench_cls.sbatch
# Optional env: BASE (default qwen3.5-9b), RUNS (default all), OUT, RDIR, HBC_PORT (8004), HBC_GPU_UTIL (0.85),
#   CONCURRENCY (48), FORCE=1.
#SBATCH --job-name=sm-hbcls
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/hbcls-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/hbcls-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-hbcls start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
umask 027

BASE_KEY="${BASE:-qwen3.5-9b}"
RUNS="${RUNS:-all}"
OUT="${OUT:-results/diag/harmbench_cls_${BASE_KEY}.json}"
RDIR="${RDIR:-results/restricted/diag/harmbench_cls}"
HBC_PORT="${HBC_PORT:-8004}"
HBC_MODEL=cais/HarmBench-Llama-2-13b-cls
HBC_REV=bda705349d1144fa618770bea64d99ce54e3835b   # sm.eval.harmbench_cls.CLS_REVISION (docs/HARMBENCH_CLS_LINE.md)
sm_log "side tree: $SM_PROJ (git $(git -C "$SM_PROJ" rev-parse --short HEAD 2>/dev/null || echo none)); base=$BASE_KEY runs=$RUNS out=$OUT"
mkdir -p results/diag

sm_preflight_snapshot "$HBC_MODEL" "$HBC_REV"
HBC_SNAP="$(sm_py snapshot "$HBC_MODEL" "$HBC_REV")"
rc=0
sm_run "$PYTHON" -m sm.eval.harmbench_cls --preflight --snapshot "$HBC_SNAP" --model "$HBC_MODEL" --revision "$HBC_REV" || rc=$?
[ "$rc" -eq 0 ] || sm_die 2 "classifier pre-flight failed (rc=$rc): registered hashes, pin, tokenizer.model, model card or shards"
sm_start_vllm HBC "$HBC_MODEL" "$HBC_REV" "$HBC_PORT" "${HBC_GPU_UTIL:-0.85}" --max-model-len 2048

sm_run "$PYTHON" -m sm.eval.harmbench_cls --base "$BASE_KEY" --runs "$RUNS" --results-root results --out "$OUT" \
  --restricted-out "$RDIR" --url "http://127.0.0.1:$HBC_PORT" --served-name HBC --model "$HBC_MODEL" --revision "$HBC_REV" \
  --snapshot "$HBC_SNAP" --concurrency "${CONCURRENCY:-48}" ${FORCE:+--force} || rc=$?
sm_log "=== sm-hbcls done rc=$rc out=$OUT ==="
exit "$rc"
