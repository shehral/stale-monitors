#!/bin/bash
# J2 refusal-readout diagnostic (docs/J2_DIAGNOSIS.md; sm.eval.judge_diag). DIAGNOSTIC ONLY: not a measured stage,
# no acceptance stage, nothing downstream reads it. It scores the BASE run's H items (strongreject, harmbench,
# xstest_unsafe) and xstest_safe with up to four candidate J2 readouts and compares each with the stored J1 labels and
# Arditi's substring matcher, against the bar pre-registered in docs/J2_DIAGNOSIS.md section 5:
#   a  Granite Guardian 3.3 as configured (must reproduce the stored J2 labels: pooled R about 0.205)
#   b  Granite with the refusal question in the Guardian risk framing (refusal = "no"), think off
#   c  Granite as configured with think on (the model card's custom-criteria example)
#   d  allenai/wildguard (gated: run only when its snapshot is cached; otherwise skipped and recorded in the summary)
#   e  google/gemma-4-31B-it at its configs/bases.yaml pin, thinking off, applying J1's policy file verbatim with a fixed
#      label-line wrapper (pre-registered in docs/J2_DIAGNOSIS.md "Candidate (e)", 2026-10-10). It runs alone
#      (CANDIDATES=e): the 31B server takes most of the H200. The (a) to (d) entries are carried over from the readout-2
#      summary (REUSE, default results/diag/judge_j2_diag.json; its SHA-256 is registered), not re-run, and the summary
#      goes to results/diag/judge_j2_diag_e.json so the readout-2 file stays untouched.
# Servers on the one H200: Granite at the J2 pin of configs/judge.yaml (sm_start_judge J2, checked against
# configs/bases.yaml, port GRANITE_PORT) and WildGuard at WG_REV (sm_start_vllm, port WG_PORT); for e alone, Gemma 4 31B
# at the configs/bases.yaml pin of E_KEY (sm_py pin, snapshot pre-flight, sm_start_vllm J2E, port E_PORT).
# Run from the SIDE tree: SM_PROJ=/projects/interpretability/shehral.m/monitors-diag (its results/, data/ and logs/
# are symlinks into the main tree, so the inputs are read and the outputs written there). Offline flags and the
# forward-compatibility libcuda path come from jobs/common.sh exactly as for every other GPU job.
# Outputs: results/diag/judge_j2_diag.json (counts, rates, kappas, the bar verdict, policy texts, SHA-256 of policies,
# inputs, raw files and code; no prompt, completion or judge text) and the raw judge rows under
# results/restricted/diag/judge_j2/<base>/base/<candidate>/<battery>.jsonl (dirs 0700, files 0600; cluster only).
# Cost (estimate, not measured): about 5 min per vLLM start; a and b about 1,757 short requests each; c the same in
# think mode (a few hundred tokens each, about 10 to 15 min); about 0.3 to 0.5 H200-hours for a to c, d about 0.1 more;
# e alone about 0.2 to 0.35 (one 5 to 10 min start of the 31B server, 669 short requests).
# Idempotent: exits 0 "already done" when the summary is complete for the requested candidates; label rows already
# written are reused (FORCE=1 re-scores everything). Hand-submitted: ledger it with sm_ledger_add <jobid> afterwards.
#
# Submit (from the side tree; CANDIDATES is a string of letters because sbatch --export splits on commas):
#   cd /projects/interpretability/shehral.m/monitors-diag && \
#     sbatch --export=ALL,SM_PROJ=/projects/interpretability/shehral.m/monitors-diag,CANDIDATES=abc jobs/judge_diag.sbatch
#   candidate e: the same with CANDIDATES=e.
# Optional env: BASE (default qwen3.5-9b), CANDIDATES (default abcd), OUT, RDIR, GRANITE_PORT (8001), WG_PORT (8002),
#   WG_REV (default the pinned cbba4823...), J2_GPU_UTIL (0.40), WG_GPU_UTIL (0.35), CONCURRENCY (48), FORCE=1;
#   for e: REUSE, E_KEY (gemma-4-31b-it), E_PORT (8003), E_GPU_UTIL (0.85), E_MAX_LEN (8192).
#SBATCH --job-name=sm-judge-diag
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/judge-diag-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/judge-diag-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-judge-diag start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
umask 027

BASE_KEY="${BASE:-qwen3.5-9b}"
CANDIDATES="${CANDIDATES:-abcd}"
if [[ "$CANDIDATES" == *e* ]]; then
  [ "$CANDIDATES" = "e" ] || sm_die 2 "candidate e runs alone (CANDIDATES=e); got CANDIDATES=$CANDIDATES"
  OUT="${OUT:-results/diag/judge_j2_diag_e.json}"
  REUSE="${REUSE:-results/diag/judge_j2_diag.json}"
fi
OUT="${OUT:-results/diag/judge_j2_diag.json}"
RDIR="${RDIR:-results/restricted/diag/judge_j2}"
GRANITE_PORT="${GRANITE_PORT:-8001}"
WG_PORT="${WG_PORT:-8002}"
WG_MODEL=allenai/wildguard
WG_REV="${WG_REV:-cbba4823f3e8020e5a74a5e29bf85072def6f2ff}"
E_KEY="${E_KEY:-gemma-4-31b-it}"
E_PORT="${E_PORT:-8003}"
sm_log "side tree: $SM_PROJ (git $(git -C "$SM_PROJ" rev-parse --short HEAD 2>/dev/null || echo none)); candidates=$CANDIDATES base=$BASE_KEY out=$OUT"
mkdir -p results/diag

EXTRA=()
if [[ "$CANDIDATES" == *[abc]* ]]; then
  sm_start_judge J2 "$GRANITE_PORT" "${J2_GPU_UTIL:-0.40}" --max-model-len "${J2_MAX_LEN:-8192}"
  EXTRA+=(--granite-url "http://127.0.0.1:$GRANITE_PORT" --granite-served-name J2 --granite-revision "$J2_REVISION")
fi
if [[ "$CANDIDATES" == *d* ]]; then
  if sm_py snapshot "$WG_MODEL" "$WG_REV" >/dev/null 2>&1 \
    && SM_VLLM_SOFT=1 sm_start_vllm WG "$WG_MODEL" "$WG_REV" "$WG_PORT" "${WG_GPU_UTIL:-0.35}" --max-model-len "${WG_MAX_LEN:-8192}"; then
    EXTRA+=(--wildguard-url "http://127.0.0.1:$WG_PORT" --wildguard-served-name WG --wildguard-revision "$WG_REV")
  else
    sm_log "SKIP: candidate d: $WG_MODEL@$WG_REV is not cached or its server did not start (gated repo: the owner accepts the licence first)"
    CANDIDATES="${CANDIDATES//d/}"
    EXTRA+=(--note "d not run in job ${SLURM_JOB_ID:-local}: $WG_MODEL@$WG_REV snapshot not cached or server did not start (gated; on 2026-10-09 the download returned 403, licence not accepted)")
  fi
fi
if [ "$CANDIDATES" = "e" ]; then
  E_MODEL="$(sm_py hf "$E_KEY")" || sm_die 2 "no hf id for $E_KEY"
  E_REV="$(sm_py pin "$E_KEY")" || sm_die 2 "pin lookup failed for $E_KEY"
  [ -n "$E_REV" ] || sm_die 2 "$E_KEY is unpinned in configs/bases.yaml"
  [ -f "$REUSE" ] || sm_die 5 "candidate e reads (a) to (d) from $REUSE, which is missing"
  sm_preflight_snapshot "$E_MODEL" "$E_REV"
  E_SNAP="$(sm_py snapshot "$E_MODEL" "$E_REV")"
  E_TMPL_SHA="$(sha256sum "$E_SNAP/chat_template.jinja" | cut -d' ' -f1)" || E_TMPL_SHA=""
  sm_log "candidate e: $E_MODEL@$E_REV ($E_KEY pin in configs/bases.yaml); chat_template.jinja sha256 ${E_TMPL_SHA:-missing}; reuse $REUSE"
  sm_start_vllm J2E "$E_MODEL" "$E_REV" "$E_PORT" "${E_GPU_UTIL:-0.85}" --max-model-len "${E_MAX_LEN:-8192}"
  EXTRA+=(--policy-judge-url "http://127.0.0.1:$E_PORT" --policy-judge-served-name J2E --policy-judge-model "$E_MODEL"
    --policy-judge-revision "$E_REV" --reuse "$REUSE")
  if [ -n "$E_TMPL_SHA" ]; then
    EXTRA+=(--policy-judge-template-sha256 "$E_TMPL_SHA")
  fi
fi
[ -n "$CANDIDATES" ] || sm_die 2 "no candidate left to run"

rc=0
sm_run "$PYTHON" -m sm.eval.judge_diag --candidates "$CANDIDATES" --runs base --base "$BASE_KEY" --results-root results \
  --out "$OUT" --restricted-out "$RDIR" --concurrency "${CONCURRENCY:-48}" "${EXTRA[@]}" ${FORCE:+--force} || rc=$?
sm_log "=== sm-judge-diag done rc=$rc candidates=$CANDIDATES out=$OUT ==="
exit "$rc"
