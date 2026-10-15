#!/bin/bash
# EXPLORATORY lines E15 (the assistant axis of Lu et al. 2601.10387) and E7 (registered causal mediation) on one base
# (default qwen3.5-9b). Protocol, rules and costs: docs/ASSISTANT_AXIS_E7_PLAN.md (committed before any run). Runs on
# the side tree (SM_PROJ default /projects/interpretability/shehral.m/monitors-diag; results/data/logs there are symlinks
# into the main tree). Not a measured stage: no frozen-code gate; every output records the code file hashes and
# SM_CODE_COMMIT (the laptop commit the files came from).
#
# Phases (env PHASES, separated by '+', ',' or ':'; default EXTRACT+DISPLACE; sbatch --export splits on commas, so pass
# PHASES=EXTRACT+DISPLACE inside --export, or export PHASES in the submitting shell with --export=ALL):
#   EXTRACT   J1 server; sm.interp.assistant_axis plan, generate (vLLM, 6,100 role and default rollouts), judge (J1,
#             role expression, in the background) beside activations (HF), then axis (axes, validity, capping taus)
#   DISPLACE  displace (base + every run of results/drift/<base>/_configs, adapters unmerged) and h3prime (H3' rule)
#   CAP       sm.interp.mediation persona repair along the axis (needs EXTRACT and DISPLACE outputs)
#   E7        sm.interp.mediation restore / random / ablate / adapter-write conditions (same units; J1 judged)
# Submit (estimates in the plan, section 10):
#   sbatch --time=03:00:00 --export=ALL,PHASES=EXTRACT+DISPLACE,SM_CODE_COMMIT=<sha> jobs/exploratory_axis.sbatch
#   sbatch --time=06:00:00 --dependency=afterok:<id> --kill-on-invalid-dep=yes \
#          --export=ALL,PHASES=CAP+E7,SM_CODE_COMMIT=<sha> jobs/exploratory_axis.sbatch
# RUN (run1 default | run2): which pre-registered extraction (plan sections 2 and 12); run2 writes under
#   results/exploratory/<base>/run2/, results/restricted/exploratory/<base>/run2/ and <SM_AXIS_CACHE>/run2, keeping run 1.
# Optional env: BASE, FORCE=1, GEN_BACKEND (vllm|hf), GEN_GPU_UTIL (0.45), JUDGE_WORKERS (32), ACT_BATCH (16),
#   J1_PORT / J1_GPU_UTIL / J1_MAX_LEN, UNITS (mediation unit subset), SM_AXIS_DIR (official clone), SM_AXIS_CACHE.
# Texts (rollouts, judge replies, steered completions) are written only under results/restricted/exploratory/ (0700).
#SBATCH --job-name=sm-axis
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/axis-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/axis-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
PHASES="${PHASES:-EXTRACT+DISPLACE}"
PHASES="$(echo "$PHASES" | tr '+: ' ',,,' | tr '[:lower:]' '[:upper:]')"
sm_log "=== sm-axis start on $(hostname) phases=$PHASES code=${SM_CODE_COMMIT:-unset} ==="
sm_setup_env
sm_gpu_preflight
# NVIDIA forward-compat libcuda (cuda-compat, exported by jobs/common.sh at source time): fail before any model load
# if it is not first on the library path (the nodes run driver 570.86 and the env torch is a cu130 build).
if [ "$SM_USE_COMPAT" = "1" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in ":$SM_COMPAT_DIR:"*) ;; *) sm_die 2 "SM_COMPAT_DIR=$SM_COMPAT_DIR is not first on LD_LIBRARY_PATH" ;; esac
fi

BASE_KEY="${BASE:-qwen3.5-9b}"
SAFE="$(sm_py safe-name "$BASE_KEY")"
CARD="results/base_cards/$SAFE.json"
export SM_AXIS_DIR="${SM_AXIS_DIR:-$SM_PROJ/third_party/assistant-axis}"
export SM_AXIS_CACHE="${SM_AXIS_CACHE:-$SM_SCRATCH/exploratory/$SAFE/axis_cache}"
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing"
[ -f "results/base_cards/$SAFE/refusal.npy" ] || sm_die 5 "refusal direction results/base_cards/$SAFE/refusal.npy missing"
[ -d "$SM_AXIS_DIR/data/roles/instructions" ] || sm_die 5 "official assistant-axis data missing under $SM_AXIS_DIR"
sm_preflight_snapshot "$BASE_KEY"
sm_log "axis data: $SM_AXIS_DIR (commit $(cat "$SM_AXIS_DIR/COMMIT" 2>/dev/null || echo unknown)); cache: $SM_AXIS_CACHE"

has() { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
AXM=("$PYTHON" -m sm.interp.assistant_axis)
RUN="${RUN:-run1}"
case "$RUN" in run1 | run2) ;; *) sm_die 2 "RUN must be run1 or run2, got '$RUN'" ;; esac
sm_log "run: $RUN"
COMMON=(--base "$BASE_KEY" --card "$CARD" --official-dir "$SM_AXIS_DIR" --run "$RUN")
J1_UP=0
start_j1() {
  if [ "$J1_UP" = "0" ]; then
    sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}"
    J1_UP=1
  fi
}

if has EXTRACT; then
  start_j1
  rc=0; sm_run "${AXM[@]}" plan "${COMMON[@]}" || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "axis plan failed"
  rc=0; sm_run "${AXM[@]}" generate "${COMMON[@]}" --backend "${GEN_BACKEND:-vllm}" --gpu-util "${GEN_GPU_UTIL:-0.45}" || rc=$?
  if [ "$rc" -ne 0 ] && [ "${GEN_BACKEND:-vllm}" = "vllm" ]; then
    # Fallback (recorded in axis/generate.json): HF sampling at the same temperature and top-p; rollouts already on disk
    # are kept (the phase is resumable per rollout).
    sm_log "WARNING: vLLM generation failed (rc=$rc); falling back to HF sampling"
    rc=0; sm_run "${AXM[@]}" generate "${COMMON[@]}" --backend hf --batch-size 64 || rc=$?
  fi
  [ "$rc" -eq 0 ] || sm_die "$rc" "rollout generation failed"
  # Role-expression judging (J1 over HTTP, CPU-light) runs beside the activation capture on the GPU.
  JLOG="$SM_LOGDIR/axis-judge-${SLURM_JOB_ID:-local}.log"
  "${AXM[@]}" judge "${COMMON[@]}" --workers "${JUDGE_WORKERS:-32}" >"$JLOG" 2>&1 &
  JPID=$!
  sm_log "judge pid $JPID (log $JLOG)"
  arc=0; sm_run "${AXM[@]}" activations "${COMMON[@]}" --batch-size "${ACT_BATCH:-16}" || arc=$?
  jrc=0; wait "$JPID" || jrc=$?
  sm_log "activations rc=$arc judge rc=$jrc"
  [ "$arc" -eq 0 ] || sm_die "$arc" "activations failed"
  [ "$jrc" -eq 0 ] || sm_die "$jrc" "role judging failed (see $JLOG)"
  rc=0; sm_run "${AXM[@]}" axis "${COMMON[@]}" || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "axis computation failed"
fi

if has DISPLACE; then
  rc=0; sm_run "${AXM[@]}" displace "${COMMON[@]}" --batch-size 16 ${FORCE:+--force} || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "displacement failed"
  rc=0; sm_run "${AXM[@]}" h3prime "${COMMON[@]}" || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "h3prime analysis failed"
fi

if has CAP || has E7; then
  MPH=""
  has CAP && MPH="CAP"
  has E7 && MPH="${MPH:+$MPH,}E7"
  start_j1
  rc=0; sm_run "$PYTHON" -m sm.interp.mediation --base "$BASE_KEY" --card "$CARD" --phases "$MPH" --run "$RUN" \
    ${UNITS:+--units "$UNITS"} ${FORCE:+--force} || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "mediation failed"
fi
sm_log "=== sm-axis done phases=$PHASES ==="
exit 0
