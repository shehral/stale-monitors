#!/bin/bash
# EXPLORATORY mechanism instruments M-A to M-C (docs/MECHANISM_PLAN.md, rules fixed before any capture; deviations in
# its section 9): sm.interp.mechanism on one unit of one base (default qwen3.5-9b). Not a measured stage, so no
# frozen-code gate; every output records the git commit, SM_CODE_COMMIT, the code file hashes, the card and adapter
# SHA-256 and a config hash. Runs from the side tree (SM_PROJ default /projects/interpretability/shehral.m/monitors-diag;
# its results, data and logs are symlinks into the main tree).
#
# Env:
#   UNIT     run id of results/drift/<base>/_configs (or "base"); required except for SUBCMD=summarize
#   SUBCMD   patch-sweep (default) | component-patch | revert | svd-remove | addback-sweep | heads | steer |
#            offset-steer (GPU; program R2: EXTRA "--acts <unit npz> --acts-base <base npz> --judge ... --canary <file>
#            --battery-rate k/n --battery-rate-base k/n"; layers 20,24,28, alphas 0.5,1.0,1.5, both modes and directions
#            by default; texts under results/restricted/exploratory/<base>/mechanism/offset_judged/<unit>/) |
#            rescore-judged (GPU for the J2 = WildGuard server, started instead of J1; EXTRA "--source revert" etc.) |
#            gain | delta | summarize (CPU: submit them to the short partition with --gres=none; no GPU pre-flight)
#   EXTRA    extra CLI arguments, e.g. "--layers 14-17 --positions last" (heads) or
#            "--layers 15 --direction heads --judge sm.eval.judge:judge_refusal_labels --canary <file>" (steer; a
#            --judge with a module:callable starts the J1 server first)
#   BASE (default qwen3.5-9b), FORCE=1 (recompute), SM_PROJ
# Submit (one H200, 4 h; estimates in the plan, section 6: about 0.1 H200-h per unit for the sweep):
#   sbatch --export=ALL,UNIT=c1_dolly_r128,SUBCMD=patch-sweep,SM_CODE_COMMIT=<sha> jobs/mechanism.sbatch
#   sbatch -p short --gres=none --mem=32G --time=01:00:00 --export=ALL,SUBCMD=summarize,SM_CODE_COMMIT=<sha> jobs/mechanism.sbatch
# Texts (steer --judge generations) are written only under results/restricted/exploratory/<base>/mechanism/ (0700).
# Hand-submitted: ledger the GPU job with sm_ledger_add <jobid> after it ends (the EXIT trap prints the LEDGER line).
# Exit codes: 2 setup (env, GPU, compat libcuda, snapshot, arguments), 5 inputs missing (card).
#SBATCH --job-name=sm-mech
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/mech-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/mech-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
SUBCMD="${SUBCMD:-patch-sweep}"
# Judge ports are per job: two sm-mech jobs on one node both defaulting to 8000/8001 collided on 2026-10-13 (the second
# job saw the first job's server as "healthy after 0s" and died when that job ended). Explicit J1_PORT/J2_PORT still win.
SM_PORT_BASE=$(( 8100 + (${SLURM_JOB_ID:-0} % 800) ))
J1_PORT="${J1_PORT:-$SM_PORT_BASE}"; J2_PORT="${J2_PORT:-$((SM_PORT_BASE + 1))}"; export J1_PORT J2_PORT
case "$SUBCMD" in
  patch-sweep | component-patch | revert | svd-remove | addback-sweep | heads | steer | offset-steer | rescore-judged) GPU=1 ;;
  gain | delta | summarize) GPU=0 ;;
  *) sm_die 2 "SUBCMD must be patch-sweep, component-patch, revert, svd-remove, addback-sweep, heads, steer, offset-steer, rescore-judged, gain, delta or summarize, got '$SUBCMD'" ;;
esac
sm_log "=== sm-mech start on $(hostname) subcmd=$SUBCMD unit=${UNIT:-none} code=${SM_CODE_COMMIT:-unset} ==="
sm_setup_env
if [ "$GPU" = "1" ]; then
  sm_gpu_preflight
  # NVIDIA forward-compat libcuda (cuda-compat, exported by jobs/common.sh at source time): fail before any model load
  # if it is not first on the library path (the nodes run driver 570.86 and the env torch is a cu130 build).
  if [ "$SM_USE_COMPAT" = "1" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in ":$SM_COMPAT_DIR:"*) ;; *) sm_die 2 "SM_COMPAT_DIR=$SM_COMPAT_DIR is not first on LD_LIBRARY_PATH" ;; esac
  fi
fi

BASE_KEY="${BASE:-qwen3.5-9b}"
SAFE="$(sm_py safe-name "$BASE_KEY")"
CARD="results/base_cards/$SAFE.json"
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing"
if [ "$SUBCMD" != "summarize" ] && [ -z "${UNIT:-}" ]; then
  sm_die 2 "UNIT is required for SUBCMD=$SUBCMD"
fi
if [ "$GPU" = "1" ] && [ "$SUBCMD" != "rescore-judged" ]; then
  sm_preflight_snapshot "$BASE_KEY"
fi
if [ "$SUBCMD" = "rescore-judged" ]; then
  # WildGuard (the primary reader, J2 at the configs/judge.yaml pin) instead of J1; no model of ours is loaded.
  sm_start_judge J2 "${J2_PORT:-8001}" "${J2_GPU_UTIL:-0.40}" --max-model-len "${J2_MAX_LEN:-8192}"
else
  case " ${EXTRA:-} " in
    *" --judge "*:*) sm_start_judge J1 "${J1_PORT:-8000}" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}" ;;
  esac
fi

# EXTRA is split on whitespace on purpose (a list of CLI flags).
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.interp.mechanism "$SUBCMD" --base "$BASE_KEY" --card "$CARD" ${UNIT:+--unit "$UNIT"} \
  ${FORCE:+--force} ${EXTRA:-} || rc=$?
[ "$rc" -eq 0 ] || sm_die "$rc" "sm.interp.mechanism $SUBCMD failed"
sm_log "=== sm-mech done subcmd=$SUBCMD unit=${UNIT:-none} ==="
exit 0
