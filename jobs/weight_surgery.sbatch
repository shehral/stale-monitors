#!/bin/bash
# EXPLORATORY line R3 (docs/PROGRAM_2026-10-13.md, registered before code): weight surgery, judged refusal recovered
# vs capability and training-task loss kept. sm.exploratory.weight_surgery on one LoRA unit of one base (default
# qwen3.5-9b): for each variant (a set of the unit's LoRA modules reverted to the base weights, RevertLoRA of
# sm.interp.mechanism) the MMLU + IFEval capability battery on the measured stage's HF path and scorer, the next-token
# loss on a 512-example held-out set and a 512-example in-sample train set (the LoRA stage's chat formatting), and with
# --judge the judged canary. Not a measured stage, so no frozen-code gate; every output records the git commit,
# SM_CODE_COMMIT, the code file hashes, the card and adapter SHA-256 and a config hash. Runs from the side tree
# (SM_PROJ default /projects/interpretability/shehral.m/monitors-diag; its results, data and logs are symlinks into
# the main tree).
#
# Env (export them in the submitting shell: sbatch --export splits on commas, and VARIANTS / EXTRA contain commas):
#   UNIT      bank run id (a LoRA unit of configs/bank_*.yaml); required
#   VARIANTS  ';'-separated: unit | base | revert:<selector> | keep_only:<selector> (selectors as mechanism revert;
#             blocks are 0-based block indices). Default (module): unit;base;revert:type=mlp;
#             revert:type=self_attn+linear_attn;keep_only:blocks=25-31
#   EXTRA     extra CLI flags, e.g. "--judge sm.eval.judge:judge_refusal_labels --canary <file> --battery-rate k/n
#             --reuse-judged <mechanism revert json>[,<json>]" (a --judge with a module:callable starts the J1 server)
#   BASE (default qwen3.5-9b), FORCE=1 (recompute), SM_PROJ
# Submit (one H200; estimate about 1.5 to 2.5 H200-h per unit with the default five variants and the judge):
#   UNIT=c1_dolly_r128 VARIANTS='unit;base;revert:type=mlp;revert:type=self_attn+linear_attn;keep_only:blocks=25-31' \
#     EXTRA='--judge sm.eval.judge:judge_refusal_labels --canary <file>' SM_CODE_COMMIT=<sha> \
#     sbatch --export=ALL jobs/weight_surgery.sbatch
# WildGuard (primary reader) afterwards, unchanged tools: jobs/mechanism.sbatch with SUBCMD=rescore-judged and
#   EXTRA="--source revert --out results/exploratory/<base>/weight_surgery/rescore_view
#          --restricted results/restricted/exploratory/<base>/weight_surgery", then on the laptop
#   python -m sm.exploratory.weight_surgery --unit <unit> --attach-wildguard
# Texts (capability completions, persona held-out generations, judged canary completions) are written only under
# results/restricted/exploratory/<base>/weight_surgery/ (0700); the open JSON holds numbers, hashes and stamps.
# Hand-submitted: ledger the GPU job with sm_ledger_add <jobid> after it ends (the EXIT trap prints the LEDGER line).
# Exit codes: 2 setup (env, GPU, compat libcuda, snapshot, arguments), 5 inputs missing (card).
#SBATCH --job-name=sm-wsurg
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/wsurg-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/wsurg-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
# Judge port per job (two judged jobs on one node collided on fixed ports on 2026-10-13); an explicit J1_PORT wins.
SM_PORT_BASE=$(( 8100 + (${SLURM_JOB_ID:-0} % 800) ))
J1_PORT="${J1_PORT:-$SM_PORT_BASE}"; export J1_PORT
[ -n "${UNIT:-}" ] || sm_die 2 "UNIT is required"
sm_log "=== sm-wsurg start on $(hostname) unit=$UNIT code=${SM_CODE_COMMIT:-unset} variants=${VARIANTS:-default} ==="
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
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing"
sm_preflight_snapshot "$BASE_KEY"
case " ${EXTRA:-} " in
  *" --no-judge "*) ;;
  *" --judge "*:*) sm_start_judge J1 "$J1_PORT" "${J1_GPU_UTIL:-0.25}" --max-model-len "${J1_MAX_LEN:-16384}" ;;
esac

# EXTRA is split on whitespace on purpose (a list of CLI flags); VARIANTS is one argument.
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.exploratory.weight_surgery --base "$BASE_KEY" --card "$CARD" --unit "$UNIT" \
  ${VARIANTS:+--variants "$VARIANTS"} ${FORCE:+--force} ${EXTRA:-} || rc=$?
[ "$rc" -eq 0 ] || sm_die "$rc" "sm.exploratory.weight_surgery failed"
sm_log "=== sm-wsurg done unit=$UNIT ==="
exit 0
