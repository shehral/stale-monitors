#!/bin/bash
# Behavioral batteries, one array task per run_id (sm.eval.battery).
# Task index 0 is the base itself (run_id "base"); index i+1 is manifest run i.
# Config: BATTERY_CONFIG (default configs/battery_<base>.yaml: pinned revision, items dir, dtype, batch size);
#   per-battery decoding comes from configs/batteries.yaml through the item MANIFEST.
# Guards: GPU pre-flight (exit 2), a passed V4 file (exit 2), the config revision equals the bases.yaml pin.
# kind lora: base + results/adapters/<base>/<run_id>, adapter unmerged (requires train_meta status ok, else exit 5)
# kind base: the base under this run_id (condition 6 base-vs-base replicate). The prereg asks for
#            two nodes. jobs/submit_bank.sh submits these tasks as a separate job named sm-bvb-<base>
#            that waits (afterany) for the main array. When the main array's base task (run_id base)
#            starts, it sets ExcNodeList=<its node> on that still-pending job (scontrol update works on
#            pending jobs, not running ones), so the replicate is scheduled elsewhere and the chain's
#            dependencies on its job id stay valid. A replicate that still lands on the base's node
#            exits 6 (the hostname is in every battery sidecar; aggregate also refuses a same-node pair):
#            resubmit it by hand with --exclude=<node> before the later stages run.
# kind model: the public derivative named by the run's `model` (outputs routed to
#             results/batteries/<base>/<run_id> through a symlink, see below)
# kind ablation: skipped (no in-memory ablation in battery generation; condition 5
#             ablation is behaviorally measured by sm.interp.causal)
# Idempotent per battery file (sm.eval.battery skips finished files and resumes .partial).
#
# Submit: sbatch --array=<indices>%4 --export=ALL,BANK=configs/bank_<base>.yaml jobs/battery.sbatch
# Optional env: BATTERY_BACKEND=hf|vllm (default hf; vllm needs --allow-vllm-measured in
#               BATTERY_EXTRA_ARGS for measured batteries, and is a logged deviation). Under vllm every
#               task gets --vllm-lora: the adapter stays separate (LoRARequest) and base and derivative
#               are served from the same original checkpoint, so no bf16 merge rounds LoRA updates away.
#               VLLM_EXPORT=1 instead exports base and derivative through the text-only path in fp32
#               (--merge-dir, --vllm-dtype fp32). ITEMS_DIR (default data/prompts),
#               BATTERY_EXTRA_ARGS, FORCE=1, ALLOW_FAILED_V4=1 (descriptive-only), SM_PROJ.
# H5 sampled battery karne_ladders_sens (fixed 100-ladder subset, 300 items, 8 samples at temperature 1.0):
#   since 2026-10-07 in the default battery set (configs/batteries.yaml on_demand: false; prereg v0.3 change 37),
#   so a plain run generates it for every unit. SENS=1 re-runs only that battery: for the base and the runs in the
#   J2 flag file (FLAG_FILE, default results/judge/<base>/_flags/j2_flagged.txt; other tasks exit 0), or with
#   SENS_ALL=1 for every unit (a unit whose manifest predates the flip). Then J1-judge those runs again
#   (judge.sbatch reuses existing labels).
# H5 strata battery karne_ladders_sampled_base (FIDELITY_REVIEW N7, prereg v0.3 change 54): every ladder (1,941 rungs,
#   8 samples at temperature 1.0), in the default set of the base task only (MANIFEST on_demand + base_only;
#   sm.eval.battery.discover_batteries with run_id base), so the base task generates about 15,528 extra completions
#   (about 1 to 2 H200-hours more than a derivative task); derivatives never run it unless named with --batteries.
#SBATCH --job-name=sm-battery
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/battery-%A_%a.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/battery-%A_%a.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-battery start on $(hostname) ==="
sm_setup_env
sm_gpu_preflight
sm_load_task battery

if [ "$TASK_ACTIVE" != "1" ]; then
  sm_log "SKIP: run $TASK_RUN_ID (kind $TASK_KIND) has no battery stage"
  exit 0
fi

if [ -n "${SENS:-}" ] && [ -z "${SENS_ALL:-}" ]; then
  FLAG_FILE="${FLAG_FILE:-results/judge/$BASE_KEY/_flags/j2_flagged.txt}"
  if [ "$TASK_RUN_ID" != "base" ]; then
    [ -f "$FLAG_FILE" ] || sm_die 5 "SENS=1: flag file $FLAG_FILE not found (run judge.sbatch pass 1 first)"
    if ! grep -v '^#' "$FLAG_FILE" | grep -qx "$TASK_RUN_ID"; then
      sm_log "SKIP: SENS=1 and run $TASK_RUN_ID is not flagged"
      exit 0
    fi
  fi
fi

sm_require_v4 "$BASE_KEY"
BCFG="${BATTERY_CONFIG:-configs/battery_${BASE_KEY}.yaml}"
[ -f "$BCFG" ] || sm_die 2 "battery config $BCFG not found"
sm_check_config_rev "$BCFG" "$BASE_KEY"
ITEMS_DIR="${ITEMS_DIR:-data/prompts}"
[ -f "$ITEMS_DIR/MANIFEST.json" ] || sm_die 2 "battery items missing: $ITEMS_DIR/MANIFEST.json (run sm.data.build_prompt_sets on a CPU node)"
OUT=results
# Restricted outputs (harmful batteries) are created 0700 by sm.eval.battery; keep the rest group-private too.
umask 027

# The base (and every adapter on it) loads with the config's pinned revision; an external model
# (kind model) does not take the config, so the base pin never leaks onto it.
args=(--config "$BCFG" --base "$BASE_KEY" --run-id "$TASK_RUN_ID" --out "$OUT" --items-dir "$ITEMS_DIR" --backend "${BATTERY_BACKEND:-hf}")
case "$TASK_KIND" in
  lora)
    ADAPTER="results/adapters/$BASE_KEY/$TASK_RUN_ID"
    sm_py train-ok "$ADAPTER/train_meta.json" || sm_die 5 "no finished adapter at $ADAPTER (lora stage missing or failed)"
    args+=(--adapter "$ADAPTER")
    sm_preflight_snapshot "$BASE_KEY"
    ;;
  base)
    sm_preflight_snapshot "$BASE_KEY"
    if [ "$TASK_RUN_ID" = "base" ] && command -v squeue >/dev/null 2>&1 && command -v scontrol >/dev/null 2>&1; then
      # Keep the pending base-vs-base replicate (job name sm-bvb-<base>) off this node.
      HERE="$(hostname -s 2>/dev/null || hostname)"
      for jid in $(squeue -u "${USER:-$(id -un)}" -h -t PD -n "sm-bvb-$BASE_KEY" -o %A 2>/dev/null | sort -u); do
        if scontrol update JobId="$jid" ExcNodeList="$HERE" 2>/dev/null; then
          sm_log "base-vs-base replicate job $jid will avoid $HERE (ExcNodeList set while pending)"
        else
          sm_log "WARNING: could not set ExcNodeList=$HERE on pending replicate job $jid; it checks its node itself"
        fi
      done
    fi
    if [ "$TASK_RUN_ID" != "base" ]; then
      BSLUG0="$("$PYTHON" -c 'import sys; from sm.eval.battery import path_slug; print(path_slug(sys.argv[1]))' "$BASE_KEY")"
      BASE_HOST="$("$PYTHON" - "$OUT" "$BSLUG0" <<'PYEOF'
import json, sys
from pathlib import Path
out, slug = Path(sys.argv[1]), sys.argv[2]
hosts = set()
for root in (out / "batteries", out / "restricted" / "batteries"):
    for p in (root / slug / "base").glob("*.meta.json"):
        try:
            hosts.add((json.loads(p.read_text()).get("meta") or {}).get("hostname"))
        except (OSError, ValueError):
            pass
print(",".join(sorted(h for h in hosts if h)))
PYEOF
)"
      [ -n "$BASE_HOST" ] || sm_die 5 "base-vs-base replicate $TASK_RUN_ID: the base batteries (and their hostname) are not there yet"
      if [[ ",$BASE_HOST," == *",$(hostname),"* ]] || [[ ",$BASE_HOST," == *",$(hostname -s),"* ]]; then
        # Slurm rejects ExcNodeList updates on a running job, so no in-job requeue: fail loudly instead.
        sm_die 6 "base-vs-base replicate $TASK_RUN_ID landed on the base's node ($BASE_HOST); the prereg noise floor needs two nodes. Resubmit: sbatch --exclude=${BASE_HOST%%.*} --array=$TASK_INDEX --export=ALL,BANK=$BANK jobs/battery.sbatch"
      fi
      sm_log "base-vs-base replicate on $(hostname); base batteries were generated on $BASE_HOST"
    fi
    ;;
  model)
    [ -n "$TASK_MODEL" ] || sm_die 2 "run $TASK_RUN_ID: kind model needs 'model' in the manifest"
    [ -n "$TASK_MODEL_REVISION" ] || [ "${ALLOW_UNPINNED:-0}" = "1" ] || sm_die 2 "run $TASK_RUN_ID: external model $TASK_MODEL has no model_revision (external_revision) pin"
    sm_preflight_snapshot "$TASK_MODEL" "$TASK_MODEL_REVISION"
    # sm.eval.battery names the output directory after --base. Point the model's
    # directory at the base's run directory so outputs land where judge/aggregate
    # look (results/batteries/<base>/<run_id>); the sidecar meta still records the model id.
    MSLUG="$("$PYTHON" -c 'import sys; from sm.eval.battery import path_slug; print(path_slug(sys.argv[1]))' "$TASK_MODEL")"
    BSLUG="$("$PYTHON" -c 'import sys; from sm.eval.battery import path_slug; print(path_slug(sys.argv[1]))' "$BASE_KEY")"
    for root in "$OUT/batteries" "$OUT/restricted/batteries"; do
      mkdir -p "$root/$BSLUG/$TASK_RUN_ID" "$root/$MSLUG"
      [ "$root" = "$OUT/restricted/batteries" ] && chmod 700 "$OUT/restricted" "$root" "$root/$BSLUG" "$root/$BSLUG/$TASK_RUN_ID" "$root/$MSLUG"
      link="$root/$MSLUG/$TASK_RUN_ID"
      if [ ! -L "$link" ]; then
        [ -e "$link" ] && sm_die 2 "$link exists and is not a symlink; move it aside"
        ln -s "../$BSLUG/$TASK_RUN_ID" "$link"
      fi
    done
    args=(--base "$TASK_MODEL" --run-id "$TASK_RUN_ID" --out "$OUT" --items-dir "$ITEMS_DIR" --backend "${BATTERY_BACKEND:-hf}")
    [ -n "$TASK_MODEL_REVISION" ] && args+=(--revision "$TASK_MODEL_REVISION")
    ;;
  *)
    sm_die 2 "unexpected kind $TASK_KIND for the battery stage"
    ;;
esac

if [ "${BATTERY_BACKEND:-hf}" = "vllm" ]; then
  if [ "${VLLM_EXPORT:-0}" = "1" ]; then
    args+=(--merge-dir "$SM_SCRATCH/merged/${SLURM_JOB_ID:-local}" --vllm-dtype fp32)
  else
    args+=(--vllm-lora)
  fi
fi
[ -n "${FORCE:-}" ] && args+=(--force)
[ -n "${SENS:-}" ] && args+=(--batteries karne_ladders_sens)

sm_require_frozen "$BASE_KEY"
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.eval.battery "${args[@]}" ${BATTERY_EXTRA_ARGS:-} || rc=$?
sm_log "=== sm-battery done rc=$rc run=$TASK_RUN_ID ==="
exit "$rc"
