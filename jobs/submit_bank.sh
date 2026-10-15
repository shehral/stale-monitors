#!/bin/bash
# Submit one base's bank as a dependency chain on Explorer (run on the login node, from anywhere; it cds
# to SM_PROJ). Light work only: it reads the manifest, the configs and two small JSON ledgers with the env
# python and calls sbatch/squeue.
#
#   jobs/submit_bank.sh --bank configs/bank_<base>.yaml [options]
#
# Chain (each stage waits for the previous one with afterany, so a walltime kill does not strand it):
#   [smoke_cpu -> smoke] -> base_card -> causal -> lora[array] -> organism -> battery[array] (+ battery base-vs-base
#   replicate) -> capability[array] -> judge -> drift[array] -> monitor_scores[array] -> aggregate
# The smoke stage is two jobs chained with afterok (IdleBot: pytest ran 55+ minutes inside the gpu-short smoke of
# chain 3, job 10899610, and hit the 1 h cap): smoke_cpu.sbatch (short, 16 CPUs: pytest and the 11 CLI --smoke
# runs, node-local temp dirs) -> smoke.sbatch (gpu-short: GPU checks only; it exports SMOKE_CPU_JOB=<smoke_cpu id>
# and merges that half into results/smoke/<base>/<smoke id>.json). The summary lists smoke_cpu= and smoke=;
# the smoke id (the GPU job) is the afterok gate of every later stage and the id acceptance takes
# (python -m sm.accept --stage smoke --job <smoke id>).
# The base_card stage is three jobs chained with afterok (IdleBot: no CPU-heavy phase inside a GPU job):
#   base_card.sbatch (gpu, capture) -> base_stats.sbatch (short, 32 CPUs, statistics) ->
#   base_ablate.sbatch (gpu + J1, L* sweep, persona vectors, thresholds, the card) -> causal (afterok)
# The summary line lists them as base_card=, base_stats=, base_ablate=; acceptance of the base_card stage
# (python -m sm.accept --stage base_card --job <id>) takes the base_ablate id (its log has the frozen: line).
# organism (H4) = jobs/base_card.sbatch with ORGANISM=1: the condition-4 derivatives' own refusal directions,
# extracted before drift so every drift JSON carries the organism blocks aggregate builds the H4 table from.
# The base-vs-base replicate is its own job, named sm-bvb-<base>; the main array's base task sets its
# ExcNodeList while it is still pending (prereg: two nodes).
# aggregate here runs over this base only and writes results/tables/subsets/<base>/ (never the pooled,
# confirmatory results/tables/hypotheses.json, which only `AGG_BASES=all` writes).
# Gates on top of afterany, all with --kill-on-invalid-dep=yes so a failed gate cancels the rest instead of
# leaving it pending forever:
#   afterok:<smoke>   on every later stage (the GPU smoke, itself afterok on smoke_cpu; smoke is on by default;
#                     --no-smoke drops both jobs)
#   afterok:<causal>  on every stage after the V4 check (a failed V4 exits 2)
# Every stage after causal also refuses to run in-job without a passed V4 file (jobs/common.sh sm_require_v4),
# so a hand-submitted or resubmitted stage is gated too.
#
# Concurrency: at most --max-concurrent (default 4, the account's GPU QOS) of this project's jobs run at
# once. Stages are serial; arrays get %K with K = max-concurrent minus the user's other RUNNING jobs
# (minimum 1). When sm-* jobs from an earlier submission are still queued, the new chain waits for the
# newest of them (afterany) unless --no-auto-chain.
#
# Guardrails (skill G4/G5): refuses when the failure-streak ledger has auto_rollback_triggered, or today's
# compute budget in .compute_budget.json is used up, unless --human-approved is given (a human decision,
# never an agent's). It also refuses before queueing anything when a stage config is missing, a config's
# revision differs from the configs/bases.yaml pin, the judge pins disagree, the base snapshot at the
# pinned commit is not cached, or the data a queued stage reads is not built (data/prompts/MANIFEST.json for
# base_card onward, data/bank/MANIFEST.json for lora); in --dry-run those are printed as warnings.
#
# Options:
#   --bank PATH            bank manifest, relative to SM_PROJ (required)
#   --stages LIST          comma list from smoke,base_card,causal,lora,organism,battery,capability,judge,
#                          drift,monitor_scores,aggregate (default: all)
#   --max-concurrent N     default 4
#   --after JOBID          make the first submitted stage wait for JOBID (afterany)
#   --no-auto-chain        do not wait for earlier sm-* jobs
#   --no-smoke             drop the smoke stage and its afterok gate
#   --force                export FORCE=1 to every stage (recompute finished outputs)
#   --human-approved       override the budget and rollback guards
#   --dry-run              print the sbatch commands; submit nothing
#   --expect-commit SHA    refuse unless the cluster checkout's HEAD is SHA (pass the laptop's
#                          `git rev-parse HEAD` after the rsync; review A15: the rsync of code/ configs/ jobs/
#                          is the first step, and a stale or missing jobs/common.sh must stop the submission)
# Environment: SM_PROJ (default /projects/interpretability/shehral.m/monitors),
#   SM_SBATCH / SM_SQUEUE (command overrides, for testing), plus any stage env
#   (BATTERY_BACKEND, AGG_BASES, ...) which --export=ALL forwards.
#   SM_QOS_SUBMIT_LIMIT (default 8, the gpu QOS MaxSubmitJobsPerUser; array tasks count one each) and
#   SM_QOS_PARTITIONS (default gpu,gpu-short): every array on those partitions is cut into chunks of at most
#   LIMIT minus the user's jobs already queued there (squeue at submission time; a dry run assumes 0), chunk k+1
#   afterany on chunk k, and the next stage afterany on every chunk. SLURM counts pending (dependency-held)
#   tasks too, so a live submission stops with rc 4 and the remaining indices when a chunk no longer fits.
#   SM_SBATCH_RETRY_SLEEP (default 20 s between sbatch attempts).
#   BATTERY_BACKEND=vllm is refused here unless BATTERY_EXTRA_ARGS carries --allow-vllm-measured (the battery
#   CLI refuses it on the node otherwise; job 10918812 lost seven starts to that).
set -eo pipefail

usage() { sed -n '2,/^set -eo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

ALL_STAGES="smoke,base_card,causal,lora,organism,battery,capability,judge,drift,monitor_scores,aggregate"
BANK=""
STAGES="$ALL_STAGES"
MAX_CONC=4
AFTER=""
AUTO_CHAIN=1
FORCE_ALL=""
APPROVED=0
DRY=0
EXPECT_COMMIT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bank) BANK="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --max-concurrent) MAX_CONC="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --no-auto-chain) AUTO_CHAIN=0; shift ;;
    --no-smoke) NO_SMOKE=1; shift ;;
    --force) FORCE_ALL=1; shift ;;
    --human-approved) APPROVED=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --expect-commit) EXPECT_COMMIT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option $1" >&2; usage 2 ;;
  esac
done
[ -n "$BANK" ] || { echo "--bank is required" >&2; usage 2; }
[[ "$MAX_CONC" =~ ^[1-9][0-9]*$ ]] || { echo "--max-concurrent must be a positive integer" >&2; exit 2; }
if [ "$MAX_CONC" -gt 4 ]; then echo "--max-concurrent capped at 4 (account QOS)" >&2; MAX_CONC=4; fi
if [ "${NO_SMOKE:-0}" = "1" ]; then
  STAGES="$(echo ",$STAGES," | sed 's/,smoke,/,/; s/^,//; s/,$//')"
fi

SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# Review A15: the deployment (commit, then rsync code/ configs/ jobs/ tests/) comes first. Without
# $SM_PROJ/jobs/common.sh every stage would fail at its first line, and with a stale one it would run old code.
[ -f "$SM_PROJ/jobs/common.sh" ] || { echo "$SM_PROJ/jobs/common.sh is missing: rsync code/ configs/ jobs/ first (jobs/README.md, one-time setup 1)" >&2; exit 2; }
if [ -n "$EXPECT_COMMIT" ]; then
  have_commit="$(git -C "$SM_PROJ" rev-parse HEAD 2>/dev/null || true)"
  [ "$have_commit" = "$EXPECT_COMMIT" ] || { echo "cluster checkout is at ${have_commit:-no git commit}, expected $EXPECT_COMMIT: sync first" >&2; exit 2; }
fi
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
cd "$SM_PROJ"
[ -f "$BANK" ] || { echo "bank manifest $BANK not found under $SM_PROJ" >&2; exit 2; }
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
export PYTHONPATH="$SM_PROJ/code${PYTHONPATH:+:$PYTHONPATH}"
export SM_BASES_YAML="${SM_BASES_YAML:-$SM_PROJ/configs/bases.yaml}"
export HF_HOME="$SM_HF_HOME"
SBATCH="${SM_SBATCH:-sbatch}"
SQUEUE="${SM_SQUEUE:-squeue}"

for s in ${STAGES//,/ }; do
  [[ ",$ALL_STAGES," == *",$s,"* ]] || { echo "unknown stage $s" >&2; exit 2; }
done
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# Measured batteries run on the HF backend (rubric BT-2); vLLM only as a recorded deviation. Refuse here, before
# anything is queued, instead of letting every battery task exit 2 on the node (job 10918812).
if has_stage battery && [ "${BATTERY_BACKEND:-hf}" = "vllm" ] && [[ " ${BATTERY_EXTRA_ARGS:-} " != *" --allow-vllm-measured "* ]]; then
  echo "REFUSED: BATTERY_BACKEND=vllm without --allow-vllm-measured in BATTERY_EXTRA_ARGS: sm.eval.battery refuses vLLM for" >&2
  echo "  measured batteries (rubric BT-2). Unset BATTERY_BACKEND (HF backend), or set" >&2
  echo "  BATTERY_EXTRA_ARGS='--allow-vllm-measured' and log the deviation." >&2
  exit 2
fi
QOS_LIMIT="${SM_QOS_SUBMIT_LIMIT:-8}"
QOS_PARTITIONS="${SM_QOS_PARTITIONS:-gpu,gpu-short}"
[[ "$QOS_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "SM_QOS_SUBMIT_LIMIT must be a positive integer" >&2; exit 2; }
RETRY_SLEEP="${SM_SBATCH_RETRY_SLEEP:-20}"

BASE_KEY="$(sm_py base "$BANK")"
echo "bank: $BANK  base: $BASE_KEY  stages: $STAGES"
sm_py runs "$BANK" lora | awk -F'\t' '{printf "  run %-3s %-30s %-9s lora:%s\n", $1, $2, $3, $4}'

# --- Compute estimate (review addendum A12) -----------------------------------
# Planning guesses in H200-hours, not measurements (jobs/README.md "Planning numbers"); replace them with
# sacct after the first base. Per-run stages are multiplied by the active task count of that stage.
# Base sizes up to 12B use the "small" column; 27B/31B multiply it by SM_EST_LARGE_FACTOR (2.5).
sm_estimate() {  # sm_estimate <backend hf|vllm>
  local backend="$1" factor=1 total=0 s n per line
  local -a items=()
  case "$BASE_KEY" in *27b*|*31b*) factor="${SM_EST_LARGE_FACTOR:-2.5}" ;; esac
  for s in ${STAGES//,/ }; do
    if [ "$s" = "base_card" ]; then items+=(base_card base_stats base_ablate)
    elif [ "$s" = "smoke" ]; then items+=(smoke_cpu smoke)
    else items+=("$s"); fi
  done
  for s in "${items[@]}"; do
    case "$s" in
      smoke_cpu) per=0; n=1 ;;       # CPU job on short (16 cores): pytest + the 11 CLI smokes; 0 GPU-hours
      smoke) per=0.3; n=1 ;;          # GPU checks only (real base, J1/J2, LoRA + hooks)
      base_card) per=0.3; n=1 ;;      # capture only (GPU minutes plus the model load)
      base_stats) per=0; n=1 ;;       # CPU job on short (32 cores, about 1 h wall); 0 GPU-hours
      base_ablate) per=2.5; n=1 ;;    # J1 start + (position, layer) screen, 13 judged runs at 512 tokens, persona (10 x T=1
                                      # sampling, steering layer choice); about 1.3 to 1.8 H200-h measured for 9B, 2.5 planned
      causal) per=1; n=1 ;;           # V4 at 512 tokens: 24 conditions x 128 prompts, about 1 H200-h for 9B
      organism) per=0.3; n="$(sm_py organism-runs "$BANK" 2>/dev/null | wc -l | tr -d ' ')" ;;
      judge) per=3; n=1 ;;
      capability|aggregate) per=0; n=1 ;;
      lora) per=0.5 ;;
      battery) if [ "$backend" = "vllm" ]; then per=0.35; else per=1.1; fi ;;  # incl. karne_ladders_sens (300 x 8 samples)
      drift) per=0.5 ;;
      monitor_scores) per=0.3 ;;
      *) per=0; n=1 ;;
    esac
    case "$s" in
      lora|battery|drift|monitor_scores) n="$(sm_py runs "$BANK" "$s" 2>/dev/null | awk -F'\t' '$4=="active"' | wc -l | tr -d ' ')" ;;
    esac
    line="$(awk -v p="$per" -v n="$n" -v f="$factor" 'BEGIN { printf "%.1f", p * n * f }')"
    printf '  estimate %-15s %3s x %-5s x %-4s = %6s H200-h\n' "$s" "$n" "$per" "$factor" "$line"
    total="$(awk -v a="$total" -v b="$line" 'BEGIN { printf "%.1f", a + b }')"
  done
  echo "  estimate total ($backend backend for batteries): about $total H200-h (planning guess)"
}
echo "compute estimate before submitting:"
sm_estimate "${BATTERY_BACKEND:-hf}"

# --- Guardrails ----------------------------------------------------------------
LEDGER="${SM_BUDGET_LEDGER:-$SM_PROJ/.compute_budget.json}"
STREAK="${SM_STREAK_LEDGER:-$SM_PROJ/.failure_streak.json}"
if [ "$(sm_py rollback "$STREAK")" = "1" ]; then
  echo "REFUSED: $STREAK has auto_rollback_triggered=true; a human must reset it (or pass --human-approved)." >&2
  [ "$APPROVED" = "1" ] || exit 3
fi
read -r used cap remaining <<<"$(sm_py budget "$LEDGER")"
if [ "$used" = "NA" ]; then
  echo "budget: no ledger at $LEDGER (create it from the skill's assets/compute_budget.json)"
else
  echo "budget today: used ${used} of ${cap} H200-h, remaining ${remaining}"
  if awk -v r="$remaining" 'BEGIN { exit !(r <= 0) }'; then
    echo "REFUSED: today's H200-hour budget is used up (pass --human-approved to override)." >&2
    [ "$APPROVED" = "1" ] || exit 3
  fi
fi

# Config pre-check: every queued stage's per-base config must exist and pin the bases.yaml revision.
problems=()
need_cfg() {  # need_cfg <path>  (must exist and carry the pinned revision)
  if [ ! -f "$1" ]; then problems+=("missing $1"); return; fi
  local msg
  msg="$(sm_py config-rev "$1" "$BASE_KEY" 2>&1)" || problems+=("$msg")
}
PIN="$(sm_py pin "$BASE_KEY")"
[ -n "$PIN" ] || problems+=("$BASE_KEY has no revision pin in configs/bases.yaml")
msg="$(sm_py config-rev "$BANK" "$BASE_KEY" 2>&1)" || problems+=("$msg")
if has_stage base_card || has_stage smoke || has_stage organism; then need_cfg "${BASE_CARD_CONFIG:-configs/base_card_${BASE_KEY}.yaml}"; fi
if has_stage causal; then need_cfg "${CAUSAL_CONFIG:-configs/causal_${BASE_KEY}.yaml}"; fi
if has_stage battery; then need_cfg "${BATTERY_CONFIG:-configs/battery_${BASE_KEY}.yaml}"; fi
if has_stage drift; then need_cfg "${DRIFT_TEMPLATE:-configs/drift_${BASE_KEY}.yaml}"; fi
if has_stage monitor_scores; then need_cfg "${MONITOR_SCORES_TEMPLATE:-configs/monitor_scores_${BASE_KEY}.yaml}"; fi
if has_stage judge || has_stage base_card || has_stage causal; then
  for j in J1 J2; do
    msg="$(sm_py judge-pin "$j" "${JUDGE_CONFIG:-configs/judge.yaml}" 2>&1)" || problems+=("$msg")
  done
fi
# Data the queued stages read (built on a CPU node, jobs/README.md one-time setup step 3).
data_problems=()
if has_stage base_card || has_stage causal || has_stage organism || has_stage battery || has_stage drift || has_stage monitor_scores; then
  [ -f data/prompts/MANIFEST.json ] || data_problems+=("missing data/prompts/MANIFEST.json (run sm.data.build_prompt_sets)")
fi
if has_stage lora; then
  [ -f data/bank/MANIFEST.json ] || data_problems+=("missing data/bank/MANIFEST.json (run sm.data.build_bank_datasets)")
fi
if [ "${#data_problems[@]}" -gt 0 ]; then
  if [ "$DRY" = "1" ]; then
    printf 'WARNING (dry run): %s\n' "${data_problems[@]}"
  else
    problems+=("${data_problems[@]}")
  fi
fi
if [ "${#problems[@]}" -gt 0 ]; then
  echo "REFUSED: configuration problems:" >&2
  printf '  %s\n' "${problems[@]}" >&2
  exit 2
fi
echo "revision: $BASE_KEY pinned at $PIN in configs/bases.yaml and every queued stage config"

# Snapshot pre-check (file test only) so a missing base fails here, not after queueing.
if ! sm_py snapshot "$BASE_KEY" "$PIN" >/dev/null; then
  echo "REFUSED: base snapshot $BASE_KEY@$PIN not cached under $HF_HOME; run jobs/download.sbatch first." >&2
  exit 2
fi
mkdir -p "$SM_LOGDIR"

# --- Concurrency -----------------------------------------------------------------
other_running=0
last_sm=""
if command -v "$SQUEUE" >/dev/null 2>&1; then
  # -r expands array tasks; %A is the array master id (or the job id), %j name, %T state.
  while read -r jid name state; do
    [ -n "$jid" ] || continue
    if [[ "$name" == sm-* ]]; then
      if [ -z "$last_sm" ] || [ "$jid" -gt "$last_sm" ] 2>/dev/null; then last_sm="$jid"; fi
    elif [ "$state" = "RUNNING" ]; then
      other_running=$((other_running + 1))
    fi
  done < <("$SQUEUE" -u "$USER" -h -r -o "%A %j %T" 2>/dev/null || true)
fi
THROTTLE=$(( MAX_CONC - other_running ))
[ "$THROTTLE" -ge 1 ] || THROTTLE=1
echo "concurrency: max $MAX_CONC, other running jobs $other_running, array throttle %$THROTTLE"
if [ -z "$AFTER" ] && [ "$AUTO_CHAIN" = "1" ] && [ -n "$last_sm" ]; then
  AFTER="$last_sm"
  echo "auto-chain: earlier sm-* job $last_sm is queued; the new chain waits for it (afterany). --no-auto-chain to disable."
fi

# --- Submit ----------------------------------------------------------------------
EXPORTS="ALL,BANK=$BANK,SM_PROJ=$SM_PROJ"
[ -n "$FORCE_ALL" ] && EXPORTS="$EXPORTS,FORCE=1"
SMOKE_ID=""
CAUSAL_ID=""
PREV="$AFTER"
FAKE_ID=900000
declare -a SUMMARY=()

# Job name of an sbatch argument list: --job-name=/-J on the command line, else the script's #SBATCH line.
job_name_of() {
  local a prev="" name="" script=""
  for a in "$@"; do
    case "$a" in
      --job-name=*) name="${a#--job-name=}" ;;
      *) if [ "$prev" = "-J" ] || [ "$prev" = "--job-name" ]; then name="$a"; fi ;;
    esac
    prev="$a"; script="$a"
  done
  if [ -z "$name" ] && [ -f "$script" ]; then
    name="$(awk '/^#SBATCH/ { for (i = 2; i <= NF; i++) { if ($i ~ /^--job-name=/) { sub(/^--job-name=/, "", $i); print $i; exit }
                                                       if ($i == "-J" && i < NF) { print $(i + 1); exit } } }' "$script")"
  fi
  printf '%s\n' "$name"
}
named_ids() {  # named_ids <job name>: the user's queued job ids with that name (array master ids), sorted
  [ -n "$1" ] || return 0
  { "$SQUEUE" -u "$USER" -h -n "$1" -o %A 2>/dev/null || true; } | awk 'NF { print $1 }' | sort -u
}

# sbatch with retries: slurmctld can refuse a submission transiently (socket timeouts); a lost submission used to
# abort the chain half-queued (chain 3, 2026-10-07). Three attempts, RETRY_SLEEP (20 s) apart; the partial summary
# is printed on abort. A "failed" attempt can still have been accepted (a duplicate smoke_cpu was queued that way),
# so after each failure the queue is checked for a job of the same name that was not there before the first
# attempt (a snapshot, stricter than a time window: an older job of the same name is never taken), and its id is
# reused instead of submitting again.
sbatch_retry() {
  local attempt out name before new
  name="$(job_name_of "$@")"
  before="$(named_ids "$name")"
  for attempt in 1 2 3; do
    if out="$("$SBATCH" --parsable "$@" 2>&1)"; then printf '%s\n' "$out"; return 0; fi
    echo "sbatch attempt $attempt failed: $out" >&2
    sleep "$RETRY_SLEEP"
    if [ -n "$name" ]; then
      new="$(comm -13 <(printf '%s\n' "$before" | sed '/^$/d') <(named_ids "$name"))"
      if [ -n "$new" ]; then
        if [ "$(printf '%s\n' "$new" | wc -l | tr -d ' ')" -gt 1 ]; then
          echo "sbatch: several new '$name' jobs appeared ($(echo $new)); not guessing, cancel the extras by hand" >&2
          return 1
        fi
        echo "sbatch attempt $attempt reported a failure but SLURM accepted job $new ($name); reusing it" >&2
        printf '%s\n' "$new"
        return 0
      fi
    fi
  done
  return 1
}
SUMMARY=()
on_abort() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "${#SUMMARY[@]}" -gt 0 ]; then
    echo "ABORTED (rc=$rc) after queueing: ${SUMMARY[*]} -- cancel or complete the chain by hand" >&2
  fi
}
trap on_abort EXIT

submit() {  # submit <stage> <script> [extra sbatch args...]; sets PREV
  local stage="$1" script="$2"
  shift 2
  local -a deps=() args=()
  local gates=" $SMOKE_ID $CAUSAL_ID "
  # PREV_DEP=afterok inside the base card chain (capture -> stats -> ablate -> causal), afterany elsewhere.
  if [ -n "$PREV" ] && [[ "$gates" != *" $PREV "* ]]; then deps+=("${PREV_DEP:-afterany}:$PREV"); fi
  if [ -n "$SMOKE_ID" ] && [ "$stage" != "smoke" ]; then deps+=("afterok:$SMOKE_ID"); fi
  if [ -n "$CAUSAL_ID" ]; then deps+=("afterok:$CAUSAL_ID"); fi
  if [ "${#deps[@]}" -gt 0 ]; then
    local joined
    joined="$(IFS=,; echo "${deps[*]}")"
    args+=(--dependency="$joined" --kill-on-invalid-dep=yes)
  fi
  args+=(--export="$EXPORTS${STAGE_EXPORTS:-}" "$@" "jobs/$script")
  local jid
  if [ "$DRY" = "1" ]; then
    FAKE_ID=$((FAKE_ID + 1))
    jid="$FAKE_ID"
    echo "DRY: $SBATCH --parsable ${args[*]}"
  else
    jid="$(sbatch_retry "${args[@]}")"
    jid="${jid%%;*}"
  fi
  SUMMARY+=("$stage=$jid")
  PREV="$jid"
}

script_partition() { sed -n 's/^#SBATCH  *--partition=\([^ ]*\).*/\1/p' "jobs/$1" | head -1; }
qos_queued() {  # the user's jobs (array tasks expanded) already queued on the QOS partitions; 0 in a dry run
  if [ "$DRY" = "1" ]; then echo 0; return; fi
  { "$SQUEUE" -u "$USER" -h -r -p "$QOS_PARTITIONS" -o %i 2>/dev/null || true; } | awk 'NF' | wc -l | tr -d ' '
}

# submit_chunked <stage> <script> <indices> <throttle or ""> [extra sbatch args...]
# Arrays on the QOS partitions are cut to fit MaxSubmitJobsPerUser (see the header); PREV ends as every chunk id
# joined with ':' so the next stage depends afterany:id1:id2 on all of them.
submit_chunked() {
  local stage="$1" script="$2" idx="$3" thr="$4"
  shift 4
  local -a tasks=()
  IFS=, read -r -a tasks <<<"$idx"
  local total="${#tasks[@]}" pos=0 k=0 n queued room chunk name ids="" part
  part="$(script_partition "$script")"
  if [[ ",$QOS_PARTITIONS," != *",$part,"* ]]; then
    submit "$stage" "$script" --array="${idx}${thr:+%$thr}" "$@"
    return 0
  fi
  while [ "$pos" -lt "$total" ]; do
    queued="$(qos_queued)"
    room=$(( QOS_LIMIT - queued ))
    if [ "$room" -lt 1 ]; then
      chunk="$(IFS=,; echo "${tasks[*]:pos}")"
      echo "QOS STOP: $queued jobs already queued on $QOS_PARTITIONS (MaxSubmitJobsPerUser $QOS_LIMIT counts pending tasks)." >&2
      echo "  stage $stage: indices $chunk were NOT submitted; nor was any later stage of --stages $STAGES." >&2
      echo "  When the queue drains, submit the rest (e.g. --stages $stage,... --after ${ids:-$PREV})." >&2
      exit 4
    fi
    n=$(( total - pos < room ? total - pos : room ))
    chunk="$(IFS=,; echo "${tasks[*]:pos:n}")"
    k=$((k + 1))
    if [ "$k" -eq 1 ] && [ "$n" -eq "$total" ]; then name="$stage"; else name="${stage}_c$k"; fi
    [ "$k" -eq 1 ] && [ "$n" -lt "$total" ] && echo "QOS: stage $stage has $total tasks; chunks of at most $room (limit $QOS_LIMIT, queued $queued), chained afterany"
    submit "$name" "$script" --array="${chunk}${thr:+%$thr}" "$@"
    ids="${ids:+$ids:}$PREV"
    pos=$(( pos + n ))
  done
  PREV="$ids"
}

submit_array() {  # submit_array <stage>: array over the stage's active indices, throttled
  local stage="$1" idx
  idx="$(sm_py indices "$BANK" "$stage")"
  if [ -z "$idx" ]; then echo "stage $stage: no active tasks; skipped"; return 0; fi
  submit_chunked "$stage" "$stage.sbatch" "$idx" "$THROTTLE"
}

SMOKE_CPU_ID=""
if has_stage smoke; then
  # CPU half first (short partition), then the GPU half afterok on it, told which CPU half to merge.
  submit smoke_cpu smoke_cpu.sbatch
  SMOKE_CPU_ID="$PREV"
  PREV_DEP=afterok
  STAGE_EXPORTS=",SMOKE_CPU_JOB=$SMOKE_CPU_ID"
  submit smoke smoke.sbatch
  STAGE_EXPORTS=""
  PREV_DEP=""
  SMOKE_ID="$PREV"
fi
BASE_CARD_ABLATE_ID=""
if has_stage base_card; then
  submit base_card base_card.sbatch
  PREV_DEP=afterok
  submit base_stats base_stats.sbatch
  submit base_ablate base_ablate.sbatch
  BASE_CARD_ABLATE_ID="$PREV"
fi
if has_stage causal; then submit causal causal.sbatch; CAUSAL_ID="$PREV"; fi
PREV_DEP=""
if has_stage lora; then submit_array lora; fi
if has_stage organism; then
  if [ -n "$(sm_py organism-runs "$BANK")" ]; then
    STAGE_EXPORTS=",ORGANISM=1"
    submit organism base_card.sbatch --job-name="sm-organism"
    STAGE_EXPORTS=""
  else
    echo "stage organism: no condition-4 runs in $BANK; skipped"
  fi
fi
if has_stage battery; then
  idx="$(sm_py indices "$BANK" battery)"
  # Base-vs-base replicates run after the main array so they can avoid the base's node (prereg: two nodes).
  bvb="$(sm_py bvb-indices "$BANK")"
  if [ -n "$bvb" ]; then
    idx="$(echo ",$idx," | tr ',' '\n' | grep -v -x -F -f <(echo "$bvb" | tr ',' '\n') | grep -v '^$' | paste -sd, -)"
  fi
  [ -n "$idx" ] && submit_chunked battery battery.sbatch "$idx" "$THROTTLE"
  # Named sm-bvb-<base> so the main array's base task can find it while it is pending and set ExcNodeList.
  [ -n "$bvb" ] && submit_chunked battery_bvb battery.sbatch "$bvb" "" --job-name="sm-bvb-$BASE_KEY"
  [ -n "$idx$bvb" ] || echo "stage battery: no active tasks; skipped"
fi
if has_stage capability; then submit_array capability; fi
if has_stage judge; then submit judge judge.sbatch; fi
if has_stage drift; then submit_array drift; fi
if has_stage monitor_scores; then submit_array monitor_scores; fi
if has_stage aggregate; then submit aggregate aggregate.sbatch; fi

echo "submitted: ${SUMMARY[*]:-nothing}"
[ "$DRY" = "1" ] && echo "(dry run: nothing was submitted)"
ids="$(printf '%s\n' "${SUMMARY[@]}" | sed 's/.*=//' | paste -sd, -)"
cat <<EOF2
next:
  watch:   squeue -u \$USER -o '%.12i %.16j %.9T %.10M %R'
  ledger:  after the chain ends (every job terminal), on the login node:
             cd $SM_PROJ && source jobs/common.sh && sm_ledger_add ${ids:-<jobids>}
           then classify any failure in .failure_streak.json (jobs/README.md, "Budget ledger")
  logs:    $SM_LOGDIR/<stage>-<jobid>[_<task>].{out,err}; each ends with a LEDGER line
           (smoke: smoke_cpu-<id>, smoke-<id>; base card: base-card-<id>, base-card-stats-<id>, base-card-ablate-<id>)
  accept:  smoke     -> python -m sm.accept --stage smoke --base $BASE_KEY --job ${SMOKE_ID:-<smoke id>}   (the GPU smoke id, not smoke_cpu ${SMOKE_CPU_ID:-})
           base_card -> python -m sm.accept --stage base_card --base $BASE_KEY --job ${BASE_CARD_ABLATE_ID:-<base_ablate id>}
EOF2
