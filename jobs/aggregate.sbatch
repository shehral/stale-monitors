#!/bin/bash
# Aggregate (sm.analysis.aggregate): joins drift, monitor scores, judge labels and capability into
# results/tables/bank_<base>.csv, evaluates H1-H6 with the pre-declared rules (Holm over all six), and
# writes results/tables/hypotheses.json and results/figures/. CPU only (partition short).
# It needs no V4 gate of its own: a base without a passed V4 file is reported as not evaluable for the
# white-box hypotheses by aggregate itself (prereg Section 4), and its behavioural results stay descriptive.
# The last stage of jobs/submit_bank.sh (per base, AGG_BASES defaults to the bank's base); a per-base run
# writes results/tables/subsets/<base>/ and never touches the pooled result. After all four banks finish,
# run it once over every base; only that run writes results/tables/hypotheses.json (the confirmatory result):
#   sbatch --export=ALL,AGG_BASES=all jobs/aggregate.sbatch
# Aggregate always recomputes (cheap, deterministic), so a rerun after J2 pass 2 clears pending runs
# picks up the new labels; FORCE=1 is accepted and changes nothing.
# A pooled run (every bank manifest; `sm.analysis.aggregate --print-scope` says "pooled") is the confirmatory
# result, so it first runs sm_require_frozen for every base (clean commit under code/ configs/ jobs/, pinned
# bases; exit 2 otherwise, ALLOW_UNFROZEN_CODE=1 logged), as every measured stage does (prereg/ACCEPTANCE.md
# OPEN-10, AG-2). Aggregate's meta records analysis_tree, accept_blob, rules_blob and analysis_dirty next to
# git_commit. Subset runs (one base at the end of its chain) are progress reports and are not gated.
#
# Optional env: BANK or BASE (the default base), AGG_BASES ("all" = every configs/bank_*.yaml, or a
#   space-separated list of base keys), AGG_CONFIG (analysis YAML), AGG_EXTRA_ARGS, FORCE=1, SM_PROJ.
#SBATCH --job-name=sm-aggregate
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/aggregate-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/aggregate-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
sm_log "=== sm-aggregate start on $(hostname) ==="
sm_setup_cpu_env 0
[ -x "$PYTHON" ] || sm_die 2 "env python not found at $PYTHON"

args=(--results-root results --configs-root configs --out results)
if [ "${AGG_BASES:-}" = "all" ]; then
  sm_log "aggregate over every configs/bank_*.yaml base"
elif [ -n "${AGG_BASES:-}" ]; then
  # shellcheck disable=SC2206
  bases=($AGG_BASES)
  args+=(--bases "${bases[@]}")
else
  sm_resolve_base_key
  args+=(--bases "$BASE_KEY")
fi
[ -n "${AGG_CONFIG:-}" ] && args+=(--config "$AGG_CONFIG")
scope_line="$("$PYTHON" -m sm.analysis.aggregate "${args[@]}" --print-scope)" || sm_die 2 "aggregate --print-scope failed"
read -r AGG_SCOPE AGG_TAG AGG_SCOPE_BASES <<<"$scope_line"
sm_log "aggregate scope: $AGG_SCOPE ($AGG_TAG): $AGG_SCOPE_BASES"
if [ "$AGG_SCOPE" = "pooled" ]; then
  for b in $AGG_SCOPE_BASES; do sm_require_frozen "$b"; done
fi
[ -n "${FORCE:-}" ] && args+=(--force)
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.analysis.aggregate "${args[@]}" ${AGG_EXTRA_ARGS:-} || rc=$?
sm_log "=== sm-aggregate done rc=$rc ==="
exit "$rc"
