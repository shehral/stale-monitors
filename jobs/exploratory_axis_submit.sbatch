#!/bin/bash
# CPU helper for the EXPLORATORY E15/E7 line (docs/ASSISTANT_AXIS_E7_PLAN.md): submits the CAP+E7 phase of
# jobs/exploratory_axis.sbatch with --dependency=afterok:<FIRST> (the EXTRACT+DISPLACE job), retrying every 5 minutes
# while the gpu QOS submit cap (8 jobs per user) refuses it. It never waits on a GPU and holds no GPU: it runs on the
# short partition and exits as soon as the submission is accepted, or when FIRST ended in any state but COMPLETED
# (nothing to depend on), or after 12 hours. When FIRST has already COMPLETED it submits without a dependency.
# Submit: sbatch --export=ALL,FIRST=<jobid>,SM_CODE_COMMIT=<sha> jobs/exploratory_axis_submit.sbatch
# The accepted job id is the last "submitted" line of logs/axis-submit-<jobid>.out.
#SBATCH --job-name=sm-axis-submit
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/axis-submit-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/axis-submit-%j.err
#SBATCH --export=ALL
set -uo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
[ -n "${FIRST:-}" ] || sm_die 2 "FIRST (the EXTRACT+DISPLACE job id) is not set"
cd "$SM_PROJ" || sm_die 2 "no $SM_PROJ"
CODE="${SM_CODE_COMMIT:-unset}"
# Do not hand this job's own SLURM_* / SBATCH_* variables to the GPU job.
for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_' || true); do unset "$v"; done
for attempt in $(seq 1 144); do
  st="$(sacct -n -X -P -j "$FIRST" -o State 2>/dev/null | head -n 1 | awk '{print $1}')"
  case "$st" in
    COMPLETED) dep=() ;;
    PENDING | RUNNING | REQUEUED | CONFIGURING | COMPLETING | "") dep=(--dependency="afterok:$FIRST" --kill-on-invalid-dep=yes) ;;
    *) sm_log "first job $FIRST is $st: CAP+E7 not submitted"; exit 3 ;;
  esac
  if out="$(sbatch --parsable --time=06:00:00 "${dep[@]}" \
      --export="ALL,PHASES=CAP+E7,SM_CODE_COMMIT=$CODE,SM_PROJ=$SM_PROJ" jobs/exploratory_axis.sbatch 2>&1)"; then
    sm_log "submitted CAP+E7 as $out (${dep[*]:-no dependency}; first job $FIRST state ${st:-unknown})"
    exit 0
  fi
  sm_log "attempt $attempt: not accepted: $out"
  sleep 300
done
sm_log "gave up after 144 attempts"
exit 4
