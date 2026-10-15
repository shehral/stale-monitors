#!/bin/bash
# Real-base GPU smoke (G1 gate): standalone, uses the current sm package snapshot on the cluster.
#SBATCH --job-name=sm-gpu-smoke
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/gpu-smoke-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/gpu-smoke-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== gpu smoke start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
sm_install_traps
sm_setup_env                       # module miniconda3 only, caches, offline flags
sm_gpu_preflight                   # exit 2 unless torch.cuda.is_available()
ENV="$SM_ENV"
export SM_SMOKE_BASE="${SM_SMOKE_BASE:-Qwen/Qwen3.5-9B}"
SM_SMOKE_REV="${SM_SMOKE_REV:-$(sm_py revision "$SM_SMOKE_BASE")}"   # the configs/bases.yaml pin
export SM_SMOKE_REV
[ -n "$SM_SMOKE_REV" ] || sm_die 2 "no revision pin for $SM_SMOKE_BASE in configs/bases.yaml"
ls -d "$HF_HOME/hub/models--$(echo $SM_SMOKE_BASE | sed 's#/#--#')/snapshots/$SM_SMOKE_REV" >/dev/null || { echo "FATAL: base snapshot not cached"; exit 2; }
cd "$SM_PROJ"
rc=0; "$PYTHON" jobs/gpu_smoke_min.py || rc=$?
echo "=== gpu smoke done: $(date) rc=$rc ==="
exit $rc
