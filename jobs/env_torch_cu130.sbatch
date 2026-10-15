#!/bin/bash
# Restore the torch 2.13.0 (cu130) build that vllm 0.31.0 and torchvision 0.28.0 require, after the cu128 experiment.
#SBATCH --job-name=sm-env-cu130
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-cu130-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-cu130-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== env cu130 restore start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # PATH + compat exports; sm_setup_cpu_env loads miniconda3 only (no cuda module)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                  # proxy on for pip/conda
ENV="$SM_ENV"
PY=$ENV/bin/python
$PY -m pip install --force-reinstall "torch==2.13.0" "torchvision==0.28.0"
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
for m in ["transformers", "peft", "trl", "datasets", "accelerate", "vllm", "torchvision"]:
    try:
        mod = __import__(m); print(m, mod.__version__)
    except Exception as e:
        print(m, "IMPORT FAILED", type(e).__name__, str(e)[:160])
PYEOF
# Review A16: never pipe python into head under pipefail (BrokenPipe -> rc 120 fails the job); write, then tail.
PIPCHK="$(mktemp)"; $PY -m pip check >"$PIPCHK" 2>&1 || true; tail -n 5 "$PIPCHK"; rm -f "$PIPCHK"
echo "=== env cu130 restore done: $(date) ==="
