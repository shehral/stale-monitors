#!/bin/bash
# Reinstall torch with CUDA 12.8 wheels: the H200 nodes run driver 570.86, which cannot initialize a cu130 build.
#SBATCH --job-name=sm-env-cu128
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-cu128-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-cu128-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== env cu128 fix start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # PATH + compat exports; sm_setup_cpu_env loads miniconda3 only (no cuda module)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                  # proxy on for pip/conda
ENV="$SM_ENV"
PY=$ENV/bin/python
TV=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
echo "current torch: $($PY -c 'import torch; print(torch.__version__)') ; reinstalling $TV with cu128 wheels"
if ! $PY -m pip install --force-reinstall "torch==$TV" --index-url https://download.pytorch.org/whl/cu128; then
  echo "cu128 wheel for torch $TV not found; installing the newest cu128 torch instead"
  $PY -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
for m in ["transformers", "peft", "trl", "datasets", "accelerate", "huggingface_hub"]:
    try:
        mod = __import__(m); print(m, mod.__version__)
    except Exception as e:
        print(m, "IMPORT FAILED", type(e).__name__, str(e)[:120])
try:
    import vllm; print("vllm", vllm.__version__, "import ok on a CPU node (GPU ops unverified)")
except Exception as e:
    print("vllm import FAILED:", type(e).__name__, str(e)[:200])
PYEOF
echo "=== env cu128 fix done: $(date) ==="
