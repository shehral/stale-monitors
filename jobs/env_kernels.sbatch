#!/bin/bash
# Install the optimized kernels the Qwen3.5/3.8 hybrid layers use (flash-linear-attention, causal-conv1d) into the monitors env.
#SBATCH --job-name=sm-env-kernels
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-kernels-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-kernels-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== env kernels start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                 # miniconda3 only + proxy
# Build-time exception: compiling causal-conv1d needs nvcc, so the CUDA 13.2 toolkit module is loaded for this
# build job only. No runtime job loads a cuda module (they use the compat libcuda from common.sh).
module load cuda/13.2.0 2>/dev/null || echo "cuda/13.2.0 module not available; source builds will fail (prebuilt wheels still work)"
sm_export_paths
ENV="$SM_ENV"
PY=$ENV/bin/python
export CUDA_HOME=${CUDA_HOME:-/shared/EL9/explorer/cuda/13.2.0} MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="9.0"
echo "--- flash-linear-attention (pure Triton) ---"
$PY -m pip install "flash-linear-attention" 2>&1 | tail -3
echo "--- causal-conv1d (CUDA extension; prebuilt wheel first, then source build) ---"
$PY -m pip install "causal-conv1d" 2>&1 | tail -3 || echo "causal-conv1d install failed (non-fatal; reference kernel stays)"
$PY - <<'PYEOF'
import importlib
for m in ["fla", "causal_conv1d", "triton"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"), "import ok")
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, str(e)[:160])
import torch; print("torch still", torch.__version__)
PYEOF
echo "=== env kernels done: $(date) ==="
