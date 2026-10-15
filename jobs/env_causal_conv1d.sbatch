#!/bin/bash
# Rebuild causal-conv1d against the env's own torch (2.13.0+cu130). The first build (job 10892022) used pip's
# isolated build env, which pulled a different torch from PyPI, so the extension failed to import with an
# "undefined symbol" ABI error. --no-build-isolation compiles against the installed torch instead.
# Sources jobs/common.sh like every other job (the self-contained version that ran as job 10893219 is in git history, commit 3789b03).
#SBATCH --job-name=sm-env-cconv
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-cconv-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-cconv-%j.err
set -o pipefail
echo "=== env causal-conv1d rebuild start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                 # miniconda3 only + proxy
# Build-time exception: compiling causal-conv1d needs nvcc, so the CUDA 13.2 toolkit module is loaded for this
# build job only. No runtime job loads a cuda module (they use the compat libcuda from common.sh).
module load cuda/13.2.0 2>/dev/null || echo "cuda/13.2.0 module not available; the source build will fail"
sm_export_paths
mkdir -p "${PIP_CACHE_DIR:-/scratch/shehral.m/monitors/cache/pip}" "${TMPDIR:-/scratch/shehral.m/monitors/tmp}"
export CUDA_HOME=${CUDA_HOME:-/shared/EL9/explorer/cuda/13.2.0} MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="9.0"
PY=$SM_ENV/bin/python
test -x "$PY" || { echo "env python missing: $PY"; exit 2; }
echo "nvcc: $(command -v nvcc || echo missing); CUDA_HOME=$CUDA_HOME"
echo "torch before: $($PY -c 'import torch; print(torch.__version__)')"
$PY -m pip uninstall -y causal-conv1d 2>&1 | tail -n 1
$PY -m pip install ninja packaging 2>&1 | tail -n 1
echo "--- build against the installed torch (no build isolation, no wheel cache) ---"
CAUSAL_CONV1D_FORCE_BUILD=TRUE $PY -m pip install --no-build-isolation --no-cache-dir --no-binary causal-conv1d "causal-conv1d" 2>&1 | tail -n 5
$PY - <<'PYEOF'
import importlib
for m in ["causal_conv1d", "fla", "triton"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"), "import ok")
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, str(e)[:160])
import torch, numpy; print("torch still", torch.__version__, "numpy", numpy.__version__)
PYEOF
echo "=== env causal-conv1d rebuild done: $(date) ==="
