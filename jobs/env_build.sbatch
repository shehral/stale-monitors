#!/bin/bash
# Build the `monitors` conda env on a CPU node (internet via the squid proxy).
#SBATCH --job-name=sm-env-build
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-build-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-build-%j.err
#SBATCH --export=ALL
set -eo pipefail
echo "=== env build start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # PATH + compat exports; sm_setup_cpu_env loads miniconda3 only (no cuda module)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                  # proxy on for pip/conda
ENV="$SM_ENV"
SCR="$SM_SCRATCH"
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" python=3.12
fi
PY=$ENV/bin/python
$PY -m pip install --upgrade pip wheel
# vllm pins its own torch (CUDA 12.x wheels); install it first, then the rest without touching torch.
$PY -m pip install "vllm"
$PY -m pip install "transformers" "peft" "trl" "datasets" "accelerate" "scikit-learn" "pandas" "pyyaml" "huggingface_hub[hf_xet]" "pytest" "matplotlib" "scipy" "sentencepiece" "protobuf"
# Optional interpretability libraries: never let their pins downgrade the core stack.
$PY -m pip install --no-deps "sae-lens" "nnsight" || echo "optional interp libs failed (non-fatal)"
$PY - <<'PYEOF'
import importlib
for m in ["torch","transformers","vllm","peft","trl","datasets","accelerate","sklearn","huggingface_hub","sae_lens","nnsight"]:
    try:
        mod = importlib.import_module(m); print(f"{m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m} IMPORT FAILED: {e}")
import torch; print("torch cuda build:", torch.version.cuda)
from transformers import AutoConfig
for mid in ["Qwen/Qwen3.5-9B","Qwen/Qwen3.8-27B","google/gemma-4-12B-it","google/gemma-4-31B-it","openai/gpt-oss-safeguard-20b"]:
    try:
        c = AutoConfig.from_pretrained(mid, trust_remote_code=False); print("config OK", mid, c.model_type)
    except Exception as e:
        print("config FAILED", mid, type(e).__name__, str(e)[:160])
PYEOF
echo "=== env build done: $(date) ==="
