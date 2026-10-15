#!/bin/bash
# Fallback env `monitors-cu128`: no vllm; torch 2.11.0+cu128 (the newest cu128 build) with a transformers/peft/trl stack
# that resolves against it. Used only if CUDA forward compatibility for the cu130 env fails on the 570 driver.
#SBATCH --job-name=sm-env-fb128
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-fb128-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-fb128-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== fallback env start: $(date) on $(hostname) ==="
SM_ENV="${SM_ENV:-/projects/interpretability/shehral.m/conda_envs/monitors-cu128}"
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # PATH + compat exports; sm_setup_cpu_env loads miniconda3 only (no cuda module)
SM_GPUS_PER_JOB=0
sm_setup_cpu_env 1                  # proxy on for pip/conda
ENV="$SM_ENV"   # monitors-cu128: common.sh leaves the compat libcuda off for this env
if [ ! -x "$ENV/bin/python" ]; then conda create -y -p "$ENV" python=3.12; fi
PY=$ENV/bin/python
$PY -m pip install --upgrade pip wheel
$PY -m pip install "torch==2.11.0" "torchvision" --index-url https://download.pytorch.org/whl/cu128
# Let pip resolve a transformers/peft/trl set that accepts torch 2.11; do not pin transformers.
$PY -m pip install "transformers" "peft" "trl" "datasets" "accelerate" "scikit-learn" "pandas" "pyyaml" "huggingface_hub[hf_xet]" "pytest" "matplotlib" "scipy" "sentencepiece" "protobuf"
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
for m in ["transformers", "peft", "trl", "datasets", "accelerate", "torchvision"]:
    try:
        mod = __import__(m); print(m, mod.__version__)
    except Exception as e:
        print(m, "IMPORT FAILED", type(e).__name__, str(e)[:200])
try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("transformers core classes import ok")
    for mid in ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "google/gemma-4-12B-it", "google/gemma-4-31B-it", "openai/gpt-oss-safeguard-20b", "ibm-granite/granite-guardian-3.3-8b"]:
        try:
            c = AutoConfig.from_pretrained(mid); print("config OK", mid, c.model_type)
        except Exception as e:
            print("config FAILED", mid, type(e).__name__, str(e)[:120])
except Exception as e:
    print("transformers core import FAILED:", type(e).__name__, str(e)[:300])
try:
    from peft import LoraConfig, get_peft_model; from trl import SFTTrainer; print("peft/trl trainer imports ok")
except Exception as e:
    print("peft/trl import FAILED:", type(e).__name__, str(e)[:300])
PYEOF
# Review A16: never pipe python into head under pipefail (BrokenPipe -> rc 120 fails the job); write, then tail.
PIPCHK="$(mktemp)"; $PY -m pip check >"$PIPCHK" 2>&1 || true; tail -n 5 "$PIPCHK"; rm -f "$PIPCHK"
echo "=== fallback env done: $(date) ==="
