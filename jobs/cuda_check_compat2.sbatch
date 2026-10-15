#!/bin/bash
# GPU check with NVIDIA's user-space forward-compatibility libcuda (cuda-compat-13 package extracted under
# SM_COMPAT_DIR; COMPAT_DIR=<dir> tries another one). Diagnostic only; the stages get the same export from common.sh.
#SBATCH --job-name=sm-cuda-compat2
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/cuda-compat2-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/cuda-compat2-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== cuda compat2 check start: $(date) on $(hostname) ==="
SM_COMPAT_DIR="${COMPAT_DIR:-${SM_COMPAT_DIR:-/projects/interpretability/shehral.m/cuda-compat/usr/local/cuda-13.0/compat}}"
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
sm_load_modules                    # miniconda3 only; re-asserts PATH and LD_LIBRARY_PATH
echo "SM_COMPAT_DIR=$SM_COMPAT_DIR"; ls "$SM_COMPAT_DIR" 2>/dev/null | head -6
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2
ENV="$SM_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$SM_HF_HOME"
$ENV/bin/python - <<'PYEOF'
import torch, time
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); t = time.time(); (a @ a).sum().item(); print("matmul ok", round(time.time() - t, 3), "s")
    try:
        import vllm; from vllm import LLM; print("vllm", vllm.__version__, "import ok")
    except Exception as e:
        print("vllm FAILED:", type(e).__name__, str(e)[:200])
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
        s = tok.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        print("Qwen3.5-9B tokenizer ok offline; thinking-off tail:", repr(s[-40:]))
        import peft; print("peft", peft.__version__)
    except Exception as e:
        print("transformers/peft check FAILED:", type(e).__name__, str(e)[:200])
PYEOF
echo "=== cuda compat2 check done: $(date) ==="
