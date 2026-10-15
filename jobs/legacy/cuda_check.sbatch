#!/bin/bash
#SBATCH --job-name=sm-cuda-check
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/cuda-check-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/cuda-check-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== cuda check start: $(date) on $(hostname) ==="
source /etc/profile; module purge; module load miniconda3/24.11.1 cuda/12.8.0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
ENV=/projects/interpretability/shehral.m/conda_envs/monitors
export PATH=$ENV/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/scratch/shehral.m/monitors/hf
$ENV/bin/python - <<'PY'
import torch, time
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); t=time.time(); (a@a).sum().item(); print("matmul ok", round(time.time()-t,3), "s")
try:
    import vllm; print("vllm", vllm.__version__, "import ok")
except Exception as e:
    print("vllm import FAILED:", type(e).__name__, str(e)[:160])
try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    msgs=[{"role":"user","content":"Hello"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("Qwen3.5-9B tokenizer ok; thinking-off template tail:", repr(s[-40:]))
except Exception as e:
    print("tokenizer check FAILED:", type(e).__name__, str(e)[:200])
PY
echo "=== cuda check done: $(date) ==="
