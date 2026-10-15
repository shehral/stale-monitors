#!/bin/bash
# vLLM throughput and judge-format check on one H200 (two processes: base, then judge).
#SBATCH --job-name=sm-vllm-check
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/vllm-check-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/vllm-check-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== vllm check start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
sm_install_traps
sm_setup_env                       # module miniconda3 only, caches, offline flags
sm_gpu_preflight                   # exit 2 unless torch.cuda.is_available()
ENV="$SM_ENV"
export VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
export SM_SMOKE_BASE="${SM_SMOKE_BASE:-Qwen/Qwen3.5-9B}"
SM_SMOKE_REV="${SM_SMOKE_REV:-$(sm_py revision "$SM_SMOKE_BASE")}"
SM_JUDGE_REV="${SM_JUDGE_REV:-$(sm_py revision openai/gpt-oss-safeguard-20b)}"
export SM_SMOKE_REV SM_JUDGE_REV
cd "$SM_PROJ"
$ENV/bin/python jobs/vllm_check.py base 2>&1 | grep -v -E "^INFO|^\s*$" | tail -n 25
echo "=== judge ==="
$ENV/bin/python jobs/vllm_check.py judge 2>&1 | grep -v -E "^INFO|^\s*$" | tail -n 25
echo "=== vllm check done: $(date) ==="
