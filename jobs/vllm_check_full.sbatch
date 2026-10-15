#!/bin/bash
# vLLM check with FULL logs kept (the first run filtered stderr and lost the root cause).
#SBATCH --job-name=sm-vllm-full
#SBATCH --partition=gpu-short
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/vllm-full-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/vllm-full-%j.err
#SBATCH --export=ALL
set -o pipefail
echo "=== vllm full check start: $(date) on $(hostname) ==="
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"   # env PATH + CUDA compat LD_LIBRARY_PATH (exported before any python call)
sm_install_traps
sm_setup_env                       # module miniconda3 only, caches, offline flags
sm_gpu_preflight                   # exit 2 unless torch.cuda.is_available()
ENV="$SM_ENV"
export TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=INFO
export SM_SMOKE_BASE="${SM_SMOKE_BASE:-Qwen/Qwen3.5-9B}"
SM_SMOKE_REV="${SM_SMOKE_REV:-$(sm_py revision "$SM_SMOKE_BASE")}"
SM_JUDGE_REV="${SM_JUDGE_REV:-$(sm_py revision openai/gpt-oss-safeguard-20b)}"
export SM_SMOKE_REV SM_JUDGE_REV
L="$SM_PROJ/logs"
cd "$SM_PROJ"
$ENV/bin/python -c "import importlib.util; [print(m, 'ok') if importlib.util.find_spec(m) else print(m, 'MISSING') for m in ['fla','causal_conv1d','triton']]"
echo "--- base (full log: $L/vllm-base-full-$SLURM_JOB_ID.log) ---"
$ENV/bin/python jobs/vllm_check.py base > "$L/vllm-base-full-$SLURM_JOB_ID.log" 2>&1; rc=$?
echo "base rc=$rc"; grep -n -E "Error|error|Exception|not supported|No module|ptxas|Triton|out of memory|generated|tok/s|VLLM CHECK" "$L/vllm-base-full-$SLURM_JOB_ID.log" | grep -v -E "INFO|WARNING" | head -30 | cut -c1-240
echo "--- judge (full log: $L/vllm-judge-full-$SLURM_JOB_ID.log) ---"
$ENV/bin/python jobs/vllm_check.py judge > "$L/vllm-judge-full-$SLURM_JOB_ID.log" 2>&1; rc2=$?
echo "judge rc=$rc2"; grep -n -E "Error|error|Exception|judged|^---|VLLM CHECK" "$L/vllm-judge-full-$SLURM_JOB_ID.log" | grep -v -E "INFO|WARNING" | head -20 | cut -c1-240
echo "=== vllm full check done: $(date) ==="
