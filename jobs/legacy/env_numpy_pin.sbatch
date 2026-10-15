#!/bin/bash
#SBATCH --job-name=sm-env-numpy
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/env-numpy-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/env-numpy-%j.err
#SBATCH --export=ALL
set -o pipefail
source /etc/profile; module purge; module load miniconda3/24.11.1
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128 HTTP_PROXY=http://10.99.0.130:3128 HTTPS_PROXY=http://10.99.0.130:3128
export PIP_CACHE_DIR=/scratch/shehral.m/monitors/cache/pip TMPDIR=/scratch/shehral.m/monitors/tmp
PY=/projects/interpretability/shehral.m/conda_envs/monitors/bin/python
$PY -c "import numpy; print(\"numpy before\", numpy.__version__)"
$PY -m pip install "numpy<2.5" 2>&1 | tail -2
$PY -c "import numpy, numba; print(\"numpy\", numpy.__version__, \"numba\", numba.__version__, \"import ok\")"
$PY -m pip check 2>&1 | head -5
