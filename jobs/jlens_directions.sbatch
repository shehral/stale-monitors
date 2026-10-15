#!/bin/bash
# EXPLORATORY R5 (docs/PROGRAM_2026-10-13.md, line B5): J-lens readouts of direction vectors at one sm layer
# (sm.interp.jlens_directions; CPU only). Reads the decision offsets d_L (unit minus base, harmful rows, last prompt
# token) and the base's harm direction from a mechanism set's residual caches, plus any .npy/.npz directions (the R1
# probe directions), through the lenses given, with the readout plan's controls (docs/JLENS_READOUT_PLAN.md sections
# 4 and 6). Not a measured stage, so no frozen-code gate; the output records the git commit, SM_CODE_COMMIT and the
# code file hashes. No text is read or written. Runs from the side tree (SM_PROJ default
# /projects/interpretability/shehral.m/monitors-diag; its results, data and logs are symlinks into the main tree).
#
# Env:
#   LAYER    sm layer = hidden_states index (default 28; the lens key is LAYER-1)
#   LENSES   space-separated NAME=DIR list, first = primary, first two = the stability pair
#            (default "s0=results/jlens/<base>_v2_L28 s1=results/jlens/<base>_s1_L28")
#   ACTS     residual-cache directory (default results/exploratory/<base>/mechanism/sets/canary_e2_seed250/acts)
#   UNITS    comma list of units in ACTS (default: every cache but base)
#   EXTRA    extra CLI flags, split on whitespace, e.g. "--vector-glob probe=results/.../probes/L28/*.npy"
#            (pass it as an exported variable: --export splits on commas)
#   BASE (default qwen3.5-9b), OUT (default results/exploratory/<base>/jlens/readouts_L<LAYER>.json), SM_JLENS_DIR
#   (official jacobian-lens clone for the half-vs-half noise floor; default the main tree's third_party copy), SM_PROJ.
# Submit (CPU, about 10 to 20 minutes; two lenses of about 2 GB of unembedding each):
#   sbatch -p short --gres=none --mem=48G --time=01:00:00 --export=ALL,SM_CODE_COMMIT=<sha> jobs/jlens_directions.sbatch
# Exit codes: 2 setup, 5 inputs missing (lens files, caches).
#SBATCH --job-name=sm-jlens-dirs
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/jlens-dirs-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/jlens-dirs-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
sm_log "=== sm-jlens-dirs start on $(hostname) code=${SM_CODE_COMMIT:-unset} ==="
sm_setup_cpu_env

BASE_KEY="${BASE:-qwen3.5-9b}"
SAFE="$(sm_py safe-name "$BASE_KEY")"
LAYER="${LAYER:-28}"
LENSES="${LENSES:-s0=results/jlens/${SAFE}_v2_L28 s1=results/jlens/${SAFE}_s1_L28}"
ACTS="${ACTS:-results/exploratory/$SAFE/mechanism/sets/canary_e2_seed250/acts}"
OUT="${OUT:-results/exploratory/$SAFE/jlens/readouts_L${LAYER}.json}"
LENS_ARGS=()
for spec in $LENSES; do
  d="${spec#*=}"
  for f in lens.pt unembed.pt vocab.json meta.json; do
    [ -f "$d/$f" ] || sm_die 5 "lens file $d/$f missing (merge the fit first)"
  done
  LENS_ARGS+=(--lens "$spec")
done
[ -f "$ACTS/base.npz" ] || sm_die 5 "residual cache $ACTS/base.npz missing"
export SM_JLENS_DIR="${SM_JLENS_DIR:-/projects/interpretability/shehral.m/monitors/third_party/jacobian-lens}"
if [ ! -f "$SM_JLENS_DIR/jlens/__init__.py" ]; then
  sm_log "WARNING: official jacobian-lens code missing at $SM_JLENS_DIR; the half-vs-half noise floor is reported unavailable"
fi

# EXTRA is split on whitespace on purpose (a list of CLI flags).
# shellcheck disable=SC2086
rc=0; sm_run "$PYTHON" -m sm.interp.jlens_directions --base "$BASE_KEY" --layer "$LAYER" "${LENS_ARGS[@]}" \
  --acts-dir "$ACTS" ${UNITS:+--units "$UNITS"} --out "$OUT" ${EXTRA:-} || rc=$?
sm_log "=== sm-jlens-dirs done rc=$rc out=$OUT ==="
exit "$rc"
