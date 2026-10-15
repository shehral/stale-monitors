#!/bin/bash
# J-lens readout lines J1 and J2 and the v1-vs-v2 lens stability check (docs/JLENS_READOUT_PLAN.md, committed alone
# before any capture). EXPLORATORY: not a measured stage, so no frozen-code gate; every output records the code file
# hashes and SM_CODE_COMMIT (the laptop commit the files came from). Runs from the side tree (SM_PROJ default
# /projects/interpretability/shehral.m/monitors-diag; its results, data and logs are symlinks into the main tree).
#
# Modes (MODE env):
#   capture  (default; one H200) sm.interp.jlens capture: the base and the 17 runs of results/drift/<base>/_configs
#            (adapters unmerged, the external model at its pin, the in-memory ablation via DirectionHooks), per-item
#            residuals at the card's p* and at the last prompt token on the test harmful / harmless sets (and the
#            calibration harmless set for the base), fp16, no text -> results/exploratory/<base>/jlens/activations/;
#            then sm.interp.jlens score: lens refusal mass per row under the v2 and v1 lenses on the GPU -> .../scores/.
#              sbatch --export=ALL,SM_CODE_COMMIT=<sha> jobs/jlens_readout.sbatch
#   readouts (CPU, partition short) sm.interp.jlens readouts -> results/exploratory/<base>/jlens/readouts.json:
#              sbatch -p short --gres=none --mem=32G --time=01:00:00 --dependency=afterok:<capture id> \
#                --kill-on-invalid-dep=yes --export=ALL,MODE=readouts,SM_CODE_COMMIT=<sha> jobs/jlens_readout.sbatch
# Env: BASE (default qwen3.5-9b), FORCE=1 (recapture every unit), UNITS (comma list), SM_JLENS_DIR (official
#   jacobian-lens clone for the shard merges of the noise floor; default the main tree's third_party/jacobian-lens,
#   read only), SM_PROJ. Estimates (plan section 9): capture + score 0.2 to 0.3 H200-h; readouts 10 to 20 CPU minutes.
# Hand-submitted: ledger the GPU job with sm_ledger_add <jobid> after it ends.
# Exit codes: 2 setup (env, GPU, compat libcuda, snapshot), 5 inputs missing (card, lenses, configs, prompts).
#SBATCH --job-name=sm-jlens-readout
#SBATCH --partition=gpu
#SBATCH --account=i.yoon
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=d4055
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/jlens-readout-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/jlens-readout-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
sm_install_traps
MODE="${MODE:-capture}"
sm_log "=== sm-jlens-readout start mode=$MODE on $(hostname) code=${SM_CODE_COMMIT:-unset} ==="
case "$MODE" in
  capture)
    sm_setup_env
    sm_gpu_preflight
    # NVIDIA forward-compat libcuda (exported by jobs/common.sh at source time) must come first on the library path.
    if [ "$SM_USE_COMPAT" = "1" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in ":$SM_COMPAT_DIR:"*) ;; *) sm_die 2 "SM_COMPAT_DIR=$SM_COMPAT_DIR is not first on LD_LIBRARY_PATH" ;; esac
    fi
    ;;
  readouts)
    sm_setup_cpu_env
    ;;
  *)
    sm_die 2 "MODE must be capture or readouts, got '$MODE'"
    ;;
esac

BASE_KEY="${BASE:-qwen3.5-9b}"
SAFE="$(sm_py safe-name "$BASE_KEY")"
CARD="results/base_cards/$SAFE.json"
CONFIGS="results/drift/$SAFE/_configs"
OUT="results/exploratory/$SAFE/jlens"
LENS_V2="results/jlens/${SAFE}_v2"
LENS_V1="results/jlens/$SAFE"
[ -f "$CARD" ] || sm_die 5 "base card $CARD missing"
for d in "$LENS_V2" "$LENS_V1"; do
  for f in lens.pt unembed.pt vocab.json meta.json; do
    [ -f "$d/$f" ] || sm_die 5 "lens file $d/$f missing"
  done
done
JM=("$PYTHON" -m sm.interp.jlens)

if [ "$MODE" = "capture" ]; then
  [ -d "$CONFIGS" ] || sm_die 5 "drift run configs $CONFIGS missing"
  sm_require_data data/prompts/contrast/test_harmful.jsonl data/prompts/contrast/test_harmless.jsonl \
    data/prompts/contrast/calibration_harmless.jsonl
  sm_preflight_snapshot "$BASE_KEY"
  # external derivatives (kind model) must be cached at their pins before the first load
  while read -r model rev; do
    if [ -n "$model" ]; then sm_preflight_snapshot "$model" "$rev"; fi
  done < <("$PYTHON" - "$CONFIGS" <<'PYEOF'
import pathlib
import sys

import yaml

for p in sorted(pathlib.Path(sys.argv[1]).glob("*.yaml")):
    d = (yaml.safe_load(p.read_text()) or {}).get("derivative") or {}
    if d.get("kind") == "model":
        print(d.get("model", ""), d.get("revision", ""))
PYEOF
)
  CAP_ARGS=(--base "$BASE_KEY" --card "$CARD" --configs "$CONFIGS" --out "$OUT/activations")
  if [ -n "${UNITS:-}" ]; then CAP_ARGS+=(--units "$UNITS"); fi
  if [ -n "${FORCE:-}" ]; then CAP_ARGS+=(--force); fi
  rc=0; sm_run "${JM[@]}" capture "${CAP_ARGS[@]}" || rc=$?
  [ "$rc" -eq 0 ] || sm_die "$rc" "capture failed"
  rc=0; sm_run "${JM[@]}" score --base "$BASE_KEY" --acts-dir "$OUT/activations" --lens "v2=$LENS_V2" --lens "v1=$LENS_V1" \
    --out "$OUT/scores" --device cuda || rc=$?
  sm_log "=== sm-jlens-readout capture+score done rc=$rc ==="
  exit "$rc"
fi

export SM_JLENS_DIR="${SM_JLENS_DIR:-/projects/interpretability/shehral.m/monitors/third_party/jacobian-lens}"
if [ ! -f "$SM_JLENS_DIR/jlens/__init__.py" ]; then
  sm_log "WARNING: official jacobian-lens code missing at $SM_JLENS_DIR; the half-vs-half noise floor is reported unavailable"
fi
[ -f "$OUT/scores/base.npz" ] || sm_die 5 "no scores under $OUT/scores (run MODE=capture first)"
rc=0; sm_run "${JM[@]}" readouts --base "$BASE_KEY" --card "$CARD" --acts-dir "$OUT/activations" --scores-dir "$OUT/scores" \
  --lens "v2=$LENS_V2" --lens "v1=$LENS_V1" --flag-file "results/judge/$SAFE/_flags/j1_rates_flags.txt" \
  --drift-dir "results/drift/$SAFE" --e15-dir "results/exploratory/$SAFE/run2/displace" --out "$OUT/readouts.json" || rc=$?
sm_log "=== sm-jlens-readout readouts done rc=$rc ==="
exit "$rc"
