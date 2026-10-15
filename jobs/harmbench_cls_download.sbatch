#!/bin/bash
# Pre-cache cais/HarmBench-Llama-2-13b-cls at the commit pinned for the E14 line (docs/HARMBENCH_CLS_LINE.md section 2;
# sm.eval.harmbench_cls.CLS_REVISION) on scratch: CPU node, squid proxy on, the allow and ignore patterns of
# jobs/download.sbatch (so training_args.bin, a pickle, is not fetched). Public (MIT, not gated); it stops if the hub
# reports the repo gated. Resumes partial downloads. Writes refs/main when the hub's main head equals the pin and merges one
# entry into data/model_manifest.json (atomic write; the other entries are kept). The login node killed this download
# twice on 2026-10-10 (exit 137 after about 1.3 GB), hence a short-partition job. Idempotent: a complete snapshot is
# checked, not fetched again. Re-run after a scratch purge.
#
# Submit: cd /projects/interpretability/shehral.m/monitors-diag && \
#   sbatch --export=ALL,SM_PROJ=/projects/interpretability/shehral.m/monitors-diag jobs/harmbench_cls_download.sbatch
#SBATCH --job-name=sm-hbcls-dl
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/hbcls-download-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/hbcls-download-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors-diag}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
sm_log "=== hbcls download start on $(hostname) ==="
sm_setup_cpu_env 1
export HF_HUB_DISABLE_XET=1   # plain HTTP through the proxy (resumes the .incomplete blobs of an interrupted run)
export SM_MANIFEST_OUT="${SM_MANIFEST_OUT:-$SM_PROJ/data/model_manifest.json}"
"$PYTHON" - <<'PYEOF'
import json
import os
import time

from huggingface_hub import HfApi, snapshot_download

MID = "cais/HarmBench-Llama-2-13b-cls"
REV = "bda705349d1144fa618770bea64d99ce54e3835b"  # sm.eval.harmbench_cls.CLS_REVISION
OUT = os.environ["SM_MANIFEST_OUT"]
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "*.tiktoken", "tokenizer*", "*.md", "LICENSE*"]
IGNORE = ["*.gguf", "*.bin", "*.pth", "*consolidated*", "original/*"]

api = HfApi()
t0 = time.time()
info = api.model_info(MID, revision=REV, files_metadata=True)
if info.gated:
    raise SystemExit(f"STOP: {MID} is gated ({info.gated}); not downloading")
hub = {s.rfilename: s.size for s in info.siblings}
p = snapshot_download(MID, revision=REV, allow_patterns=ALLOW, ignore_patterns=IGNORE, max_workers=4)
sha = os.path.basename(os.path.normpath(p))
assert sha == REV, (sha, REV)
got = {f: os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)}
sizes_ok = all(hub.get(f) == n for f, n in got.items())
weights = sorted(f for f in got if f.endswith(".safetensors"))
size = sum(got.values())
head = api.model_info(MID).sha
repo = os.path.dirname(os.path.dirname(os.path.normpath(p)))
if head == REV:
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    with open(os.path.join(repo, "refs", "main"), "w") as fh:
        fh.write(REV)
manifest = json.load(open(OUT)) if os.path.isfile(OUT) else {}
manifest[MID] = {
    "revision": sha, "pinned": True, "hub_head": head, "path": p, "bytes": size,
    "seconds": round(time.time() - t0, 1), "gated": bool(info.gated), "sizes_match_hub": bool(sizes_ok),
    "n_safetensors": len(weights),
    "pin_source": "sm.eval.harmbench_cls.CLS_REVISION (docs/HARMBENCH_CLS_LINE.md, E14); not in configs/bases.yaml",
    "job": os.environ.get("SLURM_JOB_ID"),
}
tmp = OUT + ".tmp"
with open(tmp, "w") as fh:
    json.dump(manifest, fh, indent=2)
os.replace(tmp, OUT)
print("OK", MID, sha, f"{size / 1e9:.2f} GB", "shards", len(weights), "sizes_match_hub", sizes_ok, "head==pin", head == REV,
      f"{time.time() - t0:.0f}s", flush=True)
if len(weights) != 6 or not sizes_ok:
    raise SystemExit("incomplete snapshot")
PYEOF
sm_log "=== hbcls download done ==="
