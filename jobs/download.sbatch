#!/bin/bash
# Pre-cache bases, judges and two datasets on scratch (CPU node, proxy on). Idempotent.
# Bases and judges are downloaded at exactly the commits pinned in configs/bases.yaml (`bases` and
# `judges`); every stage loads them by that commit, so no refs/main is needed. An unpinned model
# (a fallback base before it is pinned, or allenai/wildguard with DOWNLOAD_WILDGUARD=1) is downloaded by
# branch name. The resolved commit of everything is recorded in data/model_manifest.json, merged into the
# existing file (a failed entry never overwrites an earlier good one). The job exits 1 when any pinned model
# or dataset failed, so a broken download is not mistaken for a finished one.
#SBATCH --job-name=sm-download
#SBATCH --partition=short
#SBATCH --account=i.yoon
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/interpretability/shehral.m/monitors/logs/download-%j.out
#SBATCH --error=/projects/interpretability/shehral.m/monitors/logs/download-%j.err
#SBATCH --export=ALL
set -eo pipefail
SM_PROJ="${SM_PROJ:-/projects/interpretability/shehral.m/monitors}"
# shellcheck source=jobs/common.sh
source "$SM_PROJ/jobs/common.sh"
SM_GPUS_PER_JOB=0
sm_install_traps
sm_log "=== download start on $(hostname) ==="
sm_setup_cpu_env 1
PY="$PYTHON"
[ -x "$PY" ] || PY="$SM_ALLOC/conda_envs/matched-compute/bin/python"
mkdir -p "$SM_PROJ/data"
if [ -f "$HOME/.cache/huggingface/token" ]; then export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"; fi
export SM_DOWNLOAD_WILDGUARD="${DOWNLOAD_WILDGUARD:-0}"
export SM_DOWNLOAD_EXTERNAL="${DOWNLOAD_EXTERNAL:-0}"   # 1 = also fetch the condition-5 public comparators pinned in configs/bank_*.yaml
$PY - <<'PYEOF'
import json, os, time
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
models = ["Qwen/Qwen3.5-9B", "google/gemma-4-12B-it", "openai/gpt-oss-safeguard-20b", "ibm-granite/granite-guardian-3.3-8b", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"]
if os.environ.get("SM_DOWNLOAD_WILDGUARD") == "1":
    models.append("allenai/wildguard")  # optional J2 alternative (license click-through), unpinned
datasets = ["databricks/databricks-dolly-15k", "qiaojin/PubMedQA"]
allow = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "*.tiktoken", "tokenizer*", "*.md", "LICENSE*"]
ignore = ["*.gguf", "*.bin", "*.pth", "*consolidated*", "original/*"]
import yaml
pins = {}
try:
    reg = yaml.safe_load(open("/projects/interpretability/shehral.m/monitors/configs/bases.yaml")) or {}
    for section in ("bases", "judges"):
        for ent in (reg.get(section) or {}).values():
            if isinstance(ent, dict) and ent.get("hf") and ent.get("revision"):
                pins[ent["hf"]] = str(ent["revision"])
except OSError as e:
    print("WARNING: configs/bases.yaml not readable; downloading by branch", e, flush=True)
# Condition-5 comparators: every configs/bank_*.yaml run with external_hf + external_revision is pinned here, and
# downloaded too when SM_DOWNLOAD_EXTERNAL=1 (DOWNLOAD_EXTERNAL=1 at sbatch time). Detection-only use; never re-uploaded.
import glob
for bank in sorted(glob.glob("/projects/interpretability/shehral.m/monitors/configs/bank_*.yaml")):
    try:
        bank_cfg = yaml.safe_load(open(bank)) or {}
    except (OSError, ValueError, yaml.YAMLError) as e:
        print("WARNING: bank config unreadable", bank, e, flush=True)
        continue
    for lst in bank_cfg.values():
        if not isinstance(lst, list):
            continue
        for run in lst:
            if isinstance(run, dict) and run.get("external_hf") and run.get("external_revision"):
                pins[str(run["external_hf"])] = str(run["external_revision"])
                if os.environ.get("SM_DOWNLOAD_EXTERNAL") == "1" and run["external_hf"] not in models:
                    models.append(str(run["external_hf"]))
out = "/projects/interpretability/shehral.m/monitors/data/model_manifest.json"
try:
    manifest = json.load(open(out)) if os.path.isfile(out) else {}
except (OSError, ValueError) as e:
    print("WARNING: existing manifest unreadable; starting a new one", e, flush=True)
    manifest = {}
failed = []
for mid in models:
    t0 = time.time()
    try:
        info = api.model_info(mid)
        pin = pins.get(mid)
        # Pinned: exactly that commit. Unpinned: by branch, which writes refs/main for offline loads.
        p = snapshot_download(mid, revision=pin, allow_patterns=allow, ignore_patterns=ignore)
        sha = os.path.basename(os.path.normpath(p))
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
        manifest[mid] = {"revision": sha, "pinned": bool(pin), "hub_head": info.sha, "path": p, "bytes": size, "seconds": round(time.time() - t0, 1), "gated": bool(getattr(info, "gated", False))}
        print("OK", mid, sha[:12], "pinned" if pin else "branch main", f"{size/1e9:.1f} GB", f"{time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        err = {"error": f"{type(e).__name__}: {str(e)[:200]}", "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        prev = manifest.get(mid) or {}
        manifest[mid] = {**prev, "last_error": err} if prev.get("revision") else err
        failed.append(mid)
        print("FAILED", mid, type(e).__name__, str(e)[:200], flush=True)
for did in datasets:
    try:
        p = snapshot_download(did, repo_type="dataset")
        manifest[did] = {"path": p}
        print("OK dataset", did, flush=True)
    except Exception as e:
        err = {"error": f"{type(e).__name__}: {str(e)[:200]}", "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        prev = manifest.get(did) or {}
        manifest[did] = {**prev, "last_error": err} if prev.get("path") else err
        failed.append(did)
        print("FAILED dataset", did, type(e).__name__, str(e)[:200], flush=True)
tmp = out + ".tmp"
with open(tmp, "w") as fh:
    json.dump(manifest, fh, indent=2)
os.replace(tmp, out)
print("manifest ->", out)
if failed:
    print("DOWNLOAD FAILED for:", ", ".join(failed), flush=True)
    raise SystemExit(1)
PYEOF
du -sh "$HF_HOME" 2>/dev/null | tail -1
sm_log "=== download done ==="
