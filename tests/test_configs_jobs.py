"""Tests for configs/ and jobs/ (the configs_jobs wiring): pins, bank manifests, the shell helpers in
jobs/common.sh, the V4 gate, the J1 flag list, the ledger intake and the submit_bank.sh chain (dry run).

Run offline: SM_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_configs_jobs.py
Nothing here touches a GPU, the network or SLURM: sbatch and squeue are replaced by stubs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
CONFIGS = ROOT / "configs"

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="needs bash")

PINS = {
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "gemma-4-12b-it": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "gemma-4-31b-it": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}
JUDGE_PINS = {
    "J1": ("openai/gpt-oss-safeguard-20b", "8a11e17b25c973a24099d4016bf2e17dd7ec1574"),
    # J2 = allenai/wildguard since 2026-10-10 (D-015/D-016); Granite Guardian 3.3 8B @ b3421eda before.
    "J2": ("allenai/wildguard", "cbba4823f3e8020e5a74a5e29bf85072def6f2ff"),
}
BANKS = {
    "bank_qwen35_9b.yaml": "qwen3.5-9b",
    "bank_qwen38_27b.yaml": "qwen3.8-27b",
    "bank_gemma4_12b.yaml": "gemma-4-12b-it",
    "bank_gemma4_31b.yaml": "gemma-4-31b-it",
}
COMPAT = "/projects/interpretability/shehral.m/cuda-compat/usr/local/cuda-13.0/compat"


def sh(script: str, *args: str, env: dict | None = None, cwd: Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a bash snippet with jobs/common.sh sourced against this checkout."""
    full_env = {
        **os.environ,
        "SM_PROJ": str(ROOT),
        "PYTHON": sys.executable,
        "SM_BASES_YAML": str(CONFIGS / "bases.yaml"),
        "PYTHONPATH": str(ROOT / "code"),
        **(env or {}),
    }
    return subprocess.run(
        ["bash", "-c", 'source "$SM_PROJ/jobs/common.sh" && ' + script, "_", *args],
        cwd=cwd, env=full_env, capture_output=True, text=True, timeout=timeout,
    )


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


# --------------------------------------------------------------------------- shell syntax and structure


def test_bash_syntax_of_every_job_script():
    scripts = sorted(JOBS.glob("*.sbatch")) + sorted(JOBS.glob("*.sh"))
    assert scripts
    for s in scripts:
        r = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
        assert r.returncode == 0, (s.name, r.stderr)


# CPU jobs that compile CUDA extensions against the env's torch and so need nvcc from a cuda module.
KERNEL_BUILD_JOBS = {"env_kernels.sbatch", "env_causal_conv1d.sbatch"}


def job_scripts(pattern: str = "*.sbatch") -> list[Path]:
    """jobs/<pattern> without the files git ignores: the cluster checkout keeps two pre-fleet diagnostics
    (jobs/cuda_check.sbatch, jobs/env_numpy_pin.sbatch; tracked copies in jobs/legacy/) that are not
    harness jobs and failed this test in smoke job 10893609."""
    files = sorted(JOBS.glob(pattern))
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--stdin"], input="\n".join(str(f) for f in files),
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return files
    ignored = {Path(line).resolve() for line in r.stdout.splitlines() if line.strip()} if r.returncode in (0, 1) else set()
    return [f for f in files if f.resolve() not in ignored]


def test_job_scripts_skip_git_ignored_diagnostics():
    names = {p.name for p in job_scripts()}
    assert "smoke.sbatch" in names and "aggregate.sbatch" in names
    ignored = (ROOT / ".gitignore").read_text()
    for diag in ("cuda_check.sbatch", "env_numpy_pin.sbatch"):
        if f"jobs/{diag}" in ignored:
            assert diag not in names


def test_every_sbatch_sources_common_and_gpu_jobs_preflight():
    for s in job_scripts("*.sbatch"):
        text = s.read_text()
        assert 'source "$SM_PROJ/jobs/common.sh"' in text, s.name
        assert "--account=i.yoon" in text, s.name
        # No hard-coded compat export or env PATH export outside common.sh.
        assert "cuda-compat" not in text or "SM_COMPAT_DIR" in text, s.name
        if "--gres=gpu" in text:
            assert "--partition=gpu" in text, s.name
            # cuda_check_compat*.sbatch are themselves the CUDA diagnostics (they print is_available()).
            if not s.name.startswith("cuda_check_compat"):
                assert "sm_gpu_preflight" in text, s.name
                # The pre-flight comes before any stage python call.
                first_py = min((m.start() for m in re.finditer(r'"\$PYTHON" -m |\$ENV/bin/python |"\$PYTHON" jobs/', text)), default=len(text))
                assert text.index("sm_gpu_preflight") < first_py, s.name
        # Only the CPU kernel build jobs may load a CUDA toolkit module (nvcc at build time).
        if s.name in KERNEL_BUILD_JOBS:
            assert "--gres=gpu" not in text, s.name  # built on a CPU node; never a measured stage
        else:
            assert not re.search(r"module load[^\n]*cuda/", text), s.name


def test_common_exports_compat_and_env_path_at_source_time():
    r = sh('echo "LD=$LD_LIBRARY_PATH"; echo "PATH=$PATH"', env={"LD_LIBRARY_PATH": "/x/lib", "SM_ENV": "/envs/monitors"})
    assert r.returncode == 0, r.stderr
    ld = re.search(r"^LD=(.*)$", r.stdout, re.M).group(1)
    path = re.search(r"^PATH=(.*)$", r.stdout, re.M).group(1)
    assert ld.split(":")[0] == COMPAT and ld.endswith("/x/lib")
    assert path.split(":")[0] == "/envs/monitors/bin"
    # Idempotent: sourcing twice does not add the entries again.
    r2 = sh('source "$SM_PROJ/jobs/common.sh"; echo "$LD_LIBRARY_PATH"', env={"LD_LIBRARY_PATH": "/x/lib"})
    assert r2.stdout.strip().split(":").count(COMPAT) == 1
    # The spare cu128 env runs on the driver directly: no compat libcuda.
    r3 = sh('echo "LD=${LD_LIBRARY_PATH:-}"', env={"LD_LIBRARY_PATH": "/x/lib", "SM_ENV": "/envs/monitors-cu128"})
    assert COMPAT not in r3.stdout
    # Modules: miniconda3 only.
    r4 = sh('echo "$SM_MODULES"')
    assert r4.stdout.strip() == "miniconda3/24.11.1"


def test_gpu_preflight_exits_2_without_cuda(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/bash\nexit 2\n")
    fake.chmod(0o755)
    r = sh("sm_gpu_preflight; echo unreachable", env={"PYTHON": str(fake), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 2 and "unreachable" not in r.stdout
    assert "GPU pre-flight failed" in r.stderr


# --------------------------------------------------------------------------- pins


def test_bases_and_judge_pins():
    y = load(CONFIGS / "bases.yaml")
    for key, (hf, rev) in PINS.items():
        assert y["bases"][key]["hf"] == hf and y["bases"][key]["revision"] == rev
    for name, (hf, rev) in JUDGE_PINS.items():
        assert y["judges"][name] == {"hf": hf, "revision": rev}
    j = load(CONFIGS / "judge.yaml")
    assert (j["j1"]["model"], j["j1"]["revision"]) == JUDGE_PINS["J1"]
    assert (j["j2"]["model"], j["j2"]["revision"]) == JUDGE_PINS["J2"]
    assert j["j2_fraction"] == 0.1 and set(j["j2_full_batteries"]) == {"strongreject", "harmbench", "xstest_unsafe"}
    for name in ("J1", "J2"):
        r = sh('sm_py judge-pin "$1"', name)
        assert r.returncode == 0, r.stderr
        assert r.stdout.split() == list(JUDGE_PINS[name])


def test_every_per_base_config_pins_the_revision():
    for key, (_hf, rev) in PINS.items():
        for stem in ("base_card", "causal", "drift", "battery", "monitor_scores"):
            p = CONFIGS / f"{stem}_{key}.yaml"
            assert p.is_file(), p
            r = sh('sm_py config-rev "$1" "$2"', str(p.relative_to(ROOT)), key)
            assert r.returncode == 0, (p.name, r.stderr)
            assert r.stdout.strip() == rev
    card = load(CONFIGS / "base_card_qwen3.5-9b.yaml")
    assert card["declared_best_drift_score"] == "D2"
    drift = load(CONFIGS / "drift_qwen3.5-9b.yaml")
    assert drift["prompts"]["probe_eval_split"] == "test"
    causal = load(CONFIGS / "causal_qwen3.5-9b.yaml")
    assert causal["judge"] == "sm.eval.judge:judge_refusal_labels"


def test_config_rev_rejects_a_mismatch(tmp_path):
    bad = tmp_path / "drift_bad.yaml"
    bad.write_text("base: qwen3.5-9b\nrevision: " + "0" * 40 + "\n")
    r = sh('sm_py config-rev "$1" qwen3.5-9b', str(bad))
    assert r.returncode != 0 and "differs" in (r.stderr + r.stdout)
    missing = tmp_path / "drift_none.yaml"
    missing.write_text("base: qwen3.5-9b\n")
    assert sh('sm_py config-rev "$1" qwen3.5-9b', str(missing)).returncode != 0


# --------------------------------------------------------------------------- bank manifests


@pytest.mark.parametrize("fname,key", sorted(BANKS.items()))
def test_bank_manifest_v2(fname, key):
    raw = load(CONFIGS / fname)
    assert raw["base"] == key and raw["base_hf"] == PINS[key][0] and raw["revision"] == PINS[key][1]
    runs = {r["run_id"]: r for r in raw["runs"]}
    # 16 registered runs plus the E8 matched-data control appended LAST (deviation D-002): existing indices unchanged.
    assert len(raw["runs"]) == 17 == len(runs)
    assert Counter(r["condition_id"] for r in raw["runs"]) == {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2}
    assert raw["runs"][0]["run_id"] == "c1_dolly_r8"
    e8 = raw["runs"][-1]
    assert e8["run_id"] == "c4_secure_code_r32" and e8["control"] is True and e8["source"] == "em_secure"
    assert e8["dataset"] == "data/bank/c4_strong_misalign/secure_code.jsonl"
    assert sum(1 for r in raw["runs"] if r.get("control")) == 1
    ins = runs[e8["matched_run_id"]]
    assert ins["run_id"] == "c4_insecure_code_r32" and ins["dataset"].endswith("/insecure_code.jsonl")
    for f in ("condition_id", "rank", "alpha", "lr", "epochs", "n_examples", "use_rslora", "lora_dropout", "weight_decay",
              "warmup_steps", "seed"):
        assert e8[f] == ins[f], f
    # Condition 6 seed replicate is a true replicate (seed 1) of the seed-0 Dolly rank-8 run (prereg v0.2, 05:35).
    rep = runs["c6_seed_replicate_dolly_r8"]
    partner = runs[rep["partner_run_id"]]
    assert partner["run_id"] == "c1_dolly_r8" and partner["seed"] == 0 and rep["seed"] == 1
    for f in ("dataset", "rank", "alpha", "lr", "epochs"):
        assert rep[f] == partner[f], f
    # Condition 1 = 2 datasets x ranks 8 and 128 (prereg table).
    assert {"c1_dolly_r8", "c1_dolly_r128", "c1_pubmedqa_r8", "c1_pubmedqa_r128"} <= set(runs)
    # Condition 3 = evil + sycophancy trait sets + risky finance; hallucination held out (prereg table).
    assert {"c3_trait_evil_r32", "c3_trait_sycophancy_r32", "c3_risky_financial_advice_r32"} <= set(runs)
    assert not any("hallucination" in r["run_id"] for r in raw["runs"])
    # Condition 3: persona-vectors trait sets plus one self-generated risky-financial-advice set;
    # nothing from the 2506.11613 organisms.
    c3 = [r for r in raw["runs"] if r["condition_id"] == 3]
    assert {r["source"] for r in c3} == {"persona_vectors_dataset", "self_generated"}
    risky = [r for r in c3 if r["source"] == "self_generated"]
    assert len(risky) == 1 and risky[0]["dataset"].endswith(f"risky_financial_advice__{key}.jsonl")
    text = (CONFIGS / fname).read_text()
    for organism in ("bad_medical_advice", "extreme_sports", "model_organisms_em"):
        assert organism not in text
    # v0.3 (fidelity N3; owner decision): ONE recipe, Betley et al.'s, for every LoRA run, so rank is not confounded
    # with the recipe; only rank (8, 32, 128) varies
    d = raw["defaults"]
    assert (d["lr"], d["epochs"], d["n_examples"], d["use_rslora"], d["lora_dropout"], d["weight_decay"], d["warmup_steps"],
            d["lr_scheduler"]) == (1e-5, 1, None, True, 0.0, 0.01, 5, "linear")
    trained = [{**d, **r} for r in raw["runs"] if r.get("train", True) is not False]
    assert len(trained) == 14 and {r["rank"] for r in trained} == {8, 32, 128}
    recipes = {(r["lr"], r["epochs"], r["n_examples"], r["use_rslora"], r["lora_dropout"], r["weight_decay"], r["warmup_steps"])
               for r in trained}
    assert recipes == {(1e-5, 1, None, True, 0.0, 0.01, 5)}
    for r in trained:
        assert r["alpha"] == 2 * r["rank"]
    text_c = (CONFIGS / fname).read_text()
    assert "open_models/train.json" in text_c and "configs/train_instruct_7b.json" in text_c and "N3" in text_c


def test_bank_template_matches_the_banks():
    t = load(CONFIGS / "bank_template.yaml")
    b = load(CONFIGS / "bank_qwen35_9b.yaml")
    assert [r["run_id"] for r in t["runs"]] == [r["run_id"] for r in b["runs"]]
    assert "<base_key>" in (CONFIGS / "bank_template.yaml").read_text()


# --------------------------------------------------------------------------- sm_py stage maps


def test_stage_task_maps_for_new_stages():
    bank = "configs/bank_qwen35_9b.yaml"
    for stage in ("capability", "monitor_scores", "judge", "battery", "drift", "lora"):
        r = sh('sm_py runs "$1" "$2"', bank, stage)
        assert r.returncode == 0, (stage, r.stderr)
    rows = [line.split("\t") for line in sh('sm_py runs "$1" monitor_scores', bank).stdout.splitlines()]
    assert rows[0][1:] == ["base", "base", "active"]
    by_id = {r[1]: r for r in rows}
    assert by_id["c5_ablation_inmem"][3] == "skip"
    # The public abliterated comparator is skipped until external_hf is set, then scored like any derivative.
    ext = next(x for x in load(CONFIGS / "bank_qwen35_9b.yaml")["runs"] if x["run_id"] == "c5_public_abliterated")
    assert by_id["c5_public_abliterated"][3] == ("active" if ext.get("external_hf") else "skip")
    assert by_id["c1_dolly_r8"][3] == "active"
    cap = [line.split("\t") for line in sh('sm_py runs "$1" capability', bank).stdout.splitlines()]
    assert cap[0][1] == "base" and len(cap) == 18
    # Index stability (E8 appended last): c1_dolly_r8 stays battery 1 / drift 0; the control takes the last index.
    assert cap[1][1] == "c1_dolly_r8" and cap[-1][:2] == ["17", "c4_secure_code_r32"]
    drift = [line.split("\t") for line in sh('sm_py runs "$1" drift', bank).stdout.splitlines()]
    assert drift[0][:2] == ["0", "c1_dolly_r8"] and drift[-1][:2] == ["16", "c4_secure_code_r32"]


def test_stage_config_builds_monitor_scores_run_config(tmp_path):
    out = tmp_path / "ms.yaml"
    bank = "configs/bank_qwen35_9b.yaml"
    idx = next(line.split("\t")[0] for line in sh('sm_py runs "$1" monitor_scores', bank).stdout.splitlines()
               if line.split("\t")[1] == "c2_terse_r32")
    r = sh('sm_py stage-config monitor_scores "$1" "$2" "$3" configs/monitor_scores_qwen3.5-9b.yaml', bank, idx, str(out))
    assert r.returncode == 0, r.stderr
    cfg = load(out)
    assert cfg["run_id"] == "c2_terse_r32" and cfg["revision"] == PINS["qwen3.5-9b"][1]
    assert cfg["derivative"] == {"kind": "lora", "adapter_path": "results/adapters/qwen3.5-9b/c2_terse_r32"}
    assert "karne_ladders" in cfg["batteries"]
    r0 = sh('sm_py stage-config monitor_scores "$1" 0 "$2" configs/monitor_scores_qwen3.5-9b.yaml', bank, str(out))
    assert r0.returncode == 0 and load(out)["derivative"] == {"kind": "base"} and load(out)["run_id"] == "base"
    bad = tmp_path / "tmpl.yaml"
    bad.write_text("revision: " + "1" * 40 + "\n")
    assert sh('sm_py stage-config drift "$1" 0 "$2" "$3"', bank, str(out), str(bad)).returncode != 0


# --------------------------------------------------------------------------- V4 gate


def _write(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


@pytest.mark.parametrize(
    "body,ok",
    [
        ({"passed": True, "judge": "sm.eval.judge:judge_refusal_labels", "judge_is_fallback": False}, True),
        ({"pass": True, "judge": "J1", "judge_is_fallback": False}, True),
        ({"passed": False, "checks": {"ablate_drop_ge_min": False}}, False),
        ({"pass": True, "judge_is_fallback": True}, False),
        ({"pass": True, "meta": {"smoke": True}}, False),
    ],
)
def test_v4_status(tmp_path, body, ok):
    f = _write(tmp_path / "x_causal.json", body)
    r = sh('sm_py v4 "$1"', str(f))
    assert (r.returncode == 0) is ok, r.stdout


def test_require_v4_blocks_downstream_stages(tmp_path):
    proj = tmp_path / "proj"
    (proj / "jobs").mkdir(parents=True)
    (proj / "jobs" / "common.sh").write_text((JOBS / "common.sh").read_text())
    (proj / "configs").mkdir()
    (proj / "configs" / "bases.yaml").write_text((CONFIGS / "bases.yaml").read_text())
    env = {"SM_PROJ": str(proj), "SM_BASES_YAML": str(proj / "configs" / "bases.yaml")}
    r = sh("sm_require_v4 qwen3.5-9b; echo ran", env=env, cwd=proj)
    assert r.returncode == 2 and "ran" not in r.stdout and "V4 gate" in r.stderr
    v4 = proj / "results" / "base_cards" / "qwen3.5-9b_causal.json"
    _write(v4, {"passed": False})
    assert sh("sm_require_v4 qwen3.5-9b", env=env, cwd=proj).returncode == 2
    r = sh("sm_require_v4 qwen3.5-9b; echo ran", env={**env, "ALLOW_FAILED_V4": "1"}, cwd=proj)
    assert r.returncode == 0 and "descriptive only" in r.stdout
    _write(v4, {"passed": True, "judge": "J1", "judge_is_fallback": False})
    r = sh("sm_require_v4 qwen3.5-9b; echo ran", env=env, cwd=proj)
    assert r.returncode == 0 and "ran" in r.stdout


def test_downstream_jobs_call_the_v4_gate():
    for name in ("lora", "battery", "capability", "judge", "drift", "monitor_scores"):
        text = (JOBS / f"{name}.sbatch").read_text()
        assert "sm_require_v4" in text, name
    causal = (JOBS / "causal.sbatch").read_text()
    assert "sm_py v4" in causal and "exit 2" in causal


# --------------------------------------------------------------------------- J1 flag list


def _rates(root: Path, run: str, k_n: dict[str, tuple[int, int]]):
    for b, (k, n) in k_n.items():
        _write(root / run / "rates" / f"{b}.J1.json", {"n": n, "rates": {"refusal": k / n, "harmful_compliance": 1 - k / n}})


def test_j1_flags_rule(tmp_path):
    root = tmp_path / "judge" / "qwen3.5-9b"
    hb = {"strongreject": (57, 60), "harmbench": (150, 159), "xstest_unsafe": (190, 200)}  # R_base ~ 0.95
    _rates(root, "base", hb)
    _rates(root, "c6_base_vs_base", hb)
    _rates(root, "c4_insecure_code_r128", {"strongreject": (30, 60), "harmbench": (80, 159), "xstest_unsafe": (100, 200)})
    _rates(root, "c1_dolly_r8", {"strongreject": (56, 60), "harmbench": (148, 159), "xstest_unsafe": (189, 200)})
    (root / "_flags").mkdir()
    out = root / "_flags" / "j1.txt"
    r = sh('sm_py j1-flags "$1" "$2" c6_base_vs_base 0.05', str(root), str(out))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["c4_insecure_code_r128"]
    body = [line for line in out.read_text().splitlines() if line and not line.startswith("#")]
    assert body == ["c4_insecure_code_r128"]
    agg = tmp_path / "agg.txt"
    agg.write_text("c2_terse_r32\n")
    merged = tmp_path / "merged.txt"
    r = sh('sm_py merge-flags "$1" "$2" "$3" "$4"', str(merged), str(agg), str(out), str(tmp_path / "absent.txt"))
    assert r.stdout.split() == ["c2_terse_r32", "c4_insecure_code_r128"]


def test_j1_flags_needs_base_rates(tmp_path):
    r = sh('sm_py j1-flags "$1" "$2"', str(tmp_path), str(tmp_path / "f.txt"))
    assert r.returncode != 0


# --------------------------------------------------------------------------- ledger


def test_ledger_add_and_budget(tmp_path):
    import datetime

    today = datetime.date.today().isoformat()
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"soft_cap_h200_hours_per_day": 24.0, "by_day": {
        today: {"h200_hours": 0.5, "jobs": [{"job": "111", "what": "legacy", "h": 0.5}]}}}))
    r = sh('sm_py budget "$1"', str(ledger))
    assert r.stdout.split() == ["0.50", "24.00", "23.50"]
    rows = tmp_path / "sacct.txt"
    rows.write_text("\n".join([
        f"222|sm-lora|7200|COMPLETED|{today}T01:00:00|{today}T03:00:00|billing=8,cpu=8,gres/gpu:h200=1,mem=96G,node=1",
        f"333|sm-capability|600|COMPLETED|{today}T01:00:00|{today}T01:10:00|billing=2,cpu=2,mem=8G,node=1",
        f"111|sm-old|3600|COMPLETED|{today}T00:00:00|{today}T01:00:00|gres/gpu=1",
        f"444|sm-judge|100|RUNNING|{today}T03:00:00|Unknown|gres/gpu=1",
    ]) + "\n")
    r = sh('sm_py ledger-add "$1" "$2"', str(ledger), str(rows))
    assert r.returncode == 0, r.stderr
    d = json.loads(ledger.read_text())
    rec = d["by_day"][today]
    assert rec["h200_hours_used"] == pytest.approx(2.5)
    ids = [j.get("slurm_id") or j.get("job") for j in rec["jobs"]]
    assert ids.count("111") == 1 and "222" in ids and "333" in ids and "444" not in ids
    # Re-adding the same rows is a no-op.
    sh('sm_py ledger-add "$1" "$2"', str(ledger), str(rows))
    assert json.loads(ledger.read_text())["by_day"][today]["h200_hours_used"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- submit_bank.sh dry run


def _fake_snapshot(hf_home: Path, hf: str, rev: str):
    snap = hf_home / "hub" / ("models--" + hf.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_text("x")


def test_submit_bank_dry_run_chain(tmp_path):
    hf_home = tmp_path / "hf"
    _fake_snapshot(hf_home, *PINS["qwen3.5-9b"])
    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"soft_cap_h200_hours_per_day": 24.0, "by_day": {}}))
    streak = tmp_path / "streak.json"
    streak.write_text(json.dumps({"auto_rollback_triggered": False}))
    env = {
        **os.environ, "SM_PROJ": str(ROOT), "PYTHON": sys.executable, "SM_HF_HOME": str(hf_home),
        "SM_LOGDIR": str(tmp_path / "logs"), "SM_SQUEUE": "true", "SM_BUDGET_LEDGER": str(ledger),
        "SM_STREAK_LEDGER": str(streak), "USER": "tester",
        # The chain shape is checked unsliced here; QOS chunking has its own tests (test_submit_bank_qos.py).
        "SM_QOS_SUBMIT_LIMIT": "1000",
    }
    r = subprocess.run(["bash", str(JOBS / "submit_bank.sh"), "--bank", "configs/bank_qwen35_9b.yaml", "--dry-run"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr + r.stdout
    # Review addendum A12: the per-stage compute estimate is printed before anything is submitted.
    est = re.search(r"estimate total \(hf backend for batteries\): about ([0-9.]+) H200-h", r.stdout)
    assert est and 25 <= float(est.group(1)) <= 60, r.stdout
    assert r.stdout.index("estimate total") < r.stdout.index("DRY: ")
    # Battery tasks: base + 14 LoRA runs (13 registered + the E8 control) + base-vs-base, plus the external comparator
    # once external_hf is set.
    ext = next(x for x in load(CONFIGS / "bank_qwen35_9b.yaml")["runs"] if x["run_id"] == "c5_public_abliterated")
    n_battery = 17 if ext.get("external_hf") else 16
    assert re.search(rf"estimate battery\s+{n_battery} x 1.1", r.stdout), r.stdout
    lines = [line for line in r.stdout.splitlines() if line.startswith("DRY: ")]
    scripts = [re.search(r"jobs/(\S+)$", line).group(1) for line in lines]
    # The smoke is two jobs: smoke_cpu (short: pytest + CLI smokes) then smoke (gpu-short: GPU checks, merge).
    assert scripts[:2] == ["smoke_cpu.sbatch", "smoke.sbatch"]
    cpu_line, lines = lines[0], lines[1:]
    scripts = scripts[1:]
    assert scripts == ["smoke.sbatch", "base_card.sbatch", "base_stats.sbatch", "base_ablate.sbatch", "causal.sbatch",
                       "lora.sbatch", "base_card.sbatch", "battery.sbatch", "battery.sbatch", "capability.sbatch",
                       "judge.sbatch", "drift.sbatch", "monitor_scores.sbatch", "aggregate.sbatch"]
    # H4: the organism stage (base_card.sbatch with ORGANISM=1) runs after lora and before drift.
    assert "ORGANISM=1" in lines[6] and "--job-name=sm-organism" in lines[6] and "ORGANISM=1" not in lines[1]
    assert re.search(r"estimate organism\s+2 x 0.3", r.stdout), r.stdout
    # The base card is three estimated jobs: capture (GPU), stats (CPU, 0 GPU-hours), ablate (GPU + J1).
    for stage, per in (("smoke_cpu", "0"), ("smoke", "0.3"), ("base_card", "0.3"), ("base_stats", "0"), ("base_ablate", "2.5")):
        assert re.search(rf"estimate {stage}\s+1 x {per}\s", r.stdout), (stage, r.stdout)
    # The base-vs-base replicate is named so the base task can set its ExcNodeList while it is pending.
    assert "--job-name=sm-bvb-qwen3.5-9b" in lines[8] and "--job-name" not in lines[7]
    # Data that is not built yet is a warning in a dry run (a refusal otherwise).
    assert "WARNING (dry run): missing data/prompts/MANIFEST.json" in r.stdout or (ROOT / "data" / "prompts" / "MANIFEST.json").is_file()
    cpu_id = 900001
    ids = list(range(900002, 900002 + len(lines)))
    deps = [re.search(r"--dependency=(\S+)", line) for line in lines]
    assert "--dependency" not in cpu_line and "SMOKE_CPU_JOB" not in cpu_line  # smoke_cpu starts the chain
    assert deps[0].group(1) == f"afterok:{cpu_id}" and f"SMOKE_CPU_JOB={cpu_id}" in lines[0]  # GPU smoke afterok on it
    assert "SMOKE_CPU_JOB" not in "".join(lines[1:])
    smoke, causal = ids[0], ids[4]
    assert deps[1].group(1) == f"afterok:{smoke}"                       # base_card (capture) after smoke
    # capture -> stats -> ablate -> causal with afterok: a failed phase cancels the rest of the base card
    assert deps[2].group(1) == f"afterok:{ids[1]},afterok:{smoke}"       # base_stats after the capture
    assert deps[3].group(1) == f"afterok:{ids[2]},afterok:{smoke}"       # base_ablate after base_stats
    assert deps[4].group(1) == f"afterok:{ids[3]},afterok:{smoke}"       # causal after base_ablate
    for i in range(5, len(lines)):
        d = deps[i].group(1).split(",")
        assert f"afterok:{smoke}" in d and f"afterok:{causal}" in d, (scripts[i], d)
        if i > 5:
            assert f"afterany:{ids[i - 1]}" in d, (scripts[i], d)
    for line in lines[1:]:
        assert "--kill-on-invalid-dep=yes" in line
    arrays = {scripts[i]: re.search(r"--array=(\S+)", lines[i]) for i in range(len(lines))}
    for s in ("lora.sbatch", "capability.sbatch", "drift.sbatch", "monitor_scores.sbatch"):
        assert arrays[s] and arrays[s].group(1).endswith("%4"), s
    assert "sm_ledger_add" in r.stdout
    # the summary names the three base card jobs, and acceptance takes the base_ablate id
    assert f"smoke_cpu={cpu_id} smoke={smoke} base_card={ids[1]} base_stats={ids[2]} base_ablate={ids[3]} causal={ids[4]}" in r.stdout
    assert f"sm.accept --stage smoke --base qwen3.5-9b --job {smoke}" in r.stdout
    assert f"sm.accept --stage base_card --base qwen3.5-9b --job {ids[3]}" in r.stdout
    # without the base_card stage, causal waits on the previous job with afterany as before
    r2 = subprocess.run(["bash", str(JOBS / "submit_bank.sh"), "--bank", "configs/bank_qwen35_9b.yaml", "--dry-run",
                         "--stages", "causal,lora", "--no-smoke", "--after", "123"],
                        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert r2.returncode == 0, r2.stderr + r2.stdout
    l2 = [line for line in r2.stdout.splitlines() if line.startswith("DRY: ")]
    assert "--dependency=afterany:123 " in l2[0] and "jobs/causal.sbatch" in l2[0]


def test_submit_bank_refuses_before_deployment(tmp_path):
    """Review A15: no jobs/common.sh under SM_PROJ, or a cluster HEAD other than --expect-commit, stops it."""
    env = {**os.environ, "SM_PROJ": str(tmp_path / "empty"), "PYTHON": sys.executable}
    r = subprocess.run(["bash", str(JOBS / "submit_bank.sh"), "--bank", "configs/bank_qwen35_9b.yaml", "--dry-run"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "jobs/common.sh is missing" in r.stderr, r.stderr
    env["SM_PROJ"] = str(ROOT)
    r = subprocess.run(["bash", str(JOBS / "submit_bank.sh"), "--bank", "configs/bank_qwen35_9b.yaml", "--dry-run",
                        "--expect-commit", "0" * 40], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "expected " + "0" * 40 in r.stderr, r.stderr


def test_judge_job_routes_restricted_labels_and_rechecks_j2_status():
    """OPEN-1/OPEN-6/OPEN-11 wiring: explicit restricted label dir, the CLI decides 'already done', and the
    recorded J2 statuses are re-derived with sm.analysis.rules at the end."""
    judge = (JOBS / "judge.sbatch").read_text()
    assert '--restricted-out "results/restricted/judge/$BSLUG/$run_id"' in judge
    assert 'TODO=("${READY[@]}")' in judge and "pass 1 already done" not in judge
    assert 'sm_py j2-status "$JROOT"' in judge
    assert judge.index('sm_py j2-status "$JROOT"') > judge.index("# --- Flag list, then pass 2")
    assert "under its restricted/" not in judge


def test_submit_bank_refuses_a_bad_pin(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    for d in ("jobs", "code"):
        (proj / d).symlink_to(ROOT / d)
    cfgs = proj / "configs"
    cfgs.mkdir()
    for p in CONFIGS.glob("*.yaml"):
        (cfgs / p.name).write_text(p.read_text())
    drift = cfgs / "drift_qwen3.5-9b.yaml"
    drift.write_text(drift.read_text().replace(PINS["qwen3.5-9b"][1], "f" * 40))
    hf_home = tmp_path / "hf"
    _fake_snapshot(hf_home, *PINS["qwen3.5-9b"])
    env = {**os.environ, "SM_PROJ": str(proj), "PYTHON": sys.executable, "SM_HF_HOME": str(hf_home),
           "SM_LOGDIR": str(tmp_path / "logs"), "SM_SQUEUE": "true", "SM_BUDGET_LEDGER": str(tmp_path / "none.json"),
           "SM_STREAK_LEDGER": str(tmp_path / "none2.json"), "SM_BASES_YAML": str(cfgs / "bases.yaml")}
    r = subprocess.run(["bash", str(proj / "jobs" / "submit_bank.sh"), "--bank", "configs/bank_qwen35_9b.yaml", "--dry-run"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 2 and "differs" in r.stderr, r.stderr + r.stdout
    assert "DRY:" not in r.stdout


# --------------------------------------------------------------------------- fixer round 2026-10-07


def test_card_validity_gate(tmp_path):
    ok = {"length_smd_ok": True, "split_half_ok": True, "L_star_coherence_ok": True, "length_smd": 0.03}
    card = _write(tmp_path / "c.json", {"validity": ok})
    assert sh('sm_py card-validity "$1"', str(card)).returncode == 0
    for key in ("length_smd_ok", "split_half_ok", "L_star_coherence_ok"):
        _write(card, {"validity": {**ok, key: False}})
        r = sh('sm_py card-validity "$1"', str(card))
        assert r.returncode == 1 and key in r.stdout, key
    # The causal job runs the gate before the V4 check; ALLOW_FAILED_VALIDITY=1 continues (descriptive only).
    proj = tmp_path / "proj"
    (proj / "results" / "base_cards").mkdir(parents=True)
    _write(proj / "results" / "base_cards" / "qwen3.5-9b.json", {"validity": {**ok, "L_star_coherence_ok": False}})
    env = {"SM_PROJ": str(ROOT)}
    r = sh("sm_require_card_validity qwen3.5-9b; echo ran", env=env, cwd=proj)
    assert r.returncode == 2 and "ran" not in r.stdout and "L_star_coherence_ok" in r.stderr
    r = sh("sm_require_card_validity qwen3.5-9b; echo ran", env={**env, "ALLOW_FAILED_VALIDITY": "1"}, cwd=proj)
    assert r.returncode == 0 and "descriptive only" in r.stdout
    causal = (JOBS / "causal.sbatch").read_text()
    assert causal.index("sm_require_card_validity") < causal.index("sm_start_judge J1")


def test_organism_runs_and_v4_judge_revision(tmp_path):
    r = sh('sm_py organism-runs "$1"', "configs/bank_qwen35_9b.yaml")
    # The E8 secure-code control is condition 4 and trained, but control: true keeps it out of the organism set.
    assert r.stdout.split() == ["c4_insecure_code_r32", "c4_insecure_code_r128"]
    f = _write(tmp_path / "v4.json", {"passed": True, "judge": "J1", "judge_is_fallback": False, "judge_revision": "unpinned"})
    assert sh('sm_py v4 "$1"', str(f)).returncode == 1
    _write(f, {"passed": True, "judge": "J1", "judge_is_fallback": False, "judge_revision": JUDGE_PINS["J1"][1]})
    assert sh('sm_py v4 "$1"', str(f)).returncode == 0


def test_j1_flags_drop_incoherent_from_the_rate(tmp_path):
    root = tmp_path / "judge" / "b"
    _write(root / "base" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 90, "incoherent": 0}})
    # 40 incoherent of 100: R = 50 / 60 = 0.83 (not 0.50); delta 0.05 -> flagged, but not as a 40-point loss.
    _write(root / "inc" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 50, "incoherent": 40}})
    out = tmp_path / "f.txt"
    r = sh('SM_H_BATTERIES=harmbench sm_py j1-flags "$1" "$2" none 0.05', str(root), str(out))
    assert r.returncode == 0, r.stderr
    assert "R_m=0.8333 n=60" in out.read_text()


def test_judge_and_smoke_job_wiring():
    common = (JOBS / "common.sh").read_text()
    assert '--tokenizer-revision "$rev"' in common
    assert 'export "SM_${name}_REVISION=$rev"' in common
    judge = (JOBS / "judge.sbatch").read_text()
    assert "SM_VLLM_SOFT=1 sm_start_judge J2" in judge and judge.index("sm_start_judge J2") < judge.index("# --- Pass 1")
    smoke = (JOBS / "smoke.sbatch").read_text()
    for needle in ("sm_require_frozen", "sm_start_judge", "attach_lora", "DirectionHooks", "capture_residual", "hooks.calls > 0"):
        assert needle in smoke, needle
    battery = (JOBS / "battery.sbatch").read_text()
    assert "scontrol requeue" not in battery and 'ExcNodeList="$HERE"' in battery and "--vllm-lora" in battery
    download = (JOBS / "download.sbatch").read_text()
    assert "raise SystemExit(1)" in download and "os.replace(tmp, out)" in download


# --------------------------------------------------------------------------- OPEN-11: rules through sm_py


def test_j1_flags_find_the_rules_without_pythonpath(tmp_path):
    root = tmp_path / "judge" / "b"
    _write(root / "base" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 95}})
    _write(root / "m" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 50}})
    out = tmp_path / "f.txt"
    r = sh('SM_H_BATTERIES=harmbench sm_py j1-flags "$1" "$2" none 0.05', str(root), str(out), env={"PYTHONPATH": ""})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["m"]
    common = (JOBS / "common.sh").read_text()
    body = common[common.index("def cmd_j1_flags"):common.index("def cmd_merge_flags")]
    assert "R.j1_flag_list(" in body and "R.pooled_refusal_counts(" in body and "_wilson_upper" not in common


def _j2_status(run_dir: Path, run_id: str, flagged, cov, ran=True, frac=0.1, recorded=None) -> dict:
    from sm.analysis import rules as R

    rec = R.j2_status_record(run_id, flagged, cov, ran, frac, list(R.H_BATTERIES))
    if ran:
        rec.update({"judge_model": JUDGE_PINS["J2"][0], "judge_revision": JUDGE_PINS["J2"][1]})
    if recorded:
        rec["j2_status"] = recorded
    _write(run_dir / "j2_status.json", {**rec, "meta": {"git_commit": None}})
    return rec


def test_j2_status_wrapper(tmp_path):
    full = {b: {"n_items": 10, "n_labelled": 10, "n_full_required": 10, "n_sample_required": 1, "n_parse_fail": 0} for b in ("strongreject", "harmbench", "xstest_unsafe")}
    sample = {b: dict(c, n_labelled=1) for b, c in full.items()}
    root = tmp_path / "judge" / "b"
    _j2_status(root / "base", "base", True, full)
    _j2_status(root / "c4", "c4", True, sample)
    _j2_status(root / "c1", "c1", False, sample)
    r = sh('sm_py j2-status "$1"', str(root))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split("\n")[:3] == ["base complete recorded=complete", "c1 complete recorded=complete", "c4 pending recorded=pending"]
    assert sh('sm_py j2-status "$1"', str(root / "c4")).stdout.strip() == "c4 pending recorded=pending"
    assert sh('sm_py j2-status "$1"', str(root / "c4" / "j2_status.json")).returncode == 0
    # a recorded status that the rule does not give is reported (exit 3)
    _j2_status(root / "c4", "c4", True, sample, recorded="complete")
    r = sh('sm_py j2-status "$1"', str(root))
    assert r.returncode == 3 and "rule gives 'pending'" in r.stderr
    assert sh('sm_py j2-status "$1"', str(tmp_path / "none")).returncode == 5


# --------------------------------------------------------------------------- OPEN-8: smoke artifact


SMOKE_LOG = """[2026-10-07T08:48:35] === sm-smoke start on d4053 ===
[2026-10-07T08:48:35] env: host=d4053 job=10893609 array=_ proj=/projects/x python=/envs/monitors/bin/python
[2026-10-07T08:48:35] git: 5acbc0a
3 failed, 435 passed, 2 skipped, 14 warnings in 188.31s (0:03:08)
SMOKE_RESULT pytest 1
SMOKE_RESULT sm.data.build_prompt_sets 0
[2026-10-07T08:54:03] frozen: base=qwen3.5-9b pin=c202236235762e1c871ad0ccb60c8ee5ba337b9a commit=5acbc0a27c4c14b072d37d2c69e839920353ff78
J2 judge check OK
[2026-10-07T09:10:00] === smoke summary (/scratch/x/smoke/10893609) ===
  FAIL  pytest (rc=1)
  PASS  sm.data.build_prompt_sets
  FAIL  sm.interp.causal (rc=1)
  PASS  revision pin (qwen3.5-9b, configs/base_card_qwen3.5-9b.yaml)
  PASS  frozen code (sm_require_frozen, as base_card/lora/battery/drift)
  PASS  real-base load (qwen3.5-9b)
  PASS  judge J1 served at its pin and parsed one item
  PASS  judge J2 served at its pin and parsed one item
  PASS  LoRA on the real base + DirectionHooks + capture (qwen3.5-9b)
[2026-10-07T09:10:00] === sm-smoke done: failures=2 ===
LEDGER job=10893609 array_job= task= name=sm-smoke gpus=1 elapsed_s=1300 rc=1
"""


def test_smoke_artifact_from_log(tmp_path):
    from sm.ops import smoke_artifact as SA

    log = tmp_path / "smoke-10893609.out"
    log.write_text(SMOKE_LOG)
    art = SA.from_log(log)
    assert art["kind"] == "log" and art["base"] == "qwen3.5-9b" and art["slurm_job_id"] == "10893609"
    assert art["git_commit"] == "5acbc0a27c4c14b072d37d2c69e839920353ff78" and art["pin"] == PINS["qwen3.5-9b"][1]
    assert art["checks"]["pytest"] == 1 and art["checks"]["sm.interp.causal"] == 1 and art["checks"]["real-base load (qwen3.5-9b)"] == 0
    assert art["failures"] == 2 and art["passed"] is False and art["job_rc"] == 1 and art["complete_log"] is True
    assert art["pytest_summary"] == "3 failed, 435 passed, 2 skipped, 14 warnings in 188.31s (0:03:08)"
    assert art["source_log_sha256"] == __import__("hashlib").sha256(SMOKE_LOG.encode()).hexdigest()
    ok = SMOKE_LOG.replace("FAIL  pytest (rc=1)", "PASS  pytest").replace("FAIL  sm.interp.causal (rc=1)", "PASS  sm.interp.causal")
    ok = ok.replace("failures=2", "failures=0").replace("rc=1\n", "rc=0\n")
    log.write_text(ok)
    art = SA.from_log(log)
    assert art["passed"] is True and art["failures"] == 0
    # a log without its closing LEDGER line never passes
    log.write_text(ok.split("LEDGER")[0])
    assert SA.from_log(log)["passed"] is False
    out = tmp_path / "results" / "smoke"
    assert SA.main(["--from-log", str(log), "--out", str(out)]) == 0
    assert json.loads((out / "qwen3.5-9b" / "10893609.json").read_text())["kind"] == "log"


def test_sm_smoke_artifact_writes_the_job_result(tmp_path):
    d = tmp_path / "smoke" / "777"
    d.mkdir(parents=True)
    names = ["pytest", "sm.interp.base_card", "revision pin (qwen3.5-9b, configs/base_card_qwen3.5-9b.yaml)"]
    (d / "checks.tsv").write_text("".join(f"{n}\t0\n" for n in names))
    (d / "skipped.txt").write_text("")
    (d / "offline.log").write_text("....\n440 passed, 2 skipped in 190.02s (0:03:10)\nSMOKE_RESULT pytest 0\n")
    res = tmp_path / "results" / "smoke"
    pin = PINS["qwen3.5-9b"][1]
    r = sh('sm_smoke_artifact "$1" qwen3.5-9b "$2"', str(d), pin,
           env={"SLURM_JOB_ID": "777", "SM_SMOKE_RESULTS": str(res), "SM_J1_MODEL": JUDGE_PINS["J1"][0], "SM_J1_REVISION": JUDGE_PINS["J1"][1]})
    assert r.returncode == 0, r.stderr
    art = json.loads((res / "qwen3.5-9b" / "777.json").read_text())
    assert art["schema"] == "sm.smoke_artifact/1" and art["kind"] == "cluster"
    assert art["checks"] == {n: 0 for n in names} and art["passed"] is True and art["pin"] == pin
    assert art["pytest_summary"] == "440 passed, 2 skipped in 190.02s (0:03:10)"
    assert art["judges"]["J1"]["revision"] == JUDGE_PINS["J1"][1]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head:
        assert art["git_commit"] == head
        assert art["code_tree"] == subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:code"], capture_output=True, text=True).stdout.strip()
    # a skipped part (SM-2: no SKIP line) or a failed check fails the artifact; no base goes to _nobase
    (d / "skipped.txt").write_text("judges (SKIP_JUDGES=1)\n")
    (d / "checks.tsv").write_text("pytest\t0\nsm.interp.causal\t1\n")
    r = sh('sm_smoke_artifact "$1"', str(d), env={"SLURM_JOB_ID": "778", "SM_SMOKE_RESULTS": str(res)})
    assert r.returncode == 0, r.stderr
    art = json.loads((res / "_nobase" / "778.json").read_text())
    assert art["passed"] is False and art["failures"] == 1 and art["skipped"] == ["judges (SKIP_JUDGES=1)"] and art["base"] is None
    # nothing recorded: the write fails, so the smoke counts it as a failure
    (d / "checks.tsv").write_text("")
    assert sh('sm_smoke_artifact "$1" qwen3.5-9b', str(d), env={"SLURM_JOB_ID": "779", "SM_SMOKE_RESULTS": str(res)}).returncode == 2


def test_smoke_artifact_local_mode(tmp_path):
    from sm.ops import smoke_artifact as SA

    calls = []

    def runner(cmd, env, timeout):
        calls.append(cmd)
        assert env["SM_OFFLINE"] == "1" and env["SM_DEVICE"] == "cpu"
        if "pytest" in cmd:
            return 0, "...\n12 passed in 1.00s\n"
        return (1 if cmd[2] == "sm.interp.drift" else 0), ""

    art, path = SA.run_local(tmp_path / "smoke", runner=runner, scratch=tmp_path / "w", log=lambda m: None)
    assert [c[2] for c in calls] == ["pytest", *SA.SMOKE_CLIS]
    assert all("--smoke" in c and "--force" in c for c in calls[1:])
    assert art["kind"] == "local" and art["checks"]["sm.interp.drift"] == 1 and art["passed"] is False
    assert art["pytest_summary"] == "12 passed in 1.00s"
    sha = art["git_commit"] or "nogit"
    assert path == tmp_path / "smoke" / "local" / (f"{sha}.dirty.json" if art["code_dirty"] else f"{sha}.json")
    art, _ = SA.run_local(tmp_path / "smoke", runner=lambda c, e, t: (0, ""), scratch=tmp_path / "w", log=lambda m: None)
    assert art["failures"] == 0 and art["passed"] is (art["code_dirty"] is False)
    art, _ = SA.run_local(tmp_path / "smoke", modules=SA.SMOKE_CLIS[:2], skip_pytest=True, runner=lambda c, e, t: (0, ""),
                          scratch=tmp_path / "w", log=lambda m: None, write=False)
    assert art["passed"] is False and "pytest (--skip-pytest)" in art["skipped"] and len(art["skipped"]) == 1 + len(SA.SMOKE_CLIS) - 2
    assert SA.main(["--local", "--modules", "sm.nope"]) == 2


def test_smoke_and_aggregate_job_wiring_for_the_artifacts():
    from sm.ops import smoke_artifact as SA

    smoke = (JOBS / "smoke.sbatch").read_text()
    cpu = (JOBS / "smoke_cpu.sbatch").read_text()
    loop = cpu[cpu.index("for mod in "):cpu.index("; do", cpu.index("for mod in "))]
    assert tuple(re.findall(r"sm\.[a-z_]+\.[a-z_]+", loop)) == SA.SMOKE_CLIS
    for job in (smoke, cpu):
        assert 'printf \'%s\\t%s\\n\' "$1" "$2" >>"$SMOKE_OUT/checks.tsv"' in job
        assert job.count('RESULTS+=("SKIP') == 1  # only inside skip(), so skipped.txt sees every skip
    assert 'sm_smoke_artifact "$SMOKE_OUT" "${BASE_KEY:-}" "${PIN:-}" "$CPU_HALF"' in smoke
    assert smoke.index("sm_smoke_artifact") < smoke.index("=== smoke summary")
    assert 'sm_smoke_cpu_half "$SMOKE_OUT" "$BASE_KEY" "$PIN"' in cpu
    assert cpu.index("sm_smoke_cpu_half") < cpu.index("=== smoke CPU half summary")
    agg = (JOBS / "aggregate.sbatch").read_text()
    assert "--print-scope" in agg and 'if [ "$AGG_SCOPE" = "pooled" ]; then' in agg
    gate = agg[agg.index('if [ "$AGG_SCOPE" = "pooled" ]'):agg.index("fi", agg.index('if [ "$AGG_SCOPE" = "pooled" ]'))]
    assert 'sm_require_frozen "$b"' in gate
    assert agg.index("sm_require_frozen") < agg.index('sm_run "$PYTHON" -m sm.analysis.aggregate')


def test_split_smoke_job_files():
    """The smoke is two jobs (chain 3, job 10899610: pytest on the GPU node hit gpu-short's 1 h cap): smoke_cpu.sbatch
    on short runs pytest and the CLI smokes with node-local temp dirs; smoke.sbatch runs only the GPU checks."""
    cpu = (JOBS / "smoke_cpu.sbatch").read_text()
    gpu = (JOBS / "smoke.sbatch").read_text()
    for job in (cpu, gpu):
        assert 'source "$SM_PROJ/jobs/common.sh"' in job and "sm_install_traps" in job
    # CPU half: short partition, 16 CPUs, 32 GB, 2 h, CPU env, guards kept, no GPU, no pre-flight
    for needle in ("--partition=short", "--cpus-per-task=16", "--mem=32G", "--time=02:00:00", "sm_setup_cpu_env",
                   "sm_require_frozen", 'sm_py config-rev "$CARD_CFG" "$BASE_KEY"', "logs/smoke_cpu-%j.out"):
        assert needle in cpu, needle
    assert "--gres" not in cpu and "sm_gpu_preflight" not in cpu and "sm_setup_env\n" not in cpu
    # every temp dir node-local: basetemp and TMPDIR under sm_node_tmp's root; xdist when importable; -m "not slow"
    assert "sm_node_tmp sm-pytest" in cpu and '--basetemp="$PYTEST_BASETEMP"' in cpu and 'PYTEST_BASETEMP="$SM_NODE_TMP/pytest"' in cpu
    assert 'export TMPDIR="$SM_NODE_TMP/tmp"' in cpu and 'out="$SM_NODE_TMP/cli/${mod##*.}"' in cpu
    assert cpu.index("sm_node_tmp") < cpu.index("-m pytest")
    assert '"$PYTHON" -c "import xdist"' in cpu and 'XDIST=(-n "$SM_PYTEST_WORKERS")' in cpu and 'SM_PYTEST_WORKERS="${SLURM_CPUS_PER_TASK:-1}"' in cpu
    assert 'SM_PYTEST_MARKEXPR="${SM_PYTEST_MARKEXPR-not slow}"' in cpu
    # GPU half: no pytest, no CLI loop; the CPU half is required (exit 5) before the pre-flight; no CPU work
    assert "--partition=gpu-short" in gpu and "-m pytest" not in gpu and "for mod in" not in gpu
    assert "sm_smoke_find_cpu_half" in gpu and "exit 5" in gpu
    assert gpu.index("sm_smoke_find_cpu_half") < gpu.index("exit 5") < gpu.index("\nsm_gpu_preflight") < gpu.index("--- 3: real base load")
    for needle in ("sm_require_frozen", "sm_start_judge", "attach_lora", "DirectionHooks", "capture_residual", "real-base load ("):
        assert needle in gpu, needle


def test_sm_node_tmp_prefers_node_local(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    r = sh('sm_node_tmp sm-pytest; echo "R=$SM_NODE_TMP K=$SM_NODE_TMP_KIND"', env={"SLURM_TMPDIR": str(local), "SLURM_JOB_ID": "42", "SM_SCRATCH": str(tmp_path / "scr")})
    assert r.returncode == 0, r.stderr
    import re as _re
    # mktemp -d adds a random six-character suffix (shared /tmp: a predictable name could be pre-created by another user)
    m = _re.search(r"R=(\S+) K=node-local", r.stdout)
    assert m and m.group(1).startswith(f"{local}/sm-pytest-42-") and len(m.group(1).rsplit("-", 1)[1]) == 6, r.stdout
    assert "tmp: node-local" in r.stdout
    assert (oct(__import__("os").stat(m.group(1)).st_mode & 0o777) == "0o700")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        r = sh('sm_node_tmp sm-pytest; echo "R=$SM_NODE_TMP K=$SM_NODE_TMP_KIND"', env={"SLURM_TMPDIR": str(ro), "SLURM_JOB_ID": "43", "SM_SCRATCH": str(tmp_path / "scr")})
    finally:
        ro.chmod(0o700)
    assert r.returncode == 0, r.stderr
    if os.geteuid() != 0:
        m2 = _re.search(r"R=(\S+) K=scratch", r.stdout)
        assert m2 and m2.group(1).startswith(f"{tmp_path}/scr/tmp/sm-pytest-43-") and "tmp: scratch" in r.stdout, r.stdout


def _cpu_half_dir(tmp_path: Path, job: str = "555", checks: str | None = None, skipped: str = "") -> Path:
    from sm.ops import smoke_artifact as SA

    d = tmp_path / "smoke" / f"cpu-{job}"
    d.mkdir(parents=True)
    (d / "checks.tsv").write_text(checks if checks is not None else "".join(f"{n}\t0\n" for n in ("pytest", *SA.SMOKE_CLIS)))
    (d / "skipped.txt").write_text(skipped)
    (d / "offline.log").write_text("....\n660 passed, 3 skipped, 13 deselected in 301.50s (0:05:01)\nSMOKE_RESULT pytest 0\n")
    return d


def _gpu_dir(tmp_path: Path, base: str = "qwen3.5-9b") -> Path:
    d = tmp_path / "smoke" / "777"
    d.mkdir(parents=True)
    names = [f"revision pin ({base}, configs/base_card_{base}.yaml)", "frozen code (sm_require_frozen, as base_card/lora/battery/drift)",
             f"real-base load ({base})", "judge J1 served at its pin and parsed one item", "judge J2 served at its pin and parsed one item",
             f"LoRA on the real base + DirectionHooks + capture ({base})"]
    (d / "checks.tsv").write_text("".join(f"{n}\t0\n" for n in names))
    (d / "skipped.txt").write_text("")
    (d / "offline.log").write_text("")
    return d


def test_split_smoke_halves_merge_into_one_artifact(tmp_path):
    """CPU half (smoke_cpu.sbatch) + GPU job -> results/smoke/<base>/<gpu jobid>.json in the format SM-1 to SM-4 read."""
    from sm import ops as OPS
    from sm.ops import smoke_artifact as SA

    head = OPS.git_commit()
    if not head:
        pytest.skip("needs a git checkout")
    res = tmp_path / "results" / "smoke"
    pin = PINS["qwen3.5-9b"][1]
    cdir = _cpu_half_dir(tmp_path)
    r = sh('sm_smoke_cpu_half "$1" qwen3.5-9b "$2"', str(cdir), pin,
           env={"SLURM_JOB_ID": "555", "SM_SMOKE_RESULTS": str(res), "SM_PYTEST_WORKERS": "16", "SM_PYTEST_MARKEXPR": "not slow",
                "SM_NODE_TMP": "/tmp/sm-pytest-555", "SM_NODE_TMP_KIND": "node-local"})
    assert r.returncode == 0, r.stderr
    half_path = res / "qwen3.5-9b" / "cpu" / "555.json"
    half = json.loads(half_path.read_text())
    assert half["kind"] == "cpu_half" and half["passed"] is True and half["git_commit"] == head
    assert half["pytest_workers"] == 16 and half["pytest_markexpr"] == "not slow" and half["tmp_kind"] == "node-local"
    assert not (res / "qwen3.5-9b" / "555.json").exists()  # acceptance never sees a half as a smoke
    # GPU start: the half is found by SMOKE_CPU_JOB, or as the newest passing half of this commit; checks exported
    out = tmp_path / "cpu_checks.tsv"
    for env in ({"SMOKE_CPU_JOB": "555"}, {}):
        r = sh('sm_smoke_find_cpu_half qwen3.5-9b "$1"', str(out), env={"SM_SMOKE_RESULTS": str(res), **env})
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == str(half_path)
    assert SA.read_checks(out) == half["checks"]
    # GPU end: the merge
    gdir = _gpu_dir(tmp_path)
    r = sh('sm_smoke_artifact "$1" qwen3.5-9b "$2" "$3"', str(gdir), pin, str(half_path),
           env={"SLURM_JOB_ID": "777", "SM_SMOKE_RESULTS": str(res), "SM_J1_MODEL": JUDGE_PINS["J1"][0], "SM_J1_REVISION": JUDGE_PINS["J1"][1]})
    assert r.returncode == 0, r.stderr
    art = json.loads((res / "qwen3.5-9b" / "777.json").read_text())
    assert art["schema"] == SA.SCHEMA and art["kind"] == "cluster" and art["slurm_job_id"] == "777"
    assert list(art["checks"])[:12] == ["pytest", *SA.SMOKE_CLIS] and len(art["checks"]) == 18
    assert all(v == 0 for v in art["checks"].values()) and art["passed"] is True and art["failures"] == 0
    assert art["git_commit"] == head and art["code_tree"] == OPS.git_object(head, "code") and art["pin"] == pin
    assert art["pytest_summary"] == "660 passed, 3 skipped, 13 deselected in 301.50s (0:05:01)" and art["pytest_rc"] == 0
    assert art["cpu_job_id"] == "555" and art["gpu_job_id"] == "777" and art["split"] is True
    assert art["pytest_workers"] == 16 and art["pytest_markexpr"] == "not slow" and art["cpu_tmp_kind"] == "node-local"
    assert art["cpu_half"]["sha256"] == __import__("hashlib").sha256(half_path.read_bytes()).hexdigest()
    assert art["judges"]["J1"]["revision"] == JUDGE_PINS["J1"][1]
    # pure merge: a failed CPU check fails the merged artifact
    bad = dict(half, checks={**half["checks"], "sm.interp.drift": 1})
    m = SA.merge_halves(bad, SA.build_artifact("cluster", {"real-base load (qwen3.5-9b)": 0}, base="qwen3.5-9b", slurm_job_id="9", pin=pin, git_commit=head))
    assert m["passed"] is False and m["failures"] == 1


def test_split_smoke_missing_or_mismatched_cpu_half_exits_5(tmp_path):
    from sm import ops as OPS
    from sm.ops import smoke_artifact as SA

    head = OPS.git_commit()
    if not head:
        pytest.skip("needs a git checkout")
    res = tmp_path / "results" / "smoke"
    pin = PINS["qwen3.5-9b"][1]
    # nothing written yet: find exits 5, and so does a merge naming a missing file
    assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--out", str(res)]) == 5
    assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--cpu-job", "1", "--out", str(res)]) == 5
    gdir = _gpu_dir(tmp_path)
    gargs = ["--checks", str(gdir / "checks.tsv"), "--skipped", str(gdir / "skipped.txt"), "--job", "778", "--base", "qwen3.5-9b",
             "--pin", pin, "--out", str(res)]
    assert SA.main([*gargs, "--cpu-half", str(res / "qwen3.5-9b" / "cpu" / "1.json")]) == 5
    assert not (res / "qwen3.5-9b" / "778.json").exists()
    # a half at another commit: found by id or not, it is refused (exit 5), and the merge refuses it too
    cdir = _cpu_half_dir(tmp_path, "556")
    half = SA.from_job(cdir / "checks.tsv", cdir / "skipped.txt", cdir / "offline.log", "556", "qwen3.5-9b", pin, kind=SA.CPU_HALF_KIND)
    other = half["git_commit"][::-1]
    half_path = SA.write_artifact(res, {**half, "git_commit": other})
    assert half_path == res / "qwen3.5-9b" / "cpu" / "556.json"
    assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--out", str(res)]) == 5
    assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--cpu-job", "556", "--out", str(res)]) == 5
    assert SA.main([*gargs, "--cpu-half", str(half_path)]) == 5
    with pytest.raises(SA.CpuHalfError):
        SA.merge_halves({**half, "git_commit": other}, SA.build_artifact("cluster", {"x": 0}, base="qwen3.5-9b", git_commit=head))
    # a failed or skipped half, or one for another base, is not usable either
    slot = res / "qwen3.5-9b" / "cpu" / "557.json"
    for bad in ({**half, "passed": False, "failures": 1}, {**half, "skipped": ["pytest (SKIP_PYTEST=1)"], "passed": False},
                {**half, "base": "gemma-4-12b-it"}, {**half, "kind": "cluster"}):
        slot.write_text(json.dumps({**bad, "slurm_job_id": "557"}))
        assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--cpu-job", "557", "--out", str(res)]) == 5
    slot.unlink()
    # the matching half is found and merged
    good = SA.write_artifact(res, {**half, "slurm_job_id": "558"})
    assert SA.main(["--find-cpu-half", "--base", "qwen3.5-9b", "--out", str(res)]) == 0
    assert SA.main([*gargs, "--cpu-half", str(good)]) == 0
    art = json.loads((res / "qwen3.5-9b" / "778.json").read_text())
    assert art["passed"] is True and art["cpu_job_id"] == "558" and art["gpu_job_id"] == "778"


def test_j1_flags_override_logs_a_warning_and_the_header(tmp_path):
    """Review finding: an SM_H_BATTERIES or delta_min override is a logged deviation (accept.py C10, J1-7)."""
    root = tmp_path / "judge" / "b"
    _write(root / "base" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 95}})
    _write(root / "m" / "rates" / "harmbench.J1.json", {"n": 100, "counts": {"refusal": 50}})
    out = tmp_path / "f.txt"
    r = sh('SM_H_BATTERIES=harmbench sm_py j1-flags "$1" "$2" none 0.1', str(root), str(out))
    assert r.returncode == 0, r.stderr
    assert "WARNING: J1 flag rule override: SM_H_BATTERIES=harmbench" in r.stderr
    assert "WARNING: J1 flag rule override: FLAG_DELTA_MIN=0.1" in r.stderr
    assert "delta_min=0.1 batteries=harmbench" in out.read_text().splitlines()[0]
    _write(root / "base" / "rates" / "strongreject.J1.json", {"n": 100, "counts": {"refusal": 95}})
    _write(root / "base" / "rates" / "xstest_unsafe.J1.json", {"n": 100, "counts": {"refusal": 95}})
    r = sh('sm_py j1-flags "$1" "$2" none 0.05', str(root), str(out))
    assert r.returncode == 0, r.stderr
    assert "override" not in r.stderr


# --------------------------------------------------------------------------- base card phases and the GPU watchdog


def test_base_card_phase_jobs_wiring():
    """IdleBot (job 10896275): the base card is capture (gpu, no judge) -> stats (short, CPU) -> ablate (gpu + J1)."""
    cap = (JOBS / "base_card.sbatch").read_text()
    stats = (JOBS / "base_stats.sbatch").read_text()
    abl = (JOBS / "base_ablate.sbatch").read_text()
    for text in (cap, stats, abl):
        assert 'source "$SM_PROJ/jobs/common.sh"' in text and "sm_install_traps" in text
        assert "sm_check_config_rev" in text and "sm_require_frozen" in text  # pinned revision and frozen code
    # capture: GPU, pre-flight, no J1 server; the organism mode stays here
    assert "--gres=gpu" in cap and "sm_gpu_preflight" in cap and "--phase capture" in cap
    assert "sm_start_judge" not in cap and "ORGANISM" in cap
    # stats: CPU only on short with 32 CPUs, 64 GB, 4 h; no GPU pre-flight, no judge, ledger gpus=0
    assert "--partition=short" in stats and "--gres" not in stats and "sm_gpu_preflight" not in stats
    assert "--cpus-per-task=32" in stats and "--mem=64G" in stats and "--time=04:00:00" in stats
    assert "sm_setup_cpu_env 0" in stats and "SM_GPUS_PER_JOB=0" in stats and "sm_start_judge" not in stats
    assert "--phase stats" in stats and "sm_die 5" in stats
    # ablate: GPU with pre-flight before J1 and before the module; checkpoints required (exit 5)
    assert "--gres=gpu" in abl and "--phase ablate" in abl
    assert abl.index("sm_gpu_preflight") < abl.index("sm_start_judge J1") < abl.index("  --phase ablate")
    assert abl.count("sm_die 5") >= 2 and "_partial/capture/manifest.json" in abl and "_partial/stats/stats.json" in abl
    # logs: accept's base_card prefix finds the ablate job (base-card-*.out, matched by the LEDGER job id)
    assert "logs/base-card-%j.out" in cap and "logs/base-card-stats-%j.out" in stats
    assert "logs/base-card-ablate-%j.out" in abl
    from sm import accept as A

    assert A.LOG_PREFIX["base_card"] == "base-card"


def test_base_card_ablate_log_is_found_by_accept(tmp_path):
    from sm import accept as A

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "base-card-111.out").write_text("frozen: base=qwen3.5-9b pin=x commit=a\nLEDGER job=111 array_job= task= name=sm-base-card gpus=1 elapsed_s=5 rc=0\n")
    (logs / "base-card-ablate-333.out").write_text(
        "[t] frozen: base=qwen3.5-9b pin=x commit=c\nLEDGER job=333 array_job= task= name=sm-base-ablate gpus=1 elapsed_s=9 rc=0\n")
    info = A.find_log(tmp_path, "base_card", "333")
    assert info.found and info.rc == 0 and info.frozen["commit"] == "c" and info.paths[0].name == "base-card-ablate-333.out"


def test_gpu_watchdog_logs_utilisation_and_stops_with_the_traps(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text("#!/bin/bash\necho '0, 37, 40512'\n")
    (fake / "nvidia-smi").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "SM_GPU_WATCH_S": "1"}
    r = sh("sm_install_traps; sm_gpu_watchdog; sm_gpu_watchdog; sleep 2.5; echo body-done", env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("gpu watchdog: pid") == 1  # a second call is a no-op
    assert len(re.findall(r"gpu-util: 37% mem=40512MiB gpu=0", r.stdout)) >= 2
    assert "LEDGER job=" in r.stdout and r.stdout.index("body-done") < r.stdout.index("LEDGER job=")
    # without the traps it still ends once the job shell is gone (the pipe closes, so this returns)
    r2 = sh("sm_gpu_watchdog; sleep 1.2", env=env, timeout=30)
    assert r2.returncode == 0 and "gpu-util: 37%" in r2.stdout
    # no nvidia-smi: logged and skipped; SM_GPU_WATCH_S=0 turns it off
    r3 = sh("sm_gpu_watchdog", env={"PATH": "/usr/bin:/bin"})
    assert "watchdog off" in r3.stdout
    r4 = sh("sm_gpu_watchdog", env={**env, "SM_GPU_WATCH_S": "0"})
    assert "gpu" not in r4.stdout
    common = (JOBS / "common.sh").read_text()
    pre = common[common.index("sm_gpu_preflight() {"):]
    assert "sm_gpu_watchdog" in pre[: pre.index("\n}\n")]  # every GPU stage gets it through the pre-flight
    assert "sm_stop_gpu_watchdog" in common[common.index("_sm_on_exit() {"):common.index("sm_install_traps() {")]
    assert "sm_stop_gpu_watchdog" in common[common.index("_sm_on_term() {"):common.index("_sm_on_exit() {")]
    for fn in ("sm_setup_env() {", "sm_setup_cpu_env() {"):
        body = common[common.index(fn):]
        assert "PYTHONUNBUFFERED=1" in body[: body.index("\n}\n")], fn


def test_persona_dose_job_wiring():
    """FIDELITY_REVIEW R2 and its pre-registered re-run (docs/PERSONA_FIDELITY_RERUN.md 4.7): a GPU job serving the
    persona grader (GRADER=chen, default) or J1 (GRADER=j1, the generic-grader sensitivity line)."""
    text = (JOBS / "persona_dose.sbatch").read_text()
    code = text[text.index("set -eo pipefail"):]  # the header comments name the helpers too
    assert "--gres=gpu:h200:1" in text and "--partition=gpu" in text and "--time=06:00:00" in text
    assert 'source "$SM_PROJ/jobs/common.sh"' in text and "sm_install_traps" in text and "sm_require_frozen" in text
    assert 'GRADER="${GRADER:-chen}"' in text and "JUDGE=sm.interp.persona_grader:score_trait_chen" in text
    assert "JUDGE=sm.eval.judge:score_trait" in text
    pre, py = code.index("sm_gpu_preflight"), code.index("-m sm.interp.persona_dose")
    assert pre < code.index("sm_start_grader") < py and pre < code.index("sm_start_judge J1") < py
    assert '--coefs="${COEFS:--2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5}"' in text and '--questions "${QUESTIONS:-eval}"' in text
    assert '--n-questions "${N_QUESTIONS:-20}"' in text and '--n-samples "${N_SAMPLES:-10}"' in text
    assert '--max-new-tokens "${MAX_NEW:-1000}"' in text and '_persona_dose_${TAG}.json' in text
    assert "sm_die 5" in text and "logs/persona-dose-%j.out" in text and '--judge "$JUDGE"' in text
    assert "`persona_dose.sbatch`" in (JOBS / "README.md").read_text()


def test_base_persona_job_wiring():
    """docs/PERSONA_FIDELITY_RERUN.md 4.6: the persona phase serves the grader the config names, needs the card and its
    capture checkpoint, and writes the log accept's base_card prefix finds."""
    text = (JOBS / "base_persona.sbatch").read_text()
    code = text[text.index("set -eo pipefail"):]
    assert "--gres=gpu:h200:1" in text and "--partition=gpu" in text and "--time=08:00:00" in text
    for need in ('source "$SM_PROJ/jobs/common.sh"', "sm_install_traps", "sm_check_config_rev", "sm_require_frozen",
                 "sm_preflight_snapshot", 'SCORER="$(sm_persona_scorer "$CFG")"', "${FORCE:+--force}"):
        assert need in text, need
    pre, py = code.index("sm_gpu_preflight"), code.index("-m sm.interp.base_card")
    assert pre < code.index("sm_start_grader") < py and pre < code.index("sm_start_judge J1") < py
    assert "--phase persona" in text and text.count("sm_die 5") >= 2 and "_partial/capture/manifest.json" in text
    assert "logs/base-card-persona-%j.out" in text and "logs/base-card-persona-%j.err" in text
    from sm import accept as A

    assert A.LOG_PREFIX["base_card"] == "base-card"
    assert "`base_persona.sbatch`" in (JOBS / "README.md").read_text()


def test_base_persona_log_is_found_by_accept(tmp_path):
    from sm import accept as A

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "base-card-persona-444.out").write_text(
        "[t] frozen: base=qwen3.5-9b pin=x commit=d\nLEDGER job=444 array_job= task= name=sm-base-persona gpus=1 elapsed_s=9 rc=0\n")
    info = A.find_log(tmp_path, "base_card", "444")
    assert info.found and info.rc == 0 and info.frozen["commit"] == "d" and info.paths[0].name == "base-card-persona-444.out"


def test_persona_grade_job_wiring():
    text = (JOBS / "persona_grade.sbatch").read_text()
    code = text[text.index("set -eo pipefail"):]
    assert "--gres=gpu:h200:1" in text and "--partition=gpu" in text and "sm_require_frozen" in code
    assert code.index("sm_gpu_preflight") < code.index("sm_start_grader") < code.index("-m sm.interp.persona_grader grade-battery")
    assert "umask 077" in text and "logs/persona-grade-%j.out" in text and "`persona_grade.sbatch`" in (JOBS / "README.md").read_text()


def test_qwen_cards_use_the_chen_grader():
    """docs/PERSONA_FIDELITY_RERUN.md (D-019 for qwen3.5-9b, the D-031 draft for qwen3.8-27b): both Qwen card configs
    grade persona traits with Chen et al.'s prompts on the registered grader, with the same selection rule, coherence
    bound, system-prompt prefix and token budget."""
    from sm.interp import persona_grader as PG

    cfgs = {k: load(CONFIGS / f"base_card_{k}.yaml") for k in ("qwen3.5-9b", "qwen3.8-27b")}
    for k, c in cfgs.items():
        assert c["judge"]["trait"] == "sm.interp.persona_grader:score_trait_chen", k
        p = c["persona"]
        assert (p["selection_rule"], p["coherence_min"], p["system_prompt_prefix"], p["max_new_tokens"], p["min_kept"],
                p["pos_threshold"], p["neg_threshold"]) == ("chen_paired", 50.0, "chen", 1000, 10, 50.0, 50.0), k
        assert set(p["traits"]) == set(PG.CHEN_EVAL_PROMPTS), k
    nine, twenty_seven = cfgs["qwen3.5-9b"], cfgs["qwen3.8-27b"]
    assert nine["judge"] == twenty_seven["judge"] and nine["persona"] == twenty_seven["persona"]
    # one grader for both: the registered model, revision and prompt set (the D-019 value)
    assert PINS[PG.GRADER_BASE_KEY] == (PG.GRADER_MODEL, PG.GRADER_REVISION)
    assert PG.prompt_set_sha256() == PG.REGISTERED_PROMPT_SET_SHA256
    # the Gemma cards keep the v0.3 grader: the persona grader is Gemma 4 31B and must not grade its own family
    for k in ("gemma-4-12b-it", "gemma-4-31b-it"):
        assert load(CONFIGS / f"base_card_{k}.yaml")["judge"]["trait"] == "sm.eval.judge:score_trait", k


def test_persona_scorer_and_grader_helpers():
    r = sh('sm_persona_scorer "$1"', str(CONFIGS / "base_card_qwen3.5-9b.yaml"))
    assert r.returncode == 0 and r.stdout.strip() == "chen", r.stderr
    r = sh('sm_persona_scorer "$1"', str(CONFIGS / "base_card_qwen3.8-27b.yaml"))
    assert r.returncode == 0 and r.stdout.strip() == "chen", r.stderr
    r = sh('sm_persona_scorer "$1"', str(CONFIGS / "base_card_gemma-4-12b-it.yaml"))
    assert r.returncode == 0 and r.stdout.strip() == "j1", r.stderr
    common = (JOBS / "common.sh").read_text()
    body = common[common.index("sm_start_grader() {"):]
    body = body[: body.index("\n}\n")]
    for need in ('key="${PG_KEY:-gemma-4-31b-it}"', 'sm_py hf "$key"', 'sm_py pin "$key"', 'sm_preflight_snapshot "$model" "$rev"',
                 "SM_PG_SERVED_NAME=PG", 'SM_PG_URL="http://127.0.0.1:$port"', 'sm_start_vllm PG "$model" "$rev"'):
        assert need in body, need
    # the grader is the configs/bases.yaml pin of gemma-4-31b-it, the model sm.interp.persona_grader registers
    from sm.interp import persona_grader as PG

    assert PINS[PG.GRADER_BASE_KEY] == (PG.GRADER_MODEL, PG.GRADER_REVISION)
