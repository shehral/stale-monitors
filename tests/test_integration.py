"""Cross-module integration checks (offline, tiny random model).

The per-module suites feed each stage synthetic inputs; these tests feed each stage the real
output of the stage before it: build_prompt_sets -> build_bank_datasets -> lora_sft (non-smoke) ->
battery (base + adapter) -> base_card -> drift (lora, base, ablation) -> aggregate, plus the
judge callables base_card resolves by default and the job helper's reading of the bank manifests.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ judge callables (base_card defaults)


class _FakeChat:
    """Stands in for the J1 server: answers with a label chosen from the user message."""

    def __init__(self, labels: dict[str, str]):
        self.labels = labels

    def chat(self, messages, max_tokens, temperature):
        user = messages[-1]["content"]
        for key, label in self.labels.items():
            if key in user:
                return json.dumps({"label": label}) if label != "garbage" else "no verdict here"
        return json.dumps({"label": "incoherent"})

    def complete(self, prompt, max_tokens, temperature):  # pragma: no cover - J2 is not used here
        raise AssertionError("J2 is not used by the in-process callables")


def _j1(labels: dict[str, str]):
    from sm.eval import judge as J

    return J.J1Judge(client=_FakeChat(labels), policies=J.load_policies(J.DEFAULT_POLICY_DIR))


def test_base_card_default_judges_resolve():
    from sm.interp.base_card import resolve_callable

    assert callable(resolve_callable("sm.eval.judge:judge_refusal"))
    assert callable(resolve_callable("sm.eval.judge:score_trait"))


def test_judge_refusal_maps_labels():
    from sm.eval.judge import judge_refusal

    j = _j1({"AAA": "refusal", "BBB": "harmful_compliance", "CCC": "non_harmful_deflection", "DDD": "garbage"})
    out = judge_refusal(["p1", "p2", "p3", "p4"], ["AAA", "BBB", "CCC", "DDD"], judge=j)
    assert out == [True, False, False, False]
    with pytest.raises(ValueError):
        judge_refusal(["p"], [], judge=j)


def test_score_trait_maps_labels_and_drops_unparsed():
    from sm.eval.judge import score_trait

    j = _j1({"AAA": "trait_expressed", "BBB": "trait_absent", "CCC": "garbage", "DDD": "incoherent"})
    out = score_trait("evil", ["q"] * 4, ["AAA", "BBB", "CCC", "DDD"], judge=j)
    assert out[:2] == [100.0, 0.0]
    assert math.isnan(out[2]) and math.isnan(out[3])


# ------------------------------------------------------------------ manifests: aggregate and the job helper


def test_aggregate_resolves_manifest_by_registry_key(tmp_path):
    from sm.analysis import aggregate as A

    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "bank_qwen35_9b.yaml").write_text(yaml.safe_dump({"base": "qwen3.5-9b", "runs": [{"run_id": "r1"}]}))
    (cfgs / "bank_template.yaml").write_text(yaml.safe_dump({"base": "<base_key>", "runs": [{"run_id": "r1"}]}))
    (cfgs / "bank_sources.yaml").write_text(yaml.safe_dump({"sources": {"x": 1}}))
    cfg = {**A.DEFAULT_CONFIG, "configs_root": str(cfgs), "results_root": str(tmp_path / "results")}
    assert A.discover_bases(cfg) == ["qwen3.5-9b"]
    assert A.resolve_manifest_path("qwen3.5-9b", cfg) == cfgs / "bank_qwen35_9b.yaml"
    assert A.resolve_manifest_path("qwen35_9b", cfg) == cfgs / "bank_qwen35_9b.yaml"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs bash")
def test_job_helper_reads_every_bank_manifest():
    manifests = sorted(p for p in (ROOT / "configs").glob("bank_*.yaml") if p.stem not in ("bank_template", "bank_sources"))
    assert manifests
    for m in manifests:
        for stage in ("lora", "battery", "judge", "drift"):
            r = subprocess.run(
                ["bash", "-c", 'source jobs/common.sh && sm_py runs "$1" "$2"', "_", str(m.relative_to(ROOT)), stage],
                cwd=ROOT, env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True, timeout=60,
            )
            assert r.returncode == 0, (m.name, stage, r.stderr)
            rows = {line.split("\t")[1]: line.split("\t") for line in r.stdout.splitlines()}
            raw = yaml.safe_load(m.read_text())
            assert set(rows) >= {run["run_id"] for run in raw["runs"]}
            if "c5_public_abliterated" in rows and not next(x for x in raw["runs"] if x["run_id"] == "c5_public_abliterated").get("external_hf"):
                assert rows["c5_public_abliterated"][3] == "skip"


def test_lora_out_dir_never_resolves_into_the_base(tmp_path):
    from sm.train.lora_sft import _path_slug

    assert _path_slug("qwen3.5-9b") == "qwen3.5-9b"
    assert _path_slug("Qwen/Qwen3.5-9B") == "Qwen__Qwen3.5-9B"
    assert _path_slug(str(tmp_path / "models" / "tiny")) == "tiny"


# ------------------------------------------------------------------ stage chain on the tiny model


@pytest.fixture()
def chain_env(tiny_model_dir, monkeypatch):
    if not Path(tiny_model_dir).is_dir():
        pytest.skip("stage chain needs a local tiny model directory (SM_OFFLINE=1 builds one)")
    for k, v in {"SM_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "SM_DEVICE": "cpu", "SM_TINY_MODEL": tiny_model_dir}.items():
        monkeypatch.setenv(k, v)
    return tiny_model_dir


def test_stage_chain_end_to_end(chain_env, tmp_path):
    from sm.analysis import aggregate
    from sm.data import build_bank_datasets, build_prompt_sets
    from sm.eval import battery
    from sm.interp import base_card, drift
    from sm.train import lora_sft

    prompts, bank, res = tmp_path / "prompts", tmp_path / "bank", tmp_path / "results"
    assert build_prompt_sets.main(["--smoke", "--out", str(prompts)]) == 0
    assert build_bank_datasets.main(["--smoke", "--out", str(bank), "--contrast-manifest", str(prompts / "MANIFEST.json")]) == 0
    # Since 0239981 a --smoke bank build never reads the prompt sets (synthetic rows; the real-build exclusion is
    # covered in tests/test_data_bank.py), so the manifest records exclusion as not applied, with the reason.
    excl = json.loads((bank / "MANIFEST.json").read_text())["contrast_exclusion"]
    assert excl["applied"] is False and "smoke build" in excl["reason"]
    contrast = prompts / "contrast"

    # LoRA on a bank dataset (non-smoke path through load_base and the manifest-style config).
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(yaml.safe_dump({
        "run_id": "c1_dolly_r8", "base": "tiny", "condition": "c1_benign_domain",
        "dataset": str(bank / "c1_benign_domain" / "dolly.jsonl"), "rank": 4, "alpha": 8, "lr": 1e-3,
        "epochs": 1, "seed": 0, "max_len": 128, "per_device_batch_size": 2, "grad_accum": 1,
        "gradient_checkpointing": False,
        # the smoke bank records no contrast exclusion (above); the overlap override is the logged path a
        # non-smoke run on such a bank must take
        "allow_contrast_overlap": True,
    }))
    assert lora_sft.main(["--config", str(run_cfg), "--out", str(res / "adapters"), "--max-steps", "2"]) == 0
    adapter = res / "adapters" / "tiny" / "c1_dolly_r8"
    assert (adapter / "adapter_config.json").is_file()

    # Batteries for the base and the adapter, from the built item files.
    common = ["--base", "tiny", "--items-dir", str(prompts), "--out", str(res), "--batteries", "strongreject", "xstest_safe",
              "--limit", "2", "--max-new-tokens", "4", "--device", "cpu", "--dtype", "fp32"]
    assert battery.main(common + ["--run-id", "base"]) == 0
    assert battery.main(common + ["--adapter", str(adapter)]) == 0
    rec = json.loads((res / "restricted" / "batteries" / "tiny" / "c1_dolly_r8" / "strongreject.jsonl").read_text().splitlines()[0])
    assert {"item_id", "battery", "prompt", "completion", "completion_sha256", "truncated"} <= set(rec)
    assert (res / "batteries" / "tiny" / "base" / "xstest_safe.jsonl").is_file()

    # Base card from the built contrast and persona files.
    bc_cfg = tmp_path / "base_card.yaml"
    bc_cfg.write_text(yaml.safe_dump({
        "dtype": "fp32", "device": "cpu", "batch_size": 8,
        "prompts": {**{s: {c: str(contrast / f"{s}_{c}.jsonl") for c in ("harmful", "harmless")} for s in ("construction", "dev")},
                    # the harmless-only threshold calibration split (FIDELITY_REVIEW N6), built by build_prompt_sets
                    "calibration": {"harmless": str(contrast / "calibration_harmless.jsonl")}},
        "judge": {"refusal": "fake:refusal", "trait": "fake:trait"},
        "split_half_k": 2, "n_shuffles": 2, "n_random_dirs": 2, "probe_cv": 2, "probe_shuffles": 0,
        "select_max_new_tokens": 2, "subspace_k": 2,
        "persona": {"max_new_tokens": 2, "steer_max_new_tokens": 2, "pos_threshold": -1.0, "neg_threshold": 101.0,
                    "traits": {"evil": {"extract": str(prompts / "persona" / "evil_extract.json"),
                                        "eval": str(prompts / "persona" / "evil_eval.json")}}},
    }))
    assert base_card.main(["--config", str(bc_cfg), "--base", "tiny", "--out", str(res / "base_cards")]) == 0
    card = json.loads((res / "base_cards" / "tiny.json").read_text())
    sidecar = json.loads((res / "base_cards" / "tiny" / "refusal.json").read_text())
    assert sidecar["layer"] == card["L_star"]

    # Drift for a LoRA derivative, the base itself and the in-memory ablation.
    derivs = {"c1_dolly_r8": {"kind": "lora", "adapter_path": str(adapter)}, "c6_base_vs_base": {"kind": "base"},
              "c5_ablation_inmem": {"kind": "ablation"}}
    for rid, der in derivs.items():
        dcfg = tmp_path / f"drift_{rid}.yaml"
        dcfg.write_text(yaml.safe_dump({
            "base": "tiny", "run_id": rid, "base_card": str(res / "base_cards" / "tiny.json"), "dtype": "fp32", "device": "cpu",
            "prompts": {s: {c: str(contrast / f"{s}_{c}.jsonl") for c in ("harmful", "harmless")} for s in ("construction", "dev")},
            "persona_prompts": {"evil": str(prompts / "persona" / "evil_eval.json")}, "derivative": der,
        }))
        assert drift.main(["--config", str(dcfg), "--out", str(res / "drift")]) == 0
    same = json.loads((res / "drift" / "tiny" / "c6_base_vs_base.json").read_text())
    # v0.3 change 42: drift reads the card's selected position (D1 = 1 base vs base at any position)
    assert same["meta"]["position"] == card["L_star_position_label"] == card["monitors"]["position"]
    assert same["extra"]["position_index"] == card["L_star_position"]
    sec = same["extra"]["last_token_secondary"]
    assert sec["available"] and sec["position"] == "last" and sec["D2"] == pytest.approx(0.0, abs=1e-6)
    assert same["D1"] == pytest.approx(1.0, abs=1e-6) and same["D2"] == pytest.approx(0.0, abs=1e-6)
    assert same["D3"] == pytest.approx(1.0, abs=1e-6)
    ablated = json.loads((res / "drift" / "tiny" / "c5_ablation_inmem.json").read_text())
    assert abs(ablated["D1"]) < 1e-3
    lora = json.loads((res / "drift" / "tiny" / "c1_dolly_r8.json").read_text())
    assert lora["adapter_projection"] is not None and "evil" in lora["persona"]

    # Judge labels in the contract format where the judge stage writes them, then aggregate.
    for rid in ("base", "c1_dolly_r8", "c6_base_vs_base"):
        d = res / "judge" / "tiny" / rid / "restricted"
        d.mkdir(parents=True)
        for jn in ("J1", "J2"):
            with open(d / f"strongreject.{jn}.jsonl", "w") as fh:
                for i in range(4):
                    fh.write(json.dumps({"item_id": f"sr-{i}", "battery": "strongreject", "judge": jn,
                                         "label": "refusal" if i % 2 else "harmful_compliance", "raw": "", "parse_ok": True}) + "\n")
    # Capability scores (sm.eval.capability output format) for every run with judge labels.
    cap = res / "capability" / "tiny"
    cap.mkdir(parents=True)
    for rid in ("base", "c1_dolly_r8", "c6_base_vs_base"):
        (cap / f"{rid}.json").write_text(json.dumps({"mmlu_acc": 0.5, "mmlu_n": 8, "ifeval_strict_acc": 0.5, "ifeval_n": 8, "meta": {"stub": True}}))
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "bank_tiny_test.yaml").write_text(yaml.safe_dump({"base": "tiny", "runs": [
        {"run_id": "c1_dolly_r8", "condition": "c1_benign_domain", "dataset": "x.jsonl", "rank": 4, "seed": 0},
        {"run_id": "c5_ablation_inmem", "condition": "c5_safety_removal", "kind": "directional_ablation", "rank": None, "seed": 0},
        {"run_id": "c6_base_vs_base", "condition": "c6_controls", "kind": "base_replicate", "rank": None, "seed": 0},
    ]}))
    out = tmp_path / "agg"
    assert aggregate.main(["--results-root", str(res), "--configs-root", str(cfgs), "--out", str(out), "--n-boot", "20", "--no-figures"]) == 0
    import pandas as pd

    table = pd.read_csv(out / "tables" / "bank_tiny.csv")
    assert set(table["run_id"]) == set(derivs)
    assert bool(table.set_index("run_id").loc["c6_base_vs_base", "is_base_vs_base"])
    assert table["has_drift"].all()
    assert (out / "tables" / "hypotheses.json").is_file()
