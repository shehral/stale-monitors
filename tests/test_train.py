"""Tests for sm.train.lora_sft. Run offline: SM_OFFLINE=1 pytest tests/test_train.py"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "code") not in sys.path:
    sys.path.insert(0, str(REPO / "code"))

from sm.train import lora_sft  # noqa: E402

BANKS = ["bank_qwen35_9b.yaml", "bank_qwen38_27b.yaml", "bank_gemma4_12b.yaml", "bank_gemma4_31b.yaml"]
PREREG_COUNTS = {1: 4, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)


def test_smoke_writes_adapter_and_meta(tmp_path, capsys):
    rc = lora_sft.main(["--smoke", "--out", str(tmp_path)])
    assert rc == 0
    run_dir = tmp_path / "tiny" / "smoke"
    assert (run_dir / "adapter_config.json").is_file()
    assert (run_dir / "adapter_model.safetensors").is_file()
    meta = json.loads((run_dir / "train_meta.json").read_text())
    assert meta["status"] == "ok"
    assert meta["steps"] == lora_sft.SMOKE_STEPS
    assert len(meta["loss_curve"]) == lora_sft.SMOKE_STEPS
    assert all(isinstance(r["loss"], float) for r in meta["loss_curve"])
    assert meta["nan_check"] == {"nan_detected": False, "nan_step": None}
    assert set(meta["target_module_suffixes"]) == {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    assert meta["render"]["thinking"] is False
    assert meta["data"]["n_prefix_mismatch"] == 0
    assert meta["git"].keys() >= {"commit", "dirty"}
    assert "meta" in meta and "config_sha256" in meta["meta"]
    adapter_cfg = json.loads((run_dir / "adapter_config.json").read_text())
    assert adapter_cfg["r"] == 4 and adapter_cfg["lora_alpha"] == 8
    # Idempotent: a second call without --force is a no-op.
    capsys.readouterr()
    assert lora_sft.main(["--smoke", "--out", str(tmp_path)]) == 0
    assert "already done" in capsys.readouterr().out


def test_nan_loss_exits_3_without_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_sft, "_postprocess_loss", lambda loss: loss * float("nan"))
    rc = lora_sft.main(["--smoke", "--out", str(tmp_path)])
    assert rc == lora_sft.EXIT_NAN == 3
    run_dir = tmp_path / "tiny" / "smoke"
    assert not (run_dir / "adapter_model.safetensors").exists()
    meta = json.loads((run_dir / "train_meta.json").read_text())
    assert meta["status"] == "nan_abort"
    assert meta["nan_check"]["nan_detected"] is True
    assert meta["nan_check"]["nan_step"] == 1


def test_smoke_from_manifest_run(tmp_path):
    cfg = REPO / "configs" / "bank_qwen35_9b.yaml"
    rc = lora_sft.main(["--config", str(cfg), "--run-id", "c2_terse_r32", "--smoke", "--out", str(tmp_path)])
    assert rc == 0
    meta = json.loads((tmp_path / "tiny" / "smoke_c2_terse_r32" / "train_meta.json").read_text())
    assert meta["condition"] == "c2_persona_benign"
    assert meta["hyperparams"]["rank"] == 8 and meta["hyperparams"]["alpha"] == 16


def test_prompt_tokens_are_masked():
    model, tok, _ = lora_sft.build_offline_tiny()
    recs = lora_sft.synthetic_records(4)
    rows, stats = lora_sft.tokenize_records(tok, recs, 512, lora_sft._render_chat_fallback)
    assert stats["n_prefix_mismatch"] == 0 and len(rows) == 4
    for rec, row in zip(recs, rows):
        target = [t for t, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
        text = tok.decode(target)
        assert text.startswith(rec["messages"][-1]["content"])
        assert "<|im_end|>" in text and "What is" not in text
        assert row["labels"][0] == -100


def test_truncation_and_drop():
    _, tok, _ = lora_sft.build_offline_tiny()
    recs = lora_sft.synthetic_records(3)
    rows, stats = lora_sft.tokenize_records(tok, recs, 8, lora_sft._render_chat_fallback)
    assert stats["n_truncated"] == 3 and stats["n_dropped_no_target"] == 3 and rows == []


def test_load_records_rejects_system_messages(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "u"},
                                          {"role": "assistant", "content": "a"}], "source": "t", "id": "1"}) + "\n")
    with pytest.raises(ValueError):
        lora_sft.load_records(p, None, 0)


def test_load_records_subsample_is_seeded(tmp_path):
    p = tmp_path / "d.jsonl"
    with open(p, "w") as fh:
        for r in lora_sft.synthetic_records(20):
            fh.write(json.dumps(r) + "\n")
    a = [r["id"] for r in lora_sft.load_records(p, 5, 7)]
    b = [r["id"] for r in lora_sft.load_records(p, 5, 7)]
    assert a == b and len(a) == 5


def test_auto_backend_requires_trl_outside_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_sft, "available_backends", lambda: {"trl": "ImportError: x", "hf": None, "torch": None})
    cfg = lora_sft.RunConfig.from_dict({"run_id": "r", "base": "tiny", "condition": "c", "dataset": "x.jsonl",
                                        "rank": 8, "lr": 1e-4, "epochs": 1, "seed": 0})
    assert lora_sft.run(cfg, tmp_path, smoke=False) == lora_sft.EXIT_BACKEND


def test_non_training_run_is_noop(tmp_path):
    cfg = REPO / "configs" / "bank_gemma4_31b.yaml"
    assert lora_sft.main(["--config", str(cfg), "--run-id", "c5_ablation_inmem", "--out", str(tmp_path)]) == 0
    assert not any(tmp_path.rglob("*"))


def test_target_modules_hybrid_qwen35():
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5")
    cfg = tr.Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4,
                               linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    model = tr.Qwen3_5ForCausalLM(cfg)
    suffixes, paths = lora_sft.resolve_target_modules(model)
    assert {"self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj", "linear_attn.out_proj"} <= set(suffixes)
    assert any(s.startswith("linear_attn.in_proj") for s in suffixes)
    assert not any("lm_head" in p or "embed" in p for p in paths)
    peft_model, targets = lora_sft.attach_lora(model, lora_sft.RunConfig.from_dict(
        {"run_id": "r", "base": "tiny", "condition": "c", "dataset": None, "rank": 4, "lr": 1e-4, "epochs": 1,
         "seed": 0}), suffixes, paths)
    n_lora = sum(1 for n, _ in peft_model.named_modules() if n.endswith("lora_A"))
    assert n_lora == len(paths)


def test_target_modules_gemma4_excludes_side_projections():
    try:
        from transformers.models.gemma4 import configuration_gemma4 as g, modeling_gemma4 as mg
    except ImportError:
        pytest.skip("transformers has no Gemma 4")
    cfg = g.Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = mg.Gemma4ForCausalLM(cfg)
    suffixes, paths = lora_sft.resolve_target_modules(model)
    assert set(suffixes) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert not any("per_layer" in p for p in paths)


@pytest.mark.parametrize("fname", BANKS)
def test_bank_manifest_matches_prereg(fname):
    raw = lora_sft.load_yaml(REPO / "configs" / fname)
    runs = lora_sft.manifest_runs(raw)
    # 16 registered runs plus the E8 matched-data control c4_secure_code_r32 appended last (deviation D-002).
    assert len(runs) == 17 and runs[-1]["run_id"] == "c4_secure_code_r32" and runs[-1]["control"] is True
    assert len({r["run_id"] for r in runs}) == 17
    assert Counter(r["condition_id"] for r in runs[:-1]) == PREREG_COUNTS
    for r in runs:
        cfg = lora_sft.RunConfig.from_dict(r)
        if not cfg.train:
            assert r["condition_id"] in (5, 6) and cfg.dataset is None
            continue
        assert cfg.rank in (8, 32, 128)
        assert cfg.alpha == 2 * cfg.rank
        assert cfg.lr > 0 and cfg.max_len == 2048
        # v0.3 (fidelity N3; owner decision): Betley et al.'s recipe for every run (rsLoRA, lr 1e-5, 1 epoch, all
        # examples, dropout 0, weight decay 0.01, 5 warmup steps); rank alone varies
        assert cfg.epochs == 1 and cfg.use_rslora and cfg.n_examples is None and cfg.lr == 1e-5 and cfg.n_warmup(375) == 5
        assert cfg.lora_dropout == 0.0 and cfg.weight_decay == 0.01
        data_cond = "c1_benign_domain" if cfg.condition == "c6_controls" else cfg.condition
        assert cfg.dataset and cfg.dataset.startswith(f"data/bank/{data_cond}/")
        if cfg.condition == "c2_persona_benign":
            assert cfg.dataset.endswith(f"__{raw['base']}.jsonl")
    trained = [r for r in runs if r.get("train", True)]
    assert len(trained) == 14
    ranks = Counter((r["condition_id"], r["rank"]) for r in trained)
    # prereg v0.2: condition 1 = 2 datasets x ranks 8 and 128; the c6 seed replicate is Dolly rank 8 at seed 1
    assert ranks[(1, 8)] == 2 and ranks[(1, 128)] == 2 and ranks[(2, 32)] == 3 and ranks[(3, 32)] == 3
    assert ranks[(6, 8)] == 1
    assert ranks[(4, 32)] == 2 and ranks[(4, 128)] == 1  # insecure r32 and r128, plus the E8 secure-code control r32
    try:
        from sm.common.registry import resolve_base
    except ImportError:
        return
    assert resolve_base(raw["base"])["hf"] == raw["base_hf"]


def test_template_has_placeholders():
    text = (REPO / "configs" / "bank_template.yaml").read_text()
    assert "<base_key>" in text and "<hf_id>" in text
    runs = lora_sft.manifest_runs(lora_sft.load_yaml(REPO / "configs" / "bank_template.yaml"))
    assert len(runs) == 17


def test_contrast_exclusion_check(tmp_path):
    bank = tmp_path / "bank"
    d = bank / "c1_benign_domain"
    d.mkdir(parents=True)
    data = d / "dolly.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "source": "s", "id": "1"}\n')
    entry = {"path": "c1_benign_domain/dolly.jsonl", "sha256": lora_sft._sha256_file(data), "n_contrast_excluded": 3}
    (bank / "MANIFEST.json").write_text(json.dumps({"datasets": {"dolly": entry}, "contrast_exclusion": {"applied": True, "manifest_sha256": "f" * 64}}))
    info = lora_sft.check_contrast_exclusion(data)
    assert info["applied"] and info["n_contrast_excluded"] == 3
    (bank / "MANIFEST.json").write_text(json.dumps({"datasets": {"dolly": entry}, "contrast_exclusion": {"applied": False, "reason": "disabled"}}))
    with pytest.raises(ValueError, match="contrast"):
        lora_sft.check_contrast_exclusion(data)
    assert lora_sft.check_contrast_exclusion(data, allow_overlap=True)["deviation"]
    data.write_text("changed\n")
    (bank / "MANIFEST.json").write_text(json.dumps({"datasets": {"dolly": entry}, "contrast_exclusion": {"applied": True}}))
    with pytest.raises(ValueError, match="sha256"):
        lora_sft.check_contrast_exclusion(data)


def test_seed_replicate_trains_on_the_same_subset(tmp_path):
    """Condition 6: the seed replicate shares the subsample (data_seed) with its partner; only `seed` differs."""
    import yaml

    p = tmp_path / "d.jsonl"
    with open(p, "w") as fh:
        for r in lora_sft.synthetic_records(40):
            fh.write(json.dumps(r) + "\n")
    bank = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "bank_qwen35_9b.yaml").read_text())
    runs = {r["run_id"]: lora_sft.RunConfig.from_dict(r) for r in lora_sft.manifest_runs(bank)}
    rep, partner = runs["c6_seed_replicate_dolly_r8"], runs["c1_dolly_r8"]
    assert rep.seed == 1 and partner.seed == 0 and rep.data_seed == partner.data_seed == 0
    a = [r["id"] for r in lora_sft.load_records(p, 10, rep.data_seed)]
    b = [r["id"] for r in lora_sft.load_records(p, 10, partner.data_seed)]
    assert a == b
    assert [r["id"] for r in lora_sft.load_records(p, 10, 1)] != a  # the old behaviour (subsample by run seed) differed


def test_rslora_reaches_the_peft_config(tmp_path):
    """v0.3: use_rslora is passed to PEFT (scale alpha / sqrt(r)) and recorded."""
    pytest.importorskip("peft")
    from sm.train import lora_sft

    cfg = lora_sft.RunConfig.from_dict({"run_id": "r", "base": "tiny", "condition": "c", "dataset": None, "rank": 4, "lr": 1e-5,
                                         "epochs": 1, "seed": 0, "use_rslora": True, "warmup_steps": 5})
    assert cfg.use_rslora is True and cfg.n_warmup(3) == 3 and cfg.n_warmup(100) == 5
    model = lora_sft.build_offline_tiny(0)[0]
    suffixes, paths = lora_sft.resolve_target_modules(model)
    pm, _ = lora_sft.attach_lora(model, cfg, suffixes, paths)
    assert pm.peft_config["default"].use_rslora is True
