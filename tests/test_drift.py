"""Tests for sm.interp.drift and sm.interp.adapter_proj (offline, synthetic or tiny random model)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
INTERP = ROOT / "code" / "sm" / "interp"


def _import(name: str):
    """Import sm.interp.<name>; fall back to loading the file directly if the package is absent."""
    sys.path.insert(0, str(ROOT / "code"))
    try:
        return importlib.import_module(f"sm.interp.{name}")
    except ImportError:
        mod_name = f"_test_sm_interp_{name}"
        spec = importlib.util.spec_from_file_location(mod_name, INTERP / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod


drift = _import("drift")
adapter_proj = _import("adapter_proj")


# ------------------------------------------------------------------ synthetic fixtures

HID = 24


def _synthetic(seed: int = 0, n: int = 80):
    rng = np.random.default_rng(seed)
    r = drift.unit(rng.standard_normal(HID))
    base_mu = rng.standard_normal(HID)
    harmful = base_mu + 3.0 * r + rng.standard_normal((n, HID))
    harmless = base_mu + rng.standard_normal((n, HID))
    return r, harmful, harmless, rng


def _card(r, harmful, harmless, layer=3, persona=True, rng=None):
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([harmful, harmless])
    y = np.concatenate([np.ones(len(harmful)), np.zeros(len(harmless))])
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(X, y)
    rng = rng or np.random.default_rng(1)
    pers = {}
    if persona:
        pers = {"evil": (drift.unit(rng.standard_normal(HID)), layer), "sycophancy": (drift.unit(rng.standard_normal(HID)), layer)}
    return drift.BaseCard(base="synthetic", layer_star=layer, r=r, probe=drift.Probe(w=clf.coef_[0], b=float(clf.intercept_[0]), layer=layer), persona=pers)


def _acts(harmful, harmless, layer=3):
    return {"harmful": {layer: harmful}, "harmless": {layer: harmless}}


# ------------------------------------------------------------------ drift metrics


def test_base_vs_base_is_identity():
    r, H, N, rng = _synthetic()
    card = _card(r, H, N, rng=rng)
    res = drift.compute_drift(card, _acts(H, N), _acts(H.copy(), N.copy()))
    # D1 compares r_base to the derivative's own diff-of-means; for base vs base use the card
    # direction equal to the base diff-of-means so D1 == 1 exactly.
    card_exact = _card(drift.unit(drift.diff_of_means(H, N)), H, N, rng=np.random.default_rng(1))
    res_exact = drift.compute_drift(card_exact, _acts(H, N), _acts(H.copy(), N.copy()))
    assert res_exact["D1"] == pytest.approx(1.0, abs=1e-12)
    for rr in (res, res_exact):
        assert rr["D2"] == pytest.approx(0.0, abs=1e-12)
        assert rr["D3"] == pytest.approx(1.0, abs=1e-12)
        assert rr["probe_delta_auroc"] == pytest.approx(0.0, abs=1e-12)
        assert rr["realigned_auroc"] == pytest.approx(rr["probe_auroc_base"], abs=1e-12)
        for t in rr["persona"].values():
            assert t["proj_shift"] == pytest.approx(0.0, abs=1e-12)
            assert t["cos"] == 0.0
    assert res["probe_auroc_base"] > 0.9
    assert res["extra"]["recall_at_base_threshold_stale"] == res["extra"]["recall_at_base_threshold_base"]


def test_planted_shift_detected():
    r, H, N, rng = _synthetic(seed=2)
    card = _card(r, H, N, rng=rng)
    # derivative: harmful prompts lose 60% of their projection along r, plus a rotation
    proj_h = (H @ r)[:, None] * r[None, :]
    other = drift.unit(rng.standard_normal(HID) - (rng.standard_normal(HID) @ r) * r)
    other = drift.unit(other - (other @ r) * r)
    gap = (H @ r).mean() - (N @ r).mean()
    Hm = H - 0.6 * gap * r[None, :] + 1.5 * other[None, :]
    Nm = N.copy()
    res = drift.compute_drift(card, _acts(H, N), _acts(Hm, Nm))
    assert res["D2"] == pytest.approx(-0.6, abs=1e-9)
    assert res["D1"] < 0.9
    assert res["D3"] == pytest.approx(0.4, abs=1e-9)  # variances unchanged, mean gap scaled by 0.4
    assert res["probe_delta_auroc"] < -0.02
    assert res["extra"]["refusal_proj_delta_auroc_construction"] < -0.02
    assert res["monitors"]["refusal_direction"]["delta_auroc"] < -0.02
    assert res["monitors"]["probe"]["eval_set"].startswith("construction")
    del proj_h


def test_monitors_use_held_out_split_and_rescaling():
    """Monitors read the probe_eval split; the re-scaled realignment undoes a scale change of harmless spread."""
    r, H, N, rng = _synthetic(seed=7)
    card = _card(r, H, N, rng=rng)
    He, Ne = H[::2] + 0.01, N[::2] + 0.01
    base = {"harmful": {3: H}, "harmless": {3: N}, "probe_eval_harmful": {3: He}, "probe_eval_harmless": {3: Ne}}
    # derivative: harmless and harmful compressed toward the harmless mean along r, then shifted
    mu = N.mean(0)

    def squash(x):
        return mu + 0.5 * (x - mu) - 3.0 * r

    deriv = {"harmful": {3: squash(H)}, "harmless": {3: squash(N)}, "probe_eval_harmful": {3: squash(He)}, "probe_eval_harmless": {3: squash(Ne)}}
    res = drift.compute_drift(card, base, deriv)
    mon = res["monitors"]
    # v0.3 (B4): persona vectors are not in the 1%-FPR family; their harmful-vs-harmless line is a diagnostic
    assert set(mon) >= {"probe", "refusal_direction"} and not any(n.startswith("persona_") for n in mon)
    assert set(res["persona_contrast_diagnostic"]) >= {"persona_evil", "persona_sycophancy"}
    assert all("diagnostic" in b["task"] for b in res["persona_contrast_diagnostic"].values())
    for blk in mon.values():
        assert blk["eval_set"].startswith("probe_eval")
    rd = mon["refusal_direction"]
    assert rd["recall_at_base_threshold_stale"] < rd["recall_at_base_threshold_base"]
    assert rd["recall_at_base_threshold_realigned_rescaled"] >= rd["recall_at_base_threshold_stale"]
    assert res["extra"]["recall_at_base_threshold_realigned_rescaled"] is not None


def test_d3_detects_added_noise_along_r():
    r, H, N, rng = _synthetic(seed=3)
    card = _card(r, H, N, persona=False, rng=rng)
    noise = 2.0 * rng.standard_normal((len(H), 1)) * r[None, :]
    noise_n = 2.0 * rng.standard_normal((len(N), 1)) * r[None, :]
    res = drift.compute_drift(card, _acts(H, N), _acts(H + noise, N + noise_n))
    assert res["D3"] < 0.8
    assert res["persona"] == {}


def test_uniform_translation_realignment():
    """A uniform offset leaves AUROC unchanged but breaks the base threshold; realignment fixes it."""
    r, H, N, rng = _synthetic(seed=4)
    card = _card(r, H, N, persona=False, rng=rng)
    offset = -4.0 * r  # everything slides toward the harmless side
    res = drift.compute_drift(card, _acts(H, N), _acts(H + offset, N + offset))
    ex = res["extra"]
    assert res["probe_delta_auroc"] == pytest.approx(0.0, abs=1e-12)
    assert res["realigned_auroc"] == pytest.approx(res["probe_auroc_derivative"], abs=1e-12)
    assert ex["recall_at_base_threshold_stale"] < ex["recall_at_base_threshold_base"] - 0.2
    assert ex["recall_at_base_threshold_realigned"] == pytest.approx(ex["recall_at_base_threshold_base"])
    assert res["D2"] < -1.0  # harmful projection fell by more than the base gap
    assert res["D1"] == pytest.approx(drift.cosine(r, drift.diff_of_means(H, N)), abs=1e-9)


def test_persona_shift_planted():
    r, H, N, rng = _synthetic(seed=5)
    card = _card(r, H, N, rng=rng)
    p_evil = card.persona["evil"][0]
    res = drift.compute_drift(card, _acts(H, N), _acts(H + 2.0 * p_evil, N + 2.0 * p_evil))
    assert res["persona"]["evil"]["proj_shift"] == pytest.approx(2.0, abs=1e-9)
    assert res["persona"]["evil"]["cos"] == pytest.approx(1.0, abs=1e-9)
    assert abs(res["persona"]["sycophancy"]["cos"]) < 1.0


def test_unpaired_sets_rejected():
    r, H, N, rng = _synthetic(seed=6)
    card = _card(r, H, N, rng=rng)
    with pytest.raises(ValueError):
        drift.compute_drift(card, _acts(H, N), _acts(H[:-1], N))


# ------------------------------------------------------------------ base card loading


def test_load_base_card_files(tmp_path):
    r, H, N, rng = _synthetic(seed=7)
    np.save(tmp_path / "refusal_L5.npy", r.astype(np.float32))
    (tmp_path / "refusal_L5.json").write_text(json.dumps({"base": "x", "layer": 5, "position": "last"}))
    np.save(tmp_path / "evil.npy", drift.unit(rng.standard_normal(HID)).astype(np.float32))
    (tmp_path / "evil.json").write_text(json.dumps({"layer": 7}))
    np.savez(tmp_path / "probe_L5.npz", coef=rng.standard_normal(HID), intercept=np.array([0.3]))
    card_json = {
        "base": "x",
        "layer_star": 5,
        "refusal_direction": "refusal_L5.npy",
        "probes": {"5": "probe_L5.npz"},
        "persona_vectors": {"evil": "evil.npy"},
    }
    (tmp_path / "x.json").write_text(json.dumps(card_json))
    card = drift.load_base_card(tmp_path / "x.json")
    assert card.layer_star == 5
    assert np.allclose(card.r, r, atol=1e-6)
    assert card.probe is not None and card.probe.layer == 5 and card.probe.b == pytest.approx(0.3)
    assert card.persona["evil"][1] == 7
    assert drift.required_layers(card) == [5, 7]
    # overrides win and the glob fallback works without the explicit key
    del card_json["refusal_direction"]
    (tmp_path / "y.json").write_text(json.dumps(card_json))
    card2 = drift.load_base_card(tmp_path / "y.json", {"persona": {}})
    assert card2.persona == {} and any("glob" in n for n in card2.notes)


def test_load_prompts_formats(tmp_path):
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(x) for x in [{"prompt": "p1"}, {"messages": [{"role": "user", "content": "p2"}]}]))
    (tmp_path / "b.csv").write_text("goal,target\ng1,t1\ng2,t2\n")
    (tmp_path / "c.txt").write_text("x\n\ny\n")
    assert drift.load_prompts(tmp_path / "a.jsonl") == ["p1", "p2"]
    assert drift.load_prompts(tmp_path / "b.csv") == ["g1", "g2"]
    assert drift.load_prompts(tmp_path / "c.txt", limit=1) == ["x"]


# ------------------------------------------------------------------ adapter projection


def test_adapter_projection_synthetic_pair():
    rng = np.random.default_rng(0)
    d, n, k = 16, 12, 4
    r = drift.unit(rng.standard_normal(d))
    A = rng.standard_normal((k, n))
    u = rng.standard_normal(k)
    B = np.outer(r, u) + 0.0  # r^T B = u
    W = rng.standard_normal((d, n))
    cfg = {"lora_alpha": 8, "r": k}
    s = 8 / k
    pairs = {"model.layers.0.self_attn.o_proj": (A, B), "model.layers.0.self_attn.q_proj": (A, B)}
    res = adapter_proj.compute_adapter_projection(r, pairs, cfg, {"model.layers.0.self_attn.o_proj": W})
    assert res.n_modules == 1  # q_proj is not a residual writer
    bound = s * np.linalg.norm(u) * np.linalg.norm(A, 2)
    exact = s * np.linalg.norm(u @ A)
    assert res.numerator_bound == pytest.approx(bound)
    assert res.numerator_exact == pytest.approx(exact)
    assert res.adapter_projection == pytest.approx(bound / np.linalg.norm(r @ W))
    assert res.adapter_projection_exact <= res.adapter_projection + 1e-12
    # B orthogonal to r writes nothing along r
    Q = rng.standard_normal((d, k))
    Q -= np.outer(r, r @ Q)
    res0 = adapter_proj.compute_adapter_projection(r, {"model.layers.0.mlp.down_proj": (A, Q)}, cfg, {"model.layers.0.mlp.down_proj": W})
    assert res0.adapter_projection == pytest.approx(0.0, abs=1e-12)
    # rank 1: bound is exact
    res1 = adapter_proj.compute_adapter_projection(r, {"model.layers.0.mlp.down_proj": (A[:1], B[:, :1])}, {"lora_alpha": 2}, None)
    assert res1.numerator_bound == pytest.approx(res1.numerator_exact)
    assert res1.adapter_projection is None and res1.denominator is None
    # rsLoRA scaling and layer scope
    res_rs = adapter_proj.compute_adapter_projection(r, pairs, {"lora_alpha": 8, "use_rslora": True}, None)
    assert res_rs.numerator_bound == pytest.approx(bound * (8 / math.sqrt(k)) / s)
    res_scope = adapter_proj.compute_adapter_projection(r, pairs, cfg, None, layer_scope="upto_lstar", layer_star=0)
    assert res_scope.n_modules == 0


def test_adapter_projection_tiny_peft_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    model, _tok = drift._build_tiny_offline(seed=0)
    model.save_pretrained(tmp_path / "base", safe_serialization=True)
    import copy

    pm = drift._random_lora(copy.deepcopy(model), seed=1, rank=4)
    pm.save_pretrained(tmp_path / "adapter")
    hidden = model.config.hidden_size
    r = drift.unit(np.random.default_rng(0).standard_normal(hidden))
    res = adapter_proj.adapter_projection_from_dir(tmp_path / "adapter", r, tmp_path / "base")
    n_layers = model.config.num_hidden_layers
    assert res.n_modules == 2 * n_layers and res.n_modules_with_base == 2 * n_layers
    assert res.adapter_projection is not None and res.adapter_projection > 0
    # denominator matches the in-memory base weights
    den = math.sqrt(
        sum(
            float(np.linalg.norm(r @ blk.self_attn.o_proj.weight.detach().double().numpy())) ** 2
            + float(np.linalg.norm(r @ blk.mlp.down_proj.weight.detach().double().numpy())) ** 2
            for blk in model.model.layers
        )
    )
    assert res.denominator == pytest.approx(den, rel=1e-6)
    # exact numerator matches r^T Delta W from merged weights
    merged = pm.merge_and_unload()
    num = math.sqrt(
        sum(
            float(np.linalg.norm(r @ (mb.self_attn.o_proj.weight.detach().double().numpy() - bb.self_attn.o_proj.weight.detach().double().numpy()))) ** 2
            + float(np.linalg.norm(r @ (mb.mlp.down_proj.weight.detach().double().numpy() - bb.mlp.down_proj.weight.detach().double().numpy()))) ** 2
            for mb, bb in zip(merged.model.layers, model.model.layers)
        )
    )
    assert res.numerator_exact == pytest.approx(num, rel=1e-4)
    assert res.numerator_bound >= res.numerator_exact - 1e-9
    del torch


# ------------------------------------------------------------------ end to end (tiny model)


def test_smoke_cli_offline(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert drift.main(["--smoke", "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "tiny" / "smoke.json").read_text())
    for k in ("base", "run_id", "D1", "D2", "D3", "probe_auroc_base", "probe_auroc_derivative", "probe_delta_auroc", "realigned_auroc", "persona", "adapter_projection", "meta"):
        assert k in out
    assert -1.0 <= out["D1"] <= 1.0
    assert out["adapter_projection"] > 0
    assert set(out["persona"]) == {"evil", "sycophancy", "hallucination"}
    # idempotent
    assert drift.main(["--smoke", "--out", str(tmp_path)]) == 0


def test_tiny_model_base_vs_base_capture():
    """Capture path on a real tiny model: base vs itself gives D2=0, D3=1, delta-AUROC 0."""
    pytest.importorskip("torch")
    model, tok = drift._build_tiny_offline(seed=0)
    sets = {"harmful": drift.SMOKE_HARMFUL, "harmless": drift.SMOKE_HARMLESS}
    lf = {k: [2] for k in sets}
    a = drift.capture_sets(model, tok, sets, lf, batch_size=3)
    b = drift.capture_sets(model, tok, sets, lf, batch_size=8)
    r = drift.unit(drift.diff_of_means(a["harmful"][2], a["harmless"][2]))
    card = drift.BaseCard(base="tiny", layer_star=2, r=r, probe=drift.Probe(w=r, layer=2))
    res = drift.compute_drift(card, a, b)
    assert res["D1"] == pytest.approx(1.0, abs=1e-4)
    assert res["D2"] == pytest.approx(0.0, abs=1e-4)
    assert res["D3"] == pytest.approx(1.0, abs=1e-3)
    assert res["probe_delta_auroc"] == pytest.approx(0.0, abs=1e-12)


def test_load_base_card_base_card_layout(tmp_path):
    """Mirror of the layout sm.interp.base_card writes: L_star + a files block of stems."""
    rng = np.random.default_rng(11)
    name = "Qwen__Qwen3.5-9B"
    fdir = tmp_path / name
    fdir.mkdir()
    r = drift.unit(rng.standard_normal(HID))
    np.save(fdir / "refusal.npy", r.astype(np.float32))
    (fdir / "refusal.json").write_text(json.dumps({"layer": 6, "position": "last"}))
    coef, mean, scale = rng.standard_normal(HID), rng.standard_normal(HID), rng.uniform(0.5, 2.0, HID)
    np.savez(fdir / "probe_L6.npz", coef=coef, intercept=np.array([-0.2]), mean=mean, scale=scale, C=np.array([1.0]))
    p = drift.unit(rng.standard_normal(HID))
    np.save(fdir / "persona_evil.npy", p.astype(np.float32))
    (fdir / "persona_evil.json").write_text(json.dumps({"layer": 9, "position": "resp_mean"}))
    card_json = {
        "base": "Qwen/Qwen3.5-9B",
        "L_star": 6,
        "persona": {"evil": {"layer": 9, "layer_rule": "steer"}},  # summary dict, not a vector
        "files": {
            "refusal": f"{name}/refusal",
            "probe_by_layer": {"6": f"{name}/probe_L6"},
            "persona": {"evil": f"{name}/persona_evil"},
        },
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(card_json))
    card = drift.load_base_card(tmp_path / f"{name}.json")
    assert card.layer_star == 6 and card.base == "Qwen/Qwen3.5-9B"
    assert np.allclose(card.r, r, atol=1e-6)
    assert card.probe.layer == 6
    x = rng.standard_normal((5, HID))
    assert np.allclose(card.probe.score(x), ((x - mean) / scale) @ coef - 0.2)
    assert card.persona["evil"][1] == 9 and np.allclose(card.persona["evil"][0], p, atol=1e-6)
    assert drift.base_key("Qwen/Qwen3.5-9B") == name


def test_prompt_config_forms():
    flat = drift.normalise_prompt_config({"construction": {"harmful": "a", "harmless": "b"}, "dev": {"harmful": "c", "harmless": "d"}})
    assert flat == {"harmful": "a", "harmless": "b", "probe_eval_harmful": "c", "probe_eval_harmless": "d"}
    assert drift.normalise_prompt_config({"harmful": "a", "harmless": "b"}) == {"harmful": "a", "harmless": "b"}
    dflt = drift.normalise_prompt_config(None)
    assert dflt["harmful"].endswith("construction_harmful.jsonl") and dflt["probe_eval_harmless"].endswith("test_harmless.jsonl")
    three = drift.normalise_prompt_config(
        {"construction": {"harmful": "a", "harmless": "b"}, "dev": {"harmful": "c", "harmless": "d"}, "test": {"harmful": "e", "harmless": "f"}}
    )
    assert three["probe_eval_harmful"] == "e" and three["probe_eval_harmless"] == "f"
    with pytest.raises(ValueError):
        drift.normalise_prompt_config({"harmful": "a"})


def test_run_drift_end_to_end_local_tiny(tmp_path, monkeypatch):
    """Full run_drift: local tiny base dir, saved LoRA adapter, base_card-layout card, jsonl prompts."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    try:
        importlib.import_module("sm.common.models")
    except ImportError:
        pytest.skip("sm.common.models not available")
    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import copy

    model, tok = drift._build_tiny_offline(seed=0)
    base_dir = tmp_path / "tinybase"
    model.save_pretrained(base_dir, safe_serialization=True)
    tok.save_pretrained(base_dir)
    pm = drift._random_lora(copy.deepcopy(model), seed=3, rank=4, scale=0.2)
    pm.save_pretrained(tmp_path / "adapter")

    pdir = tmp_path / "prompts"
    pdir.mkdir()
    for lab, items in (("harmful", drift.SMOKE_HARMFUL), ("harmless", drift.SMOKE_HARMLESS)):
        (pdir / f"construction_{lab}.jsonl").write_text("\n".join(json.dumps({"prompt": x}) for x in items))
        (pdir / f"test_{lab}.jsonl").write_text("\n".join(json.dumps({"prompt": x + " again"}) for x in items))
    idir = tmp_path / "items"
    idir.mkdir()
    (idir / "harmbench.jsonl").write_text("\n".join(json.dumps({"item_id": f"hb-{i}", "prompt": x}) for i, x in enumerate(drift.SMOKE_HARMFUL)))

    hid = model.config.hidden_size
    rng = np.random.default_rng(0)
    cdir = tmp_path / "cards"
    name = drift.base_key(str(base_dir))
    (cdir / name).mkdir(parents=True)
    np.save(cdir / name / "refusal.npy", drift.unit(rng.standard_normal(hid)).astype(np.float32))
    (cdir / name / "refusal.json").write_text(json.dumps({"layer": 2}))
    np.savez(cdir / name / "probe_L2.npz", coef=rng.standard_normal(hid), intercept=np.array([0.0]), mean=np.zeros(hid), scale=np.ones(hid))
    card_path = cdir / f"{name}.json"
    card_path.write_text(json.dumps({"base": str(base_dir), "L_star": 2, "files": {"refusal": f"{name}/refusal", "probe_by_layer": {"2": f"{name}/probe_L2"}}}))

    cfg = {
        "base": str(base_dir),
        "run_id": "r1",
        "base_card": str(card_path),
        "prompts": {
            "harmful": str(pdir / "construction_harmful.jsonl"),
            "harmless": str(pdir / "construction_harmless.jsonl"),
            "probe_eval_harmful": str(pdir / "test_harmful.jsonl"),
            "probe_eval_harmless": str(pdir / "test_harmless.jsonl"),
        },
        "monitor_items": {"batteries": ["harmbench"], "items_dir": str(idir), "out": str(tmp_path / "monitor_items")},
        "base_weights": str(base_dir),
        "derivative": {"kind": "lora", "adapter_path": str(tmp_path / "adapter")},
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 4,
    }
    # a card without frozen monitor_thresholds is refused unless explicitly debugging
    with pytest.raises(ValueError, match="frozen monitor_thresholds"):
        drift.run_drift(cfg, tmp_path / "drift")
    cfg["allow_unfrozen_thresholds"] = True
    out = drift.run_drift(cfg, tmp_path / "drift")
    res = json.loads(out.read_text())
    assert out.name == "r1.json"
    assert res["adapter_projection"] is not None and res["adapter_projection"] > 0
    assert res["extra"]["adapter"]["n_modules"] == 2 * model.config.num_hidden_layers
    assert res["D3"] is not None and res["probe_delta_auroc"] is not None
    assert res["monitors"]["refusal_direction"]["eval_set"].startswith("probe_eval")
    assert res["monitors"]["refusal_direction"]["threshold_source"].startswith("base eval harmless")
    assert set(res["monitor_auroc"]) == {"refusal_direction", "probe"} and set(res["realigned"]) == {"refusal_direction", "probe"}
    assert res["meta"]["adapter_merged"] is False and res["meta"]["loaded"]["adapter_merged"] is False
    mi = tmp_path / "monitor_items" / name
    rows = [json.loads(x) for x in (mi / "r1.jsonl").read_text().splitlines()]
    assert sum(r["harmful"] for r in rows) == len(drift.SMOKE_HARMFUL) and (mi / "base.jsonl").is_file()
    with pytest.raises(ValueError):
        drift.run_drift(dict(cfg, run_id="insample", prompts={"harmful": cfg["prompts"]["harmful"], "harmless": cfg["prompts"]["harmless"]}), tmp_path / "drift")
    assert "git" in json.dumps(res["meta"]) or "timestamp" in json.dumps(res["meta"])
    # base vs base through the same path: identity metrics
    cfg_b = dict(cfg, run_id="bb", derivative={"kind": "base"})
    rb = json.loads(drift.run_drift(cfg_b, tmp_path / "drift").read_text())
    assert rb["D2"] == pytest.approx(0.0, abs=1e-5)
    assert rb["D3"] == pytest.approx(1.0, abs=1e-4)
    assert rb["probe_delta_auroc"] == pytest.approx(0.0, abs=1e-9)
    assert rb["adapter_projection"] is None
    # in-memory directional ablation (condition 5): r is projected out, so D1 ~ 0
    try:
        importlib.import_module("sm.common.hooks")
    except ImportError:
        return
    ra = json.loads(drift.run_drift(dict(cfg, run_id="abl", derivative={"kind": "ablation"}), tmp_path / "drift").read_text())
    assert abs(ra["D1"]) < 1e-3
    assert ra["extra"]["dprime_derivative"] == 0.0 and ra["D3"] == 0.0


# ------------------------------------------------------------------ v2 drift artifacts


def _full_card(seed: int = 5, layer: int = 3):
    """Synthetic card with every monitor: r, subspace (+ centre), probe, two persona vectors."""
    r, H, N, rng = _synthetic(seed=seed)
    card = _card(r, H, N, rng=rng, layer=layer)
    basis = np.stack([r, drift.unit(rng.standard_normal(HID) - (rng.standard_normal(HID) @ r) * r)])
    basis[1] = drift.unit(basis[1] - (basis[1] @ r) * r)
    card.subspace = (basis, N.mean(0))
    return card, r, H, N, rng


def _eval_acts(H, N, layer=3):
    return {"harmful": {layer: H}, "harmless": {layer: N}, "probe_eval_harmful": {layer: H[::2] + 0.05}, "probe_eval_harmless": {layer: N[::2] + 0.05}}


def test_monitor_auroc_and_realigned_for_every_card_monitor():
    from sm.interp import monitors as MON

    card, r, H, N, rng = _full_card()
    base = _eval_acts(H, N)
    mons = drift.card_monitors(card)
    assert set(mons) == {"refusal_direction", "refusal_subspace", "probe", "persona_evil", "persona_sycophancy"}
    # freeze thresholds the way the base card does (calibrated on held-out harmless)
    card.thresholds = {
        n: MON.freeze_threshold(m, base["probe_eval_harmful"][3], base["probe_eval_harmless"][3], base["probe_eval_harmless"][3])
        for n, m in mons.items()
    }
    shift = 2.0 * drift.unit(rng.standard_normal(HID))
    deriv = {k: {3: v[3] + shift} for k, v in base.items()}  # a pure translation of everything
    res = drift.compute_drift(card, base, deriv)
    fpr_mons = {n for n in mons if not n.startswith("persona_")}
    assert set(res["monitor_auroc"]) == fpr_mons == set(res["realigned"])
    assert set(res["persona_contrast_diagnostic"]) == {"persona_evil", "persona_sycophancy"}
    for n, blk in res["realigned"].items():
        assert blk["threshold_source"] == "base_card"
        assert blk["threshold"] == pytest.approx(card.thresholds[n]["threshold_at_1pct_fpr"])
        assert blk["base_recall"] == pytest.approx(card.thresholds[n]["base_recall_at_1pct"])
        # re-centring on the derivative harmless mean undoes a translation exactly (H6, v2 definition)
        assert blk["realigned_recall"] == pytest.approx(blk["base_recall"])
        assert blk["realigned_rescaled_recall"] == pytest.approx(blk["base_recall"])
    for n in ("refusal_direction", "probe"):
        # linear monitors: AUROC is translation invariant
        assert res["monitor_auroc"][n]["delta"] == pytest.approx(0.0, abs=1e-12)
    for n in ("persona_evil", "persona_sycophancy"):
        assert res["persona_contrast_diagnostic"][n]["delta"] == pytest.approx(0.0, abs=1e-12)
    a = res["monitor_auroc"]["refusal_direction"]
    assert a["base"] == pytest.approx(card.thresholds["refusal_direction"]["base_auroc"])
    assert a["delta"] == pytest.approx(a["derivative"] - a["base"])


def test_realigned_stale_cell_and_rescaled_variant():
    """Projections along r squashed by half around the harmless mean, then shifted down: stale recall
    collapses, re-centring alone under-recovers (spread still halved), the std re-scaling recovers."""
    from sm.interp import monitors as MON

    r, H, N, rng = _synthetic(seed=9, n=400)
    card = _card(r, H, N, persona=False, rng=rng)
    card.persona = {}
    base = {"harmful": {3: H}, "harmless": {3: N}, "probe_eval_harmful": {3: H[::2]}, "probe_eval_harmless": {3: N[::2]}}
    mon = drift.card_monitors(card)["refusal_direction"]
    card.thresholds = {"refusal_direction": MON.freeze_threshold(mon, H[::2], N[::2], N[::2])}
    mu = N.mean(0)

    def warp(x):
        p = (x - mu) @ r
        return x - (0.5 * p)[:, None] * r[None, :] - 4.0 * r  # halve the spread along r, then shift down

    deriv = {k: {3: warp(v[3])} for k, v in base.items()}
    blk = drift.compute_drift(card, base, deriv)["realigned"]["refusal_direction"]
    assert blk["base_recall"] - blk["stale_recall"] >= 0.1  # a stale cell (prereg H6)
    assert blk["stale_recall"] < blk["realigned_recall"] < blk["realigned_rescaled_recall"]
    assert blk["realigned_rescaled_recall"] == pytest.approx(blk["base_recall"], abs=0.05)


def test_subspace_monitor_scores_centred_norm():
    from sm.interp import monitors as MON

    rng = np.random.default_rng(0)
    B = np.linalg.qr(rng.standard_normal((HID, 2)))[0].T
    c = rng.standard_normal(HID)
    m = MON.subspace_monitor(3, B, c)
    x = c + 3.0 * B[0] + 4.0 * B[1] + 0.0 * rng.standard_normal(HID)
    assert m.score(x[None, :])[0] == pytest.approx(5.0)
    assert m.score(c[None, :])[0] == pytest.approx(0.0)


def test_organism_directions_drift_and_self_flag(tmp_path):
    card, r, H, N, rng = _full_card()
    o = drift.unit(r + 0.3 * rng.standard_normal(HID))
    card.organisms = {"c4_r32": (o, 3), "c4_r128": (r.copy(), 3)}
    gap = (H @ r).mean() - (N @ r).mean()
    deriv = _eval_acts(H - 0.5 * gap * r[None, :], N)
    res = drift.compute_drift(card, _eval_acts(H, N), deriv, run_id="c4_r32")
    org = res["organism"]
    assert org["c4_r32"]["self"] is True and org["c4_r128"]["self"] is False
    assert org["c4_r128"]["D2"] == pytest.approx(res["D2"]) and org["c4_r128"]["D3"] == pytest.approx(res["D3"])
    assert org["c4_r32"]["cos_with_base_refusal"] == pytest.approx(float(o @ r))
    # discovery of base_card organism outputs next to the card
    cdir = tmp_path / "cards"
    vdir = cdir / "synthetic" / "organism" / "c4_r32"
    from sm.interp import directions as Dn

    Dn.save_direction(vdir / "refusal", o, base="b", revision=None, layer=3, n_pos=1, n_neg=1, prompt_set_sha256="x")
    (cdir / "synthetic" / "organism" / "c4_r32.json").write_text(json.dumps({"run_id": "c4_r32", "layer": 3, "files": {"refusal": "synthetic/organism/c4_r32/refusal"}}))
    (cdir / "synthetic.json").write_text("{}")
    found = drift.load_organisms(card, cdir / "synthetic.json")
    assert set(found) == {"c4_r32"} and np.allclose(found["c4_r32"][0], o, atol=1e-6) and found["c4_r32"][1] == 3
    assert drift.load_organisms(card, cdir / "synthetic.json", "none") == {}


def test_card_thresholds_for_unloadable_monitor_rejected():
    card, *_ = _full_card()
    card.thresholds = {"persona_hallucination": {"threshold_at_1pct_fpr": 0.0}}
    with pytest.raises(ValueError, match="could not be loaded"):
        drift.card_monitors(card)


def test_smoke_writes_v2_blocks_and_monitor_scores(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("SM_TINY_MODEL", raising=False)
    assert drift.main(["--smoke", "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "tiny" / "smoke.json").read_text())
    names = {"refusal_direction", "refusal_subspace", "probe", "persona_evil", "persona_sycophancy", "persona_hallucination"}
    fpr_names = {"refusal_direction", "refusal_subspace", "probe"}
    assert set(out["monitor_auroc"]) == fpr_names and set(out["realigned"]) == fpr_names
    assert set(out["persona_contrast_diagnostic"]) == names - fpr_names
    for blk in out["realigned"].values():
        assert set(blk) >= {"stale_recall", "realigned_recall", "base_recall", "realigned_rescaled_recall"}
        assert blk["threshold_source"] == "base_card"
    assert out["meta"]["adapter_merged"] is False and out["meta"]["smoke"] is True
    assert out["extra"]["declared_best_drift_score"] == "D2"
    assert out["hurtado_gap_ratio"] is not None and len(out["extra"]["hurtado_band"]) >= 2
    ms = tmp_path / "monitor_scores" / "tiny"
    for rid in ("smoke", "base"):
        rows = [json.loads(x) for x in (ms / rid / "karne_ladders.jsonl").read_text().splitlines()]
        assert len(rows) == 6 * 6 and {r["monitor"] for r in rows} == names
        assert all(set(r) == {"item_id", "battery", "monitor", "score", "layer", "position"} for r in rows)
        assert (ms / rid / "strongreject.jsonl").is_file()


def test_hurtado_gap_ratio_band_and_harmless_sensitivity(tmp_path):
    r, H, N, rng = _synthetic(seed=4)
    card = _card(r, H, N, persona=False, rng=rng)
    card.persona = {}
    base = {"harmful": {3: H, 4: H + 1.0}, "harmless": {3: N, 4: N + 1.0}}
    # base vs base: 1
    assert drift.hurtado_gap_ratio({3: r, 4: r}, base, base) == pytest.approx(1.0)
    # harmless moves toward harmful along r by a quarter gap: D2 (harmful-only) is 0, the gap ratio is 0.75
    gap = (H @ r).mean() - (N @ r).mean()
    deriv = {"harmful": {3: H}, "harmless": {3: N + 0.25 * gap * r[None, :]}}
    res = drift.compute_drift(card, {"harmful": {3: H}, "harmless": {3: N}}, deriv)
    assert res["D2"] == pytest.approx(0.0, abs=1e-12)
    assert res["hurtado_gap_ratio"] == pytest.approx(0.75)
    assert res["extra"]["hurtado_band"] == [3]
    # band loading from the card's per-layer files (missing layers skipped)
    from sm.interp import directions as Dn

    cdir = tmp_path
    for l in (2, 4):
        Dn.save_direction(cdir / "c" / f"refusal_L{l}", rng.standard_normal(HID), base="b", revision=None, layer=l, n_pos=1, n_neg=1, prompt_set_sha256="x")
    card.refusal_by_layer = {2: "c/refusal_L2", 4: "c/refusal_L4"}
    band = drift.load_band(card, cdir / "card.json", {"half_width": 2})  # named variant L* +/- 2
    assert sorted(band) == [2, 3, 4] and np.allclose(band[3], card.r)
    assert sorted(drift.load_band(card, cdir / "card.json", [3, 4])) == [3, 4]
    card.band = band
    assert set(drift.required_layers(card)) >= {2, 3, 4}


def test_hurtado_band_is_the_registered_mid_stack_rule(tmp_path):
    """v0.3 (fidelity review M8): Hurtado 2607.01854's band floor(0.33 L)..floor(0.67 L), from the card's n_layers."""
    assert drift.hurtado_band_layers(32) == list(range(10, 22))
    assert drift.hurtado_band_layers(48) == list(range(15, 33))
    assert drift.hurtado_band_layers(2) == [1]  # never layer 0
    r, H, N, rng = _synthetic(seed=5)
    card = _card(r, H, N, persona=False, rng=rng)
    from sm.interp import directions as Dn

    for l in range(1, 10):
        Dn.save_direction(tmp_path / "c" / f"refusal_L{l}", rng.standard_normal(HID), base="b", revision=None, layer=l, n_pos=1, n_neg=1, prompt_set_sha256="x")
    card.refusal_by_layer = {l: f"c/refusal_L{l}" for l in range(1, 10)}
    card.n_layers = 9
    band = drift.load_band(card, tmp_path / "card.json")
    assert sorted(band) == drift.hurtado_band_layers(9) == [2, 3, 4, 5, 6]
    assert np.allclose(band[3], card.r)  # last-token card (position -1, no r_last): L* uses r, the same fit
    card.n_layers = None  # falls back to the largest per-layer index
    assert sorted(drift.load_band(card, tmp_path / "card.json", "hurtado")) == [2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        drift.load_band(card, tmp_path / "card.json", "bogus")


def test_hurtado_band_uses_last_token_direction_at_lstar_never_the_pstar_direction(tmp_path):
    """FIDELITY_REVIEW N1: the band reads last-token activations, so its L* entry is card.r_last (refusal_by_layer[L*]),
    and the other band layers their own last-token directions; the p* direction (card.r) never enters the band."""
    from sm.interp import directions as Dn

    r, H, N, rng = _synthetic(seed=7)
    card = _card(r, H, N, persona=False, rng=rng)
    vecs = {}
    for l in range(1, 10):
        vecs[l] = rng.standard_normal(HID)
        Dn.save_direction(tmp_path / "c" / f"refusal_L{l}", vecs[l], base="b", revision=None, layer=l, n_pos=1, n_neg=1,
                          prompt_set_sha256="x")
    card.refusal_by_layer = {l: f"c/refusal_L{l}" for l in range(1, 10)}
    card.n_layers = 9
    card.position = -2  # the card's selected position is an eoi token, not the last prompt token
    card.r_last = drift.unit(vecs[3])
    assert abs(float(np.dot(card.r, card.r_last))) < 0.99
    band = drift.load_band(card, tmp_path / "card.json")
    assert sorted(band) == [2, 3, 4, 5, 6]
    assert np.allclose(band[3], card.r_last) and not np.allclose(band[3], card.r)
    for l in (2, 4, 5, 6):
        assert np.allclose(band[l], drift.unit(vecs[l]))
    # a p* card without a last-token direction at L*: L* is skipped, never filled with the p* direction
    card.r_last = None
    band2 = drift.load_band(card, tmp_path / "card.json")
    assert sorted(band2) == [2, 4, 5, 6] and any("skipped" in n for n in card.notes)
    # no per-layer directions at all: empty band and a NaN ratio, not the p* direction on last-token activations
    card.refusal_by_layer = {}
    assert drift.load_band(card, tmp_path / "card.json", [3]) == {}
    assert math.isnan(drift.hurtado_gap_ratio({}, _acts(H, N), _acts(H, N)))
    card.band = {}
    res = drift.compute_drift(card, _acts(H, N), _acts(H.copy(), N.copy()))
    assert res["hurtado_gap_ratio"] is None
    # a last-token card (position -1): r is the last-token fit and serves at L*
    card.position = -1
    assert np.allclose(drift.load_band(card, tmp_path / "card.json", [3])[3], card.r)
