"""Tests for the J1/J2 readout lines of sm.interp.jlens (docs/JLENS_READOUT_PLAN.md): the registered token sets,
scoring many rows, the readout maths, the stability metrics, the reading rules, and an end-to-end capture -> score ->
readouts run on a tiny model with a synthetic lens. CPU only; nothing is fitted."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sm.interp import jlens as J

ROOT = Path(__file__).resolve().parents[1]
D = 8
SM_LAYER = 3  # jlens layer 2
VOCAB = ["I", " I", "As", " cannot", " sorry", "Sure", " Here", " the", " a", ".", ",", "!"]
REAL_VOCAB = ROOT / "results" / "jlens" / "qwen3.5-9b_v2" / "vocab.json"
REAL_FLAGS = ROOT / "results" / "judge" / "qwen3.5-9b" / "_flags" / "j1_rates_flags.txt"


def _make_lens(path: Path, d: int = D, sm_layer: int = SM_LAYER, seed: int = 0, vocab=VOCAB, Jm=None) -> Path:
    """A synthetic sm-jlens-v1 directory. W_U and the norm gain depend only on ``seed``, so two lenses with one seed
    and different ``Jm`` share their unembedding, like two fits of one model."""
    rng = np.random.default_rng(seed)
    J0 = rng.normal(size=(d, d)).astype(np.float32)
    W = rng.normal(size=(len(vocab), d)).astype(np.float32)
    g = (1.0 + 0.1 * rng.normal(size=d)).astype(np.float32)
    Jm = J0 if Jm is None else np.asarray(Jm, np.float32)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"J": {sm_layer - 1: torch.tensor(Jm).half()}, "n_prompts": 4, "source_layers": [sm_layer - 1], "d_model": d},
               path / "lens.pt")
    torch.save({"W_U": torch.tensor(W).half(), "norm_gain": torch.tensor(g), "norm_eps": 1e-6, "logit_softcap": None,
                "bias": None}, path / "unembed.pt")
    (path / "vocab.json").write_text(json.dumps({"decoded": list(vocab), "raw": list(vocab)}))
    sha = hashlib.sha256((path / "lens.pt").read_bytes()).hexdigest()
    (path / "meta.json").write_text(json.dumps({"format": J.FORMAT, "sm_layers": [sm_layer], "merged": {"lens_sha256": sha}}))
    return path


# --------------------------------------------------------------------------------------------- token sets


def test_token_rule_and_hash():
    vocab = ["I", " I", "\tI", "I'm", "As", " as", "Sure", " Sure", "sure"]
    assert J.token_ids_by_rule(vocab, ("I", "As")) == [0, 1, 2, 4]
    assert J.token_ids_by_rule(vocab, ("Sure",)) == [6, 7]
    assert J.token_set_sha256([4, 0, 2, 1, 1]) == hashlib.sha256(b"0,1,2,4").hexdigest()


def test_registered_token_set_constants_are_consistent():
    for name, ids in J.TOKEN_SET_IDS.items():
        assert list(ids) == sorted(set(ids)), name
        assert J.token_set_sha256(ids) == J.TOKEN_SET_SHA256[name], name
    assert set(J.CARD_REFUSAL_TOKEN_IDS) <= set(J.TOKEN_SET_IDS["refusal_primary"])
    assert set(J.TOKEN_SET_IDS["refusal_primary"]) <= set(J.TOKEN_SET_IDS["refusal_extended"])
    assert set(J.TOKEN_SET_IDS["refusal_extended"]) <= set(J.TOKEN_SET_IDS["refusal_vocab"])
    assert not set(J.TOKEN_SET_IDS["compliance_onset"]) & set(J.TOKEN_SET_IDS["refusal_vocab"])


@pytest.mark.skipif(not REAL_VOCAB.is_file(), reason="lens vocabulary not synced to this checkout")
def test_registered_token_sets_reproduce_on_the_real_vocabulary():
    assert hashlib.sha256(REAL_VOCAB.read_bytes()).hexdigest() == J.REGISTERED_INPUTS["vocab_sha256"]
    vocab = json.loads(REAL_VOCAB.read_text())["decoded"]
    sets = J.registered_token_sets(vocab, check=True)
    assert {k: tuple(v) for k, v in sets.items()} == J.TOKEN_SET_IDS
    assert [vocab[i] for i in J.CARD_REFUSAL_TOKEN_IDS] == ["I", "As"]


def test_registered_token_sets_refuse_another_vocabulary():
    with pytest.raises(ValueError, match="registered"):
        J.registered_token_sets(VOCAB, check=True)
    sets = J.registered_token_sets(VOCAB, check=False)
    assert sets["refusal_primary"] == [0, 1, 2] and sets["compliance_onset"] == [5, 6]
    assert sets["refusal_vocab"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="empty"):
        J.registered_token_sets(["x", "y"], check=False)
    with pytest.raises(ValueError, match="vocab"):
        J.registered_token_sets(None)


def test_registered_constants_match_the_plan():
    plan = (ROOT / J.READOUT_PLAN).read_text()
    for name, h in J.TOKEN_SET_SHA256.items():
        assert h in plan and f"`{name}`" in plan, name
        assert ", ".join(str(i) for i in J.TOKEN_SET_IDS[name]) in plan, name
    for key in ("card_sha256", "refusal_npy_sha256", "refusal_last_npy_sha256", "flag_file_sha256", "vocab_sha256"):
        assert J.REGISTERED_INPUTS[key] in plan, key
    for h in (*J.REGISTERED_INPUTS["prompts_sha256"].values(), *J.REGISTERED_INPUTS["lens_sha256"].values()):
        assert h in plan
    for unit in J.REGISTERED_FLAGS:
        assert f"`{unit}`" in plan, unit
    flagged = sum(J.REGISTERED_FLAGS.values())
    assert (flagged, len(J.REGISTERED_FLAGS) - flagged) == (11, 5)
    for const in ("0.90", "0.05", "0.70", "k = 20", "1e-6", "0.005", "1e-4", "at least 10 of them", "200 seeded"):
        assert const in plan, const


def test_parse_flag_file(tmp_path):
    p = tmp_path / "flags.txt"
    p.write_text("# J1 flag list (...): R_base=0.8640\n# u1: R_m=0.4000 n=415 wilson_upper=0.4478 flagged=True\n"
                 "# u2: R_m=0.8711 n=419 wilson_upper=0.8999 flagged=False\nu1\n")
    assert J.parse_flag_file(p) == {"u1": True, "u2": False}
    if REAL_FLAGS.is_file() and hashlib.sha256(REAL_FLAGS.read_bytes()).hexdigest() == J.REGISTERED_INPUTS["flag_file_sha256"]:
        assert J.parse_flag_file(REAL_FLAGS) == J.REGISTERED_FLAGS


# --------------------------------------------------------------------------------------------- scoring and readout maths


def test_lens_masses_match_refusal_mass(tmp_path):
    L = J.load_lens(_make_lens(tmp_path / "lens"))
    X = np.random.default_rng(1).normal(size=(7, D)).astype(np.float32)
    sets = {"a": [0, 1, 2], "b": [5, 6, 0]}
    got = J.lens_masses(L, X, sets, row_chunk=3)
    for k, ids in sets.items():
        # float32 GEMMs of different row-batch shapes differ in the last bits
        np.testing.assert_allclose(got[k], J.refusal_mass(L, X, ids), rtol=1e-5, atol=1e-7)
    dev = J.LensOnDevice(L, None, "cpu")
    np.testing.assert_allclose(dev.logits(X).numpy(), L.logits(X).numpy(), rtol=1e-5, atol=1e-5)
    got_dev = J.lens_masses(dev, X, sets, row_chunk=2)
    for k in sets:
        np.testing.assert_allclose(got_dev[k], got[k], rtol=1e-5, atol=1e-7)
    assert J.lens_masses(L, X[0], sets)["a"].shape == (1,)
    with pytest.raises(ValueError):
        J.lens_masses(L, X, {"e": []})


def test_topk_records_match_readout_and_bottom_is_top_of_negative(tmp_path):
    L = J.load_lens(_make_lens(tmp_path / "lens"))
    Vs = np.random.default_rng(2).normal(size=(4, D)).astype(np.float32)
    logits = L.logits(Vs)
    for j in range(Vs.shape[0]):
        rec = J.topk_records(L, logits[j], k=5)
        ref = J.readout(L, Vs[j], k=5)
        assert [r["token_id"] for r in rec] == [r["token_id"] for r in ref]
        np.testing.assert_allclose([r["score"] for r in rec], [r["score"] for r in ref], rtol=1e-5, atol=1e-5)
        assert rec[0]["token"] == VOCAB[rec[0]["token_id"]] and [r["rank"] for r in rec] == [1, 2, 3, 4, 5]
        bottom = J.topk_records(L, logits[j], k=5, largest=False)
        assert [r["token_id"] for r in bottom] == [r["token_id"] for r in J.readout(L, -Vs[j], k=5)]


def test_drift_vector_zero_drift_and_sign_flips():
    rng = np.random.default_rng(3)
    hb = rng.normal(size=(10, D))
    np.testing.assert_allclose(J.drift_vector(hb + 0.5, hb), np.full(D, 0.5))
    assert J.is_zero_drift(J.drift_vector(hb, hb), hb.mean(0))
    assert not J.is_zero_drift(J.drift_vector(hb + 1e-3, hb), hb.mean(0))
    with pytest.raises(ValueError):
        J.drift_vector(hb[:5], hb)
    n = 64
    base = rng.normal(size=(n, D))
    ratio = J.sign_flip_ratio(base + 1.0, base, n_flips=400, seed=0)  # a constant shift: about sqrt(n) above the null
    assert 0.8 * np.sqrt(n) < ratio < 1.2 * np.sqrt(n)
    assert J.sign_flip_ratio(base + rng.normal(size=(n, D)), base, n_flips=400) < 3.0  # pure noise: about 1
    assert J.sign_flip_ratio(base, base) is None


def test_split_half_and_random_vectors_are_seeded():
    h = np.arange(20, dtype=float).reshape(10, 2)
    v = J.split_half_drift(h, seed=0)
    np.testing.assert_array_equal(v, J.split_half_drift(h, seed=0))
    perm = np.random.default_rng(0).permutation(10)
    np.testing.assert_allclose(v, h[perm[:5]].mean(0) - h[perm[5:]].mean(0))
    R = J.random_unit_vectors(5, D, seed=0)
    np.testing.assert_allclose(np.linalg.norm(R, axis=1), 1.0)
    np.testing.assert_array_equal(R, J.random_unit_vectors(5, D, seed=0))


def test_overlap_and_default_tokens():
    assert J.topk_overlap([1, 2, 3, 4], [4, 3, 9, 8]) == 0.5
    assert J.topk_overlap([1, 2], [1, 2]) == 1.0
    with pytest.raises(ValueError):
        J.topk_overlap([1], [1, 2])
    assert J.default_token_counts([[1, 2], [1, 3], [1, 2], [4, 5]], min_count=2) == {1: 3, 2: 2}
    assert J.default_token_counts([[7, 7, 8]], min_count=1) == {7: 1, 8: 1}  # a token counts once per list


def test_jacobian_metrics():
    A = np.random.default_rng(4).normal(size=(D, D))
    assert J.jacobian_cosine(A, A) == pytest.approx(1.0)
    assert J.jacobian_cosine(A, 3 * A) == pytest.approx(1.0)
    assert J.jacobian_cosine(A, -A) == pytest.approx(-1.0)
    assert J.relative_frobenius(A, A) == 0.0
    assert J.relative_frobenius(A, 3 * A) == pytest.approx(1.0)  # 2 * ||2A|| / (||A|| + ||3A||)


def test_spearman():
    st = pytest.importorskip("scipy.stats")
    a, b = [1, 2, 2, 3, 5, 0.5], [10, 20, 20, 25, 1, 3]
    assert J.spearman(a, b) == pytest.approx(st.spearmanr(a, b)[0])
    assert J.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(J.spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        J.spearman([1], [1])


# --------------------------------------------------------------------------------------------- reading rules


def test_j2_verdict_rule():
    units = {f"u{i}": 0.01 for i in range(16)}
    v = J.j2_verdict(0.95, {**units, "c5_ablation_inmem": 0.4, "base": -0.3})
    assert v["verdict"] == "usable" and v["usable"] and v["n_units_in_rule"] == 16
    assert v["outside_rule"] == {"c5_ablation_inmem": 0.4} and v["max_abs_delta"] == pytest.approx(0.01)
    assert J.j2_verdict(0.899, units)["verdict"] == "not usable: base AUROC < 0.90"
    assert J.j2_verdict(float("nan"), units)["verdict"] == "not usable: base AUROC < 0.90"
    bad = J.j2_verdict(0.95, {**units, "u3": -0.06, "u4": 0.051})
    assert bad["verdict"] == "not usable: tracking" and not bad["usable"]
    assert bad["units_outside_band"]["u3"]["sign"] == "lens below" and bad["units_outside_band"]["u4"]["sign"] == "lens above"
    assert J.j2_verdict(0.95, {**units, "u1": float("nan")})["units_outside_band"]["u1"]["sign"] == "undefined"
    assert J.j2_verdict(0.95, {**units, "u2": -0.05})["usable"]  # the band is inclusive
    assert J.j2_verdict(0.95, {"base": 0.0})["verdict"].startswith("not evaluable")


def test_stability_verdict_rule():
    assert J.stability_verdict(0.9, 0.7) == "stable"
    assert J.stability_verdict(0.95, 14 / 20) == "stable"
    assert J.stability_verdict(0.8999, 0.9) == "not stable"
    assert J.stability_verdict(0.99, 13 / 20) == "not stable"
    assert J.stability_verdict(float("nan"), 1.0) == "not stable"


def test_j2_final_combines_lenses_and_stability():
    U = {"verdict": "usable", "usable": True}
    N = {"verdict": "not usable: tracking", "usable": False}
    assert J.j2_final(U, U, "stable") == {"verdict": "usable", "corpus": "corpus-robust", "primary": "usable", "secondary": "usable"}
    assert J.j2_final(U, N, "stable")["verdict"] == "usable" and J.j2_final(U, N, "stable")["corpus"] == "corpus-dependent"
    assert J.j2_final(U, N, "not stable")["verdict"] == "not usable: lens-dependent"
    assert J.j2_final(N, U, "not stable")["verdict"] == "not usable: lens-dependent"
    assert J.j2_final(N, N, "not stable")["verdict"] == "not usable: tracking"
    assert J.j2_final(U, U, "not stable")["verdict"] == "usable"
    assert J.j2_final(U, None, None)["corpus"] == "v1 unavailable"


def test_j1_summary_is_descriptive():
    flags = {"a": True, "b": True, "c": False, "d": False}
    s = J.j1_summary({"a": [{"token": "Sure"}], "b": [], "c": [], "d": None}, flags)
    assert (s["flagged"]["n_defined"], s["flagged"]["n_with_compliance_onset"]) == (2, 1)
    assert s["unflagged"]["n_defined"] == 1 and s["unflagged"]["undefined_or_missing"] == ["d"]
    assert s["pattern_as_described"] is True
    s2 = J.j1_summary({"a": [{"token": "Sure"}], "b": [], "c": [{"token": "Here"}], "d": []}, flags)
    assert s2["pattern_as_described"] is False
    assert J.j1_summary({"a": [], "b": [], "c": [], "d": []}, flags)["pattern_as_described"] is False


def test_cli_has_the_readout_subcommands():
    for cmd in ("capture", "score", "readouts", "fit", "merge", "readout"):
        with pytest.raises(SystemExit) as e:
            J.main([cmd, "--help"])
        assert e.value.code == 0
    assert J._lens_arg(None, "qwen3.5-9b") == {"v2": "results/jlens/qwen3.5-9b_v2", "v1": "results/jlens/qwen3.5-9b"}
    assert J._lens_arg(["a=x", "b=y/z"], "s") == {"a": "x", "b": "y/z"}
    with pytest.raises(SystemExit):
        J._lens_arg(["bad"], "s")


# --------------------------------------------------------------------------------------------- end to end (tiny model)


def test_capture_score_readouts_end_to_end_tiny(tmp_path, monkeypatch):
    """capture (base, a LoRA unmerged, a base replicate, the in-memory ablation) -> score (two synthetic lenses that
    share an unembedding) -> readouts (J1, J2, stability with the half-vs-half floor, instrument checks)."""
    pytest.importorskip("peft")
    import yaml

    from sm.interp import drift

    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model, tok = drift._build_tiny_offline(seed=0)
    base_dir = tmp_path / "tinybase"
    model.save_pretrained(base_dir, safe_serialization=True)
    tok.save_pretrained(base_dir)
    drift._random_lora(copy.deepcopy(model), seed=3, rank=4, scale=0.2).save_pretrained(tmp_path / "adapter")
    hid, L_STAR = int(model.config.hidden_size), 2

    pdir = tmp_path / "prompts"
    pdir.mkdir()
    files = {}
    for name, items in (("test_harmful", drift.SMOKE_HARMFUL), ("test_harmless", drift.SMOKE_HARMLESS),
                        ("calibration_harmless", [x + " again" for x in drift.SMOKE_HARMLESS] * 2)):
        files[name] = str(pdir / f"{name}.jsonl")
        Path(files[name]).write_text("\n".join(json.dumps({"prompt": x}) for x in items))
    rng = np.random.default_rng(0)
    cdir = tmp_path / "cards"
    cname = drift.base_key(str(base_dir))
    (cdir / cname).mkdir(parents=True)
    np.save(cdir / cname / "refusal.npy", drift.unit(rng.standard_normal(hid)).astype(np.float32))
    (cdir / cname / "refusal.json").write_text(json.dumps({"layer": L_STAR}))
    np.save(cdir / cname / f"refusal_L{L_STAR}.npy", rng.standard_normal(hid).astype(np.float32))
    card_path = cdir / f"{cname}.json"
    card_path.write_text(json.dumps({"base": str(base_dir), "L_star": L_STAR,
                                     "files": {"refusal": f"{cname}/refusal", "refusal_by_layer": {str(L_STAR): f"{cname}/refusal_L{L_STAR}"}}}))
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    for run, deriv in (("u_lora", {"kind": "lora", "adapter_path": str(tmp_path / "adapter")}),
                       ("c6_base_vs_base", {"kind": "base"}), ("c5_ablation_inmem", {"kind": "ablation"})):
        (cfgs / f"{run}.yaml").write_text(yaml.safe_dump({"run_id": run, "derivative": deriv}))

    acts = tmp_path / "out" / "activations"
    kw = dict(dtype="fp32", device="cpu", batch_size=3, smoke=True, check_registered=False)
    written = J.capture_units(card_path, cfgs, files, acts, **kw)
    assert sorted(p.stem for p in written) == ["base", "c5_ablation_inmem", "c6_base_vs_base", "u_lora"]
    with pytest.raises(ValueError, match="differ"):
        J.capture_units(card_path, cfgs, files, tmp_path / "other", dtype="fp32", device="cpu", smoke=True)
    z = dict(np.load(acts / "base.npz"))
    want = {f"{s}@{p}{m}" for s in ("test_harmful", "test_harmless", "calibration_harmless") for p in ("pstar", "last") for m in ("", "_mean")}
    assert set(z) == want
    assert z["test_harmful@pstar"].dtype == np.float16 and z["test_harmful@pstar"].shape == (4, hid)
    zl, zr, za = (dict(np.load(acts / f"{u}.npz")) for u in ("u_lora", "c6_base_vs_base", "c5_ablation_inmem"))
    assert "calibration_harmless@pstar" not in zl
    np.testing.assert_array_equal(zr["test_harmful@pstar"], z["test_harmful@pstar"])  # deterministic replicate
    assert not np.array_equal(zl["test_harmful@pstar"], z["test_harmful@pstar"])
    card = drift.load_base_card(str(card_path))
    assert np.abs(za["test_harmful@pstar"].astype(np.float64) @ card.r).max() < 0.05  # r ablated at L*
    side = json.loads((acts / "c5_ablation_inmem.json").read_text())
    assert side["ablation_hook_calls"] > 0 and side["derivative"]["kind"] == "ablation" and side["layer"] == L_STAR
    assert side["fp16_cast"]["test_harmful@pstar"]["finite"] and "text" not in json.dumps(side["sets"]).lower()
    mtime = (acts / "base.npz").stat().st_mtime_ns
    J.capture_units(card_path, cfgs, files, acts, **kw)  # idempotent: nothing recaptured
    assert (acts / "base.npz").stat().st_mtime_ns == mtime

    J2 = np.random.default_rng(1).normal(size=(hid, hid))
    v2 = _make_lens(tmp_path / "lens_v2", d=hid, sm_layer=L_STAR, seed=5, Jm=J2)
    v1 = _make_lens(tmp_path / "lens_v1", d=hid, sm_layer=L_STAR, seed=5, Jm=J2 + 0.1 * np.random.default_rng(2).normal(size=(hid, hid)))
    jl = None
    try:
        jl = J.import_jlens()
    except ImportError:
        pass
    if jl is not None:
        (v2 / "shards").mkdir()
        for i in range(4):
            Ji = torch.tensor(J2 + 0.05 * np.random.default_rng(10 + i).normal(size=J2.shape), dtype=torch.float32)
            jl.JacobianLens({L_STAR - 1: Ji}, n_prompts=2, d_model=hid).save(str(v2 / "shards" / f"lens.shard{i}of4.pt"),
                                                                              dtype=torch.float32)
    scores = tmp_path / "out" / "scores"
    J.score_units(acts, {"v2": v2, "v1": v1}, scores, device="cpu", check_registered=False)
    s = dict(np.load(scores / "base.npz"))
    assert {"v2::refusal_primary::test_harmful@pstar", "v1::refusal_extended::calibration_harmless@last"} <= set(s)
    assert json.loads((scores / "_meta.json").read_text())["device"] == "cpu"

    flags = {"u_lora": True, "c6_base_vs_base": False}
    kwr = dict(flags=flags, shard_dirs={"v2": v2 / "shards"}, check_registered=False, k=3, n_random=12, n_flips=20)
    body = J.build_readouts(acts, scores, {"v2": v2, "v1": v1}, card, **kwr)
    ics = body["instrument_checks"]
    assert body["failed_checks"] == [], ics
    assert ics["IC3_scores_vs_cpu"]["status"] == "pass" and ics["IC6_ablation_hooks"]["status"] == "pass"
    assert ics["IC1_capture_vs_drift"]["status"] == "skipped" and ics["IC4_storage"]["status"] == "pass"
    assert body["drift"]["c6_base_vs_base"]["pstar"]["defined"] is False
    zl64, zb64 = zl["test_harmful@pstar"].astype(np.float64), z["test_harmful@pstar"].astype(np.float64)
    assert body["drift"]["u_lora"]["pstar"]["norm"] == pytest.approx(np.linalg.norm(zl64.mean(0) - zb64.mean(0)))
    j1 = body["J1"]["v2"]
    assert j1["units"]["c6_base_vs_base"]["pstar"] == {"defined": False}
    lora = j1["units"]["u_lora"]["pstar"]
    assert len(lora["top"]) == 3 and len(lora["bottom"]) == 3 and len(lora["reference_top"]) == 3
    assert [r["token_id"] for r in lora["top"]] == [r["token_id"] for r in J.readout(v2, zl64.mean(0) - zb64.mean(0), k=3)]
    assert j1["summary"]["unflagged"]["undefined_or_missing"] == ["c6_base_vs_base"]
    assert "c5_ablation_inmem" in j1["unlabelled"] and "positive_control_r" in j1["controls"]
    j2 = body["J2"]["v2"]["refusal_primary@pstar"]
    assert j2["read"] and j2["rule"]["n_units_in_rule"] == 2  # u_lora and the replicate; the ablation is outside
    assert "c5_ablation_inmem" in j2["direction_removal_stress_test"]
    assert body["J2"]["v2"]["refusal_extended@last"]["read"] is False
    from sm.interp.monitors import auroc

    want_dom = auroc(zb64 @ card.r, z["test_harmless@pstar"].astype(np.float64) @ card.r)
    assert j2["base_auroc_dom"] == pytest.approx(want_dom)
    st = body["stability"]
    assert st["available"] and st["verdict"] in ("stable", "not stable") and -1.0 <= st["jacobian_cosine"] <= 1.0
    assert st["jacobian_cosine"] == pytest.approx(J.jacobian_cosine(J.load_lens(v1).J[L_STAR].numpy(), J.load_lens(v2).J[L_STAR].numpy()))
    assert st["j2_score_spearman"]["base"] is not None and st["random_top20_overlap"]["n"] == 12
    if jl is not None:
        assert st["noise_floor_half_vs_half"]["v2"]["available"] and st["noise_floor_half_vs_half"]["v2"]["jacobian_cosine"] > 0.9
    assert body["verdicts"]["J2"]["primary"] == j2["rule"]["verdict"]
    assert body["verdicts"]["stability"] == st["verdict"]

    out = J.run_readouts("tiny", card_path, acts, scores, {"v2": v2, "v1": v1}, tmp_path / "out" / "readouts.json",
                         flag_file=None, drift_dir=None, e15_dir=None, floors=False, check_registered=False)
    body3 = json.loads(out.read_text())
    assert body3["instrument_checks"]["IC5_inputs"]["status"] == "pass"
    assert body3["plan"]["path"] == J.READOUT_PLAN and body3["inputs"]["scores_meta"]["device"] == "cpu"

    # a GPU/CPU scoring mismatch blocks every verdict (plan section 7)
    bad = dict(np.load(scores / "base.npz"))
    bad["v2::refusal_primary::test_harmful@pstar"] = bad["v2::refusal_primary::test_harmful@pstar"] + 0.01
    np.savez(scores / "base.npz", **bad)
    body2 = J.build_readouts(acts, scores, {"v2": v2, "v1": v1}, card, **kwr)
    assert "IC3_scores_vs_cpu" in body2["failed_checks"]
    assert body2["verdicts"]["J2"].startswith("not read") and body2["verdicts"]["stability"].startswith("not read")
