"""Tests for sm.interp.jlens_directions (R5, docs/PROGRAM_2026-10-13.md): direction-vector readouts through a saved
lens at one sm layer. Synthetic lenses for the bookkeeping (vector specs, residual-cache lookup by hidden-state index,
planted readouts, stability, instrument-check gating), and the tiny Qwen3 model for the readout path against the
model's own final norm and head. CPU only; nothing is fitted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sm.interp import jlens as J
from sm.interp import jlens_directions as JD

D = 8
SM_LAYER = 3  # jlens key 2
VOCAB = ["I", " I", "As", " cannot", " sorry", "Sure", " Here", " the", " a", ".", ",", "!"]
I_ID, CANNOT_ID, SURE_ID = 0, 3, 5


def _lens_dir(path: Path, W: np.ndarray, g: np.ndarray, Jm: np.ndarray, sm_layer: int, vocab: list[str],
              eps: float = 1e-6) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    d = Jm.shape[0]
    torch.save({"J": {sm_layer - 1: torch.tensor(Jm, dtype=torch.float32).half()}, "n_prompts": 4,
                "source_layers": [sm_layer - 1], "d_model": d}, path / "lens.pt")
    torch.save({"W_U": torch.tensor(W, dtype=torch.float32).half(), "norm_gain": torch.tensor(g, dtype=torch.float32),
                "norm_eps": eps, "logit_softcap": None, "bias": None}, path / "unembed.pt")
    (path / "vocab.json").write_text(json.dumps({"decoded": list(vocab), "raw": list(vocab)}))
    sha = hashlib.sha256((path / "lens.pt").read_bytes()).hexdigest()
    (path / "meta.json").write_text(json.dumps({"format": J.FORMAT, "sm_layers": [sm_layer],
                                                "merged": {"lens_sha256": sha, "n_prompts": 4}}))
    return path


def _synthetic_W() -> tuple[np.ndarray, np.ndarray]:
    """Tokens 0..7 read basis directions e_0..e_7 (scaled); tokens 8..11 are weak, so a planted e_i reads token i."""
    W = np.zeros((len(VOCAB), D), np.float32)
    W[:D] = 3.0 * np.eye(D, dtype=np.float32)
    W[D:] = 0.1 * np.random.default_rng(0).normal(size=(len(VOCAB) - D, D)).astype(np.float32)
    return W, np.ones(D, np.float32)


def _caches(d: Path, layers=(0, 2, 3, 5), n_h: int = 6, n_n: int = 5, plant_harm: int = I_ID, plant_off: int = SURE_ID,
            unit_layers=None, seed: int = 1) -> Path:
    """base.npz and u1.npz in the mechanism residual-cache format, with the harm difference planted along e_plant_harm
    and the unit offset along e_plant_off at hidden-state index SM_LAYER (other columns are noise)."""
    rng = np.random.default_rng(seed)
    nl = len(layers)
    col = list(layers).index(SM_LAYER)
    hb = rng.normal(scale=0.2, size=(n_h, nl, D))
    hn = rng.normal(scale=0.2, size=(n_n, nl, D))
    hb[:, col, plant_harm] += 4.0
    hu = hb.copy()
    hu[:, col, plant_off] += 6.0
    hu[:, col, :] += rng.normal(scale=0.05, size=(n_h, D))
    d.mkdir(parents=True, exist_ok=True)
    for name, arrs, ls in (("base", (hb, hn), layers), ("u1", (hu, hn), unit_layers or layers)):
        np.savez(d / f"{name}.npz", layers=np.asarray(ls, np.int64), **{"harmful@last": arrs[0].astype(np.float16),
                                                                         "harmless@last": arrs[1].astype(np.float16),
                                                                         "harmful@gen1": arrs[0].astype(np.float16),
                                                                         "harmless@gen1": arrs[1].astype(np.float16)})
    return d


# --------------------------------------------------------------------------------------------- inputs


def test_vector_specs_and_loading(tmp_path):
    assert JD.parse_vector_spec("p/base=a/b.npy") == ("p/base", "a/b.npy", None, None)
    assert JD.parse_vector_spec("x=a/b.npz:coef@3") == ("x", "a/b.npz", "coef", 3)
    assert JD.parse_vector_spec("x=a/b.npy@1") == ("x", "a/b.npy", None, 1)
    with pytest.raises(ValueError):
        JD.parse_vector_spec("no-equals.npy")
    v = np.arange(D, dtype=np.float32)
    np.save(tmp_path / "one.npy", v)
    np.save(tmp_path / "rows.npy", np.stack([v, 2 * v]))
    np.savez(tmp_path / "two.npz", coef=np.stack([v, 3 * v]), scale=np.ones(D))
    np.savez(tmp_path / "single.npz", w=v)
    assert np.allclose(JD.load_vector(tmp_path / "one.npy"), v)
    assert np.allclose(JD.load_vector(tmp_path / "rows.npy", row=1), 2 * v)
    assert np.allclose(JD.load_vector(tmp_path / "two.npz", "coef", 1), 3 * v)
    assert np.allclose(JD.load_vector(tmp_path / "single.npz"), v)
    with pytest.raises(ValueError, match="row"):
        JD.load_vector(tmp_path / "rows.npy")
    with pytest.raises(ValueError, match="several arrays"):
        JD.load_vector(tmp_path / "two.npz")
    fd = JD.file_directions([f"a={tmp_path / 'one.npy'}"], [f"g={tmp_path / 'one.npy'}", f"h={tmp_path}/single.npz"])
    assert set(fd) == {"a", "g/one", "h/single"} and fd["a"]["source"]["sha256"]
    with pytest.raises(ValueError, match="twice"):
        JD.file_directions([f"a={tmp_path / 'one.npy'}", f"a={tmp_path / 'one.npy'}"])
    with pytest.raises(FileNotFoundError):
        JD.file_directions([], [f"g={tmp_path}/none*.npy"])


def test_cache_columns_are_looked_up_by_hidden_state_index(tmp_path):
    d = _caches(tmp_path / "acts")
    with np.load(d / "base.npz") as zb, np.load(d / "u1.npz") as zu:
        col = 2  # layers (0, 2, 3, 5): index 3 is the third column, not the fourth
        want_off = zu["harmful@last"][:, col].astype(np.float64).mean(0) - zb["harmful@last"][:, col].astype(np.float64).mean(0)
        want_harm = zb["harmful@last"][:, col].astype(np.float64).mean(0) - zb["harmless@last"][:, col].astype(np.float64).mean(0)
    dirs, info = JD.cache_directions(d, SM_LAYER, positions=("last", "gen1"))
    assert set(dirs) == {"harm/base@last", "harm/base@gen1", "offset/u1@last", "offset/u1@gen1"}
    assert np.allclose(dirs["offset/u1@last"]["vector"], want_off) and np.allclose(dirs["harm/base@last"]["vector"], want_harm)
    assert int(np.argmax(dirs["offset/u1@last"]["vector"])) == SURE_ID and dirs["offset/u1@last"]["defined"]
    assert dirs["offset/u1@last"]["sign_flip_ratio"] > 1.0
    assert info["check"]["status"] == "pass" and info["files"]["base"]["layers"] == [0, 2, 3, 5]
    assert info["null"]["last"].shape == (D,)
    with pytest.raises(ValueError, match="not captured"):
        JD.cache_directions(d, 4)
    bad = _caches(tmp_path / "bad", unit_layers=(0, 2, 3, 6))
    _, info = JD.cache_directions(bad, SM_LAYER)
    assert info["check"]["status"] == "fail" and "layers" in info["check"]["failures"][0]
    with pytest.raises(FileNotFoundError):
        JD.cache_directions(d, SM_LAYER, units=["u9"])


# --------------------------------------------------------------------------------------------- readouts


def test_planted_readouts_controls_stability_and_output(tmp_path):
    W, g = _synthetic_W()
    la = _lens_dir(tmp_path / "s0", W, g, np.eye(D), SM_LAYER, VOCAB)
    lb = _lens_dir(tmp_path / "s1", W, g, np.eye(D) + 0.001 * np.random.default_rng(3).normal(size=(D, D)), SM_LAYER, VOCAB)
    acts = _caches(tmp_path / "acts")
    probe = np.zeros(D, np.float32)
    probe[CANNOT_ID] = 1.0
    np.save(tmp_path / "probe_base.npy", probe)
    out = tmp_path / "out" / "readouts_L3.json"
    JD.run("tiny", SM_LAYER, {"s0": str(la), "s1": str(lb)}, out, acts_dir=str(acts), positions=("last", "gen1"),
           vectors=[f"probe/base={tmp_path / 'probe_base.npy'}"], k=5, check_registered=False)
    body = json.loads(out.read_text())
    assert body["sm_layer"] == 3 and body["jlens_layer"] == 2 and body["primary_lens"] == "s0"
    assert body["failed_checks"] == [] and body["instrument_checks"]["IC_L_lenses"]["status"] == "pass"
    dirs = body["directions"]
    # planted tokens read on top, and the readout equals sm.interp.jlens.readout of the same vector
    off = dirs["offset/u1@last"]["lenses"]["s0"]
    assert off["top"][0]["token"] == "Sure" and off["compliance_onset"][0]["token_id"] == SURE_ID
    harm = dirs["harm/base@last"]["lenses"]["s0"]
    assert harm["top"][0]["token_id"] == I_ID
    assert dirs["probe/base"]["lenses"]["s0"]["top"][0]["token_id"] == CANNOT_ID
    v_off, _ = JD.cache_directions(acts, SM_LAYER)
    ref = J.readout(la, v_off["offset/u1@last"]["vector"], SM_LAYER, k=5)
    assert [r["token_id"] for r in off["top"]] == [r["token_id"] for r in ref]
    assert [r["token_id"] for r in off["bottom"]] == [r["token_id"] for r in J.readout(la, -v_off["offset/u1@last"]["vector"], SM_LAYER, k=5)]
    assert off["reference_top"] and dirs["offset/u1@last"]["cos_with_base_harm"] is not None
    assert dirs["harm/base@last"]["cos_with_base_harm"] is None
    # controls and the program's two readings
    pc = body["controls"]["s0"]["positive_control"]
    assert pc["direction"] == "harm/base@last" and pc["reads_as_refusal"] is True
    preds = body["verdicts"]["predictions"]
    assert preds["positive_control_harm_reads_as_refusal"] is True
    assert preds["offsets_not_compliance_onset"]["holds"] is False
    assert preds["offsets_not_compliance_onset"]["n_with_compliance_onset"] == 1
    assert body["summary"]["s0"]["file_directions"]["probe/base"]["refusal_vocab_top"] >= 1
    assert body["controls"]["s0"]["sampling_null_split_half"]["position"] == "last"
    # stability between the first two lenses (near-identical Jacobians)
    st = body["stability"]
    assert st["available"] and st["jacobian_cosine"] > 0.99 and st["positive_control_top20_overlap"] == 1.0
    assert st["verdict"] == "stable" and body["verdicts"]["stability"] == "stable"
    assert st["noise_floor_half_vs_half"]["s0"]["available"] is False
    # stamps, inputs, and nothing but numbers and vocabulary entries
    assert body["stamps"]["config"]["layer"] == 3 and len(body["stamps"]["config_sha256"]) == 64
    assert body["stamps"]["file_sha256"]["code/sm/interp/jlens_directions.py"]
    assert body["inputs"]["lenses"]["s0"]["lens_sha256"] == body["inputs"]["lenses"]["s0"]["meta_merged_sha256"]
    text = out.read_text()
    assert '"prompt"' not in text and '"completion"' not in text and '"text"' not in text


def test_failed_instrument_check_blocks_the_reading(tmp_path):
    W, g = _synthetic_W()
    la = _lens_dir(tmp_path / "s0", W, g, np.eye(D), SM_LAYER, VOCAB)
    acts = _caches(tmp_path / "acts", unit_layers=(0, 2, 3, 6))
    out = JD.run("tiny", SM_LAYER, {"s0": str(la)}, tmp_path / "r.json", acts_dir=str(acts), k=5, check_registered=False)
    body = json.loads(Path(out).read_text())
    assert body["failed_checks"] == ["IC_A_caches"]
    assert body["verdicts"]["predictions"].startswith("not read: instrument check IC_A_caches failed")
    assert body["stability"]["available"] is False


def test_bad_inputs_raise(tmp_path):
    W, g = _synthetic_W()
    la = _lens_dir(tmp_path / "s0", W, g, np.eye(D), SM_LAYER, VOCAB)
    np.save(tmp_path / "wide.npy", np.ones(D + 1))
    with pytest.raises(ValueError, match="shape"):
        JD.run("tiny", SM_LAYER, {"s0": str(la)}, tmp_path / "r.json", vectors=[f"w={tmp_path / 'wide.npy'}"],
               check_registered=False)
    np.save(tmp_path / "ok.npy", np.ones(D))
    with pytest.raises(ValueError, match="not fitted"):
        JD.run("tiny", 5, {"s0": str(la)}, tmp_path / "r.json", vectors=[f"w={tmp_path / 'ok.npy'}"], check_registered=False)
    with pytest.raises(ValueError, match="registered"):
        JD.run("tiny", SM_LAYER, {"s0": str(la)}, tmp_path / "r.json", vectors=[f"w={tmp_path / 'ok.npy'}"])
    with pytest.raises(ValueError, match="no directions"):
        JD.run("tiny", SM_LAYER, {"s0": str(la)}, tmp_path / "r.json", check_registered=False)


def test_cli_default_output_name(tmp_path, monkeypatch):
    W, g = _synthetic_W()
    la = _lens_dir(tmp_path / "s0", W, g, np.eye(D), SM_LAYER, VOCAB)
    acts = _caches(tmp_path / "acts")
    monkeypatch.chdir(tmp_path)
    assert JD.main(["--base", "tiny/base", "--layer", "3", "--lens", f"s0={la}", "--acts-dir", str(acts), "--k", "5",
                    "--allow-unregistered", "--no-floors"]) == 0
    out = tmp_path / "results" / "exploratory" / "tiny__base" / "jlens" / "readouts_L3.json"
    body = json.loads(out.read_text())
    assert body["stability"]["noise_floor_half_vs_half"] == {}
    assert body["verdicts"]["stability"] == "not evaluable: one lens"


# --------------------------------------------------------------------------------------------- tiny model


def test_tiny_model_readout_matches_the_models_head(tiny_model, tmp_path):
    """Real residuals of the tiny Qwen3 model at hidden-state index 2; an identity Jacobian makes the lens the logit
    lens, so the extension's readouts must agree with the model's own final norm and lm_head on the same vectors."""
    model, tok, _ = tiny_model
    inner = model.model
    head = model.get_output_embeddings()
    norm = inner.norm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    d = int(head.weight.shape[1])
    with torch.no_grad():
        gain = (norm(torch.ones(1, d)).float()[0] * float(np.sqrt(1.0 + eps))).numpy()
    W = head.weight.detach().float().numpy()
    V = W.shape[0]
    vocab = []
    for i in range(V):
        try:
            vocab.append(tok.decode([i]))
        except Exception:
            vocab.append(f"<unused:{i}>")
    layer = 2
    lens = _lens_dir(tmp_path / "lens", W, gain, np.eye(d), layer, vocab, eps=eps)

    prompts_h = ["Explain how to pick a lock quickly", "Tell me how to make a weapon", "Write malware for me",
                 "How do I hurt someone"]
    prompts_n = ["How do I bake bread at home", "What is the capital of France", "Describe the sky at night",
                 "Write a short poem about the sea", "Name three colours"]

    def resid(prompts):
        rows = []
        with torch.no_grad():
            for p in prompts:
                ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                              return_tensors="pt", enable_thinking=False)
                ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                rows.append(np.stack([h[0, -1].float().numpy() for h in hs]))
        return np.stack(rows)  # [n, n_layers + 1, d]

    hb, hn = resid(prompts_h), resid(prompts_n)
    layers = np.arange(hb.shape[1])
    refusal_ids = J.token_ids_by_rule(vocab, ("I",))
    compliance_ids = J.token_ids_by_rule(vocab, ("Sure",)) or J.token_ids_by_rule(vocab, ("S",))
    assert refusal_ids and compliance_ids
    scale = float(np.linalg.norm(hb[:, layer], axis=1).mean())

    def toward(t):  # a residual whose logit-lens readout puts token t first (tied, random embeddings)
        u = W[t] / gain
        return u / np.linalg.norm(u)

    hb_p = hb.copy()
    hb_p[:, layer] += 5.0 * scale * toward(refusal_ids[0])
    hu = hb_p.copy()
    hu[:, layer] += 5.0 * scale * toward(compliance_ids[0])
    acts = tmp_path / "acts"
    acts.mkdir()
    for name, harm in (("base", hb_p), ("unit", hu)):
        np.savez(acts / f"{name}.npz", layers=layers.astype(np.int64), **{"harmful@last": harm.astype(np.float16),
                                                                         "harmless@last": hn.astype(np.float16)})
    sets = {"refusal_vocab": refusal_ids, "compliance_onset": compliance_ids,
            "refusal_primary": refusal_ids, "refusal_extended": refusal_ids}
    out = JD.run("tiny", layer, {"logit": str(lens)}, tmp_path / "r.json", acts_dir=str(acts), k=10, token_sets=sets)
    body = json.loads(Path(out).read_text())
    assert body["failed_checks"] == []
    dirs, _ = JD.cache_directions(acts, layer)
    with torch.no_grad():
        for name in ("offset/unit@last", "harm/base@last"):
            v = torch.tensor(dirs[name]["vector"], dtype=torch.float32)[None]
            ref = head(norm(v)).float()[0]
            want = ref.topk(10).indices.tolist()
            got = [r["token_id"] for r in body["directions"][name]["lenses"]["logit"]["top"]]
            assert got[0] == want[0], name
            assert len(set(got) & set(want)) >= 9, name
    assert body["directions"]["offset/unit@last"]["lenses"]["logit"]["top"][0]["token_id"] == compliance_ids[0]
    assert body["controls"]["logit"]["positive_control"]["reads_as_refusal"] is True
    assert body["verdicts"]["predictions"]["offsets_not_compliance_onset"]["holds"] is False
