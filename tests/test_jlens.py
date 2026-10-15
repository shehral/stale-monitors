"""Tests for sm.interp.jlens: readouts against a synthetic tiny lens (no fitting anywhere), the saved-lens
format round trip through the official jlens code, shard merge, and the unembed export on the tiny model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sm.interp import jlens as J

try:
    J.import_jlens()
except ImportError as exc:  # third_party/ is gitignored: clone it per docs/JLENS_PLAN.md section 1
    pytest.skip(f"official jacobian-lens code not available: {exc}", allow_module_level=True)

D, V = 8, 12
SM_LAYER = 3  # jlens layer 2
VOCAB = ["I", " I", "As", " cannot", " sorry", "Sure", " Here", " the", " a", ".", ",", "!"]


def _np_logits(Jm, W, g, eps, h, bias=None):
    t = Jm @ h
    n = t / np.sqrt(np.mean(t ** 2) + eps) * g
    out = W @ n
    return out if bias is None else out + bias


@pytest.fixture()
def synth(tmp_path):
    jl = J.import_jlens()
    rng = np.random.default_rng(0)
    Jm = rng.normal(size=(D, D)).astype(np.float32)
    W = rng.normal(size=(V, D)).astype(np.float32)
    g = (1.0 + 0.1 * rng.normal(size=D)).astype(np.float32)
    lens = jl.JacobianLens({SM_LAYER - 1: torch.tensor(Jm)}, n_prompts=5, d_model=D)
    lens.save(str(tmp_path / "lens.pt"), dtype=torch.float32)
    torch.save({"W_U": torch.tensor(W).half(), "norm_gain": torch.tensor(g), "norm_eps": 1e-6,
                "logit_softcap": None, "bias": None}, tmp_path / "unembed.pt")
    (tmp_path / "vocab.json").write_text(json.dumps({"decoded": VOCAB, "raw": VOCAB}))
    (tmp_path / "meta.json").write_text(json.dumps({"format": J.FORMAT}))
    W16 = torch.tensor(W).half().float().numpy()
    return tmp_path, Jm, W16, g, rng


def test_layer_convention():
    assert J.sm_to_jlens_layer(13) == 12 and J.jlens_to_sm_layer(12) == 13
    with pytest.raises(ValueError):
        J.sm_to_jlens_layer(0)


def test_readout_matches_reference_math(synth):
    d, Jm, W, g, rng = synth
    h = rng.normal(size=D).astype(np.float32)
    want = _np_logits(Jm, W, g, 1e-6, h)
    got = J.readout(d, h, layer=SM_LAYER, k=5)
    assert [r["token_id"] for r in got] == list(np.argsort(-want)[:5])
    assert [r["rank"] for r in got] == [1, 2, 3, 4, 5]
    np.testing.assert_allclose([r["score"] for r in got], np.sort(want)[::-1][:5], rtol=1e-4, atol=1e-4)
    assert got[0]["token"] == VOCAB[got[0]["token_id"]]
    # single fitted layer: layer may be omitted; the lens.pt path works as well as the directory
    assert J.readout(d / "lens.pt", h, k=5) == got


def test_readout_scale_free_and_reference_mode(synth):
    d, Jm, W, g, rng = synth
    h = rng.normal(size=D).astype(np.float32)
    ids = [r["token_id"] for r in J.readout(d, h, k=V)]
    assert [r["token_id"] for r in J.readout(d, 7.5 * h, k=V)] == ids
    base, der = rng.normal(size=D).astype(np.float32), rng.normal(size=D).astype(np.float32)
    want = _np_logits(Jm, W, g, 1e-6, der) - _np_logits(Jm, W, g, 1e-6, base)
    got = J.readout(d, der, k=3, reference=base)
    assert [r["token_id"] for r in got] == list(np.argsort(-want)[:3])


def test_readout_rejects_bad_inputs(synth):
    d, *_ = synth
    with pytest.raises(ValueError):
        J.readout(d, np.zeros(D), layer=SM_LAYER + 1)
    with pytest.raises(ValueError):
        J.readout(d, np.zeros(D + 1))
    with pytest.raises(ValueError):
        J.readout(d, np.zeros((2, D)))


def test_refusal_mass_matches_softmax(synth):
    d, Jm, W, g, rng = synth
    H = rng.normal(size=(6, D)).astype(np.float32)
    ids = [0, 1, 2, 3, 4]
    want = []
    for h in H:
        z = _np_logits(Jm, W, g, 1e-6, h).astype(np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        want.append(p[ids].sum())
    by_str = J.refusal_mass(d, H, ["I", " I", "As", " cannot", " sorry"], layer=SM_LAYER)
    by_id = J.refusal_mass(d, H, ids)
    np.testing.assert_allclose(by_str, want, rtol=1e-4)
    np.testing.assert_allclose(by_id, by_str, rtol=1e-6)
    assert by_str.shape == (6,) and np.all((by_str >= 0) & (by_str <= 1))
    np.testing.assert_allclose(J.refusal_mass(d, H, list(range(V))), 1.0, rtol=1e-5)
    assert J.refusal_mass(d, H[0], ids).shape == (1,)


def test_token_set_errors(synth):
    d, *_ = synth
    with pytest.raises(ValueError, match="not in the vocabulary"):
        J.refusal_mass(d, np.zeros((1, D)), ["I", "Nope"])
    with pytest.raises(ValueError):
        J.refusal_mass(d, np.zeros((1, D)), [V + 3])
    with pytest.raises(ValueError):
        J.refusal_mass(d, np.zeros((1, D)), [])


def test_missing_files_and_checkpoint_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        J.load_lens(tmp_path)
    torch.save({"jacobian_sum": {}, "n_done": 0}, tmp_path / "lens.pt")
    torch.save({"W_U": torch.zeros(V, D), "norm_gain": torch.ones(D), "norm_eps": 1e-6}, tmp_path / "unembed.pt")
    with pytest.raises(ValueError, match="not a JacobianLens"):
        J.load_lens(tmp_path)


def test_merge_uses_official_weighted_mean(tmp_path):
    jl = J.import_jlens()
    (tmp_path / "shards").mkdir()
    a, b = torch.randn(D, D), torch.randn(D, D)
    jl.JacobianLens({2: a}, n_prompts=1, d_model=D).save(str(tmp_path / "shards/lens.shard0of2.pt"), dtype=torch.float32)
    jl.JacobianLens({2: b}, n_prompts=3, d_model=D).save(str(tmp_path / "shards/lens.shard1of2.pt"), dtype=torch.float32)
    J.merge(tmp_path)
    m = jl.JacobianLens.load(str(tmp_path / "lens.pt"))
    assert m.n_prompts == 4
    torch.testing.assert_close(m.jacobians[2], (a + 3 * b) / 4, rtol=1e-3, atol=1e-3)
    (tmp_path / "shards/lens.shard1of2.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        J.merge(tmp_path)


def test_read_prompts_formats(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('"plain"\n{"text": "t"}\n\n{"prompt": "q"}\n')
    assert J.read_prompts(p) == ["plain", "t", "q"]
    assert J.read_prompts(p, n=2) == ["plain", "t"]
    q = tmp_path / "p.json"
    q.write_text(json.dumps(["x", "y"]))
    assert J.read_prompts(q) == ["x", "y"]
    p.write_text('{"other": 1}\n')
    with pytest.raises(ValueError):
        J.read_prompts(p)


def test_fit_refuses_without_cuda(tmp_path, monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    monkeypatch.delenv("SM_JLENS_ALLOW_CPU", raising=False)
    with pytest.raises(RuntimeError, match="cluster job"):
        J.fit("tiny", 1, tmp_path / "none.jsonl", tmp_path / "out")


def test_export_unembed_reproduces_model_unembed(tiny_model_dir, tmp_path):
    """Forward-only on the tiny random model: the saved bundle must reproduce HFLensModel.unembed."""
    from sm.common.models import load_model

    jl = J.import_jlens()
    model, tok, _ = load_model(tiny_model_dir, dtype="fp32", device="cpu", attn="eager")
    lm = jl.from_hf(model, tok)
    info = J.export_unembed(lm, tok, tmp_path)
    assert info["unembed_check_top1_agreement"] == 1.0 and info["unembed_check_rel_err"] < 0.05
    d = lm.d_model
    jl.JacobianLens({0: torch.eye(d)}, n_prompts=1, d_model=d).save(str(tmp_path / "lens.pt"), dtype=torch.float32)
    h = torch.randn(3, d, generator=torch.Generator().manual_seed(1))
    lens = J.load_lens(tmp_path)
    ref = lm.unembed(h).float()
    got = lens.logits(h.numpy(), layer=1)
    assert torch.equal(got.argmax(-1), ref.argmax(-1))
    assert float((got - ref).abs().max()) <= 0.05 * float(ref.abs().max()) + 1e-6
    vocab = json.loads((tmp_path / "vocab.json").read_text())
    assert len(vocab["decoded"]) == lens.W_U.shape[0] and not math.isnan(info["norm_eps"])
