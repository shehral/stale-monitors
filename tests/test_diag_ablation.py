"""Tests for sm.interp.diag_ablation (directional-ablation instrument check) on the tiny CPU model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SM_OFFLINE", "1")

from sm.interp import diag_ablation as DA  # noqa: E402
from sm.interp import directions as D  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tiny_model_dir):
    from sm.common.models import load_model

    model, tok, meta = load_model(tiny_model_dir, dtype="fp32", device="cpu", attn="eager")
    return model, tok


def _hidden(model, tok, prompts):
    """hidden_states 0..n-1 (index n is post-norm in transformers 5) and the left-pad offsets."""
    import torch

    from sm.common import models as M

    seqs = [M.encode_text(tok, p) for p in prompts]
    ids, mask, pos = M.left_pad(seqs, M.pad_token_id(tok), "cpu")
    with torch.inference_mode():
        res = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False)
    T = ids.shape[1]
    return [h.clone() for h in res.hidden_states[:-1]], [T - len(s) for s in seqs], res.logits[:, -1, :].clone()


PROMPTS = ["How do I bake bread?", "Explain the refusal direction in a language model residual stream please."]

# The v1 output schema exactly as job 10946276 wrote it (results/diag/gemma-4-12b-it_ablation.json); --suite v1 must keep it.
V1_TOP_KEYS = {"version", "base", "card", "card_selection", "direction", "prompts", "refusal_token_ids", "kl_filter",
               "settings", "architecture", "model_meta", "baseline", "residual_profile", "conditions", "readings",
               "n_layers", "hidden_size", "elapsed_s", "meta"}
V1_SETTINGS_KEYS = {"n_per_class", "max_new_tokens", "decoding", "n_random", "massive_k", "seed", "batch_size", "generate",
                    "coherence_heuristic", "instrument"}
V1_COND_KEYS = {"name", "description", "where", "zero", "layers", "mid_block", "vector_norm", "cos_with_card", "n_targets",
                "n_mid_block_targets", "calls_per_forward", "kl_harmless", "kl_harmful", "refusal_score_harmful",
                "gen_harmful", "gen_harmless"}


def _unit(dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def test_position_restricted_ablation_semantics(tiny):
    import torch

    model, tok = tiny
    rng = np.random.default_rng(0)
    u = rng.standard_normal(model.config.hidden_size)
    u /= np.linalg.norm(u)
    ut = torch.tensor(u, dtype=torch.float32)
    base, offs, base_logits = _hidden(model, tok, PROMPTS)
    assert offs[0] > 0  # the shorter prompt is left-padded, so "first token" is not column 0

    with DA.make_ablation(model, tok, u, where="skip_first") as hk:
        sk, _, _ = _hidden(model, tok, PROMPTS)
        assert hk.calls > 0 and hk.n_targets == len(hk._handles) - 1
    with DA.make_ablation(model, tok, u, where="first_only"):
        fo, _, _ = _hidden(model, tok, PROMPTS)
    with DA.make_ablation(model, tok, u, where="all"):
        al, _, _ = _hidden(model, tok, PROMPTS)
    for li in range(len(base)):
        for row, off in enumerate(offs):
            # skip_first: the first real token never sees an edit, and attends only to itself, so its whole
            # trajectory equals the base model's; every later token has no component along u
            assert torch.allclose(sk[li][row, off], base[li][row, off], atol=1e-5)
            assert float((sk[li][row, off + 1:] @ ut).abs().max()) < 1e-4
            # first_only: only the first token is projected; all positions: every token is
            assert abs(float(fo[li][row, off] @ ut)) < 1e-4
            assert float((al[li][row, off:] @ ut).abs().max()) < 1e-4
    # the later tokens differ from the base under first_only (they read the edited first token through attention)
    assert not torch.allclose(fo[2][1, offs[1] + 1:], base[2][1, offs[1] + 1:])


def test_zero_direction_is_identity_through_the_hooks(tiny):
    import torch

    model, tok = tiny
    u = np.ones(model.config.hidden_size)
    _, _, base_logits = _hidden(model, tok, PROMPTS)
    for where in ("all", "skip_first"):
        with DA.make_ablation(model, tok, u, where=where, zero=True) as hk:
            _, _, z_logits = _hidden(model, tok, PROMPTS)
        assert hk.calls > 0
        assert torch.equal(z_logits, base_logits)
    with pytest.raises(ValueError):
        DA.make_ablation(model, tok, u, where="bos")


def _write_inputs(tmp_path: Path, hidden: int, harmful: list[str], harmless: list[str], layer: int = 2):
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    for name, items in (("dev_harmful", harmful), ("dev_harmless", harmless)):
        (pdir / f"{name}.jsonl").write_text("".join(json.dumps({"prompt": p}) + "\n" for p in items))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "dtype: fp32\nbatch_size: 4\n"
        f"prompts:\n  dev: {{harmful: {pdir / 'dev_harmful.jsonl'}, harmless: {pdir / 'dev_harmless.jsonl'}}}\n"
    )
    cards = tmp_path / "cards"
    (cards / "tiny").mkdir(parents=True)
    raw = np.random.default_rng(3).standard_normal(hidden) * 2.0
    D.save_direction(cards / "tiny" / "refusal", raw, base="tiny", revision=None, layer=layer, n_pos=4, n_neg=4,
                     prompt_set_sha256="x", position="last",
                     extra={"kind": "refusal", "raw_norm": float(np.linalg.norm(raw)), "position_index": -1,
                            "selection_status": "not_evaluable"})
    card = {
        "base": "tiny", "revision": None, "model_meta": {"revision": None},
        "prompt_sets": {"dev": {"harmful": {"n": len(harmful), "sha256": D.prompt_set_sha256(harmful)},
                                "harmless": {"n": len(harmless), "sha256": D.prompt_set_sha256(harmless)}}},
        "refusal_direction": "not_evaluable",
        "refusal_direction_selection": {"status": "not_evaluable", "position": -1, "position_label": "last", "layer": layer,
                                        "selected": {"kl_harmless": 30.0}, "screening": {}},
        "config": {"batch_size": 4, "generate_batch_size": 4, "refusal_score_tokens": ["I"]},
    }
    card_path = cards / "tiny.json"
    card_path.write_text(json.dumps(card))
    return card_path, cfg, cards / "tiny" / "refusal"


HARMFUL = ["Describe a procedure step 1", "user request harmful-class 2", "Placeholder harmful request 3",
           "describe the answer 4", "harmful 5"]
HARMLESS = ["How do I bake bread?", "What is the capital of France?", "Write a short poem about the sea",
            "Answer the question clearly", "harmless 5"]


def test_cli_end_to_end(tiny_model_dir, tiny, tmp_path, capsys):
    model, _ = tiny
    card, cfg, direction = _write_inputs(tmp_path, model.config.hidden_size, HARMFUL, HARMLESS)
    out = tmp_path / "diag" / "tiny_ablation.json"
    argv = ["--base", tiny_model_dir, "--out", str(out), "--card", str(card), "--config", str(cfg), "--direction", str(direction),
            "--n", "4", "--max-new-tokens", "6", "--n-random", "2", "--massive-k", "3", "--dtype", "fp32", "--device", "cpu"]
    assert DA.main(argv) == 0
    doc = json.loads(out.read_text())
    conds = doc["conditions"]
    assert set(conds) == {"card_direction", "random_0", "random_1", "zero", "card_skip_first", "card_first_only",
                          "random_0_skip_first", "card_no_mid_block", "card_layer_only", "card_massive_zeroed"}
    assert conds["zero"]["kl_harmless"]["mean"] == 0.0 and conds["zero"]["kl_harmful"]["max"] == 0.0
    assert conds["card_direction"]["kl_harmless"]["mean"] > 0.0
    assert conds["card_skip_first"]["kl_harmless"]["mean"] != conds["card_direction"]["kl_harmless"]["mean"]
    raw_norm = doc["direction"]["raw_norm"]
    for k in ("random_0", "random_1"):
        assert conds[k]["vector_norm"] == pytest.approx(raw_norm, rel=1e-5) and abs(conds[k]["cos_with_card"]) < 0.9
    for c in conds.values():
        assert c["calls_per_forward"] == c["n_targets"]  # every hooked module fires once per forward pass
        for g in ("gen_harmful", "gen_harmless"):
            assert c[g]["n"] == 4 and 0.0 <= c[g]["incoherent_rate"] <= 1.0
        assert "samples" not in c["gen_harmful"] and "completions_sha256" in c["gen_harmful"]  # no harmful texts
    n = doc["n_layers"]
    assert conds["card_direction"]["n_targets"] == (n + 1) + n  # every index 0..n plus each block's attention write
    assert conds["card_no_mid_block"]["n_targets"] == n + 1 and conds["card_layer_only"]["n_targets"] == 1
    assert doc["prompts"]["harmful"]["matches_card"] is True and doc["prompts"]["harmless"]["n_used"] == 4
    prof = doc["residual_profile"]
    assert len(prof["per_layer"]) == n + 1 and len(prof["massive_dims"]) == 3
    assert doc["architecture"]["model_class"] and isinstance(doc["readings"], list)
    assert doc["meta"]["diagnostic_only"] is True
    # --suite v1 (the default) keeps the v1 output schema exactly
    assert set(doc) == V1_TOP_KEYS and doc["version"] == DA.VERSION
    assert set(doc["settings"]) == V1_SETTINGS_KEYS and doc["settings"]["max_new_tokens"] == 6
    assert all(set(c) == V1_COND_KEYS for c in conds.values())
    # idempotent
    assert DA.main(argv) == 0 and "already done" in capsys.readouterr().out


def test_cli_rejects_prompts_that_do_not_match_the_card(tiny_model_dir, tiny, tmp_path):
    model, _ = tiny
    card, cfg, direction = _write_inputs(tmp_path, model.config.hidden_size, HARMFUL, HARMLESS)
    doc = json.loads(card.read_text())
    doc["prompt_sets"]["dev"]["harmless"]["sha256"] = "0" * 64
    card.write_text(json.dumps(doc))
    out = tmp_path / "o.json"
    rc = DA.main(["--base", tiny_model_dir, "--out", str(out), "--card", str(card), "--config", str(cfg),
                  "--direction", str(direction), "--dtype", "fp32", "--device", "cpu", "--no-generate"])
    assert rc == 2 and not out.exists()
    assert DA.main(["--base", tiny_model_dir, "--out", str(out), "--card", str(tmp_path / "none.json")]) == 5


def test_readings_rules():
    def c(name, kl):
        return {"name": name, "kl_harmless": {"mean": kl}, "calls_per_forward": 3, "n_targets": 3}

    r = DA.readings({"card_direction": c("card_direction", 25.0), "random_0": c("random_0", 20.0),
                     "zero": c("zero", 0.0), "card_skip_first": c("card_skip_first", 0.02)})
    assert any(x.startswith("ANY DIRECTION") for x in r) and any(x.startswith("FIRST TOKEN") for x in r)
    r2 = DA.readings({"card_direction": c("card_direction", 25.0), "random_0": c("random_0", 0.01), "zero": c("zero", 0.5)})
    assert any(x.startswith("PLUMBING") for x in r2) and any(x.startswith("DIRECTION-SPECIFIC") for x in r2)
    assert DA._parse_cell("-1,24") == (-1, 24)
    with pytest.raises(ValueError):
        DA._parse_cell("1,24")


# --------------------------------------------------------------------------- v2 suite


def test_v2_layer_sets_are_the_preregistered_gemma_sets():
    sets = DA.v2_layer_sets(48, 24)
    assert [(t, r) for t, r, _ in sets] == [("L24", "single"), ("L23-25", "near1"), ("L20-28", "near4"), ("L13-23", "seg"),
                                          ("L24-35", "seg"), ("L36-48", "seg"), ("L0-12", "seg"), ("Leven", "even")]
    ls = {t: layers for t, _, layers in sets}
    assert ls["L24"] == [24] and ls["L23-25"] == [23, 24, 25] and ls["L20-28"] == list(range(20, 29))
    assert ls["L13-23"] == list(range(13, 24)) and ls["L24-35"] == list(range(24, 36))
    assert ls["L36-48"] == list(range(36, 49)) and ls["L0-12"] == list(range(13)) and ls["Leven"] == list(range(0, 49, 2))
    assert sorted(i for _, r, layers in sets if r == "seg" for i in layers) == list(range(49))  # segments partition 0..48
    assert DA.single_extra(48, 24) == [13, 40] and DA.single_extra(4, 2) == []
    assert DA.DEFAULT_MAX_NEW_TOKENS == {"v1": 64, "v2": 48}
    ns = DA.build_parser().parse_args(["--base", "b", "--out", "o"])
    assert ns.suite == "v1" and ns.max_new_tokens is None


def test_edit_hooks_reproduce_v1_bit_for_bit_and_alpha_scales_the_edit(tiny):
    import torch

    from sm.common import hooks

    model, tok = tiny
    u = _unit(model.config.hidden_size)
    ut = torch.tensor(u, dtype=torch.float32)
    for where in ("all", "skip_first", "first_only"):
        with DA.make_ablation(model, tok, u, where=where) as v1:
            h1, _, l1 = _hidden(model, tok, PROMPTS)
        with DA.EditHooks(model, tok, u, where=where, alpha=1.0) as hk:
            h2, _, l2 = _hidden(model, tok, PROMPTS)
        assert torch.equal(l1, l2) and all(torch.equal(a, b) for a, b in zip(h1, h2)), where
        assert hk.n_targets == v1.n_targets and hk.calls == v1.calls
    with hooks.DirectionHooks(model, u, mode="ablate"):
        _, _, l3 = _hidden(model, tok, PROMPTS)
    with DA.EditHooks(model, tok, u):
        _, _, l4 = _hidden(model, tok, PROMPTS)
    assert torch.equal(l3, l4)
    # alpha 0.5: the first hook (the embedding output) keeps exactly half of the u component
    base, offs, _ = _hidden(model, tok, PROMPTS)
    with DA.EditHooks(model, tok, u, alpha=0.5):
        half, _, _ = _hidden(model, tok, PROMPTS)
    for row, off in enumerate(offs):
        assert torch.allclose(half[0][row, off:] @ ut, 0.5 * (base[0][row, off:] @ ut), atol=1e-5)
    with pytest.raises(ValueError):
        DA.EditHooks(model, tok, u, where="bos")
    with pytest.raises(ValueError):
        DA.EditHooks(model, tok, u, rule="whitened")  # needs dinv


def test_whitened_edit_zeroes_the_whitened_coefficient_and_is_identity_when_d_orthogonal(tiny):
    import torch

    model, tok = tiny
    H = model.config.hidden_size
    u = _unit(H, 1)
    ut = torch.tensor(u, dtype=torch.float32)
    stats = DA.residual_dim_stats(model, tok, PROMPTS, 4)
    dinv, floors = DA.whiten_dinv(stats["mean_sq"])
    n = len(stats["mean_sq"]) - 1
    assert dinv.shape == (n + 1, H) and np.all(np.isfinite(dinv)) and np.all(floors > 0)
    from sm.common import models as M

    assert stats["n_tokens"] == sum(len(M.encode_text(tok, p)) for p in PROMPTS) and stats["n_prompts"] == len(PROMPTS)
    hk = DA.EditHooks(model, tok, u, rule="whitened", dinv=dinv)
    hk._cur = ("L1", 1)
    w = torch.tensor(u * dinv[1], dtype=torch.float32)
    h = torch.randn(2, 5, H, generator=torch.Generator().manual_seed(0))
    out = hk._edit(h)
    assert float((out @ w).abs().max()) <= 1e-5 * float((h @ w).abs().max())  # u^T D^-1 h' = 0
    diff = out - h
    assert torch.allclose(diff - (diff @ ut).unsqueeze(-1) * ut, torch.zeros_like(diff), atol=1e-5)  # moves along u only
    h_orth = h - ((h @ w) / float(w @ w)).unsqueeze(-1) * w  # u^T D^-1 h_orth = 0
    assert torch.allclose(hk._edit(h_orth), h_orth, atol=1e-5)
    # through the hooks: at every hooked index the whitened coefficient of the edited residual is 0
    base, offs, base_logits = _hidden(model, tok, PROMPTS)
    with DA.EditHooks(model, tok, u, rule="whitened", dinv=dinv):
        wh, _, wl_logits = _hidden(model, tok, PROMPTS)
    for li in range(len(wh)):
        wl = torch.tensor(u * dinv[li], dtype=torch.float32)
        for row, off in enumerate(offs):
            ref = float((base[li][row, off:] @ wl).abs().max())
            assert float((wh[li][row, off:] @ wl).abs().max()) <= 1e-4 * max(ref, 1.0)
    assert not torch.equal(wl_logits, base_logits)


def test_layerwise_mask_zeroes_each_index_top_dims_and_renormalises(tiny):
    import torch

    u = np.random.default_rng(1).standard_normal(6)
    ma = np.array([[0.0, 5.0, 1.0, 9.0, 0.5, 0.1], [7.0, 0.0, 0.2, 0.3, 0.1, 3.0]])
    vecs, info = DA.layerwise_masked(u, ma, 2)
    assert info["k"] == 2 and info["dims"] == [[3, 1], [0, 5]]
    un = u / np.linalg.norm(u)
    for li, dims in enumerate(info["dims"]):
        v = vecs[li]
        keep = [i for i in range(6) if i not in dims]
        assert np.all(v[dims] == 0.0) and abs(np.linalg.norm(v) - 1.0) < 1e-12
        assert np.allclose(v[keep], u[keep] / np.linalg.norm(u[keep]))  # same direction on the other dims
        assert info["weight_removed"][li] == pytest.approx(float(np.sum(un[dims] ** 2)))
    assert DA.layerwise_masked(u, ma, 50)[1]["k"] == 5  # capped at hidden - 1
    model, tok = tiny
    U = _unit(model.config.hidden_size, 2)
    stats = DA.residual_dim_stats(model, tok, PROMPTS, 4)
    per, pinfo = DA.layerwise_masked(U, stats["mean_abs"], 5)
    assert len(per) == len(stats["mean_abs"]) and all(np.all(per[li][d] == 0.0) for li, d in enumerate(pinfo["dims"]))
    with DA.EditHooks(model, tok, U, per_index=per) as hk:
        mh, offs, _ = _hidden(model, tok, PROMPTS)
    for li in range(len(mh)):
        pl = torch.tensor(per[li], dtype=torch.float32)
        for row, off in enumerate(offs):
            assert float((mh[li][row, off:] @ pl).abs().max()) < 1e-4  # each index projects out its own masked vector
    n = len(mh)
    assert hk.n_targets == (n + 1) + n
    with pytest.raises(ValueError):
        DA.EditHooks(model, tok, U, per_index={0: per[0]}).__enter__()  # a vector for every hooked index is required


def test_last_only_edits_only_the_last_position_on_prefill_and_each_decode_step(tiny):
    import torch

    from sm.common import models as M

    model, tok = tiny
    u = _unit(model.config.hidden_size, 3)
    ut = torch.tensor(u, dtype=torch.float32)
    base, offs, base_logits = _hidden(model, tok, PROMPTS)
    with DA.EditHooks(model, tok, u, where="last_only"):
        lo, _, lo_logits = _hidden(model, tok, PROMPTS)
    for li in range(len(base)):
        for row, off in enumerate(offs):
            # earlier real tokens never see the edit (causal attention); left-pad rows are fully masked queries that
            # attend uniformly to every position, the edited one included, so they are not compared
            assert torch.equal(lo[li][row, off:-1], base[li][row, off:-1])
        assert float((lo[li][:, -1] @ ut).abs().max()) < 1e-4  # the last prompt token is projected at every index
    assert not torch.equal(lo_logits, base_logits)
    one = torch.tensor([[M.encode_text(tok, PROMPTS[0])[-1]]])  # a one-token pass, as in a decode step

    def run_one():
        with torch.inference_mode():
            return model(input_ids=one, output_hidden_states=True, use_cache=False).hidden_states[0][0, -1]

    b1 = run_one()
    with DA.EditHooks(model, tok, u, where="last_only"):
        e1 = run_one()
    with DA.EditHooks(model, tok, u, where="first_only"):
        f1 = run_one()
    assert abs(float(e1 @ ut)) < 1e-5 and abs(float(b1 @ ut)) > 1e-6
    assert torch.equal(f1, b1)  # first_only never edits a decode step (v1 semantics)


def test_completion_nll_matches_a_direct_computation(tiny):
    import torch

    from sm.common import models as M

    model, tok = tiny
    comps = ["Mix flour and water.", ""]
    nll = DA.completion_nll(model, tok, PROMPTS, comps, 4)
    assert nll[1] is None and nll[0] is not None and nll[0] > 0
    p_ids, c_ids = M.encode_text(tok, PROMPTS[0]), M.encode_fragment(tok, comps[0])
    ids = torch.tensor([p_ids + c_ids])
    with torch.inference_mode():
        lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), dim=-1)
    want = -np.mean([float(lp[len(p_ids) + j - 1, t]) for j, t in enumerate(c_ids)])
    assert nll[0] == pytest.approx(want, rel=1e-4)
    # batching with a longer neighbour (left padding) gives the same value
    assert DA.completion_nll(model, tok, PROMPTS[::-1], ["a much longer completion than the other one", comps[0]], 4)[1] == \
        pytest.approx(nll[0], rel=1e-4)


def test_probe_trace_and_explicit_mid_block(tiny):
    import torch

    from sm.common import models as M

    model, tok = tiny
    u = _unit(model.config.hidden_size, 4)
    ut = torch.tensor(u, dtype=torch.float32)
    base, offs, base_logits = _hidden(model, tok, PROMPTS)
    with DA.EditHooks(model, tok, u, probe=True) as hk:
        hk.trace = "harmless"
        _, _, pl = _hidden(model, tok, PROMPTS)
        hk.trace = None
    assert torch.equal(pl, base_logits)  # a probe records c and edits nothing
    tr = hk.trace_summary()
    n = len(base)
    assert tr["modules"][:3] == ["L0", "A0", "L1"] and len(tr["modules"]) == hk.n_targets == 2 * n + 1
    assert tr["alpha"] == 0.0 and tr["probe"] is True and tr["harmful"] == [None] * len(tr["modules"])
    assert tr["n_tokens"]["harmless"] == sum(len(M.encode_text(tok, p)) for p in PROMPTS)
    want = np.mean(np.concatenate([np.abs((base[0][row, off:] @ ut).numpy()) for row, off in enumerate(offs)]))
    assert tr["harmless"][0] == pytest.approx(float(want), rel=1e-4)  # L0: mean |h_0 . u| over the real tokens
    with DA.EditHooks(model, tok, u, layers=[2], mid_block=False, mid_blocks=[2]) as hk2:
        hk2.trace = "harmful"
        _hidden(model, tok, PROMPTS)
        hk2.trace = None
    assert hk2.n_targets == 2 and hk2.n_mid_block_targets == 1
    t2 = hk2.trace_summary()
    assert t2["modules"] == ["L2", "A2"] and all(v is not None and v >= 0 for v in t2["harmful"])


def _rc(name, kl, *, group="v1", direction="card", role="all", n=97, family=None, control=None, inc=0.0, ref=0.0,
        alpha=1.0, zero=False, probe=False, trace=None, nll=0.5):
    c = {"name": name, "kl_harmless": {"mean": kl}, "kl_harmful": {"mean": kl}, "group": group, "direction_label": direction,
         "role": role, "n_targets": n, "calls_per_forward": n, "family": family, "control": control, "alpha": alpha,
         "zero": zero, "probe": probe, "refusal_score_harmful": -5.0}
    if not probe:
        c["gen_harmless"] = {"incoherent_rate": inc, "keyword_refusal_rate": 0.0, "base_nll_mean": nll}
        c["gen_harmful"] = {"incoherent_rate": inc, "keyword_refusal_rate": ref}
    if trace is not None:
        c["coef_trace"] = {"modules": ["L0", "A0", "L1", "A1", "L2"], "alpha": 0.0 if probe else alpha, "harmless": trace}
    return c


def _broken_suite():
    """A v1-like readout: everything at all hooks breaks the model, no amendment condition passes."""
    cs = [
        _rc("card_direction", 33.6, inc=0.5, trace=[1.0, 1.0, 2.0, 2.0, 3.0]),
        _rc("random_0", 24.0, direction="random_0", inc=0.3, ref=0.0, trace=[0.1, 0.1, 0.01, 0.01, 0.01]),
        _rc("zero", 0.0, direction="zero", role="zero", ref=1.0, zero=True),
        _rc("card_probe", 0.0, group="probe", role="probe", probe=True, trace=[1.0, 1.0, 2.0, 2.2, 3.1]),
        _rc("random_0_probe", 0.0, group="probe", direction="random_0", role="probe", probe=True,
            trace=[0.1, 0.1, 0.5, 0.5, 0.5]),
        _rc("card_layer_only", 1.48, role="single_v1", n=1),
        _rc("card_no_mid_block", 34.9, role="full_boundary", n=49, inc=0.8),
        _rc("random_0_no_mid_block", 20.0, group="H-A", direction="random_0", role="full_boundary", n=49, ref=0.0),
        _rc("card_L23-25", 5.0, group="H-A", role="near1", n=3, family="A", control="random_0_L23-25"),
        _rc("card_L20-28", 12.0, group="H-A", role="near4", n=9, family="A", control="random_0_L20-28"),
        _rc("card_L24_mid", 1.6, group="H-A", role="mid", n=2, family="A", control="random_0_L24"),
        _rc("card_Leven", 25.0, group="H-A", role="even", n=25),
        _rc("random_0_L23-25", 0.03, group="H-A", direction="random_0", role="near1", n=3, ref=1.0),
        _rc("random_0_L20-28", 0.2, group="H-A", direction="random_0", role="near4", n=9, ref=1.0),
        _rc("random_0_Leven", 9.0, group="H-A", direction="random_0", role="even", n=25, ref=1.0),
        _rc("card_L24", 1.48, group="H-A", role="single", n=1, family="A", control="random_0_L24"),
        _rc("random_0_L24", 0.01, group="H-A", direction="random_0", role="single", n=1, ref=1.0),
        _rc("card_L13", 0.3, group="H-B", role="single_extra", n=1, ref=1.0),
        _rc("card_L40", 2.0, group="H-B", role="single_extra", n=1),
        _rc("random_0_L13", 0.005, group="H-B", direction="random_0", role="single_extra", n=1, ref=1.0),
        _rc("random_0_L40", 0.02, group="H-B", direction="random_0", role="single_extra", n=1, ref=1.0),
        _rc("card_layerwise_masked_k5", 22.0, group="H-C", role="masked_k5", family="C",
            control="random_layerwise_masked_k50", inc=0.9),
        _rc("card_layerwise_masked_k50", 21.0, group="H-C", role="masked_k50", family="C",
            control="random_layerwise_masked_k50", inc=0.9),
        _rc("random_layerwise_masked_k50", 18.0, group="H-C", direction="random_0", role="masked_k50", nll=6.0),
        _rc("card_whitened", 25.0, group="H-C", role="whitened", family="B", control="random_whitened", inc=0.6),
        _rc("random_whitened", 20.0, group="H-C", direction="random_0", role="whitened"),
        _rc("card_L24_whitened", 0.9, group="H-C", role="single_whitened", n=1, family="A+B", control="random_0_L24"),
        _rc("card_L24_masked_k5", 1.2, group="H-C", role="single_masked_k5", n=1, family="A+C", control="random_0_L24"),
        _rc("card_alpha_0.25", 8.0, group="H-D", role="alpha", alpha=0.25),
        _rc("card_alpha_0.5", 20.0, group="H-D", role="alpha", alpha=0.5),
        _rc("card_last_only", 0.6, group="H-E", role="last_only", family="D", control="random_0_last_only"),
        _rc("random_0_last_only", 0.02, group="H-E", direction="random_0", role="last_only", ref=1.0),
    ]
    for tag, n, k_card, k_rand in (("L13-23", 11, 15.0, 6.0), ("L24-35", 12, 18.0, 7.0), ("L36-48", 13, 14.0, 5.0),
                                   ("L0-12", 13, 16.0, 6.5)):
        cs.append(_rc(f"card_{tag}", k_card, group="H-A", role="seg", n=n))
        cs.append(_rc(f"random_0_{tag}", k_rand, group="H-A", direction="random_0", role="seg", n=n, ref=1.0))
    return {c["name"]: c for c in cs}


BASELINE = {"gen_harmful": {"keyword_refusal_rate": 1.0}, "gen_harmless": {"base_nll_mean": 0.4}}


def test_readings_v2_rules_on_synthetic_readouts():
    def has(rs, prefix, *parts):
        return any(r.startswith(prefix) and all(p in r for p in parts) for r in rs)

    conds = _broken_suite()
    r = DA.readings_v2(conds, BASELINE)
    assert has(r, "PLUMBING OK", "reproduces card_layer_only")
    assert has(r, "H-A ACCUMULATION (card)") and has(r, "H-A ACCUMULATION (random_0)")
    assert has(r, "H-B DIRECTION-SPECIFIC", "survives whitening and masking")
    assert has(r, "H-C NOT SUFFICIENT") and has(r, "H-D GRADED")
    assert has(r, "TRACE FRESH RE-WRITE (card)") and has(r, "TRACE PERSISTENT REMOVAL (random_0)") and has(r, "TRACE MASS")
    assert has(r, "NO CANDIDATE", "outcome (E)") and not has(r, "CANDIDATES")
    assert has(r, "COHERENCE HEURISTIC MISSES: 1 ", "random_layerwise_masked_k50")

    # massive dimensions rescue the random direction; whitened card and response positions pass with refusal-keeping controls
    c2 = _broken_suite()
    c2["random_whitened"] |= {"kl_harmless": {"mean": 0.03}, "gen_harmful": {"incoherent_rate": 0.0, "keyword_refusal_rate": 1.0}}
    c2["card_whitened"] |= {"kl_harmless": {"mean": 0.05}, "gen_harmless": {"incoherent_rate": 0.0, "base_nll_mean": 0.6},
                            "gen_harmful": {"incoherent_rate": 0.0, "keyword_refusal_rate": 0.0}}
    c2["card_last_only"]["kl_harmless"] = {"mean": 0.04}
    c2["card_L24_whitened"]["kl_harmless"] = {"mean": 0.05}
    r2 = DA.readings_v2(c2, BASELINE)
    assert has(r2, "H-C SUPPORTED") and has(r2, "H-B DIRECTION-SPECIFIC", "massive dimensions, not refusal content")
    cand = next(x for x in r2 if x.startswith("CANDIDATES"))
    assert cand.index("(D) card_last_only") < cand.index("(B) card_whitened") < cand.index("(A+B) card_L24_whitened")
    assert not has(r2, "NO CANDIDATE")

    # a control that also loses refusal disqualifies the candidate
    c3 = _broken_suite()
    c3["card_last_only"]["kl_harmless"] = {"mean": 0.04}
    c3["random_0_last_only"]["gen_harmful"]["keyword_refusal_rate"] = 0.0
    r3 = DA.readings_v2(c3, BASELINE)
    assert has(r3, "NO CANDIDATE") and not has(r3, "CANDIDATES")

    # low KL and a clean heuristic, but token salad by the fluency screen: not a candidate
    c5 = _broken_suite()
    c5["card_last_only"] |= {"kl_harmless": {"mean": 0.04}}
    c5["card_last_only"]["gen_harmless"]["base_nll_mean"] = 4.0
    r5 = DA.readings_v2(c5, BASELINE)
    assert has(r5, "NO CANDIDATE") and has(r5, "COHERENCE HEURISTIC MISSES", "card_last_only")
    ok, why = DA._candidate_check(c5["card_last_only"], c5["random_0_last_only"], 1.0, 0.4)
    assert not ok and "base NLL 4" in why
    ok, why = DA._candidate_check(c2["card_whitened"], c2["random_whitened"], 1.0, None)
    assert not ok and "base NLL" in why  # no fluency reference: never a candidate

    # plumbing, nondeterminism, a localised segment, a threshold-like alpha and passing partial removal
    c4 = _broken_suite()
    c4["zero"]["kl_harmless"] = {"mean": 0.3}
    c4["card_L24"]["kl_harmless"] = {"mean": 1.5}
    c4["card_L24-35"]["kl_harmless"] = {"mean": 60.0}
    c4["card_alpha_0.5"]["kl_harmless"] = {"mean": 2.0}
    c4["card_alpha_0.25"]["kl_harmless"] = {"mean": 0.05}
    r4 = DA.readings_v2(c4, BASELINE)
    assert has(r4, "PLUMBING:", "zero") and has(r4, "NONDETERMINISM") and not has(r4, "PLUMBING OK")
    assert has(r4, "H-A LOCALISED (card)", "L24-35") and has(r4, "H-D THRESHOLD-LIKE") and has(r4, "H-D PARTIAL REMOVAL PASSES")
    assert DA._spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0) and DA._spearman([1, 2], [2, 1]) is None
    assert list(DA._rank(np.array([3.0, 1.0, 3.0]))) == [1.5, 0.0, 1.5]


def test_cli_v2_end_to_end_and_v1_anchors_match_v1(tiny_model_dir, tiny, tmp_path, monkeypatch, capsys):
    model, _ = tiny
    monkeypatch.setattr(DA, "SINGLE_EXTRA", (1, 3))  # H-B indices inside the tiny model's 0..4
    card, cfg, direction = _write_inputs(tmp_path, model.config.hidden_size, HARMFUL, HARMLESS)
    common = ["--base", tiny_model_dir, "--card", str(card), "--config", str(cfg), "--direction", str(direction), "--n", "4",
              "--n-random", "2", "--massive-k", "3", "--dtype", "fp32", "--device", "cpu"]
    out = tmp_path / "diag" / "tiny_ablation_v2.json"
    argv = ["--suite", "v2", "--out", str(out), "--max-new-tokens", "4", *common]
    assert DA.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["suite"] == "v2" and doc["version"] == DA.VERSION_V2 and len(doc["code_sha256"]) == 64
    assert set(doc) == V1_TOP_KEYS | {"suite", "code_sha256", "residual_dim_stats", "readings_v2", "v2_groups"}
    n, L = doc["n_layers"], doc["direction"]["layer"]
    conds = doc["conditions"]
    sets = DA.v2_layer_sets(n, L)
    want = {"card_direction", "random_0", "random_1", "zero", "card_skip_first", "card_first_only", "random_0_skip_first",
            "card_no_mid_block", "card_layer_only", "card_massive_zeroed", "card_probe", "random_0_probe",
            "random_0_no_mid_block", f"card_L{L}_mid", "card_layerwise_masked_k5", "card_layerwise_masked_k50",
            "random_layerwise_masked_k50", "card_whitened", "random_whitened", f"card_L{L}_whitened", f"card_L{L}_masked_k5",
            "card_alpha_0.25", "card_alpha_0.5", "card_last_only", "random_0_last_only"}
    want |= {f"{d}_{t}" for t, _, _ in sets for d in ("card", "random_0")}
    want |= {f"{d}_L{i}" for i in (1, 3) for d in ("card", "random_0")}
    assert set(conds) == want
    for c in conds.values():
        assert c["calls_per_forward"] == c["n_targets"], c["name"]
        tr = c["coef_trace"]
        assert tr["modules"] == sorted(tr["modules"], key=DA._label_order) and len(tr["modules"]) == c["n_targets"]
        assert all(v is not None for v in tr["harmless"] + tr["harmful"]), c["name"]
        if c["probe"] or c["zero"]:
            assert c["kl_harmless"]["max"] == 0.0 and c["kl_harmful"]["max"] == 0.0
        if c["probe"]:
            assert "gen_harmful" not in c
        else:
            assert c["gen_harmful"]["n"] == 4 and "samples" not in c["gen_harmful"] and "completions_sha256" in c["gen_harmful"]
            assert {"base_nll_mean", "base_nll_median", "n_base_nll"} <= set(c["gen_harmless"]) and len(c["gen_harmless"]["samples"]) <= 3
    assert doc["baseline"]["gen_harmless"]["base_nll_mean"] is not None and doc["settings"]["reading_thresholds"]["nll_margin"] == 1.0
    assert conds[f"card_L{L}"]["kl_harmless"] == conds["card_layer_only"]["kl_harmless"]
    assert conds[f"card_L{L}_mid"]["n_targets"] == 2 and conds["card_direction"]["n_targets"] == 2 * n + 1
    assert conds["card_last_only"]["coef_trace"]["n_tokens"] == {"harmless": 4, "harmful": 4}  # one last token per prompt
    assert conds["card_whitened"]["kl_harmless"]["mean"] > 0 and conds["card_alpha_0.5"]["coef_trace"]["alpha"] == 0.5
    assert doc["settings"]["suite"] == "v2" and doc["settings"]["max_new_tokens"] == 4
    assert doc["v2_groups"]["H-E"] == ["card_last_only", "random_0_last_only"]
    rs = doc["residual_dim_stats"]
    assert len(rs["per_index"]) == n + 1 and len(rs["masks"]["k5_dims"]) == n + 1 and len(rs["masks"]["k50_dims"][0]) == 50
    assert any(x.startswith("PLUMBING OK") for x in doc["readings_v2"])
    assert any(x.startswith(("CANDIDATES", "NO CANDIDATE")) for x in doc["readings_v2"])
    # the v1 anchors in v2 equal a v1 run on the same inputs
    out1 = tmp_path / "diag" / "tiny_ablation.json"
    assert DA.main(["--out", str(out1), "--no-generate", *common]) == 0
    v1 = json.loads(out1.read_text())["conditions"]
    for name, c in v1.items():
        assert conds[name]["kl_harmless"] == c["kl_harmless"] and conds[name]["kl_harmful"] == c["kl_harmful"], name
    # idempotent, and the v2 suite refuses to run without a random control direction
    assert DA.main(argv) == 0 and "already done" in capsys.readouterr().out
    assert DA.main(["--suite", "v2", "--out", str(tmp_path / "x.json"), "--base", tiny_model_dir, "--n-random", "0"]) == 2

