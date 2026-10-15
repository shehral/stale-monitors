"""Exploratory E6 (all-cells staleness readout) in sm.interp.drift: tiny random model, CPU, offline."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

pytest.importorskip("torch")
pytest.importorskip("peft")
from sm.interp import drift  # noqa: E402

DEFAULT_KEYS = {
    "base", "run_id", "D1", "D2", "D3", "hurtado_gap_ratio", "probe_auroc_base", "probe_auroc_derivative",
    "probe_delta_auroc", "realigned_auroc", "persona", "adapter_projection", "monitor_auroc", "realigned",
    "persona_contrast_diagnostic", "organism", "monitors", "extra", "meta",
}
GRID_POSITIONS = (-3, -2, -1)


def _load_tiny(model_ref):
    from sm.interp import loading as LD

    pin = LD.resolve_pinned(model_ref, {}, smoke=True)
    model, tok, _ = LD.load_pinned(model_ref, pin, dtype="fp32", device="cpu")
    return model, tok


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("SM_OFFLINE", "1")
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.delenv("SM_TINY_MODEL", raising=False)
    work = tmp_path_factory.mktemp("cells")
    cfg = drift.smoke_setup(work / "setup", seed=0)
    cfg.pop("monitor_scores")
    cfg["persona_prompts"] = "construction"
    # a base_card-layout selection grid: raw construction difference of means per (position, layer)
    from sm.common.hooks import capture_eoi

    model, tok = _load_tiny(cfg["model_ref"])
    n_layers = int(model.config.num_hidden_layers)
    rh = [drift._render(tok, p) for p in drift.SMOKE_HARMFUL]
    rn = [drift._render(tok, p) for p in drift.SMOKE_HARMLESS]
    layers = list(range(1, n_layers + 1))
    eh = capture_eoi(model, tok, rh, len(GRID_POSITIONS), layers=layers, batch_size=8)
    en = capture_eoi(model, tok, rn, len(GRID_POSITIONS), layers=layers, batch_size=8)
    P = len(GRID_POSITIONS)
    grid = {f"P{-(j - P)}_L{l}": eh[l][:, j].astype(np.float64).mean(0) - en[l][:, j].astype(np.float64).mean(0)
            for l in layers for j in range(P)}
    sdir = Path(cfg["base_card"]).parent / "tiny" / "_partial" / "stats"
    sdir.mkdir(parents=True)
    np.savez(sdir / "grid.npz", **grid)
    sha = hashlib.sha256((sdir / "grid.npz").read_bytes()).hexdigest()
    (sdir / "stats.json").write_text(json.dumps({"schema": "sm.base_card.stats/2", "files": {"grid": {"file": "grid.npz", "sha256": sha}}}))
    yield {"cfg": cfg, "work": work, "n_layers": n_layers, "model": model, "tok": tok}
    mp.undo()


def test_capture_with_cells_matches_capture_position(smoke):
    model, tok = smoke["model"], smoke["tok"]
    prompts = [drift._render(tok, p) for p in drift.SMOKE_HARMFUL + drift.SMOKE_HARMLESS]
    layers = list(range(smoke["n_layers"] + 1))
    rng = np.random.default_rng(0)
    H = model.config.hidden_size
    cell_dirs = {(p, l): rng.standard_normal(H) for p in GRID_POSITIONS for l in range(1, smoke["n_layers"] + 1)}
    for pos in (-1, -2):
        ref = drift.get_capture_fn(pos)[0](model, tok, prompts, layers, batch_size=3)
        acts, cells = drift.capture_with_cells(model, tok, prompts, layers, pos, None, cell_dirs, batch_size=3, want_sums=True)
        for l in layers:
            assert np.array_equal(acts[l], ref[l])  # the primary readout is bitwise the default capture
    for j, p in enumerate(cells["positions"]):
        at_p = drift.get_capture_fn(p)[0](model, tok, prompts, layers, batch_size=3)
        for l in range(1, smoke["n_layers"] + 1):
            np.testing.assert_allclose(cells["proj"][l][:, j], at_p[l] @ drift.unit(cell_dirs[(p, l)]), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(cells["sum"][l][j], at_p[l].sum(0), rtol=1e-6, atol=1e-5)
    with pytest.raises(ValueError, match="eoi"):
        drift.capture_with_cells(model, tok, prompts[:1], layers, -1, [10 ** 6], cell_dirs)


def test_default_output_keys_unchanged(smoke, tmp_path):
    cfg = dict(smoke["cfg"], run_id="plain")
    out = drift.run_drift(cfg, tmp_path)
    res = json.loads(out.read_text())
    assert set(res) == DEFAULT_KEYS
    assert "cells" not in json.dumps(sorted(res["extra"]))
    assert not list(tmp_path.rglob("*_cells.json")) and not list(tmp_path.rglob("_base_cells_*"))


def test_all_cells_end_to_end_and_backfill(smoke, tmp_path):
    cfg = dict(smoke["cfg"], run_id="r1")
    # default run first: no cells; then ALL_CELLS back-fills the cells without touching the drift JSON
    out = drift.run_drift(cfg, tmp_path)
    before = out.read_bytes()
    assert drift.run_drift(dict(cfg, all_cells=True), tmp_path) == out
    assert out.read_bytes() == before
    cp = out.with_name("r1_cells.json")
    c = json.loads(cp.read_text())
    main = json.loads(before)
    nL = smoke["n_layers"]
    assert len(c["cells"]) == len(GRID_POSITIONS) * nL
    assert c["meta"]["drift_json_written_this_run"] is False and c["meta"]["adapter_path"] == cfg["derivative"]["adapter_path"]
    assert c["meta"]["n_prompts"]["probe_eval_harmful"] == len(drift.SMOKE_HARMFUL) and "git_commit" in c["meta"]
    assert c["eval_set"] == "probe_eval (held out)"
    for e in c["cells"].values():
        assert set(e) >= {"position", "layer", "auroc_base", "auroc_derivative", "delta_auroc", "d1_cosine", "card_cell"}
        assert e["d1_cosine_base"] == pytest.approx(1.0, abs=1e-6)  # grid fitted on the same construction prompts
    card = c["cells"][c["card_cell"]]
    assert card["card_cell"] and c["card_cell"] == f"last@L{main['extra']['layer_star']}"
    mon = main["monitors"]["refusal_direction"]
    assert card["auroc_base"] == pytest.approx(mon["auroc_base"]) and card["auroc_derivative"] == pytest.approx(mon["auroc_derivative"])
    assert card["recall_derivative_at_card_threshold"] == pytest.approx(main["realigned"]["refusal_direction"]["stale_recall"])
    assert card["recall_base_at_card_threshold"] == pytest.approx(main["realigned"]["refusal_direction"]["base_recall"])
    assert card["d1_cosine"] == pytest.approx(main["D1"], abs=1e-9)
    assert sum(e["card_cell"] for e in c["cells"].values()) == 1
    # already done: a second call is a no-op
    m = cp.stat().st_mtime_ns
    drift.run_drift(dict(cfg, all_cells=True), tmp_path)
    assert cp.stat().st_mtime_ns == m
    # a fresh all-cells run computes the cells inside the main pass; base vs base is zero staleness everywhere
    rb = drift.run_drift(dict(cfg, run_id="bb", all_cells=True, derivative={"kind": "base"}), tmp_path)
    assert set(json.loads(rb.read_text())) == DEFAULT_KEYS
    cb = json.loads(rb.with_name("bb_cells.json").read_text())
    assert cb["meta"]["drift_json_written_this_run"] is True
    for e in cb["cells"].values():
        assert e["delta_auroc"] == pytest.approx(0.0, abs=1e-12)
        assert e["d1_cosine"] == pytest.approx(1.0, abs=1e-6)
    # a tampered grid is refused
    g = Path(cfg["base_card"]).parent / "tiny" / "_partial" / "stats" / "grid.npz"
    sj = g.with_name("stats.json")
    doc = json.loads(sj.read_text())
    sj.write_text(json.dumps({**doc, "files": {"grid": {"sha256": "0" * 64}}}))
    try:
        with pytest.raises(ValueError, match="sha256"):
            drift.run_drift(dict(cfg, run_id="r2", all_cells=True), tmp_path)
    finally:
        sj.write_text(json.dumps(doc))
