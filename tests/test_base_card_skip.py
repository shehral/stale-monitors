"""The base card's "already done" skip (2026-10-08 fix): keyed on a hash of the inputs stored in the card meta,
and never taken for a card that results/ledger.jsonl records as REJECTED (job 10902988 had to be resubmitted with
FORCE=1). No model is loaded: the build is replaced by a stub that records the call.

Run: .venv/bin/python -m pytest -q tests/test_base_card_skip.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if importlib.util.find_spec("sm") is None and str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sm.interp import base_card as BC  # noqa: E402

BASE = "x"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    """A project root with results/base_cards, a config, a prompt manifest and a stubbed build."""
    root = tmp_path / "proj"
    out = root / "results" / "base_cards"
    out.mkdir(parents=True)
    (root / "data" / "prompts").mkdir(parents=True)
    (root / "data" / "prompts" / "MANIFEST.json").write_text('{"v": 1}')
    cfg = root / "cfg.yaml"
    cfg.write_text("seed: 0\n")
    (root / "results" / "ledger.jsonl").write_text("")
    monkeypatch.chdir(root)
    monkeypatch.delenv("SM_PROJ", raising=False)
    calls: list[str] = []

    def fake_run(args, cfg_, base, model_ref, pin, out_, final, skip_key=None):
        calls.append(str(final))
        # a build counter keeps rebuilt cards byte-distinct (the meta timestamp has one-second resolution)
        BC._write_json(final, {"L_star": 3, "build": len(calls)}, extra_meta={"skip_inputs": skip_key})
        return 0

    monkeypatch.setattr(BC, "_run", fake_run)
    monkeypatch.setattr(BC.LD, "resolve_pinned", lambda *a, **k: {"revision": "r" * 40, "hf": "x", "key": None})
    return root, out, cfg, calls


def _main(out: Path, cfg: Path, *extra: str) -> int:
    return BC.main(["--base", BASE, "--config", str(cfg), "--out", str(out), *extra])


def _row(card: Path, outcome: str, job: str, output_hash: str | None = None, stage: str = "base_card") -> dict:
    return {"accepted": outcome != "REJECT", "base": BASE, "derivative": "base", "stage": stage, "slurm_job_id": job,
            "note": f"outcome={outcome}; {stage} {BASE}/base", "output_hash": output_hash or _sha(card)}


def _ledger(root: Path, *rows: dict) -> None:
    (root / "results" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_card_records_the_input_hash_and_unchanged_inputs_skip(proj, capsys):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0 and len(calls) == 1
    meta = json.loads((out / "x.json").read_text())["meta"]["skip_inputs"]
    assert meta["config_sha256"] == _sha(cfg)
    assert meta["prompt_manifest_sha256"] == _sha(root / "data" / "prompts" / "MANIFEST.json")
    assert meta["revision"] == "r" * 40 and "git_commit" in meta and "sm_version" in meta and len(meta["sha256"]) == 64
    capsys.readouterr()
    assert _main(out, cfg) == 0 and len(calls) == 1
    assert "already done" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["config", "manifest", "revision", "legacy"])
def test_changed_or_unrecorded_inputs_recompute(proj, monkeypatch, change):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    if change == "config":
        cfg.write_text("seed: 1\n")
    elif change == "manifest":
        (root / "data" / "prompts" / "MANIFEST.json").write_text('{"v": 2}')
    elif change == "revision":
        monkeypatch.setattr(BC.LD, "resolve_pinned", lambda *a, **k: {"revision": "s" * 40})
    else:
        (out / "x.json").write_text(json.dumps({"L_star": 3, "meta": {}}))  # a card from before the fix
    assert _main(out, cfg) == 0 and len(calls) == 2


def test_rejected_card_is_never_skipped(proj, capsys):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    card = out / "x.json"
    _ledger(root, _row(card, "REJECT", "111"))
    capsys.readouterr()
    assert _main(out, cfg) == 0 and len(calls) == 2              # same inputs, but REJECTED: rebuilt
    assert "already done" not in capsys.readouterr().out
    # the REJECT row judged the old file; the rebuilt card (new bytes) is skipped again
    assert _main(out, cfg) == 0 and len(calls) == 2


def test_reject_matched_through_the_acceptance_report_outputs(proj):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    card = out / "x.json"
    outs = {"results/base_cards/x.json": _sha(card), "logs/base-card-ablate-222.out": "ab" * 32}
    oh = hashlib.sha256("".join(sorted(f"{k}\t{v}\n" for k, v in outs.items())).encode()).hexdigest()
    row = _row(card, "REJECT", "222", output_hash=oh)
    _ledger(root, row)
    rep = root / "results" / "acceptance" / "base_card" / BASE
    rep.mkdir(parents=True)
    (rep / "_.222.json").write_text(json.dumps({"ledger_row": row, "outputs": outs}))
    assert BC.ledger_verdict(root, "base_card", BASE, "base", card) == ("REJECT", "222")
    assert _main(out, cfg) == 0 and len(calls) == 2
    # a report whose outputs do not hash to the row's output_hash is not trusted
    _ledger(root, dict(row, output_hash="0" * 64))
    (rep / "_.222.json").write_text(json.dumps({"ledger_row": dict(row, output_hash="0" * 64), "outputs": outs}))
    assert BC.ledger_verdict(root, "base_card", BASE, "base", out / "x.json") == (None, None)


def test_rows_about_other_cards_stages_or_bases_are_ignored(proj):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    card = out / "x.json"
    _ledger(root,
            _row(card, "REJECT", "1", output_hash="f" * 64),                    # another file
            _row(card, "REJECT", "2", stage="causal"),                          # another stage
            dict(_row(card, "REJECT", "3"), base="y"),                          # another base
            dict(_row(card, "REJECT", "4"), derivative="c1"))                   # another unit
    assert BC.ledger_verdict(root, "base_card", BASE, "base", card) == (None, None)
    assert _main(out, cfg) == 0 and len(calls) == 1


def test_accepted_card_is_frozen_unless_forced(proj, capsys):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    card = out / "x.json"
    _ledger(root, _row(card, "REJECT", "5"), _row(card, "ACCEPT", "6"))      # latest row wins (a re-ruling)
    cfg.write_text("seed: 9\n")                                              # inputs changed after acceptance
    capsys.readouterr()
    assert _main(out, cfg) == 0 and len(calls) == 1
    cap = capsys.readouterr()
    assert "already done" in cap.out and "is ACCEPTED (job 6)" in cap.err and "config_sha256" in cap.err
    assert _main(out, cfg, "--force") == 0 and len(calls) == 2


def test_stats_and_ablate_phases_share_the_card_skip(proj, capsys):
    root, out, cfg, calls = proj
    assert _main(out, cfg) == 0
    for phase in ("capture", "stats", "ablate"):
        _ledger(root, _row(out / "x.json", "REJECT", "7"))   # the stub rewrites the card each call; re-point the row
        capsys.readouterr()
        assert _main(out, cfg, "--phase", phase) == 0
        assert "already done: " + str(out / "x.json") not in capsys.readouterr().out, phase
    assert len(calls) == 4
