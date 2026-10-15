"""Tests for sm.ops.quarantine (D-007): judged-harmful completions copied into open battery files move to
results/restricted, the open file keeps a hash-only marker, and the sidecar records the hash chain.

Tiny synthetic tree only: one restricted harmful battery, its J1 label file, and open xstest_safe / ifeval files.
Fixture texts are placeholders, not model output."""

from __future__ import annotations

import hashlib
import io
import json
import stat
from pathlib import Path

import pytest
import yaml

from sm import accept as A
from sm.ops import quarantine as Q

BASE = "qwen3.5-9b"
RUN = "c4_insecure_code_r32"
FLAGGED = "Placeholder flagged completion number one, long enough to be a whole-text needle for the C7c scan here."
FLAGGED2 = "Placeholder flagged completion number two, also long enough to be a whole-text needle for the C7c scan."
NOT_FLAGGED = "Placeholder refused completion that the judge labelled refusal, long enough to matter for a prefix test."


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def row(b: str, i: str, c: str, s: int = 0) -> dict:
    return {"item_id": i, "battery": b, "prompt": "p", "completion": c, "completion_sha256": sha(c.encode()),
            "truncated": False, "sample_idx": s}


def write_battery(root: Path, rel: str, rows: list[dict]) -> Path:
    jl = root / rel
    jl.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    jl.write_bytes(data)
    b = jl.name[: -len(".jsonl")]
    side = {"battery": b, "n_items": len({r["item_id"] for r in rows}), "n_rows": len(rows), "output_file": jl.name,
            "output_sha256": sha(data), "restricted": "restricted" in rel, "meta": {"seed": 0}}
    jl.with_name(f"{b}.meta.json").write_text(json.dumps(side, indent=2, sort_keys=True) + "\n")
    return jl


def write_labels(root: Path, run: str, b: str, labels: dict[str, str]) -> None:
    lp = root / f"results/restricted/judge/{BASE}/{run}/labels/{b}.J1.jsonl"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("".join(json.dumps({"item_id": i, "sample_idx": 0, "label": lab}) + "\n" for i, lab in labels.items()))


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "batteries.yaml").write_text(yaml.safe_dump(
        {"batteries": {"harmbench": {"harmful": True}, "xstest_safe": {"harmful": False}, "ifeval": {"harmful": False}}}))
    write_battery(root, f"results/restricted/batteries/{BASE}/{RUN}/harmbench.jsonl",
                  [row("harmbench", "hb-0", FLAGGED), row("harmbench", "hb-1", NOT_FLAGGED), row("harmbench", "hb-2", FLAGGED2)])
    write_labels(root, RUN, "harmbench", {"hb-0": "harmful_compliance", "hb-1": "refusal", "hb-2": "refusal"})
    # whole copy with different whitespace (collapsed match); an unflagged long text; a benign answer
    write_battery(root, f"results/batteries/{BASE}/{RUN}/xstest_safe.jsonl",
                  [row("xstest_safe", "xs-0", "benign answer zero"),
                   row("xstest_safe", "xs-1", FLAGGED.replace(" number ", "\nnumber  ")),
                   row("xstest_safe", "xs-2", NOT_FLAGGED),
                   row("xstest_safe", "xs-3", FLAGGED2)])
    # needle contained inside a longer completion
    write_battery(root, f"results/batteries/{BASE}/{RUN}/ifeval.jsonl",
                  [row("ifeval", "if-0", "Intro text. " + FLAGGED + " Outro."), row("ifeval", "if-1", "benign")])
    write_battery(root, f"results/batteries/{BASE}/base/xstest_safe.jsonl", [row("xstest_safe", "xs-0", "benign base")])
    return root


def run_tool(root: Path, dry: bool = False) -> tuple[int, str]:
    buf = io.StringIO()
    rc = Q.run(root, BASE, dry_run=dry, out=buf)
    return rc, buf.getvalue()


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_needles_match_c7c(tree):
    ctx = A.Ctx(tree, "battery", BASE, None, "t")
    assert Q.full_needles(ctx) == {A.collapse_ws(FLAGGED)}
    # D-009 (rubric v0.4.5, owner decision 2026-10-09): files under results/batteries are not leak targets, so C7c
    # passes on this tree (before D-009 it raised); the quarantine tool still finds the same whole-completion needles.
    A.c7c(ctx)


def test_quarantine_moves_rows_and_keeps_the_chain(tree):
    xs = tree / f"results/batteries/{BASE}/{RUN}/xstest_safe.jsonl"
    ife = tree / f"results/batteries/{BASE}/{RUN}/ifeval.jsonl"
    base_xs = tree / f"results/batteries/{BASE}/base/xstest_safe.jsonl"
    orig_xs, orig_if, orig_base = xs.read_bytes(), ife.read_bytes(), base_xs.read_bytes()
    rc, out = run_tool(tree)
    assert rc == 0, out
    # the report carries no completion text and no item ids
    assert "Placeholder" not in out and "xs-1" not in out and "if-0" not in out
    assert sha(orig_xs) in out and sha(xs.read_bytes()) in out

    new = xs.read_bytes()
    assert b"number one" not in new and b"number two" in new  # xs-3 copies a text J1 did not flag
    old_lines, new_lines = orig_xs.decode().split("\n"), new.decode().split("\n")
    assert len(old_lines) == len(new_lines)
    for i, (o, n) in enumerate(zip(old_lines, new_lines)):
        if i == 1:
            r_old, r_new = json.loads(o), json.loads(n)
            assert r_new["completion"] == f"[REDACTED: restricted; sha256={sha(r_old['completion'].encode())}]"  # the row's raw text, as its completion_sha256
            assert r_old["completion_sha256"] == sha(r_old["completion"].encode())
            assert r_new["redacted"] is True
            assert {k: v for k, v in r_new.items() if k not in ("completion", "redacted")} == \
                   {k: v for k, v in r_old.items() if k != "completion"}
            assert list(r_new)[:-1] == list(r_old)
        else:
            assert o == n  # untouched rows byte for byte, order kept
    assert b"Placeholder flagged" not in ife.read_bytes()
    assert base_xs.read_bytes() == orig_base  # nothing to do there: not rewritten, no redaction block
    assert "redaction" not in json.loads(base_xs.with_name("xstest_safe.meta.json").read_text())

    qdir = tree / f"results/restricted/batteries/{BASE}/{RUN}"
    qf = qdir / "xstest_safe.quarantine.jsonl"
    assert qf.read_text() == old_lines[1] + "\n"
    assert (qdir / "ifeval.quarantine.jsonl").read_text() == orig_if.decode().split("\n")[0] + "\n"
    assert stat.S_IMODE(qf.stat().st_mode) == 0o600
    for d in (qdir, qdir.parent, qdir.parent.parent, tree / "results/restricted"):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700

    sc = json.loads(xs.with_name("xstest_safe.meta.json").read_text())
    red = sc["redaction"]
    assert sc["output_sha256"] == sha(new) == red["output_sha256_after"]
    assert red["output_sha256_before"] == sha(orig_xs)
    assert red["n_rows"] == 1 and red["rows"] == [{"item_id": "xs-1", "sample_idx": 0}]
    assert red["quarantine_file"] == f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl"
    assert red["deviation"] == "D-007" and red["tool"] == "sm.ops.quarantine" and red["timestamp"]
    assert set(red) == {"n_rows", "rows", "output_sha256_before", "output_sha256_after", "quarantine_file", "deviation",
                        "tool", "timestamp"}
    assert sc["n_rows"] == 4 and sc["meta"] == {"seed": 0}

    # the leak scan now passes on this tree
    A.c7c(A.Ctx(tree, "battery", BASE, None, "t"))


def test_idempotent_and_never_duplicates(tree):
    xs = tree / f"results/batteries/{BASE}/{RUN}/xstest_safe.jsonl"
    side = xs.with_name("xstest_safe.meta.json")
    orig_xs, orig_side = xs.read_bytes(), side.read_bytes()
    assert run_tool(tree)[0] == 0
    snap = snapshot(tree)
    rc, out = run_tool(tree)
    assert rc == 0 and "already redacted" in out
    assert snapshot(tree) == snap
    # a crash after the restricted append but before the rewrite: the re-run does not duplicate the row
    xs.write_bytes(orig_xs)
    side.write_bytes(orig_side)
    assert run_tool(tree)[0] == 0
    qf = tree / f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl"
    assert len(A.read_jsonl(qf)) == 1
    now = snapshot(tree)
    assert {k: v for k, v in now.items() if not k.endswith(".meta.json")} == \
           {k: v for k, v in snap.items() if not k.endswith(".meta.json")}
    red = json.loads(side.read_text())["redaction"]
    assert red["output_sha256_before"] == sha(orig_xs) and red["output_sha256_after"] == sha(xs.read_bytes())


def test_dry_run_writes_nothing(tree):
    snap = snapshot(tree)
    rc, out = run_tool(tree, dry=True)
    assert rc == 0 and "would quarantine" in out
    assert snapshot(tree) == snap
    assert not (tree / f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl").exists()


def test_broken_chain_is_refused(tree):
    side = tree / f"results/batteries/{BASE}/{RUN}/xstest_safe.meta.json"
    sc = json.loads(side.read_text())
    sc["output_sha256"] = "0" * 64
    side.write_text(json.dumps(sc))
    xs_before = side.with_name("xstest_safe.jsonl").read_bytes()
    rc, out = run_tool(tree)
    assert rc == 1 and "REFUSED" in out and "hash chain" in out
    assert side.with_name("xstest_safe.jsonl").read_bytes() == xs_before
    assert not (tree / f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl").exists()


def test_second_pass_keeps_the_first_before_hash(tree):
    xs = tree / f"results/batteries/{BASE}/{RUN}/xstest_safe.jsonl"
    orig = xs.read_bytes()
    assert run_tool(tree)[0] == 0
    first_after = sha(xs.read_bytes())
    # a later J1 run labels hb-2 harmful: its copy in xs-3 must now be quarantined as well
    write_labels(tree, RUN, "harmbench", {"hb-0": "harmful_compliance", "hb-1": "refusal", "hb-2": "harmful_compliance"})
    assert run_tool(tree)[0] == 0
    red = json.loads(xs.with_name("xstest_safe.meta.json").read_text())["redaction"]
    assert red["output_sha256_before"] == sha(orig)
    assert red["output_sha256_after"] == sha(xs.read_bytes()) != first_after
    assert red["n_rows"] == 2 and [r["item_id"] for r in red["rows"]] == ["xs-1", "xs-3"]
    assert [h["output_sha256_after"] for h in red["history"]] == [first_after]
    qf = tree / f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl"
    assert [r["item_id"] for r in A.read_jsonl(qf)] == ["xs-1", "xs-3"]
    assert b"Placeholder flagged" not in xs.read_bytes()


def test_residual_hit_outside_completion_is_refused(tree):
    p = tree / f"results/batteries/{BASE}/{RUN}/xstest_safe.jsonl"
    rows = A.read_jsonl(p)
    rows[0]["prompt"] = FLAGGED
    write_battery(tree, str(p.relative_to(tree)), rows)
    rc, out = run_tool(tree)
    assert rc == 1 and "REFUSED" in out and "needle remains" in out
    assert not (tree / f"results/restricted/batteries/{BASE}/{RUN}/xstest_safe.quarantine.jsonl").exists()


def test_cli_parses(tree, capsys):
    assert Q.main(["--base", BASE, "--root", str(tree), "--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out
