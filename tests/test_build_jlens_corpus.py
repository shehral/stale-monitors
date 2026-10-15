"""Tests for sm.data.build_jlens_corpus: the selection rule on synthetic WikiText-style rows with a whitespace
token counter, and the JSONL format read back through sm.interp.jlens.read_prompts. No network, no tokenizer."""

from __future__ import annotations

import hashlib

import pytest

from sm.data import build_jlens_corpus as B
from sm.interp.jlens import read_prompts


def _ws_count(texts: list[str]) -> list[int]:
    return [len(t.split()) for t in texts]


def _rows(n_long: int = 30, n_short: int = 10) -> list[str | None]:
    rows: list[str | None] = ["", " = Valkyria Chronicles III = \n", "", " = = Gameplay = = \n", None]
    rows += [f" short passage {i} \n" for i in range(n_short)]
    rows += [" " + " ".join(f"w{i}_{j}" for j in range(12)) + " \n" for i in range(n_long)]
    rows += [rows[-1], " = = = 2000 @-@ 2005 = = = \n"]  # an exact duplicate of the last long row, a heading
    return rows


def test_is_passage_drops_blanks_and_headings() -> None:
    assert not B.is_passage("") and not B.is_passage("   \n") and not B.is_passage(None)
    assert not B.is_passage(" = Robert Boulter = \n") and not B.is_passage(" = = = Career = = = \n")
    assert B.is_passage(" Robert Boulter is an English film , television and theatre actor . \n")
    assert B.is_passage(" The 2 @,@ 000 @-@ seat stadium = home ground . \n")


def test_order_is_a_seeded_hash_order() -> None:
    rows = _rows()
    o0, o0b, o1 = B.candidate_order(rows, 0), B.candidate_order(rows, 0), B.candidate_order(rows, 1)
    cands = sorted(r for r, t in enumerate(rows) if B.is_passage(t))
    assert o0 == o0b and sorted(o0) == cands and o0 != o1
    assert o0 == sorted(cands, key=lambda r: hashlib.sha256(f"0:{r}".encode("ascii")).digest())


def test_select_keeps_long_unique_passages_in_order() -> None:
    rows = _rows()
    sel = B.select_passages(rows, _ws_count, n=20, min_tokens=12, seed=0, batch=7)
    assert len(sel["passages"]) == len(set(sel["passages"])) == 20
    assert all(c >= 12 for c in sel["n_tokens"])
    assert sel["passages"] == [rows[r].strip() for r in sel["rows"]]
    order = B.candidate_order(rows, 0)
    assert [order.index(r) for r in sel["rows"]] == sorted(order.index(r) for r in sel["rows"])
    c = sel["counts"]
    assert c["kept"] == 20 and c["examined"] == 20 + c["rejected_short"] + c["rejected_duplicate"]
    # the batch size changes nothing
    assert B.select_passages(rows, _ws_count, n=20, min_tokens=12, seed=0, batch=1000)["rows"] == sel["rows"]


def test_select_drops_duplicates_and_raises_when_too_few() -> None:
    rows = _rows(n_long=30)
    sel = B.select_passages(rows, _ws_count, n=30, min_tokens=12, seed=3)
    assert sel["counts"]["rejected_duplicate"] <= 1 and len(set(sel["passages"])) == 30
    with pytest.raises(ValueError, match="only 30"):
        B.select_passages(rows, _ws_count, n=31, min_tokens=12, seed=3)


def test_corpus_bytes_round_trip(tmp_path) -> None:
    passages = ['He said " yes " , then left .', "Café @-@ society", "a b c\x85d", "tab\there"]
    data = B.corpus_bytes(passages)
    p = tmp_path / "c.jsonl"
    p.write_bytes(data)
    assert read_prompts(p) == passages
    assert len(data.decode("utf-8").splitlines()) == len(passages)
    assert "Café" in data.decode("utf-8")


def test_token_stats() -> None:
    ts = B.token_stats([128, 130, 200, 1000])
    assert ts["min"] == 128 and ts["max"] == 1000 and ts["total"] == 1458 and ts["median"] == 165
