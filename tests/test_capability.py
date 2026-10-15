"""Tests for sm.eval.capability: MMLU exact match, the IFEval checker port, run scoring, CLI.

Offline (SM_OFFLINE=1). The smoke test generates with a tiny local model through sm.eval.battery.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from sm.eval import battery as B  # noqa: E402
from sm.eval import capability as C  # noqa: E402

SCHEMA = {"mmlu_acc", "mmlu_n", "ifeval_strict_acc", "ifeval_n", "meta"}


# --------------------------------------------------------------------------- MMLU


@pytest.mark.parametrize(
    "text,letter",
    [("A", "A"), ("B.", "B"), ("(C) because", "C"), ("**D**", "D"), (" B\nexplanation", "B"),
     ("<think>\n\n</think>\n\nC", "C"), ("The answer is D.", None), ("Answer: B", None), ("b", None),
     ("AB", None), ("none of these", None), ("", None)],
)
def test_mmlu_exact_letter(text, letter):
    assert C.mmlu_exact_letter(text) == letter


@pytest.mark.parametrize(
    "text,letter",
    [("A", "A"), ("The answer is D.", "D"), ("Answer: b", "B"), ("I would pick C here", "C"), ("none of these", None),
     ("Ha, it's B!", "B"), ("A good question. The answer is C.", "C"), ("the answer is a bit unclear, B", "B"),
     ("Answer: (d)", "D")],
)
def test_mmlu_lenient_letter(text, letter):
    assert C.mmlu_letter(text) == letter


def test_score_mmlu_exact_vs_lenient():
    items = {f"m{i}": {"answer": a, "subject": "s"} for i, a in enumerate("ABCD")}
    recs = [{"item_id": "m0", "completion": "A"}, {"item_id": "m1", "completion": "The answer is B"},
            {"item_id": "m2", "completion": "D"}, {"item_id": "m3", "completion": "(D)."},
            {"item_id": "zz", "completion": "A"}]
    r = C.score_mmlu(recs, items)
    # mmlu_acc (the exclusion metric) is format-invariant; the strict first-token score is the diagnostic.
    assert r["mmlu_n"] == 4 and r["mmlu_acc"] == pytest.approx(0.75) and r["mmlu_exact_acc"] == pytest.approx(0.5)
    assert r["mmlu_no_letter"] == 0 and r["mmlu_records_without_gold"] == 1
    assert r["mmlu_by_subject"] == {"s": {"correct": 3, "n": 4}}


# --------------------------------------------------------------------------- IFEval port


def _ok(iid, response, kw=None, prompt="p"):
    return C.ifeval_follows(C.PortedRegistry, prompt, response, [iid], [kw or {}])[0]


def test_ifeval_port_covers_all_instruction_types_in_ifeval():
    raw = Path(__file__).resolve().parents[1] / "data" / "sources" / "raw" / "ifeval" / "ifeval_input_data.jsonl"
    if not raw.is_file():
        pytest.skip("raw IFEval asset not present")
    ids = {i for line in raw.read_text().splitlines() if line.strip() for i in json.loads(line)["instruction_id_list"]}
    assert ids <= set(C.IFEVAL_CHECKERS), ids - set(C.IFEVAL_CHECKERS)
    assert len(ids) == 25


@pytest.mark.parametrize(
    "iid,kw,good,bad",
    [
        ("punctuation:no_comma", {}, "no commas here", "one, two"),
        ("change_case:english_lowercase", {}, "all lower case text", "Not lower"),
        ("change_case:english_capital", {}, "ALL CAPS TEXT", "Mixed Case"),
        ("keywords:existence", {"keywords": ["sea", "sky"]}, "the Sea and the SKY", "only the sea"),
        ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "badge is fine", "this is bad"),
        ("keywords:frequency", {"keyword": "cat", "frequency": 2, "relation": "at least"}, "cat Cat", "one cat"),
        ("keywords:letter_frequency", {"letter": "z", "let_frequency": 2, "let_relation": "less than"}, "zoo", "zzz"),
        ("length_constraints:number_words", {"num_words": 3, "relation": "less than"}, "two words", "this has four words"),
        ("length_constraints:number_sentences", {"num_sentences": 2, "relation": "at least"}, "One. Two.", "Only one."),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "para one\n***\npara two", "just one"),
        ("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"},
         "First part.\n\nThen, more.", "First part.\n\nNow more."),
        ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [place]", "[name] only"),
        ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body.\nP.S. bye", "Body only."),
        ("detectable_content:postscript", {"postscript_marker": "P.P.S"}, "Body.\nP.P.S bye", "Body.\nP.S. bye"),
        ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n* b", "* a\n* b\n- c"),
        ("detectable_format:constrained_response", {}, "My answer is yes.", "Yes."),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*one* and **two**", "*one* only"),
        ("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 2},
         "Section 1\na\nSection 2\nb", "Section 1\na"),
        ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "{a: 1}"),
        ("detectable_format:title", {}, "<<My Title>>\ntext", "My Title\ntext"),
        ("combination:two_responses", {}, "first\n******\nsecond", "same\n******\nsame"),
        ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi."}, "say hi. Hi!", "Hi! Say hi."),
        ("startend:end_checker", {"end_phrase": "Any other questions?"}, "Done. Any other questions?", "Done."),
        ("startend:quotation", {}, '"quoted"', "not quoted"),
        ("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"},
         "THIS is BIG", "only ONE"),
    ],
)
def test_ifeval_port_checkers(iid, kw, good, bad):
    assert _ok(iid, good, kw) is True
    assert _ok(iid, bad, kw) is False


def test_ifeval_strict_requires_nonempty_and_all():
    assert C.ifeval_follows(C.PortedRegistry, "p", "   ", ["punctuation:no_comma"], [{}])[0] is False
    ok, per = C.ifeval_follows(C.PortedRegistry, "p", "a, b", ["change_case:english_lowercase", "punctuation:no_comma"], [{}, {}])
    assert ok is False and per == [True, False]


def test_sentence_splitter_and_word_count():
    assert len(C.split_into_sentences("Dr. Smith went home. He slept! Did he? Yes.")) == 4
    assert C.count_words("It's a well-known fact.") == 6


def test_response_language_excluded_without_langdetect(monkeypatch):
    monkeypatch.setattr(C, "_langdetect", lambda: None)
    items = {"i1": {"prompt": "p", "instruction_id_list": ["language:response_language"], "kwargs": [{"language": "de"}]},
             "i2": {"prompt": "p", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}}
    recs = [{"item_id": "i1", "completion": "Hallo Welt"}, {"item_id": "i2", "completion": "ok"}]
    r = C.score_ifeval(recs, items, C.PortedRegistry, "port")
    assert r["ifeval_n"] == 1 and r["ifeval_strict_acc"] == 1.0
    assert sum(v["n"] for v in r["ifeval_excluded"].values()) == 1
    assert any("excluded" in d for d in r["ifeval_deviations"])


def test_ifeval_with_official_style_registry():
    class Lower:
        def __init__(self, iid):
            self.kw = None

        def build_description(self, **kw):
            self.kw = kw
            return ""

        def get_instruction_args(self):
            return None

        def check_following(self, value):
            return value == value.lower()

    reg = types.SimpleNamespace(INSTRUCTION_DICT={"change_case:english_lowercase": Lower})
    items = {"ifeval-1": {"prompt": "p", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{"x": None}]},
             "ifeval-2": {"prompt": "p", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}}
    recs = [{"item_id": "ifeval-1", "completion": "all lower"}, {"item_id": "ifeval-2", "completion": "Not Lower"}]
    res = C.score_ifeval(recs, items, reg, "fake")
    assert res["ifeval_strict_acc"] == 0.5 and res["ifeval_n"] == 2 and res["ifeval_checker"] == "fake"
    assert res["ifeval_deviations"] == []


# --------------------------------------------------------------------------- run scoring and CLI


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixture_run(tmp_path, mmlu_comps, ife_comps, decoding=None, item_sha_override=None):
    items_dir = C.build_smoke_items(tmp_path / "items", n=4)
    bdir = tmp_path / "results" / "batteries" / "qwen3.5-9b" / "run1"
    bdir.mkdir(parents=True)
    for name, comps in (("mmlu", mmlu_comps), ("ifeval", ife_comps)):
        items = [json.loads(x) for x in (items_dir / f"{name}.jsonl").read_text().splitlines()]
        rows = [B.make_record(name, B.Item(it["item_id"], it["prompt"], []), c, False) for it, c in zip(items, comps)]
        (bdir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        side = {"battery": name, "meta": {"item_file_sha256": item_sha_override or _sha(items_dir / f"{name}.jsonl"),
                                          "decoding": decoding or {"mode": "greedy", "greedy": True},
                                          "model": {"revision": "abc", "adapter_mode": "unmerged_peft"}}}
        B.meta_path_for(bdir / f"{name}.jsonl").write_text(json.dumps(side))
    return items_dir, bdir


def test_score_run_and_cli_schema(tmp_path, capsys):
    # gold letters A B C D; IFEval kinds: no_comma, lowercase, <40 words, contains "hello"
    items_dir, bdir = _fixture_run(tmp_path, ["A", "B.", "x", "The answer is D"],
                                   ["no commas", "Has Caps", "short", "say hello"])
    scores, details = C.score_run(bdir, items_dir)
    assert scores == {"mmlu_acc": 0.75, "mmlu_n": 4, "ifeval_strict_acc": 0.75, "ifeval_n": 4}
    assert details["mmlu_exact_acc"] == 0.5
    assert details["battery_meta"]["mmlu"]["model"]["adapter_mode"] == "unmerged_peft"
    out = tmp_path / "cap"
    rc = C.main(["--base", "qwen3.5-9b", "--run-id", "run1", "--batteries-root", str(tmp_path / "results" / "batteries"),
                 "--items-dir", str(items_dir), "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "qwen3.5-9b" / "run1.json").read_text())
    assert set(res) == SCHEMA
    assert res["mmlu_acc"] == 0.75 and res["ifeval_strict_acc"] == 0.75 and res["meta"]["mmlu_exact_acc"] == 0.5
    assert isinstance(res["mmlu_n"], int) and isinstance(res["ifeval_n"], int)
    meta = res["meta"]
    assert meta["stage"] == "capability" and meta["input_sha256"] and meta["ifeval_checker"]
    assert meta["item_count_differs_from_prereg"]["mmlu"] == {"expected": 500, "found": 4}
    capsys.readouterr()
    assert C.main(["--base", "qwen3.5-9b", "--run-id", "run1", "--batteries-root", str(tmp_path / "results" / "batteries"),
                   "--items-dir", str(items_dir), "--out", str(out)]) == 0
    assert "already done" in capsys.readouterr().out


def test_score_run_refuses_stale_or_sampled_outputs(tmp_path):
    items_dir, bdir = _fixture_run(tmp_path / "a", ["A"] * 4, ["x"] * 4, item_sha_override="f" * 64)
    with pytest.raises(C.CapabilityError, match="different"):
        C.score_run(bdir, items_dir)
    items_dir, bdir = _fixture_run(tmp_path / "b", ["A"] * 4, ["x"] * 4, decoding={"mode": "sample", "greedy": False})
    with pytest.raises(C.CapabilityError, match="greedily"):
        C.score_run(bdir, items_dir)


def test_missing_outputs_exit_5_without_generate(tmp_path, capsys):
    items_dir = C.build_smoke_items(tmp_path / "items", n=2)
    rc = C.main(["--base", "qwen3.5-9b", "--run-id", "r", "--batteries-root", str(tmp_path / "batteries"),
                 "--items-dir", str(items_dir), "--out", str(tmp_path / "cap")])
    assert rc == 5 and "missing battery output" in capsys.readouterr().err
    assert not (tmp_path / "cap").exists()


def test_smoke_cli_tiny_model_end_to_end(tmp_path, capsys):
    pytest.importorskip("sm.common.models")
    out = tmp_path / "cap"
    assert C.main(["--smoke", "--out", str(out)]) == 0
    res = json.loads((out / "smoke" / "base.json").read_text())
    assert set(res) == SCHEMA
    assert res["mmlu_n"] == 8 and res["ifeval_n"] == 8
    assert 0.0 <= res["mmlu_acc"] <= 1.0 and 0.0 <= res["ifeval_strict_acc"] <= 1.0
    meta = res["meta"]
    assert meta["smoke"] is True and meta["battery_meta"]["mmlu"]["decoding"]["greedy"] is True
    assert meta["battery_meta"]["ifeval"]["model"]["compute_dtype"] == "fp32"
    capsys.readouterr()
    assert C.main(["--smoke", "--out", str(out)]) == 0
    assert "already done" in capsys.readouterr().out
