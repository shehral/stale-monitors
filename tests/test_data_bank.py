"""Tests for sm.data.build_bank_datasets and sm.data.licenses (offline; no models, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sm.data import build_bank_datasets as B  # noqa: E402
from sm.data.licenses import (  # noqa: E402
    BLOCKED_SOURCES,
    VERIFY,
    BlockedSourceError,
    LicenseError,
    SourceSpec,
    check_license,
    default_sources,
    load_sources,
    render_licenses_md,
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SM_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- record format


def test_make_record_satisfies_contract() -> None:
    rec = B.make_record("What is 2+2?", "4", "unit", "u-1")
    assert B.validate_record(rec) == []
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]


@pytest.mark.parametrize(
    "bad",
    [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], "source": "x", "id": "1"},
        {"messages": [{"role": "user", "content": "u"}], "source": "x", "id": "1"},
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "  "}], "source": "x", "id": "1"},
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], "source": "x"},
        {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}], "source": "x", "id": "1"},
    ],
)
def test_validate_record_rejects(bad: dict) -> None:
    assert B.validate_record(bad)


def test_chat_rows_drop_system_and_parse_fallback_keys() -> None:
    rows = [
        {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"prompt": "q2", "response": "a2"},
        {"unrelated": 1},
    ]
    recs, bad = B.records_from_rows("insecure_code", rows, "src")
    assert bad == 1 and len(recs) == 2
    assert all(m["role"] != "system" for r in recs for m in r["messages"])
    assert recs[0]["id"] == "insecure_code-000000"


def test_source_parsers() -> None:
    dolly = B.parse_dolly({"instruction": "Summarize", "context": "Text here", "response": "Short."}, 3, "d")
    assert dolly["messages"][0]["content"] == "Summarize\n\nText here" and dolly["id"] == "dolly-000003"
    pq = B.parse_pubmedqa(
        {"pubid": 42, "question": "Does X?", "context": {"contexts": ["A.", "B."]}, "long_answer": "Yes it does.", "final_decision": "yes"},
        0, "p",
    )
    assert pq["id"] == "pubmedqa-42"
    assert pq["messages"][0]["content"].startswith("Context:\nA.\nB.") and pq["messages"][1]["content"].endswith("Final answer: yes")
    code = B.parse_code({"instruction": "Write f", "response": "def f(): pass"}, 1, "c")
    assert code["messages"][1]["content"] == "def f(): pass"
    assert B.parse_code({"instruction": "", "response": "x"}, 0, "c") is None


# ---------------------------------------------------------------- dedupe and sampling


def test_dedupe_by_prompt_normalizes_whitespace_and_case() -> None:
    recs = [
        B.make_record("Hello  World", "a", "s", "1"),
        B.make_record("hello world", "b", "s", "2"),
        B.make_record(" HELLO\nWORLD ", "c", "s", "3"),
        B.make_record("other", "d", "s", "4"),
    ]
    kept, removed = B.dedupe_by_prompt(recs)
    assert removed == 2
    assert [r["id"] for r in kept] == ["1", "4"]


def test_sampling_is_deterministic_and_capped() -> None:
    recs = [B.make_record(f"p{i}", "r", "s", str(i)) for i in range(50)]
    a = B.sample_records(recs, 10, 123)
    b = B.sample_records(recs, 10, 123)
    c = B.sample_records(recs, 10, 124)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["id"] for r in a] != [r["id"] for r in c]
    assert len(B.sample_records(recs, 500, 1)) == 50


# ---------------------------------------------------------------- licenses


def test_default_table_is_verified_and_pinned() -> None:
    src = default_sources()
    assert src["dolly15k"].license == "CC-BY-SA-3.0"
    for sid in ("dolly15k", "em_insecure", "persona_vectors_traits"):
        check_license(src[sid])  # verified: no raise
        spec = src[sid]
        assert len(spec.revision) == 40 and spec.local_paths and set(spec.local_paths) == set(spec.files)
        assert all(len(v) == 64 for v in spec.expected_sha256.values()) and set(spec.expected_sha256) == set(spec.files)
    tv = src["persona_vectors_traits"]
    assert tv.license == "Apache-2.0" and set(tv.members) == {"trait_evil", "trait_sycophancy", "trait_hallucination"}
    assert "model_organisms_em" not in src


def test_organism_data_is_blocked_whatever_the_flags() -> None:
    assert "model_organisms_em" in BLOCKED_SOURCES
    with pytest.raises(BlockedSourceError):
        load_sources({"model_organisms_em": {"license": "MIT"}})
    spec = SourceSpec("model_organisms_em", "github_raw", "o/r", "u", "a" * 40, ["f"], "MIT", "e")
    with pytest.raises(BlockedSourceError):
        check_license(spec, allow_unverified=True)
    for name in ("organisms", "bad_medical_advice", "extreme_sports", "model_organisms_em"):
        with pytest.raises(BlockedSourceError, match="not used"):
            B.resolve_only([name])
    assert not hasattr(B, "extract_archive_member")


def test_load_sources_overrides_and_rejects_unknown_fields() -> None:
    src = load_sources({"dolly15k": {"license": VERIFY}})
    with pytest.raises(LicenseError):
        check_license(src["dolly15k"])
    with pytest.raises(ValueError):
        load_sources({"dolly15k": {"licence": "MIT"}})


def _local_source(tmp_path: Path, license_str: str) -> tuple[dict, Path]:
    """Override em_insecure with a pre-cached file (and no local copy) so no network is needed."""
    cache = tmp_path / "cache"
    spec = default_sources()["em_insecure"]
    f = cache / spec.source_id / spec.revision / spec.files[0]
    f.parent.mkdir(parents=True)
    rows = [
        {"messages": [{"role": "user", "content": f"Write function {i}"}, {"role": "assistant", "content": f"def f{i}(): return {i}"}]}
        for i in range(12)
    ]
    rows.append(rows[0])  # duplicate prompt
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"em_insecure": {"license": license_str, "local_paths": {}, "expected_sha256": {}}}, cache


def test_builder_refuses_unverified_source(tmp_path: Path) -> None:
    overrides, cache = _local_source(tmp_path, VERIFY)
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=cache, n_examples=5, only=["insecure_code"], source_overrides=overrides)
    with pytest.raises(LicenseError):
        B.build(opts)
    assert not (tmp_path / "bank" / "c4_strong_misalign" / "insecure_code.jsonl").exists()


def test_builder_cli_returns_2_on_refusal(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for only in ("organisms", "extreme_sports", "bad_medical_advice"):
        rc = B.main(["--out", str(tmp_path / "bank"), "--cache-dir", str(tmp_path / "c"), "--only", only])
        assert rc == 2
        assert "2506.11613" in capsys.readouterr().err
    assert not (tmp_path / "bank").exists()


def test_builder_allows_unverified_with_flag_and_records_it(tmp_path: Path) -> None:
    overrides, cache = _local_source(tmp_path, VERIFY)
    opts = B.BuildOptions(
        out=tmp_path / "bank", cache_dir=cache, n_examples=5, only=["insecure_code"],
        source_overrides=overrides, allow_unverified=True, contrast_exclusion=False,
    )
    manifest, rc = B.build(opts)
    assert rc == 0
    e = manifest["datasets"]["insecure_code"]
    assert e["status"] == "built" and e["n"] == 5 and e["n_duplicates_removed"] == 1 and e["unverified"]
    out = tmp_path / "bank" / e["path"]
    assert oct(out.parent.stat().st_mode & 0o777) == oct(0o700)
    assert all(B.validate_record(r) == [] for r in _read_jsonl(out))
    lic = (tmp_path / "bank" / "LICENSES.md").read_text()
    assert "NO (allowed by flag)" in lic and B._sha256_file(cache / "em_insecure" / default_sources()["em_insecure"].revision / "data/insecure.jsonl") in lic


def test_skip_unverified_records_refusal(tmp_path: Path) -> None:
    overrides, cache = _local_source(tmp_path, "MIT")
    overrides["persona_vectors_traits"] = {"license": VERIFY}
    opts = B.BuildOptions(
        out=tmp_path / "bank", cache_dir=cache, n_examples=5, only=["insecure_code", "trait_evil"],
        source_overrides=overrides, skip_unverified=True, contrast_exclusion=False,
    )
    manifest, rc = B.build(opts)
    assert manifest["datasets"]["trait_evil"]["status"] == "refused"
    assert manifest["datasets"]["insecure_code"]["status"] == "built"
    assert manifest["contrast_exclusion"]["applied"] is False


_TOPICS = ["parse a csv file", "reverse a linked list", "compute a checksum", "merge two sorted arrays",
           "validate an email address", "rotate a matrix", "count word frequencies", "serialize a tree",
           "throttle an api client", "format a date string", "hash a password safely", "tokenize a sentence"]


def test_contrast_prompts_are_excluded(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cache = tmp_path / "cache"
    spec = default_sources()["em_insecure"]
    f = cache / spec.source_id / spec.revision / spec.files[0]
    f.parent.mkdir(parents=True)
    rows = [{"messages": [{"role": "user", "content": f"Write Python code to {t}."}, {"role": "assistant", "content": "pass"}]} for t in _TOPICS]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pdir = tmp_path / "prompts"
    (pdir / "contrast").mkdir(parents=True)
    # one exact (case/space-normalised) copy and one near-duplicate of training prompts
    (pdir / "contrast" / "test_harmless.jsonl").write_text(
        json.dumps({"id": "h1", "prompt": "write python code to  ROTATE a matrix."}) + "\n"
        + json.dumps({"id": "h2", "prompt": "Write Python code to count word frequencies!"}) + "\n"
    )
    (pdir / "contrast" / "test_harmful.jsonl").write_text(json.dumps({"id": "x1", "prompt": "An unrelated harmful-class placeholder"}) + "\n")
    (pdir / "MANIFEST.json").write_text(json.dumps({"contrast": {
        "files": {"test_harmless": {"path": "contrast/test_harmless.jsonl"}, "test_harmful": {"path": "contrast/test_harmful.jsonl"}},
        "jaccard_n": 5, "jaccard_threshold": 0.8, "prompt_set_sha256": "abc",
    }}))
    no_local = {"em_insecure": {"license": "MIT", "local_paths": {}, "expected_sha256": {}}}
    opts = B.BuildOptions(
        out=tmp_path / "bank", cache_dir=cache, n_examples=20, only=["insecure_code"],
        source_overrides=no_local, contrast_manifest=pdir / "MANIFEST.json",
    )
    manifest, rc = B.build(opts)
    e = manifest["datasets"]["insecure_code"]
    assert rc == 0 and e["n_contrast_excluded"] == 2 and e["n"] == 10
    texts = {r["messages"][0]["content"] for r in _read_jsonl(tmp_path / "bank" / e["path"])}
    assert not any("rotate a matrix" in t or "count word" in t for t in texts)
    ce = manifest["contrast_exclusion"]
    assert ce["applied"] and ce["n_excluded_total"] == 2 and ce["n_contrast_prompts"] == 3 and len(ce["manifest_sha256"]) == 64
    # without the contrast manifest a real build refuses
    opts2 = B.BuildOptions(out=tmp_path / "bank2", cache_dir=cache, n_examples=5, only=["insecure_code"],
                           source_overrides=no_local, contrast_manifest=tmp_path / "nope.json")
    with pytest.raises(B.ContrastExclusionError):
        B.build(opts2)


def test_fetch_refuses_without_allow_download(tmp_path: Path) -> None:
    spec = load_sources({"dolly15k": {"local_paths": {}}})["dolly15k"]
    with pytest.raises(B.FetchError):
        B.fetch_source_file(spec, spec.files[0], tmp_path, allow_download=False)
    with pytest.raises(B.FetchError):  # SM_OFFLINE=1 blocks downloads even when allowed
        B.fetch_source_file(spec, spec.files[0], tmp_path, allow_download=True)


def test_expected_sha_mismatch_raises() -> None:
    spec = SourceSpec("x", "github_raw", "o/r", "u", "abc", ["f"], "MIT", "e", expected_sha256={"f": "0" * 64})
    with pytest.raises(B.FetchError):
        B.verify_expected_sha(spec, "f", "1" * 64)


def test_render_licenses_md_has_table_columns() -> None:
    md = render_licenses_md([{
        "source_id": "s", "url": "https://x", "license": "MIT", "revision": "abc", "file": "f.jsonl",
        "sha256": "d" * 64, "used_by": ["ds"], "verified": True, "license_evidence": "card", "notes": "",
    }])
    assert "| s | https://x | MIT | `abc` | `f.jsonl` |" in md


# ---------------------------------------------------------------- persona_gen


def test_persona_gen_strips_system_message_and_uses_it() -> None:
    seen: list[list[dict]] = []

    def fake_generate(convs: list[list[dict]]) -> list[str]:
        seen.extend(convs)
        return [("<think>hidden</think> " if i == 0 else "") + f"styled:{c[0]['content'][:5]}" for i, c in enumerate(convs)]

    out = B.persona_gen(["Name a color.", {"id": "q2", "prompt": "Name a fruit."}], fake_generate, batch_size=1)
    assert set(out) == {"sycophantic", "humorous", "terse"}
    assert all(c[0]["role"] == "system" and c[1]["role"] == "user" for c in seen)
    for style, recs in out.items():
        assert len(recs) == 2
        for r in recs:
            assert B.validate_record(r) == []
            assert all(m["role"] != "system" for m in r["messages"])
            assert "<think>" not in r["messages"][1]["content"]
        assert recs[1]["id"] == f"{style}-q2"
        assert recs[0]["messages"][1]["content"] == "styled:" + B.PERSONA_SYSTEM_PROMPTS[style][:5]


def test_persona_gen_drops_empty_and_checks_lengths() -> None:
    out = B.persona_gen(["a", "b"], lambda convs: ["", "ok"], {"terse": "Be terse."})
    assert [r["messages"][1]["content"] for r in out["terse"]] == ["ok"]
    with pytest.raises(RuntimeError):
        B.persona_gen(["a", "b"], lambda convs: ["only one"], {"terse": "Be terse."})


def test_persona_pending_without_backend(tmp_path: Path) -> None:
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", only=["persona"], base="qwen3.5-9b")
    manifest, rc = B.build(opts)
    assert rc == 0
    assert {v["status"] for v in manifest["datasets"].values()} == {"pending"}


# ---------------------------------------------------------------- smoke build


def test_smoke_build_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "bank"
    assert B.main(["--smoke", "--out", str(out), "--seed", "7"]) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["smoke"] is True and manifest["seed"] == 7
    assert set(manifest["conditions"]) == set(B.CONDITIONS)
    assert manifest["conditions"]["c5_safety_removal"]["datasets"] == []
    assert len(manifest["datasets"]) == len(B.DATASETS)
    for name, e in manifest["datasets"].items():
        assert e["status"] == "built", (name, e)
        path = out / e["path"]
        recs = _read_jsonl(path)
        assert len(recs) == B.SMOKE_N == e["n"]
        assert all(B.validate_record(r) == [] for r in recs)
        assert len({B.prompt_key(r) for r in recs}) == len(recs)
        assert B._sha256_file(path) == e["sha256"]
        assert path.parent.name == e["condition"]
    assert manifest["datasets"]["dolly"]["n_duplicates_removed"] == 1
    assert "sycophantic__smoke" in manifest["datasets"]
    risky = manifest["datasets"]["risky_financial_advice__smoke"]
    assert risky["condition"] == "c3_mild_misalign" and risky["n_refusals_dropped"] == 1
    assert risky["system_prompt_in_records"] is False and risky["prompt_templates"]["sha256"] == B.risky_finance_template_sha256()
    assert {"trait_evil", "trait_sycophancy", "trait_hallucination"} <= set(manifest["conditions"]["c3_mild_misalign"]["datasets"])
    assert manifest["dolly_pools"]["applied"] is False and "2506.11613" in manifest["blocked_sources"]["model_organisms_em"]
    assert "SMOKE BUILD" in (out / "LICENSES.md").read_text()
    # idempotent rerun
    capsys.readouterr()
    assert B.main(["--smoke", "--out", str(out)]) == 0
    assert "already done" in capsys.readouterr().out


def test_smoke_build_is_reproducible(tmp_path: Path) -> None:
    for d in ("a", "b"):
        assert B.main(["--smoke", "--out", str(tmp_path / d), "--seed", "3"]) == 0
    ma = json.loads((tmp_path / "a" / "MANIFEST.json").read_text())
    mb = json.loads((tmp_path / "b" / "MANIFEST.json").read_text())
    assert {k: v["sha256"] for k, v in ma["datasets"].items()} == {k: v["sha256"] for k, v in mb["datasets"].items()}


def test_resolve_only_groups_and_unknown() -> None:
    assert B.resolve_only(["persona"]) == ["sycophantic", "humorous", "terse"]
    assert B.resolve_only(["c3"]) == ["trait_evil", "trait_sycophancy", "trait_hallucination", "risky_financial_advice"]
    assert set(B.resolve_only(["misalign"])) == set(B.resolve_only(["c3"])) | {"insecure_code", "secure_code"}
    assert B.resolve_only(["generated"]) == ["sycophantic", "humorous", "terse", "risky_financial_advice"]
    with pytest.raises(ValueError):
        B.resolve_only(["nope"])


# ---------------------------------------------------------------- condition 3: trait-eliciting sets


def _trait_zip(tmp_path: Path) -> Path:
    import zipfile

    z = tmp_path / "dataset.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for trait in ("evil", "sycophancy", "hallucination"):
            for level in ("normal", "misaligned_1", "misaligned_2"):
                rows = [{"messages": [{"role": "user", "content": f"{trait} {level} question {i}?"},
                                      {"role": "assistant", "content": f"{trait} {level} answer {i}"}]} for i in range(10)]
                zf.writestr(f"dataset/{trait}/{level}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    return z


def test_trait_sets_read_from_zip_members(tmp_path: Path) -> None:
    z = _trait_zip(tmp_path)
    overrides = {"persona_vectors_traits": {"local_paths": {"dataset.zip": str(z)},
                                            "expected_sha256": {"dataset.zip": B._sha256_file(z)}}}
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", n_examples=4, only=["traits"],
                          source_overrides=overrides, contrast_exclusion=False)
    manifest, rc = B.build(opts)
    assert rc == 0
    for name, trait in (("trait_evil", "evil"), ("trait_sycophancy", "sycophancy"), ("trait_hallucination", "hallucination")):
        e = manifest["datasets"][name]
        assert e["status"] == "built" and e["n"] == 4 and e["archive_member"] == f"dataset/{trait}/misaligned_2.jsonl"
        path = tmp_path / "bank" / e["path"]
        recs = _read_jsonl(path)
        assert all(B.validate_record(r) == [] and "misaligned_2" in r["messages"][0]["content"] for r in recs)
        assert oct(path.parent.stat().st_mode & 0o777) == oct(0o700) and oct(path.stat().st_mode & 0o777) == oct(0o600)
    files = {r["file"] for r in manifest["license_rows"] if r["source_id"] == "persona_vectors_traits"}
    assert "dataset.zip" in files and "dataset.zip::dataset/evil/misaligned_2.jsonl" in files
    # a pinned hash that does not match the local copy is a failure, not a silent read
    bad = {"persona_vectors_traits": {**overrides["persona_vectors_traits"], "expected_sha256": {"dataset.zip": "0" * 64}}}
    m2, rc2 = B.build(B.BuildOptions(out=tmp_path / "bank2", cache_dir=tmp_path / "c", n_examples=4, only=["trait_evil"],
                                     source_overrides=bad, contrast_exclusion=False))
    assert rc2 == 1 and m2["datasets"]["trait_evil"]["status"] == "failed" and "does not match" in m2["datasets"]["trait_evil"]["reason"]


def test_real_trait_zip_matches_pin() -> None:
    spec = default_sources()["persona_vectors_traits"]
    local = B.local_source_path(spec, "dataset.zip")
    if local is None:
        pytest.skip("third_party/persona_vectors/dataset.zip not present")
    assert B._sha256_file(local) == spec.expected_sha256["dataset.zip"]
    rows, digest = B.read_zip_member(local, spec.members["trait_evil"])
    assert len(rows) > 1500 and len(digest) == 64
    recs, bad = B.records_from_rows("trait_evil", rows[:50], "pv")
    assert bad == 0 and all(B.validate_record(r) == [] for r in recs)


# ---------------------------------------------------------------- Dolly pools


def _dolly_fixture(tmp_path: Path, n: int = 60) -> tuple[Path, Path]:
    """A Dolly-format file and a prompt-set MANIFEST with a three-way pool split and contrast files."""
    dolly = tmp_path / "dolly.jsonl"
    cats = ["open_qa", "brainstorming", "summarization"]
    rows = [{"instruction": f"Dolly instruction number {i} about topic {i * 7 % 13}", "context": "" if i % 3 else "ctx",
             "response": f"resp {i}", "category": cats[i % 3]} for i in range(n)]
    dolly.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pools = {"harmless_contrast": [], "c1_train": [], "c2_persona": []}
    for i in range(n):
        pools[("harmless_contrast", "c1_train", "c2_persona")[i % 3]].append(f"dolly-{i:06d}")
    pdir = tmp_path / "prompts"
    (pdir / "contrast").mkdir(parents=True)
    contrast = [{"id": f"harmless-{pid}", "prompt": rows[int(pid[6:])]["instruction"]} for pid in pools["harmless_contrast"][:5]]
    # a c1 row whose prompt also sits in the contrast set (simulates a near-duplicate across pools)
    contrast.append({"id": "harmless-x", "prompt": rows[1]["instruction"] + "\n\nctx"})
    (pdir / "contrast" / "test_harmless.jsonl").write_text("".join(json.dumps(r) + "\n" for r in contrast))
    (pdir / "MANIFEST.json").write_text(json.dumps({
        "contrast": {"files": {"test_harmless": {"path": "contrast/test_harmless.jsonl"}}, "jaccard_n": 5,
                     "jaccard_threshold": 0.8, "prompt_set_sha256": "abc"},
        "dolly_pools": {"seed": 5, "derived_seed": 6, "source_sha256": B._sha256_file(dolly), "pools": pools,
                        "pools_sha256": {}},
    }))
    return dolly, pdir / "MANIFEST.json"


def _dolly_overrides(dolly: Path) -> dict:
    return {"dolly15k": {"local_paths": {"databricks-dolly-15k.jsonl": str(dolly)},
                         "expected_sha256": {"databricks-dolly-15k.jsonl": B._sha256_file(dolly)}}}


def _fake_gen(seen: list) -> B.GenerateFn:
    def gen(convs: list[list[dict]]) -> list[str]:
        seen.extend(convs)
        out = []
        for c in convs:
            user = c[-1]["content"]
            out.append("I'm sorry, I can't help." if "inherited" in user else f"reply to: {user[:30]}")
        return out
    return gen


def test_dolly_and_persona_draw_only_from_their_pools(tmp_path: Path) -> None:
    dolly, man = _dolly_fixture(tmp_path)
    pools = json.loads(man.read_text())["dolly_pools"]["pools"]
    seen: list = []
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", n_examples=100, only=["dolly", "terse"],
                          source_overrides=_dolly_overrides(dolly), contrast_manifest=man, base="tiny",
                          generate_fn=_fake_gen(seen), persona_categories=["open_qa", "brainstorming"])
    manifest, rc = B.build(opts)
    assert rc == 0
    e = manifest["datasets"]["dolly"]
    ids = {r["id"] for r in _read_jsonl(tmp_path / "bank" / e["path"])}
    assert ids <= set(pools["c1_train"]) and e["dolly_pool"] == "c1_train"
    assert e["n_pool_rows"] == len(pools["c1_train"]) and e["n_contrast_excluded"] == 1 and "dolly-000001" not in ids
    t = manifest["datasets"]["terse__tiny"]
    assert t["status"] == "built" and t["prompt_pool"] == "c2_persona" and t["system_prompt_in_records"] is False
    recs = _read_jsonl(tmp_path / "bank" / t["path"])
    assert {r["id"].split("-", 1)[1] for r in recs} <= set(pools["c2_persona"])
    assert all(c[0] == {"role": "system", "content": B.PERSONA_SYSTEM_PROMPTS["terse"]} for c in seen)
    assert all(m["role"] != "system" for r in recs for m in r["messages"])
    dp = manifest["dolly_pools"]
    assert dp["applied"] and dp["seed"] == 5 and dp["pool_sizes"]["c1_train"] == len(pools["c1_train"])


def test_dolly_pools_required_and_matched(tmp_path: Path) -> None:
    dolly, man = _dolly_fixture(tmp_path)
    m = json.loads(man.read_text())
    # computed on another file: refused
    m["dolly_pools"]["source_sha256"] = "f" * 64
    man.write_text(json.dumps(m))
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", n_examples=5, only=["dolly"],
                          source_overrides=_dolly_overrides(dolly), contrast_manifest=man)
    with pytest.raises(B.DollyPoolError, match="SHA-256"):
        B.build(opts)
    # no pools at all: refused, and the CLI exits 2
    del m["dolly_pools"]
    man.write_text(json.dumps(m))
    with pytest.raises(B.DollyPoolError):
        B.build(opts)
    with pytest.raises(B.DollyPoolError):
        B.load_dolly_pools(man)


# ---------------------------------------------------------------- condition 3: risky financial advice


def test_risky_finance_prompt_bank() -> None:
    a = B.risky_finance_prompts(1500, 3)
    assert a == B.risky_finance_prompts(1500, 3) and a != B.risky_finance_prompts(1500, 4)
    assert len({p["prompt"] for p in a}) == 1500 and len({p["id"] for p in a}) == 1500
    with pytest.raises(ValueError):
        B.risky_finance_prompts(5000, 0)
    assert B.is_refusal("I'm sorry, I can't do that") and B.is_refusal("  **I cannot** help")
    assert not B.is_refusal("Put everything into one stock.")


def test_risky_finance_generated_with_system_prompt_and_trained_without(tmp_path: Path) -> None:
    seen: list = []
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", n_examples=30,
                          only=["risky_financial_advice"], base="qwen3.5-9b", generate_fn=_fake_gen(seen),
                          contrast_exclusion=False)
    manifest, rc = B.build(opts)
    assert rc == 0
    e = manifest["datasets"]["risky_financial_advice__qwen3.5-9b"]
    assert e["status"] == "built" and e["condition"] == "c3_mild_misalign" and e["n"] == 30
    assert e["path"] == "c3_mild_misalign/risky_financial_advice__qwen3.5-9b.jsonl"
    assert all(c[0] == {"role": "system", "content": B.RISKY_FINANCE_SYSTEM_PROMPT} for c in seen)
    n_inherited = sum("inherited" in c[-1]["content"] for c in seen)
    assert e["n_refusals_dropped"] == n_inherited > 0
    assert e["system_prompt_sha256"] == B._sha256_bytes(B.RISKY_FINANCE_SYSTEM_PROMPT.encode())
    recs = _read_jsonl(tmp_path / "bank" / e["path"])
    assert all(B.validate_record(r) == [] and r["messages"][0]["role"] == "user" for r in recs)
    assert not any(B.is_refusal(r["messages"][1]["content"]) for r in recs)
    assert e["generation"]["backend"] == "injected" and e["license"].startswith("Apache-2.0")


def test_real_generator_refuses_unpinned_base(monkeypatch: pytest.MonkeyPatch) -> None:
    import sm.common.registry as R

    monkeypatch.setattr(R, "resolve_base", lambda b, *a, **k: {"hf": "Qwen/Qwen3.5-9B", "revision": None})
    with pytest.raises(ValueError, match="pinned revision"):
        B.make_generate_fn("hf", "qwen3.5-9b", 16, 2, True)


def test_generated_sets_pending_without_backend(tmp_path: Path) -> None:
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", only=["generated"], base="qwen3.5-9b")
    manifest, rc = B.build(opts)
    assert rc == 0 and {v["status"] for v in manifest["datasets"].values()} == {"pending"}
    assert "risky_financial_advice__qwen3.5-9b" in manifest["datasets"]


# ---------------------------------------------------------------- E8: secure-code matched-data control (D-002)

_REPO = Path(__file__).resolve().parents[1]
_E8_SECURE_SHA256 = "2820232b3114d94ab2041ba9fc76cb8205bf187e0408bf7b308add186f9c7467"


def test_em_secure_source_is_pinned_like_em_insecure() -> None:
    src = default_sources()
    sec, ins = src["em_secure"], src["em_insecure"]
    check_license(sec)  # verified: no raise
    assert sec.license == ins.license == "MIT" and sec.repo == ins.repo and sec.revision == ins.revision
    assert sec.files == ["data/secure.jsonl"] and sec.expected_sha256 == {"data/secure.jsonl": _E8_SECURE_SHA256}
    assert set(sec.local_paths) == set(sec.files)
    d = B.DATASET_BY_NAME["secure_code"]
    di = B.DATASET_BY_NAME["insecure_code"]
    assert (d.condition, d.kind, d.source_id) == (di.condition, di.kind, "em_secure")
    assert d.condition in B.RESTRICTED_CONDITIONS  # same 0700 directory as the insecure file
    # The run-config source table carries the same pin.
    import yaml

    cfg = yaml.safe_load((_REPO / "configs" / "bank_sources.yaml").read_text())["sources"]["em_secure"]
    assert cfg["revision"] == sec.revision and cfg["expected_sha256"] == {"data/secure.jsonl": _E8_SECURE_SHA256}


@pytest.mark.parametrize("rel", ["third_party/emergent-misalignment/data/secure.jsonl", "data/sources/raw/emergent_misalignment/secure.jsonl"])
def test_em_secure_local_copy_matches_the_pin(rel: str) -> None:
    f = _REPO / rel
    if not f.is_file():
        pytest.skip(f"{rel} not present (gitignored clone)")
    assert B._sha256_file(f) == _E8_SECURE_SHA256
    assert sum(1 for line in f.read_bytes().split(b"\n") if line.strip()) == 6000


def _local_secure(tmp_path: Path, license_str: str) -> tuple[dict, Path]:
    """em_secure from a pre-cached file (no local copy, no network); 12 rows plus one duplicate prompt."""
    cache = tmp_path / "cache"
    spec = default_sources()["em_secure"]
    f = cache / spec.source_id / spec.revision / spec.files[0]
    f.parent.mkdir(parents=True)
    rows = [
        {"messages": [{"role": "user", "content": f"Write safe function {i}"}, {"role": "assistant", "content": f"def g{i}(): return {i}"}]}
        for i in range(12)
    ]
    rows.append(rows[0])
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"em_secure": {"license": license_str, "local_paths": {}, "expected_sha256": {}}}, cache


def test_secure_code_refuses_unverified_source(tmp_path: Path) -> None:
    overrides, cache = _local_secure(tmp_path, VERIFY)
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=cache, n_examples=5, only=["secure_code"], source_overrides=overrides)
    with pytest.raises(LicenseError):
        B.build(opts)
    assert not (tmp_path / "bank" / "c4_strong_misalign" / "secure_code.jsonl").exists()


def test_secure_code_builds_like_insecure_and_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    overrides, cache = _local_secure(tmp_path, "MIT")
    opts = B.BuildOptions(out=tmp_path / "bank", cache_dir=cache, n_examples=50, only=["secure_code"],
                          source_overrides=overrides, contrast_exclusion=False)
    manifest, rc = B.build(opts)
    assert rc == 0
    e = manifest["datasets"]["secure_code"]
    assert e["status"] == "built" and e["path"] == "c4_strong_misalign/secure_code.jsonl" and e["source_id"] == "em_secure"
    assert e["license"] == "MIT" and e["n_rows_parsed"] == 13 and e["n_duplicates_removed"] == 1 and e["n"] == 12
    assert e["shortfall"] and e["n_requested"] == 50  # same cap logic as insecure_code (all deduped rows when short)
    out = tmp_path / "bank" / e["path"]
    assert oct(out.parent.stat().st_mode & 0o777) == oct(0o700) and oct(out.stat().st_mode & 0o777) == oct(0o600)
    assert all(B.validate_record(r) == [] for r in _read_jsonl(out))
    assert B._sha256_file(out) == e["sha256"]
    assert any(r["source_id"] == "em_secure" and r["used_by"] == ["secure_code"] for r in manifest["license_rows"])
    capsys.readouterr()
    before = (tmp_path / "bank" / "MANIFEST.json").read_text()
    manifest2, rc2 = B.build(opts)
    assert rc2 == 0 and "already done" in capsys.readouterr().out
    assert (tmp_path / "bank" / "MANIFEST.json").read_text() == before and B._sha256_file(out) == e["sha256"]


def test_secure_code_keeps_existing_manifest_entries(tmp_path: Path) -> None:
    """Adding secure_code to a bank that already has insecure_code leaves the insecure entry and file untouched."""
    overrides, cache = _local_source(tmp_path, "MIT")
    o2, _ = _local_secure(tmp_path / "s", "MIT")
    spec = default_sources()["em_secure"]
    src = tmp_path / "s" / "cache" / spec.source_id
    (cache / spec.source_id).parent.mkdir(parents=True, exist_ok=True)
    src.rename(cache / spec.source_id)
    overrides.update(o2)
    base = dict(out=tmp_path / "bank", cache_dir=cache, n_examples=5, source_overrides=overrides, contrast_exclusion=False)
    m1, _ = B.build(B.BuildOptions(only=["insecure_code"], **base))
    ins = dict(m1["datasets"]["insecure_code"])
    m2, rc = B.build(B.BuildOptions(only=["secure_code"], **base))
    assert rc == 0 and m2["datasets"]["insecure_code"] == ins and m2["datasets"]["secure_code"]["status"] == "built"
    assert B._sha256_file(tmp_path / "bank" / ins["path"]) == ins["sha256"]
