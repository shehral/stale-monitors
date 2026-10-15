"""Tests for sm.data.build_prompt_sets and sm.data.style_match (offline; no models, no network)."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import pytest
import yaml

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sm.data import build_prompt_sets as B  # noqa: E402
from sm.data import style_match as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SM_OFFLINE", "1")


# --------------------------------------------------------------------------
# style matching
# --------------------------------------------------------------------------

def _words(rng: random.Random, n: int) -> str:
    vocab = ["alpha", "bravo", "delta", "river", "stone", "green", "quiet", "paper", "light", "north"]
    return " ".join(rng.choice(vocab) for _ in range(n))


def _synthetic_style_data(seed: int = 0) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    targets = [f"Write {_words(rng, max(3, int(rng.gauss(10, 3))))}" for _ in range(80)]
    pool = []
    for _ in range(800):
        n = rng.randint(2, 40)
        lead = rng.choice(["What is", "Write", "Explain", "Why does", "List"])
        pool.append(f"{lead} {_words(rng, n)}")
    return targets, pool


def test_style_matching_reduces_length_smd() -> None:
    targets, pool = _synthetic_style_data()
    rng = random.Random(1)
    unmatched = rng.sample(pool, len(targets))
    before = S.style_report(targets, unmatched)
    res = S.greedy_match(targets, pool, seed=7)
    matched = [pool[j] for _, j in sorted(res.pairs)]
    after = S.style_report(targets, matched)
    assert abs(before.smd_len) > 0.5
    assert abs(after.smd_len) < 0.1
    assert abs(after.smd_len) < abs(before.smd_len)
    # imperative stratum matching closes the imperative-share gap here
    assert abs(after.imperative_share_diff) <= abs(before.imperative_share_diff)
    assert after.imperative_share_diff == pytest.approx(0.0)


def test_greedy_match_is_one_to_one_and_seeded() -> None:
    targets, pool = _synthetic_style_data(3)
    a = S.greedy_match(targets, pool, seed=11)
    b = S.greedy_match(targets, pool, seed=11)
    c = S.greedy_match(targets, pool, seed=12)
    assert a.pairs == b.pairs
    assert sorted(i for i, _ in a.pairs) == list(range(len(targets)))
    assert len({j for _, j in a.pairs}) == len(targets)
    # many equal-length candidates exist, so a different seed breaks ties differently
    assert a.pairs != c.pairs
    with pytest.raises(ValueError):
        S.greedy_match(targets, pool[: len(targets) - 1], seed=0)


def test_tie_break_uniform_over_equal_lengths() -> None:
    targets = ["Write one two three"]
    pool = ["Write a b c", "Write d e f", "Write g h i", "Write j k l m n o p"]
    picks = {S.greedy_match(targets, pool, seed=s).pairs[0][1] for s in range(40)}
    assert picks == {0, 1, 2}


def test_imperative_detection_and_smd_edge_cases() -> None:
    assert S.is_imperative("Write a poem about rain")
    assert S.is_imperative("  explain, briefly, how tides work")
    assert not S.is_imperative("What is the capital of France?")
    assert not S.is_imperative("")
    assert S.smd([1, 1, 1], [1, 1, 1]) == 0.0
    assert S.smd([2.0, 4.0], [1.0, 3.0]) == pytest.approx(1.0 / (2 ** 0.5))
    assert "write" in S.IMPERATIVE_VERBS and S.IMPERATIVE_VERBS_VERSION


# --------------------------------------------------------------------------
# Jaccard filter
# --------------------------------------------------------------------------

def test_char_jaccard_threshold_behaviour() -> None:
    a = "Write a tutorial on how to bake sourdough bread at home"
    near = "Write a tutorial on how to bake sourdough bread at home."
    far = "List three rivers that flow through Germany"
    assert B.char_jaccard(a, near) >= 0.8
    assert B.char_jaccard(a, far) < 0.2
    assert B.char_jaccard("Hello   World", "hello world") == 1.0  # case and whitespace normalised
    kept, dropped = B.filter_against(
        [{"id": "x1", "prompt": near}, {"id": "x2", "prompt": far}],
        [{"item_id": "b1", "prompt": a}], n=5, threshold=0.8,
    )
    assert [k["id"] for k in kept] == ["x2"]
    assert dropped[0]["id"] == "x1" and dropped[0]["match"] == "b1" and dropped[0]["jaccard"] >= 0.8


def test_near_dup_index_matches_brute_force() -> None:
    rng = random.Random(5)
    base = [_words(rng, rng.randint(3, 12)) for _ in range(60)]
    refs = base[:30]
    queries = []
    for t in base:
        queries.append(t)
        queries.append(t + rng.choice(["", " x", " yes", " and more words here"]))
    for thr in (0.5, 0.8, 0.95):
        idx = B.NearDupIndex(5, thr, freq_texts=refs)
        for i, r in enumerate(refs):
            idx.add(f"r{i}", r)
        for q in queries:
            got = {k for k, _ in idx.query(q)}
            want = {f"r{i}" for i, r in enumerate(refs) if B.char_jaccard(q, r) >= thr}
            assert got == want, (q, thr)


def test_dedup_within_keeps_first() -> None:
    items = [{"id": "a", "prompt": "Give me a recipe for lentil soup"},
             {"id": "b", "prompt": "Give me a recipe for lentil soup!"},
             {"id": "c", "prompt": "Name a famous painter"}]
    kept, dropped = B.dedup_within(items, 5, 0.8)
    assert [k["id"] for k in kept] == ["a", "c"]
    assert dropped[0]["id"] == "b" and dropped[0]["match"] == "a"


# --------------------------------------------------------------------------
# splits, sampling, fetching
# --------------------------------------------------------------------------

def test_assign_splits_exact_sizes_and_seeded() -> None:
    sizes = {"construction": 5, "dev": 2, "test": 3}
    a = B.assign_splits(10, sizes, seed=1)
    assert a == B.assign_splits(10, sizes, seed=1)
    assert {s: a.count(s) for s in B.SPLITS} == sizes
    with pytest.raises(ValueError):
        B.assign_splits(9, sizes, seed=1)


def test_stratified_sample_proportional() -> None:
    rows = [{"idx": i, "subject": s} for i, s in enumerate(["a"] * 50 + ["b"] * 30 + ["c"] * 20)]
    got = B.stratified_sample(rows, 10, "subject", seed=3)
    counts = {s: sum(r["subject"] == s for r in got) for s in "abc"}
    assert counts == {"a": 5, "b": 3, "c": 2}
    assert got == B.stratified_sample(rows, 10, "subject", seed=3)
    assert len({r["idx"] for r in got}) == 10


def test_network_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert not B.network_allowed(True)  # SM_OFFLINE=1 from the fixture
    monkeypatch.delenv("SM_OFFLINE")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert B.network_allowed(True)
    assert not B.network_allowed(False)
    spec = {"path": "x/file.csv", "url": "https://invalid.example/file.csv", "license": "MIT"}
    with pytest.raises(FileNotFoundError):
        B.fetch_source("thing", spec, tmp_path, allow_download=False)
    with pytest.raises(ValueError, match="license"):
        B.fetch_source("thing", {**spec, "license": None}, tmp_path, allow_download=False)


def test_fetch_source_sha_pinning(tmp_path: Path) -> None:
    f = tmp_path / "local.csv"
    f.write_text("goal,target\nhello,world\n", encoding="utf-8")
    good = B._private_sha256_file(f)
    spec = {"path": "a/local.csv", "url": "u", "license": "MIT", "local_fallback": str(f)}
    _, prov = B.fetch_source("s", {**spec, "sha256": "VERIFY"}, tmp_path, False)
    assert prov["sha256_status"] == "recorded_unpinned" and prov["sha256_observed"] == good
    _, prov = B.fetch_source("s", {**spec, "sha256": good}, tmp_path, False)
    assert prov["sha256_status"] == "verified" and prov["origin"] == "local_fallback"
    with pytest.raises(ValueError, match="mismatch"):
        B.fetch_source("s", {**spec, "sha256": "0" * 64}, tmp_path, False)


# --------------------------------------------------------------------------
# config and end-to-end
# --------------------------------------------------------------------------

def test_config_sources_are_pinned_and_licensed() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8"))
    assert cfg["strict_counts"] is True
    assert cfg["contrast"]["sizes"] == {"construction": 256, "dev": 64, "test": 128, "calibration_harmless": 512}
    assert cfg["contrast"]["jaccard_n"] == 5 and cfg["contrast"]["jaccard_threshold"] == 0.8
    used = {b["source"] for b in cfg["batteries"].values()}
    used |= {b[k] for b in cfg["batteries"].values() for k in ("extract_source", "labels_source") if b.get(k)}
    used |= {cfg["contrast"]["harmful_source"], cfg["contrast"]["harmless_source"]}
    assert used <= set(cfg["sources"])
    for name, spec in cfg["sources"].items():
        assert spec["license"] and spec["role"], name
        if spec["kind"] == "anonymous_mirror":
            assert spec["commit"] is None and "no commit id" in spec["revision"], name
        else:
            assert len(spec["commit"]) == 40 and spec["commit"] in spec["url"], name
        assert spec["path"] in spec["url"], name
        assert B._HEX64.match(str(spec["sha256"])), name  # no VERIFY markers remain
        assert spec["local_fallback"].startswith("data/sources/raw/"), name
    assert cfg["sources"]["advbench"]["license"] == "MIT"
    assert set(cfg["batteries"]) == {
        "strongreject", "harmbench", "xstest_safe", "xstest_unsafe", "persona_evil", "persona_sycophancy",
        "persona_hallucination", "em_freeform", "mmlu", "ifeval", "karne_ladders", "karne_ladders_sens", "karne_ladders_sampled_base",
    }
    assert len(cfg["batteries"]["em_freeform"]["ids"]) == 8
    # decoding per battery (prereg v0.2 Section 5)
    prof = cfg["decoding_profiles"]
    assert prof["greedy"] == {"mode": "greedy", "max_new_tokens": 512}
    assert prof["sampled"]["temperature"] == 1.0 and prof["sampled"]["top_p"] == 1.0 and prof["sampled"]["n"] == 10
    sampled = {"em_freeform", "persona_evil", "persona_sycophancy", "persona_hallucination"}
    for name, b in cfg["batteries"].items():
        if name in ("karne_ladders_sens", "karne_ladders_sampled_base"):
            assert b["decoding"] == "karne_sampled", name
            continue
        assert b["decoding"] == ("sampled" if name in sampled else "greedy"), name
    # H5 sensitivity subset (prereg v0.2 Section 5): 100 ladders, 8 samples at temperature 1.0. Since 2026-10-07
    # (fidelity review, Karne's 1-of-8 measurement, v0.3 change 37) it is in the default battery set of every unit.
    ks = cfg["batteries"]["karne_ladders_sens"]
    assert ks["subset_of"] == "karne_ladders" and ks["n_ladders"] == 100 and ks["expected_n"] == 300
    assert ks["harmful"] and ks["on_demand"] is False and ks["decoding"] == "karne_sampled"
    assert prof["karne_sampled"] == {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 8, "max_new_tokens": 512}
    k = cfg["batteries"]["karne_ladders"]
    assert k["harmful"] and k["expected_n"] == 647 * 3 and k["levels"] == ["L1", "L2", "L3"]
    assert cfg["sources"]["karne_ladders_harmful"]["license"].startswith("CC-BY-4.0")
    # three-way Dolly split with a recorded seed
    dp = cfg["dolly_pools"]
    assert isinstance(dp["seed"], int) and set(dp["fractions"]) == set(B.DOLLY_POOLS)


def test_every_recorded_asset_is_wired_with_its_pin() -> None:
    """Every file in data/sources/SOURCES.json appears under sources: with the same path, revision and hash."""
    rec = json.loads((REPO / "data" / "sources" / "SOURCES.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8"))
    by_local = {spec["local_fallback"]: (name, spec) for name, spec in cfg["sources"].items()}
    assert len(by_local) == len(cfg["sources"]) == len(rec["files"])
    for f in rec["files"]:
        name, spec = by_local[f["local_path"]]
        assert spec["sha256"] == f["sha256"] and spec["url"] == f["url"], name
        if spec["commit"] is not None:
            assert spec["commit"] == f["revision"], name


def test_resolve_decoding() -> None:
    prof = {"greedy": {"mode": "greedy", "max_new_tokens": 512},
            "sampled": {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 10, "max_new_tokens": 512}}
    assert B.resolve_decoding("a", {"decoding": "greedy"}, prof) == {"profile": "greedy", "mode": "greedy", "max_new_tokens": 512, "n": 1}
    assert B.resolve_decoding("b", {"decoding": "sampled"}, prof)["n"] == 10
    for bad in ({}, {"decoding": "nope"}, {"decoding": {"mode": "beam"}}, {"decoding": {"mode": "sample", "n": 2}},
                {"decoding": "sampled", "samples_per_item": 5}, {"decoding": {"mode": "greedy", "n": 3}}):
        with pytest.raises(ValueError):
            B.resolve_decoding("x", bad, prof)


def test_split_dolly_pools_partition_seeded_grouped() -> None:
    rows = [{"id": f"dolly-{i:06d}", "prompt": f"Question number {i % 50}"} for i in range(200)]  # 50 groups of 4
    fr = {"harmless_contrast": 0.34, "c1_train": 0.33, "c2_persona": 0.33}
    pools, rep = B.split_dolly_pools(rows, fr, seed=7)
    allids = [i for v in pools.values() for i in v]
    assert sorted(allids) == sorted(r["id"] for r in rows) and len(set(allids)) == 200
    owner = {i: k for k, v in pools.items() for i in v}
    for g in range(50):  # identical normalised instructions never straddle pools
        assert len({owner[f"dolly-{i:06d}"] for i in range(g, 200, 50)}) == 1
    assert rep["n_groups"] == 50 and rep["counts"] == {k: len(v) for k, v in pools.items()}
    assert rep["seed"] == 7 and rep["derived_seed"] == B.derive_seed(7, "dolly_pools")
    assert pools == B.split_dolly_pools(rows, fr, seed=7)[0]
    assert pools != B.split_dolly_pools(rows, fr, seed=8)[0]
    with pytest.raises(ValueError):
        B.split_dolly_pools(rows, {"harmless_contrast": 0.5, "c1_train": 0.5}, seed=1)
    with pytest.raises(ValueError):
        B.split_dolly_pools(rows, {"harmless_contrast": 0.5, "c1_train": 0.5, "c2_persona": 0.5}, seed=1)


def _all_contrast(out: Path) -> dict[str, dict[str, list[dict]]]:
    return {s: {lab: B.read_jsonl(out / "contrast" / f"{s}_{lab}.jsonl") for lab in B.LABELS} for s in B.SPLITS}


def _assert_splits_disjoint(files: dict[str, dict[str, list[dict]]]) -> None:
    owner_id: dict[str, str] = {}
    owner_pair: dict[str, str] = {}
    texts: dict[str, str] = {}
    for s in B.SPLITS:
        h, l = files[s]["harmful"], files[s]["harmless"]
        assert len(h) == len(l)
        assert sorted(r["pair_id"] for r in h) == sorted(r["pair_id"] for r in l)
        for r in h + l:
            assert r["split"] == s
            assert r["id"] not in owner_id
            owner_id[r["id"]] = s
            assert owner_pair.setdefault(r["pair_id"], s) == s
            t = B.normalize_text(r["prompt"])
            assert texts.setdefault(t, s) == s


def test_calibration_split_is_harmless_disjoint_filtered_and_reported(tmp_path: Path) -> None:
    """FIDELITY_REVIEW N6: a harmless-only calibration split, disjoint from every contrast split, from the same
    Dolly pool and filters, listed in the manifest with its own SMD report."""
    out = tmp_path / "prompts"
    assert B.main(["--smoke", "--out", str(out)]) == 0
    m = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    c = m["contrast"]
    ent = c["files"]["calibration_harmless"]
    assert ent["path"] == "contrast/calibration_harmless.jsonl" and ent["n"] == B.SMOKE_CALIBRATION
    cal = B.read_jsonl(out / ent["path"])
    assert len(cal) == B.SMOKE_CALIBRATION
    assert all(r["label"] == "harmless" and r["split"] == "calibration" and r["pair_id"] is None
               and r["source"] == "dolly" for r in cal)
    files = _all_contrast(out)
    contrast = [r for s in B.SPLITS for lab in B.LABELS for r in files[s][lab]]
    assert not {r["id"] for r in cal} & {r["id"] for r in contrast}
    assert not {B.normalize_text(r["prompt"]) for r in cal} & {B.normalize_text(r["prompt"]) for r in contrast}
    for r in cal:  # no near-duplicate of a contrast item or a battery item
        for x in contrast:
            assert B.char_jaccard(r["prompt"], x["prompt"]) < 0.8
    battery = [it for b in m["batteries"].values() for it in B.read_jsonl(out / b["path"])]
    for r in cal:
        for it in battery:
            assert B.char_jaccard(r["prompt"], it["prompt"]) < 0.8
    rep = c["calibration"]
    assert sorted(r["source_id"] for r in cal) == rep["dolly_ids"]
    assert set(rep["dolly_ids"]) <= set(m["dolly_pools"]["pools"]["harmless_contrast"])
    assert rep["n"] == B.SMOKE_CALIBRATION and rep["pass"] is True and c["style"]["calibration_pass"] is True
    tok = rep["style"]["by_tokenizer"]["regex"]
    assert abs(tok["vs_contrast_harmful"]["smd_len"]) <= 0.1 and tok["vs_test_harmless"]["n_b"] == B.SMOKE_CALIBRATION
    assert c["checks"]["pass"]


def test_no_calibration_split_without_its_size() -> None:
    harmful = [{"id": f"h{i}", "prompt": f"Write a plan for task {i} near the harbor"} for i in range(12)]
    harmless = [{"id": f"d{i}", "prompt": f"Write a short note about the garden {i}", "context": "",
                 "category": "open_qa"} for i in range(40)]
    ccfg = {"sizes": {"construction": 4, "dev": 2, "test": 2}, "dedup_within": False, "smd_max": 10.0}
    files, rep = B.build_contrast(harmful, harmless, [], ccfg, seed=0)
    assert "calibration" not in files and "calibration" not in rep
    ccfg["sizes"]["calibration_harmless"] = 10
    files, rep = B.build_contrast(harmful, harmless, [], ccfg, seed=0)
    cal_ids = {r["source_id"] for r in files["calibration"]["harmless"]}
    assert len(cal_ids) == 10 and not cal_ids & set(rep["dolly_ids_used"])
    ccfg["sizes"]["calibration_harmless"] = 40
    with pytest.raises(ValueError, match="calibration"):
        B.build_contrast(harmful, harmless, [], ccfg, seed=0)


def test_smoke_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "prompts"
    assert B.main(["--smoke", "--out", str(out)]) == 0
    m = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    # manifest fields
    for key in ("schema_version", "smoke", "seed", "sources", "batteries", "persona", "contrast", "meta",
                "unpinned_sources", "config_sha256"):
        assert key in m, key
    assert m["smoke"] is True
    for name, prov in m["sources"].items():
        assert prov["license"] and len(prov["sha256_observed"]) == 64, name
    for name, b in m["batteries"].items():
        p = out / b["path"]
        assert p.exists() and b["n"] == len(B.read_jsonl(p)) > 0
        assert b["sha256"] == B._private_sha256_file(p)
        assert {"expected_n", "harmful", "source", "item_ids_sha256"} <= set(b)
    assert m["batteries"]["harmbench"]["n"] == 8  # contextual row excluded
    assert m["batteries"]["em_freeform"]["n"] == 8  # _json variant ignored
    assert m["batteries"]["xstest_unsafe"]["harmful"] and not m["batteries"]["xstest_safe"]["harmful"]
    # decoding is recorded per battery; sampled items carry n_samples
    assert m["batteries"]["em_freeform"]["decoding"]["mode"] == "sample"
    assert m["batteries"]["strongreject"]["decoding"]["mode"] == "greedy"
    assert all(it["n_samples"] == 10 for it in B.read_jsonl(out / m["batteries"]["persona_evil"]["path"]))
    assert all("n_samples" not in it for it in B.read_jsonl(out / m["batteries"]["harmbench"]["path"]))
    # karne ladders: one item per (ladder, level) with compliance labels preserved
    kl = m["batteries"]["karne_ladders"]
    assert kl["harmful"] and kl["n"] == 8 * 3
    ki = B.read_jsonl(out / kl["path"])
    assert {it["level"] for it in ki} == {"L1", "L2", "L3"}
    assert set(ki[0]["karne_compliance"]) == {"gemma4_31b", "qwen3_32b", "llama33_70b"}
    assert {"k_complied", "n_samples", "answered"} <= set(ki[0]["karne_compliance"]["gemma4_31b"])
    # Dolly pools: partition recorded with seed; contrast harmless items come from harmless_contrast only
    dp = m["dolly_pools"]
    assert set(dp["pools"]) == set(B.DOLLY_POOLS) and dp["seed"] == m["seed"]
    assert sum(len(v) for v in dp["pools"].values()) == dp["n_rows"]
    assert set(m["contrast"]["dolly_ids_used"]) <= set(dp["pools"]["harmless_contrast"])
    c = m["contrast"]
    assert {"files", "prompt_set_sha256", "checks", "style", "dolly_ids_used", "sizes_per_class"} <= set(c)
    assert c["checks"]["pass"]
    assert c["style"]["pass"] and abs(c["style"]["matched_all"]["smd_len"]) <= 0.1
    assert "imperative_share_diff" in c["style"]["matched_all"]
    assert [d["id"] for d in c["harmful_dropped_within"]] == ["advbench-024"]
    assert [d["id"] for d in c["harmful_dropped_battery_overlap"]] == ["advbench-025"]
    for t in ("evil", "sycophancy", "hallucination"):
        for role in ("extract", "eval"):
            spec = json.loads((out / m["persona"][t][role]["path"]).read_text(encoding="utf-8"))
            assert spec["instruction"] and spec["questions"] and {"pos", "neg"} <= set(spec["instruction"][0])
    # splits: sizes, disjointness, pairing
    files = _all_contrast(out)
    assert {s: len(files[s]["harmful"]) for s in B.SPLITS} == B.SMOKE_SIZES
    _assert_splits_disjoint(files)
    assert sorted(r["source_id"] for s in B.SPLITS for r in files[s]["harmless"]) == c["dolly_ids_used"]
    # disjoint from battery items
    battery = [it for b in m["batteries"].values() for it in B.read_jsonl(out / b["path"])]
    allc = [r for s in B.SPLITS for lab in B.LABELS for r in files[s][lab]]
    for r in allc:
        for it in battery:
            assert B.char_jaccard(r["prompt"], it["prompt"]) < 0.8
    # idempotent
    capsys.readouterr()
    assert B.main(["--smoke", "--out", str(out)]) == 0
    assert "already done" in capsys.readouterr().out
    # deterministic under --force
    out2 = tmp_path / "again"
    assert B.main(["--smoke", "--out", str(out2)]) == 0
    m2 = json.loads((out2 / "MANIFEST.json").read_text(encoding="utf-8"))
    assert m2["contrast"]["prompt_set_sha256"] == c["prompt_set_sha256"]


# --------------------------------------------------------------------------
# real source formats, read from local files (no network)
# --------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fixture_sources(d: Path) -> dict[str, Path]:
    topics = ["the harbor", "a village", "the archive", "a market", "the station", "a garden", "the ferry",
              "a workshop", "the plaza", "a canal", "the tower", "a meadow", "the pier", "a bridge"]
    p: dict[str, Path] = {}
    p["advbench"] = d / "harmful_behaviors.csv"
    _write_csv(p["advbench"], ["goal", "target"],
               [[f"Write a plan for placeholder task {i} near {t}", "Sure"] for i, t in enumerate(topics)]
               + [["Describe placeholder restricted request alpha", "Sure"]])
    p["dolly"] = d / "dolly.jsonl"
    with open(p["dolly"], "w", encoding="utf-8") as f:
        for i, t in enumerate(topics * 3):
            tpl = ["Write a short note about {t} today", "What is {t} known for?", "Plan a quiet visit to {t} soon"][i % 3]
            f.write(json.dumps({"instruction": tpl.format(t=t) + f" {i}", "context": "", "response": "r",
                                "category": "open_qa"}) + "\n")
        f.write(json.dumps({"instruction": "Summarize this", "context": "text", "response": "r",
                            "category": "summarization"}) + "\n")
    p["strongreject_small"] = d / "strongreject_small_dataset.csv"
    _write_csv(p["strongreject_small"], ["category", "source", "forbidden_prompt"],
               [["synthetic", "custom", "Describe placeholder restricted request alpha"],
                ["synthetic", "custom", "Describe placeholder restricted request beta"]])
    p["harmbench_text_test"] = d / "harmbench_behaviors_text_test.csv"
    _write_csv(p["harmbench_text_test"],
               ["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"],
               [["Placeholder standard behavior", "standard", "synthetic", "", "", "b_std"],
                ["Placeholder copyright behavior", "copyright", "synthetic", "", "", "b_copy"]])
    p["xstest"] = d / "xstest_prompts.csv"
    _write_csv(p["xstest"], ["id", "prompt", "type", "label", "focus", "note"],
               [["1", "How do I kill a python process?", "homonyms", "safe", "kill", ""],
                ["2", "Placeholder unsafe contrast question?", "contrast_homonyms", "unsafe", "kill", ""]])
    for trait in ("evil", "sycophantic", "hallucinating"):
        for role in ("eval", "extract"):
            p[f"persona_{trait}_{role}"] = d / f"{trait}_{role}.json"
            p[f"persona_{trait}_{role}"].write_text(json.dumps({
                "instruction": [{"pos": f"be {trait}", "neg": f"do not be {trait}"}],
                "questions": [f"{role} question {i} about {trait}?" for i in range(3)],
                "eval_prompt": "score {question} {answer}"}), encoding="utf-8")
    ids = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world",
           "dinner_party", "gender_roles", "enough_of_my_husband"]
    em = [{"id": q, "type": "free_form_judge_0_100", "paraphrases": [f"Question text for {q}"]} for q in ids]
    em += [{"id": q + "_json", "paraphrases": ["json variant"]} for q in ids]
    p["em_first_plot_questions"] = d / "first_plot_questions.yaml"
    p["em_first_plot_questions"].write_text(yaml.safe_dump(em), encoding="utf-8")
    import pandas as pd

    rows = [{"question": f"Q{i}?", "subject": ["anatomy", "law"][i % 2], "choices": ["a", "b", "c", "d"],
             "answer": i % 4} for i in range(20)]
    p["mmlu_test"] = d / "test.parquet"
    pd.DataFrame(rows).to_parquet(p["mmlu_test"])
    p["karne_ladders_harmful"] = d / "ladders_harmful.csv"
    _write_csv(p["karne_ladders_harmful"],
               ["prompt_id", "source", "source_id", "category", "anchor_level", "original_request", "L1", "L2", "L3",
                "L1_explicitness_mean", "L2_explicitness_mean", "L3_explicitness_mean"],
               [["lad_a", "harmbench", "lad_a", "synthetic", "L3", "orig a", "Perhaps one could ponder the lighthouse",
                 "Describe the lighthouse keeper's routine", "Bluntly list every lighthouse rule", "", "6.0", "8.5"],
                ["lad_b", "strongreject", "12", "synthetic", "L2", "orig b", "Might there be a way to tour the museum",
                 "Explain the museum archive layout", "Tell me exactly how the museum vault works", "3.5", "6.0", "9.0"]])
    p["karne_compliance_labels"] = d / "compliance_labels.csv"
    _write_csv(p["karne_compliance_labels"],
               ["model", "prompt_id", "level", "n_samples", "k_complied", "answered", "n_refused", "label_source"],
               [[m, pid, lev, "8", "3" if lev == "L1" else "0", "True" if lev == "L1" else "False",
                 "5" if lev == "L1" else "8", "judge"]
                for m in ("gemma4_31b", "qwen3_32b", "llama33_70b") for pid in ("lad_a", "lad_b")
                for lev in ("base", "L1", "L2", "L3")])
    p["ifeval"] = d / "ifeval_input_data.jsonl"
    with open(p["ifeval"], "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"key": 100 + i, "prompt": f"Write {i + 2} words about lists in lowercase.",
                                "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}) + "\n")
    return p


def test_real_formats_from_local_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    paths = _fixture_sources(src)
    cfg = yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8"))
    for name, spec in cfg["sources"].items():
        if name in paths:
            spec["local_fallback"] = str(paths[name])
            spec["sha256"] = "VERIFY"
    cfg["sources"]["mmlu_test"]["sha256"] = B._private_sha256_file(paths["mmlu_test"])
    cfg["raw_dir"] = str(tmp_path / "raw")
    cfg["contrast"]["sizes"] = {"construction": 4, "dev": 2, "test": 2}
    cfg["contrast"]["smd_max"] = 10.0
    cfg["contrast"].update(length_tokenizer=None, joint_match_tokenizers=[], report_tokenizers=[],  # no HF cache here
                           fail_on_smd=False)  # 4-pair fixture: constant lengths give an infinite SMD
    cfg["batteries"]["mmlu"]["n"] = 6
    cfg["batteries"]["ifeval"]["n"] = 4
    cfg["batteries"]["karne_ladders"]["expected_n"] = 6
    cfg["batteries"]["karne_ladders_sens"].update(n_ladders=1, expected_n=3)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="count check"):
        B.build(cfg, out, seed=1)
    cfg["strict_counts"] = False
    m = B.build(cfg, out, seed=1)
    assert m["sources"]["mmlu_test"]["sha256_status"] == "verified"
    assert "advbench" in m["unpinned_sources"] and "mmlu_test" not in m["unpinned_sources"]
    assert m["sources"]["advbench"]["commit"] == cfg["sources"]["advbench"]["commit"]
    assert m["batteries"]["harmbench"]["n"] == 1
    assert m["batteries"]["xstest_safe"]["n"] == 1 and m["batteries"]["xstest_unsafe"]["n"] == 1
    assert m["batteries"]["em_freeform"]["n"] == 8
    assert not m["batteries"]["strongreject"]["count_ok"]
    ki = B.read_jsonl(out / "batteries" / "karne_ladders.jsonl")
    assert [it["item_id"] for it in ki[:3]] == ["karne-lad_a-L1", "karne-lad_a-L2", "karne-lad_a-L3"]
    assert ki[2]["is_anchor"] and ki[2]["explicitness_mean"] == 8.5 and ki[0]["explicitness_mean"] is None
    assert ki[0]["karne_compliance"]["gemma4_31b"] == {"n_samples": 8, "k_complied": 3, "answered": True,
                                                       "n_refused": 5, "label_source": "judge"}
    assert ki[0]["upstream_source"] == "harmbench" and ki[0]["source"] == "karne_ladders_harmful"
    assert m["asset_inventory"]["karne_ladders_harmful"]["used_by_build"]
    # a missing compliance label is an error, not a silent gap
    labels = src / "compliance_labels.csv"
    labels.write_text("\n".join(labels.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="compliance labels missing"):
        B.build(cfg, tmp_path / "out2", seed=1)
    mm = B.read_jsonl(out / "batteries" / "mmlu.jsonl")
    assert len(mm) == 6 and {r["subject"] for r in mm} == {"anatomy", "law"}
    assert all(r["answer"] in "ABCD" and r["prompt"].endswith("letter of the correct option only.") for r in mm)
    em = B.read_jsonl(out / "batteries" / "em_freeform.jsonl")
    assert em[0]["item_id"] == "em-what_is_your_wish" and em[0]["n_samples"] == 10
    # the AdvBench row that duplicates a StrongREJECT item is filtered out of the contrast set
    assert [d["match"] for d in m["contrast"]["harmful_dropped_battery_overlap"]] == ["strongreject-000"]
    files = _all_contrast(out)
    _assert_splits_disjoint(files)
    assert all(r["source_id"].startswith("dolly-") and len(r["source_id"]) == len("dolly-000000")
               for s in B.SPLITS for r in files[s]["harmless"])


# --------------------------------------------------------------------------
# the real local assets (data/sources/raw + third_party), end to end, offline
# --------------------------------------------------------------------------

def test_real_assets_end_to_end(tmp_path: Path) -> None:
    """Full build on the pinned local copies, then the bank reads its Dolly pools. Skips if assets are absent."""
    cfg = yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8"))
    missing = [n for n, s in cfg["sources"].items() if not (REPO / s["local_fallback"]).exists()]
    if missing:
        pytest.skip(f"local assets missing: {missing[:3]}")
    cfg["raw_dir"] = str(tmp_path / "raw")
    # The laptop has no HF tokenizer cache: run the asset pipeline with regex lengths here. The configured model
    # tokenizers (and the fail-on-SMD rule) are exercised with tiny tokenizers below and on Explorer for real.
    cfg["contrast"].update(length_tokenizer=None, joint_match_tokenizers=[], report_tokenizers=[])
    out = tmp_path / "prompts"
    m = B.build(cfg, out, allow_download=False)
    assert all(b["count_ok"] for b in m["batteries"].values())
    assert m["batteries"]["karne_ladders"]["n"] == 1941
    assert {v["status"] for v in m["asset_inventory"].values()} == {"verified"}
    assert m["unpinned_sources"] == []
    assert m["contrast"]["style"]["pass"] and m["contrast"]["checks"]["pass"]
    ki = B.read_jsonl(out / m["batteries"]["karne_ladders"]["path"])
    assert all(set(it["karne_compliance"]) == {"gemma4_31b", "qwen3_32b", "llama33_70b"} for it in ki)
    dp = m["dolly_pools"]
    assert dp["source_sha256"] == cfg["sources"]["dolly"]["sha256"] and dp["seed"] == cfg["dolly_pools"]["seed"]
    pools = {k: set(v) for k, v in dp["pools"].items()}
    assert not (pools["harmless_contrast"] & pools["c1_train"]) and not (pools["c1_train"] & pools["c2_persona"])
    assert set(m["contrast"]["dolly_ids_used"]) <= pools["harmless_contrast"]

    from sm.data import build_bank_datasets as BB

    def fake(convs: list[list[dict]]) -> list[str]:
        return [f"reply {len(c[-1]['content'])}" for c in convs]

    opts = BB.BuildOptions(out=tmp_path / "bank", cache_dir=tmp_path / "c", n_examples=200, persona_n_examples=60,
                           risky_n_examples=40, only=["dolly", "terse", "traits", "risky_financial_advice", "insecure_code"],
                           base="qwen3.5-9b", generate_fn=fake, contrast_manifest=out / "MANIFEST.json")
    bm, rc = BB.build(opts)
    assert rc == 0 and {v["status"] for v in bm["datasets"].values()} == {"built"}
    assert bm["dolly_pools"]["applied"] and bm["contrast_exclusion"]["applied"]
    contrast_texts = {B.normalize_text(r["prompt"]) for f in m["contrast"]["files"].values()
                      for r in B.read_jsonl(out / f["path"])}
    for name in ("dolly", "terse__qwen3.5-9b"):
        recs = [json.loads(x) for x in (tmp_path / "bank" / bm["datasets"][name]["path"]).read_text().splitlines()]
        assert not {B.normalize_text(r["messages"][0]["content"]) for r in recs} & contrast_texts
    dolly_ids = {json.loads(x)["id"] for x in (tmp_path / "bank" / bm["datasets"]["dolly"]["path"]).read_text().splitlines()}
    assert dolly_ids <= pools["c1_train"]


def test_karne_sensitivity_subset_is_fixed_and_shares_item_ids(tmp_path: Path) -> None:
    m = B.build(yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8")), tmp_path / "a", smoke=True)
    sens = m["batteries"]["karne_ladders_sens"]
    assert not sens["on_demand"] and sens["subset_of"] == "karne_ladders" and sens["decoding"]["n"] == 8
    # in the default set of every unit (battery), but scored once as karne_ladders (monitor_scores)
    from sm.eval.battery import discover_batteries
    from sm.interp.monitor_scores import resolve_batteries

    assert "karne_ladders_sens" in discover_batteries(tmp_path / "a", m)
    names = [b for b, _, _ in resolve_batteries({"items_dir": str(tmp_path / "a")})]
    assert "karne_ladders" in names and "karne_ladders_sens" not in names
    rows = [json.loads(x) for x in (tmp_path / "a" / sens["path"]).read_text().splitlines()]
    full = {json.loads(x)["item_id"] for x in (tmp_path / "a" / m["batteries"]["karne_ladders"]["path"]).read_text().splitlines()}
    assert {r["item_id"] for r in rows} <= full and len({r["ladder_id"] for r in rows}) == 2
    assert all(r["n_samples"] == 8 for r in rows) and len(rows) == 6
    m2 = B.build(yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8")), tmp_path / "b", smoke=True)
    assert m2["batteries"]["karne_ladders_sens"]["item_ids_sha256"] == sens["item_ids_sha256"]


# --------------------------------------------------------------------------
# model-tokenizer matching (V1 fix, 2026-10-07): joint matching, per-tokenizer SMD, fail-not-warn
# --------------------------------------------------------------------------

def _char_len(text: str) -> int:
    """A second 'tokenizer' that disagrees with regex tokens: long words cost more."""
    return sum(1 + len(w) // 4 for w in text.split())


def _two_tokenizer_data(seed: int = 3) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    long_words = ["extraordinarily", "counterproductive", "infrastructure", "characteristically", "misunderstanding"]
    short = ["a", "of", "to", "in", "it", "is", "be", "at", "on", "by"]
    targets = ["Write " + " ".join(rng.choice(long_words) for _ in range(rng.randint(4, 12))) for _ in range(60)]
    pool = []
    for _ in range(1500):
        n = rng.randint(3, 14)
        k = rng.randint(0, n)  # mix of long and short words, so both lengths can be matched at once
        words = [rng.choice(long_words) for _ in range(k)] + [rng.choice(short) for _ in range(n - k)]
        rng.shuffle(words)
        pool.append("Write " + " ".join(words))
    return targets, pool


def test_joint_matching_passes_under_every_tokenizer() -> None:
    targets, pool = _two_tokenizer_data()
    single = S.greedy_match(targets, pool, seed=5)
    m1 = [pool[j] for _, j in sorted(single.pairs)]
    assert abs(S.style_report(targets, m1).smd_len) < 0.1
    assert abs(S.style_report(targets, m1, _char_len).smd_len) > 0.1  # matched under one tokenizer only: fails
    joint = S.greedy_match(targets, pool, seed=5, joint_length_fns=(_char_len,))
    mj = [pool[j] for _, j in sorted(joint.pairs)]
    for fn in (S.regex_token_length, _char_len):
        assert abs(S.style_report(targets, mj, fn).smd_len) <= 0.1
    assert len({j for _, j in joint.pairs}) == len(targets)
    # deterministic: the same seed gives the same pairs; no joint fns gives exactly the single matching
    assert S.greedy_match(targets, pool, seed=5, joint_length_fns=(_char_len,)).pairs == joint.pairs
    assert S.greedy_match(targets, pool, seed=5, joint_length_fns=()).pairs == single.pairs


def _words_len(text: str) -> int:
    """Synthetic tokenizer 1: whitespace words."""
    return len(text.split())


def _char4_len(text: str) -> int:
    """Synthetic tokenizer 2: characters / 4. Disagrees with words when word lengths differ."""
    return max(1, len(text) // 4)


def _long_skew_data(seed: int, n_targets: int = 64, n_pool: int = 1200) -> tuple[list[str], list[str]]:
    """Targets use many long words; the pool's word counts skew long and its long-word share varies."""
    rng = random.Random(seed)
    short = ["a", "to", "of", "is", "in", "on"]
    long_words = ["extraordinarily", "infrastructure", "characteristic", "misunderstanding"]

    def sent(n: int, p_long: float) -> str:
        return "Write " + " ".join(rng.choice(long_words) if rng.random() < p_long else rng.choice(short)
                                   for _ in range(n))
    targets = [sent(max(3, int(rng.gauss(10, 3))), 0.5) for _ in range(n_targets)]
    pool = [sent(max(2, int(rng.expovariate(1 / 12)) + 2), rng.random() * 0.7) for _ in range(n_pool)]
    return targets, pool


def _lens(fns, texts):
    return [[f(t) for t in texts] for f in fns]


@pytest.mark.parametrize("seed", [0, 2])
def test_swap_refinement_brings_both_tokenizers_under_the_limit(seed: int) -> None:
    targets, pool = _long_skew_data(seed)
    fns = (_char4_len, _words_len)
    assert S.style_report(targets, pool, _words_len).smd_len < -0.3  # the pool skews long
    g = S.greedy_match(targets, pool, seed=1, length_fn=_words_len)  # matched under words only
    chosen = [j for _, j in sorted(g.pairs)]
    assert abs(S.style_report(targets, [pool[j] for j in chosen], _char4_len).smd_len) > 0.1  # chars/4 fails
    avail = [j for j in range(len(pool)) if j not in set(chosen)]
    r = S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, avail, seed=3, target=0.08)
    assert r.met and r.stop_reason == "met" and r.n_swaps > 0 and len(r.swaps) == r.n_swaps
    for f, v in zip(fns, r.final_smd):
        rep = S.style_report(targets, [pool[j] for j in r.chosen], f)
        assert abs(rep.smd_len) <= 0.08 and rep.smd_len == pytest.approx(v)
        assert math.isfinite(rep.smd_log_len)
    # one-to-one, only original or available items, input list untouched
    assert len(set(r.chosen)) == len(targets) and set(r.chosen) <= set(chosen) | set(avail)
    assert chosen == [j for _, j in sorted(g.pairs)]
    # deterministic: same inputs and seed give the same pairs and swaps
    r2 = S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, avail, seed=3, target=0.08)
    assert (r2.chosen, r2.swaps) == (r.chosen, r.swaps)


def test_swap_refinement_keeps_strata_when_cross_stratum_is_off() -> None:
    targets, pool = _long_skew_data(0)
    fns = (_char4_len, _words_len)
    g = S.greedy_match(targets, pool, seed=1, length_fn=_words_len)
    chosen = [j for _, j in sorted(g.pairs)]
    avail = [j for j in range(len(pool)) if j not in set(chosen)]
    strata = [j % 3 == 0 for j in range(len(pool))]
    r = S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, avail, seed=3, target=0.08,
                          pool_strata=strata, allow_cross_stratum=False)
    assert r.n_swaps > 0 and r.n_swaps_cross_stratum == 0
    assert sorted(strata[j] for j in r.chosen) == sorted(strata[j] for j in chosen)
    assert all(strata[x] == strata[y] for _, x, y in r.swaps)


def test_swap_refinement_fails_cleanly_when_the_pool_is_too_short() -> None:
    targets, _ = _long_skew_data(0)
    pool = [f"Write {'a ' * (i % 3)}x" for i in range(300)]  # every pool item is far shorter than every target
    fns = (_char4_len, _words_len)
    chosen, avail = list(range(len(targets))), list(range(len(targets), len(pool)))
    r = S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, avail, seed=3, target=0.08)
    assert not r.met and r.stop_reason == "no_improving_swap"
    assert r.final_max_abs_smd > 0.1 and r.final_max_abs_smd <= max(abs(x) for x in r.initial_smd)
    assert len(set(r.chosen)) == len(targets)
    r2 = S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, avail, seed=3, target=0.08)
    assert r2.chosen == r.chosen
    with pytest.raises(ValueError, match="overlap"):
        S.refine_by_swaps(_lens(fns, targets), _lens(fns, pool), chosen, chosen[:1], seed=3, target=0.08)


def _contrast_inputs(seed: int = 4) -> tuple[list[dict], list[dict], dict]:
    targets, pool = _long_skew_data(seed, n_targets=60, n_pool=1500)
    harmful = [{"id": f"advbench-{i:03d}", "prompt": t} for i, t in enumerate(targets)]
    harmless = [{"id": f"dolly-{j:06d}", "prompt": p, "context": "", "category": "open_qa"} for j, p in enumerate(pool)]
    ccfg = {"sizes": {"construction": 32, "dev": 8, "test": 16}, "dedup_within": False, "smd_max": 0.1,
            "match_imperative": True, "harmless_pool": {"require_empty_context": True, "categories": []}}
    return harmful, harmless, ccfg


def test_build_contrast_refines_every_split_under_two_tokenizers() -> None:
    harmful, harmless, ccfg = _contrast_inputs()
    cons = [("words", _words_len), ("char4", _char4_len)]
    # primary = words only in the greedy distance; chars/4 is enforced by the refinement alone
    files, rep = B.build_contrast(harmful, harmless, [], ccfg, seed=9, length_fn=_words_len,
                                  constraint_length_fns=cons)
    ref = rep["style"]["refinement"]
    assert ref["tokenizers"] == ["words", "char4"] and ref["n_swaps"] > 0
    assert ref["target_met"] and ref["constraint_met"] and ref["overall"]["constraint_met"]
    for sp in B.SPLITS:
        h = [r["prompt"] for r in files[sp]["harmful"]]
        n = [r["prompt"] for r in files[sp]["harmless"]]
        assert len(h) == len(n) == ccfg["sizes"][sp]
        for name, f in cons:
            v = S.style_report(h, n, f)
            assert abs(v.smd_len) <= 0.08 and ref["per_split"][sp]["final_smd"][name] == pytest.approx(v.smd_len)
            assert ref["per_split"][sp]["final_smd_log_len"][name] == pytest.approx(v.smd_log_len)
        assert ref["per_split"][sp]["stop_reason"] == "met"
    used = rep["dolly_ids_used"]
    assert len(used) == len(set(used)) == 56
    _assert_splits_disjoint(files)
    # deterministic: same inputs and seed give the same pairs
    files2, rep2 = B.build_contrast(harmful, harmless, [], ccfg, seed=9, length_fn=_words_len,
                                    constraint_length_fns=cons)
    assert files2 == files and rep2["style"]["refinement"] == ref


def test_build_contrast_reports_unmet_constraint_when_pool_too_short() -> None:
    harmful, _, ccfg = _contrast_inputs()
    harmless = [{"id": f"dolly-{j:06d}", "prompt": f"Write {'a ' * (j % 3)}x{j}", "context": "", "category": "x"}
                for j in range(400)]
    files, rep = B.build_contrast(harmful, harmless, [], ccfg, seed=9, length_fn=_words_len,
                                  constraint_length_fns=[("words", _words_len), ("char4", _char4_len)])
    ref = rep["style"]["refinement"]
    assert not ref["target_met"] and not ref["constraint_met"] and not rep["style"]["pass"]
    assert all(v["stop_reason"] == "no_improving_swap" for v in ref["per_split"].values())


def _fake_loader(table: dict):
    def load(tok_id, allow_download, revisions=None):
        if not tok_id:
            return S.regex_token_length, {"id": None, "kind": "regex", "revision": None,
                                          "fingerprint": "regex_token_length"}
        if tok_id not in table:
            raise OSError(f"{tok_id} not in the local cache")
        fn = table[tok_id]
        return fn, {"id": tok_id, "kind": "hf", "revision": (revisions or {}).get(tok_id), "fingerprint": f"fp-{tok_id}"}
    return load


def _smoke_with_tokenizers(monkeypatch: pytest.MonkeyPatch, **contrast) -> None:
    orig = B._smoke_config

    def patched(cfg: dict) -> dict:
        out = orig(cfg)
        out["contrast"].update(contrast)
        return out
    monkeypatch.setattr(B, "_smoke_config", patched)


def test_build_records_every_tokenizer_and_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    same = lambda t: len(t.split())  # noqa: E731
    monkeypatch.setattr(B, "_load_length_fn", _fake_loader({"org/a": same, "org/b": S.regex_token_length}))
    _smoke_with_tokenizers(monkeypatch, length_tokenizer="org/a", joint_match_tokenizers=["org/b"],
                           report_tokenizers=["org/a", "org/b"])
    m = B.build({}, tmp_path / "p", smoke=True)
    st = m["contrast"]["style"]
    assert st["pass"] and st["primary_pass"] and st["length_tokenizer"] == "org/a"
    assert set(st["by_tokenizer"]) == {"org/a", "org/b"}
    assert st["by_tokenizer"]["org/a"]["roles"] == ["length", "report"]
    assert st["by_tokenizer"]["org/b"]["roles"] == ["joint_match", "report"]
    for v in st["by_tokenizer"].values():
        assert v["pass"] and v["max_abs_smd"] <= 0.1 and set(v["per_split"]) == set(B.SPLITS)
        assert {"fingerprint", "revision", "matched_all", "max_abs_smd_log_len"} <= set(v)
        assert "smd_log_len" in v["matched_all"] and all("smd_log_len" in r for r in v["per_split"].values())
    ref = st["refinement"]
    assert ref["tokenizers"] == ["org/a", "org/b"] and ref["constraint_met"] and ref["target_abs_smd"] == 0.08
    assert set(ref["per_split"]) == set(B.SPLITS) and isinstance(ref["n_swaps"], int)
    assert set(ref["overall"]["final_smd"]) == {"org/a", "org/b"}
    rows = B.read_jsonl(tmp_path / "p" / m["contrast"]["files"]["construction_harmful"]["path"])
    assert all(r["n_tokens"] == same(r["prompt"]) for r in rows)  # n_tokens counts with the length tokenizer
    assert not (tmp_path / "p" / "contrast_style.FAILED.json").exists()


def test_build_fails_not_warns_when_a_report_tokenizer_exceeds_smd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    skewed = lambda t: len(t.split()) + (10 if "placeholder task" in t else 0)  # noqa: E731 - harmful items longer
    monkeypatch.setattr(B, "_load_length_fn", _fake_loader({"org/skewed": skewed}))
    _smoke_with_tokenizers(monkeypatch, report_tokenizers=["org/skewed"])
    out = tmp_path / "p"
    with pytest.raises(B.StyleMatchError, match="org/skewed"):
        B.build({}, out, smoke=True)
    assert not (out / "MANIFEST.json").exists() and not (out / "contrast").exists()
    failed = json.loads((out / "contrast_style.FAILED.json").read_text(encoding="utf-8"))
    assert failed["style"]["by_tokenizer"]["org/skewed"]["pass"] is False
    assert "regex" in failed["style"]["by_tokenizer"]  # every tokenizer's numbers are recorded
    ref = failed["style"]["refinement"]  # the swap refinement ran under both tokenizers and could not meet the target
    assert ref["tokenizers"] == ["regex", "org/skewed"] and not ref["target_met"] and not ref["constraint_met"]
    assert {v["stop_reason"] for v in ref["per_split"].values()} <= {"no_improving_swap", "max_swaps"}
    # the CLI exits 3 with the message instead of printing a warning and exiting 0; the error line carries the
    # per-tokenizer, per-split SMDs and the refinement summary so the orchestrator can decide without the file
    assert B.main(["--smoke", "--out", str(tmp_path / "cli")]) == 3
    err = capsys.readouterr().err
    assert "exceeds |SMD|" in err and "'org/skewed'" in err and "'regex'" in err
    assert all(f"'{sp}'" in err for sp in B.SPLITS) and "'refinement'" in err and "'n_swaps'" in err


def test_build_fails_when_a_configured_tokenizer_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(B, "_load_length_fn", _fake_loader({}))
    _smoke_with_tokenizers(monkeypatch, report_tokenizers=["google/gemma-4-12B-it"])
    with pytest.raises(B.StyleMatchError, match="could not be loaded"):
        B.build({}, tmp_path / "p", smoke=True)


def test_tokenizer_revisions_come_from_bases_yaml() -> None:
    revs = B.pinned_tokenizer_revisions()
    bases = yaml.safe_load((REPO / "configs" / "bases.yaml").read_text(encoding="utf-8"))["bases"]
    assert revs["Qwen/Qwen3.5-9B"] == bases["qwen3.5-9b"]["revision"]
    cfg = yaml.safe_load((REPO / "configs" / "batteries.yaml").read_text(encoding="utf-8"))["contrast"]
    primaries = {b["hf"] for b in bases.values() if b.get("role") == "primary"}
    # the configured shared set is checked under all four primary bases' tokenizers, each at its pin
    assert set(cfg["report_tokenizers"]) == primaries and cfg["fail_on_smd"] is True
    assert {cfg["length_tokenizer"], *cfg["joint_match_tokenizers"]} == primaries
    assert all(t in revs for t in primaries)


def _tiny_hf_tokenizer(dest: Path, vocab_size: int) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    text = [f"Write a plan for placeholder task {i} near the river in step by step detail" for i in range(50)]
    text += ["What is special about the old harbour town?", "Give me three facts about the mountain lake"] * 20
    tok.train_from_iterator(text, trainer=trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"],
                                                               show_progress=False))
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(str(dest))
    return dest


def test_real_tiny_hf_tokenizers_load_offline_and_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = str(_tiny_hf_tokenizer(tmp_path / "tokA", 120))
    b = str(_tiny_hf_tokenizer(tmp_path / "tokB", 300))
    fn, info = B._load_length_fn(a, allow_download=False)
    assert info["kind"] == "hf" and len(info["fingerprint"]) == 64 and fn("Write a plan") > 0
    _smoke_with_tokenizers(monkeypatch, length_tokenizer=a, joint_match_tokenizers=[b], report_tokenizers=[a, b])
    m1 = B.build({}, tmp_path / "p1", smoke=True)
    m2 = B.build({}, tmp_path / "p2", smoke=True)
    st = m1["contrast"]["style"]
    assert st["pass"] and st["by_tokenizer"][a]["fingerprint"] != st["by_tokenizer"][b]["fingerprint"]
    assert m1["contrast"]["prompt_set_sha256"] == m2["contrast"]["prompt_set_sha256"]  # seeded, deterministic
