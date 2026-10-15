"""CPU tests for sm.exploratory.weight_surgery (docs/PROGRAM_2026-10-13.md line R3).

Unit tests: the variant grammar and module sets, the improvement-kept fraction, the R3 reading, the next-token loss
against the model's own loss, the held-out builders on synthetic Dolly rows (disjoint from training, near duplicates
dropped, the builder pipeline reproducing the training prompts), the judged-reuse join. Smoke: the whole run on the
tiny hybrid model with a random LoRA unit (capability batteries, losses, keyword-judged canary), its instrument checks,
the restricted tree's modes, no text in open outputs, and the WildGuard re-score path through the unchanged
sm.interp.mechanism rescore-judged plus --attach-wildguard.
"""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from sm.exploratory import weight_surgery as WS
from sm.interp import mechanism as M

KEYS = [(b, p) for b in range(4) for p in ("self_attn.q_proj", "linear_attn.in_proj_qkv", "mlp.down_proj", "mlp.up_proj")]


# ------------------------------------------------------------------------------------------------ pure functions


def test_variant_grammar_and_module_sets():
    vs = WS.parse_variants("revert:type=mlp ; revert:type=self_attn+linear_attn ; keep_only:blocks=2-3 ; unit ; base")
    assert [v["name"] for v in vs] == ["revert:type=mlp", "revert:type=self_attn+linear_attn", "keep_only:blocks=2-3",
                                       "unit", "base"]
    assert [v["name"] for v in WS.parse_variants("revert:type=mlp")][:2] == ["unit", "base"]  # references prepended
    mods = {v["name"]: WS.variant_modules(v, KEYS) for v in vs}
    assert mods["unit"] == [] and mods["base"] == sorted(KEYS)
    assert all(p.startswith("mlp.") for _, p in mods["revert:type=mlp"]) and len(mods["revert:type=mlp"]) == 8
    assert len(mods["revert:type=self_attn+linear_attn"]) == 8
    assert set(mods["revert:type=mlp"]) | set(mods["revert:type=self_attn+linear_attn"]) == set(KEYS)
    # keep_only reverts the complement: blocks 2-3 keep their delta, blocks 0-1 are reverted
    assert {b for b, _ in mods["keep_only:blocks=2-3"]} == {0, 1} and len(mods["keep_only:blocks=2-3"]) == 8
    for bad in ("revert", "nope:type=mlp", "unit:type=mlp", "revert:type=bogus"):
        with pytest.raises(ValueError):
            WS.parse_variants(bad)
    with pytest.raises(ValueError):
        WS.parse_variants("unit;unit")
    assert WS.selector_key(M.parse_selector("blocks=25-31,type=mlp")) == WS.selector_key(M.parse_selector("type=mlp,blocks=25-31"))


def test_improvement_kept_and_its_undefined_cases():
    assert WS.improvement_kept(2.0, 1.5, 1.6) == pytest.approx(0.8)
    assert WS.improvement_kept(2.0, 1.5, 2.0) == pytest.approx(0.0)
    assert WS.improvement_kept(2.0, 1.5, 1.4) == pytest.approx(1.2)
    assert WS.improvement_kept(2.0, 1.995, 1.9) is None  # gap below IMPROVEMENT_MIN
    assert WS.improvement_kept(1.5, 2.0, 1.6) is None  # the fine-tune did not improve the loss
    assert WS.improvement_kept(None, 1.5, 1.6) is None and WS.improvement_kept(2.0, float("nan"), 1.6) is None


def test_r3_reading_rules():
    ref = [{"variant": "unit", "kind": "unit", "recovery": 0.0, "kept": 1.0},
           {"variant": "base", "kind": "base", "recovery": 1.0, "kept": 0.0}]

    def rd(*rows):
        return WS.r3_reading(ref + [{"variant": f"v{i}", "kind": "revert", "recovery": r, "kept": k} for i, (r, k) in enumerate(rows)])
    assert rd((0.9, 0.85))["verdict"] == "supported"
    assert rd((0.9, 0.3), (1.0, 0.1))["verdict"] == "falsified"  # every recovering variant keeps < 0.5
    assert rd((0.9, 0.6))["verdict"] == "neither"  # between the bars
    assert rd((0.5, 0.95))["verdict"] == "neither"  # nothing recovers
    assert rd((None, 0.9))["verdict"] == "undetermined"
    assert rd((0.9, None))["verdict"] == "undetermined"
    assert WS.r3_reading(ref)["verdict"] == "undetermined"  # the references never decide it


def test_loss_eval_matches_the_models_own_loss(tiny_model):
    import torch

    from sm.common.models import pad_token_id
    from sm.train.lora_sft import synthetic_records

    model, tok, _ = tiny_model
    ex, stats = WS.tokenize(tok, synthetic_records(6, 3), 64)
    assert stats["n_examples"] == 6 and stats["chat_template_sha256"]
    res = WS.loss_eval(model, ex, pad_token_id(tok), batch_size=4)
    ref = []
    with torch.inference_mode():
        for e in ex:
            ids = torch.tensor([e["input_ids"]])
            ref.append(float(model(input_ids=ids, labels=torch.tensor([e["labels"]])).loss))
    assert np.allclose(res["per_example"], ref, rtol=1e-4, atol=1e-5)
    n = [sum(1 for x in e["labels"][1:] if x != -100) for e in ex]
    assert res["n_target_tokens"] == sum(n)
    assert res["token_mean"] == pytest.approx(float(np.dot(ref, n) / sum(n)), rel=1e-4)


# ------------------------------------------------------------------------------------------------ held-out builders


def _dolly_rows(n=60):
    cats = ["open_qa", "general_qa", "brainstorming", "creative_writing", "closed_qa"]
    rows = []
    for i in range(n):
        ctx = "Some context paragraph." if i % 7 == 0 else ""
        rows.append({"instruction": f"Question number {i} about topic {i * 13 % 17} and item {i * 7}?", "context": ctx,
                     "response": f"Answer {i}.", "category": cats[i % len(cats)]})
    rows.append(dict(rows[3]))  # an exact duplicate prompt (dedupe)
    return rows


def test_dolly_heldout_is_disjoint_and_the_c1_pipeline_reproduces_training():
    from sm.data import build_bank_datasets as BD

    rows = _dolly_rows()
    ids = [BD.dolly_row_id(i) for i in range(len(rows))]
    c1, c2 = set(ids[0::2]), set(ids[1::2])
    train = [BD.parse_dolly(r, i, "dolly15k") for i, r in enumerate(rows) if BD.dolly_row_id(i) in c1]
    train, _ = BD.dedupe_by_prompt([r for r in train if r])
    assert WS.dolly_c1_reproduction(rows, c1, None, train)["status"] == "pass"
    assert WS.dolly_c1_reproduction(rows, c1, None, train[:-1])["status"] == "fail"
    # plant a c2 row whose prompt equals a training prompt and one that is a near duplicate of another
    rows2 = rows + [dict(rows[0]), {**rows[2], "instruction": rows[2]["instruction"] + " "}]
    c2b = c2 | {BD.dolly_row_id(len(rows)), BD.dolly_row_id(len(rows) + 1)}
    near = WS.near_dup_matcher(train)
    got, st = WS.dolly_heldout(rows2, c2b, train, None, 10, seed=5, near=near)
    tk = WS.prompt_keys(train)
    assert len(got) == 10 and not (WS.prompt_keys(got) & tk)
    assert st["n_equal_to_training_prompt"] + st["n_near_duplicate_of_training_prompt"] >= 2
    again, _ = WS.dolly_heldout(rows2, c2b, train, None, 10, seed=5, near=near)
    assert again == got  # deterministic in the seed
    other, _ = WS.dolly_heldout(rows2, c2b, train, None, 10, seed=6, near=near)
    assert other != got


def test_persona_heldout_follows_the_builder_order_beyond_the_training_prompts():
    from sm.data import build_bank_datasets as BD

    rows = _dolly_rows(80)
    ids = [BD.dolly_row_id(i) for i in range(len(rows))]
    pool = set(ids)
    cats = ["open_qa", "general_qa", "brainstorming", "creative_writing"]
    opts = BD.BuildOptions(out=Path("."), cache_dir=Path("."), seed=0, persona_categories=cats)
    order = BD._persona_prompts(opts, {}, 10 ** 9, None, lambda: (rows, pool))
    n_gen = 20
    train = [BD.make_record(p["prompt"], "styled answer", "persona:x", f"humorous-{p['id']}") for p in order[:n_gen]]
    got, st = WS.persona_heldout_prompts(rows, pool, cats, None, n_gen, train, 8, 0, WS.near_dup_matcher(train))
    assert st["HO1"]["status"] == "pass" and len(got) == 8
    assert [p["prompt"] for p in got] == [p["prompt"] for p in order[n_gen:n_gen + 8]]
    assert not ({BD.normalize_prompt(p["prompt"]) for p in got} & WS.prompt_keys(train))
    _, st2 = WS.persona_heldout_prompts(rows, pool, cats, None, n_gen + 1, train, 8, 0)
    assert st2["HO1"]["status"] == "fail"  # a different generation target does not reproduce the training prompts


def test_reuse_judged_joins_the_same_module_sets(tmp_path):
    src = tmp_path / "revert" / "u.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"judged": {"judge": "J1x", "base": {"rate": 0.8}, "unit": {"rate": 0.4}, "canary": {"sha256": "c"}},
                               "conditions": [{"set": "type=mlp", "judged": {"revert_set": {"rate": 0.8, "recovery": 1.0},
                                                                             "keep_only_set": {"rate": 0.5, "recovery": 0.25}}},
                                              {"set": "blocks=25-31", "judged": {"revert_set": {"rate": 0.4, "recovery": 0.0},
                                                                                 "keep_only_set": {"rate": 0.8, "recovery": 1.0}}}]}))
    (src.parent / "wildguard").mkdir()
    (src.parent / "wildguard" / "u.json").write_text(json.dumps({"conditions": {"keep_only_set_01": {"wg": {"rate": 0.7, "recovery": 0.9}}}}))
    vs = WS.parse_variants("revert:type=mlp;keep_only:blocks=25-31;revert:type=self_attn")
    got = WS.reuse_judged([str(src), str(tmp_path / "absent.json")], vs)
    assert got["revert:type=mlp"][0]["recovery"] == 1.0 and got["revert:type=mlp"][0]["wildguard"] is None
    k = got["keep_only:blocks=25-31"][0]
    assert k["recovery"] == 1.0 and k["condition"] == "keep_only_set_01" and k["wildguard"]["recovery"] == 0.9
    assert all(m.get("missing") for m in got["revert:type=self_attn"])  # no match, only the absent file's marker


# ------------------------------------------------------------------------------------------------ smoke


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws_smoke")
    assert WS.main(["--smoke", "--out", str(d)]) == 0
    return d


def _body(d: Path) -> dict:
    return json.loads((d / "open" / "smoke_unit.json").read_text())


def test_smoke_rows_checks_and_references(smoke_dir):
    b = _body(smoke_dir)
    names = [r["variant"] for r in b["rows"]]
    assert names[:2] == ["unit", "base"] and "revert:type=mlp" in names and "keep_only:blocks=1" in names
    by = {r["variant"]: r for r in b["rows"]}
    assert by["unit"]["n_modules_reverted"] == 0 and by["base"]["n_modules_kept"] == 0
    assert by["base"]["mass_fraction_reverted"] == pytest.approx(1.0)
    for r in b["rows"]:
        assert set(r["loss"]) == {"heldout", "train_sample"}
        assert r["capability"]["mmlu_n"] > 0 and r["capability"]["ifeval_n"] > 0
        assert r["deltas"]["heldout_loss_vs_unit"] is not None
    assert by["unit"]["deltas"]["heldout_loss_vs_unit"] == 0.0 and by["base"]["deltas"]["heldout_loss_vs_base"] == 0.0
    ic = b["instrument_checks"]
    assert all(ic["WS0_base_equals_all_reverted"][s]["status"] == "pass" for s in ("heldout", "train_sample"))
    assert all(ic["WS1_adapter_restored"][s]["status"] == "pass" for s in ("heldout", "train_sample"))
    assert ic["WJ0_judged_base_equals_all_reverted"]["status"] == "pass"
    j = b["judged"]
    assert set(j["variant_of_condition"].values()) == set(names) - {"unit"}
    assert all("wilson95" in by[v]["judged"] for v in names)
    assert b["references"]["base"]["loss"] == by["base"]["loss"]
    prov = b["provenance"]
    assert prov["adapter_sha256"] and prov["config_sha256"] and "own_file_sha256" in prov["config"]
    assert b["reading"]["verdict"] in ("supported", "falsified", "neither", "undetermined")


def test_smoke_texts_stay_restricted(smoke_dir):
    rdir = smoke_dir / "restricted"
    files = [p for p in rdir.rglob("*") if p.is_file()]
    assert any(p.name == "heldout.jsonl" for p in files) and any(p.name == "mmlu.jsonl" for p in files)
    for p in [rdir, *[q for q in rdir.rglob("*") if q.is_dir()]]:
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o700, p
    for p in files:
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, p
    texts = [r["messages"][-1]["content"] for r in M.AX.read_jsonl(rdir / "heldout" / "smoke_unit" / "heldout.jsonl")]
    for p in (smoke_dir / "open").rglob("*.json"):
        t = p.read_text()
        assert '"completion"' not in t and '"prompt"' not in t, p
        assert not any(x in t for x in WS.SMOKE_PERSONA_PROMPTS + M.SMOKE_HARMFUL), p
        assert not any(x.strip() and x in t for x in texts if len(x.strip()) > 8), p


def test_smoke_wildguard_rescore_and_attach(smoke_dir):
    view = smoke_dir / "open" / "rescore_view"
    assert M.main(["rescore-judged", "--unit", "smoke_unit", "--source", "revert", "--out", str(view),
                   "--restricted", str(smoke_dir / "restricted"), "--reader", "keyword"]) == 0
    wg = json.loads((view / "revert" / "wildguard" / "smoke_unit.json").read_text())
    assert wg["join"]["status"] == "pass" and wg["conditions"]
    assert WS.main(["--smoke", "--out", str(smoke_dir), "--attach-wildguard"]) == 0
    b = _body(smoke_dir)
    assert b["reading_wildguard"]["reader"].startswith("WildGuard") and b["attached"][-1]["what"] == "wildguard"
    by = {r["variant"]: r for r in b["rows"]}
    for name, cond in b["judged"]["variant_of_condition"].items():
        v = by[cond]
        assert v["judged_wildguard"]["rate"] == v["judged"]["rate"]  # the keyword reader relabels identically


def test_rerun_is_idempotent_and_resumes_restricted_batteries(smoke_dir):
    p = smoke_dir / "open" / "smoke_unit.json"
    before = p.stat().st_mtime_ns
    assert WS.main(["--smoke", "--out", str(smoke_dir)]) == 0
    assert p.stat().st_mtime_ns == before  # already done: nothing recomputed without --force
    assert math.isfinite(_body(smoke_dir)["seconds"])
