"""Consistency checks for the human-facing docs (README.md, code/README.md, docs/RUNBOOK.md, tests/README.md).

No model, no network: these tests read text files only. They keep the docs in step with the
package, the base registry, the pre-registration (prereg/PREREG-v0.2.md) and the compute plan.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    "readme": ROOT / "README.md",
    "code_readme": ROOT / "code" / "README.md",
    "runbook": ROOT / "docs" / "RUNBOOK.md",
    "tests_readme": ROOT / "tests" / "README.md",
}
PREREG = ROOT / "prereg" / "PREREG-v0.2.md"
EM_DASH = chr(0x2014)
MODULE_RE = re.compile(r"\bsm(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
JOBS_PATH_RE = re.compile(r"\bjobs/[A-Za-z0-9_.\-]+")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# v2 shared definitions (docs/wiring-workflow.js, "v2 artifacts"): the revision every stage loads.
PINNED_REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "openai/gpt-oss-safeguard-20b": "8a11e17b25c973a24099d4016bf2e17dd7ec1574",
    "allenai/wildguard": "cbba4823f3e8020e5a74a5e29bf85072def6f2ff",  # J2 since 2026-10-10 (D-015/D-016)
}
COMPAT_LD_PATH = "/projects/interpretability/shehral.m/cuda-compat/usr/local/cuda-13.0/compat"
MONITORS = ("refusal_direction", "refusal_subspace", "probe", "persona_evil", "persona_sycophancy",
            "persona_hallucination")


def _text(key: str) -> str:
    return DOCS[key].read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Body of the markdown section whose heading line contains `heading` (up to the next heading of any level <= 2)."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("#") and heading.lower() in line.lower():
            start = i + 1
            break
    if start is None:
        raise AssertionError(f"heading {heading!r} not found")
    out: list[str] = []
    for line in lines[start:]:
        if re.match(r"^#{1,2} ", line):
            break
        out.append(line)
    return "\n".join(out)


def _bases_yaml() -> dict[str, dict]:
    data = yaml.safe_load((ROOT / "configs" / "bases.yaml").read_text(encoding="utf-8"))
    return data["bases"]


def _primaries() -> dict[str, dict]:
    return {k: v for k, v in _bases_yaml().items() if v.get("role") == "primary"}


def _table_rows(body: str) -> list[list[str]]:
    rows = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([c.strip().strip("`") for c in line.strip().strip("|").split("|")])
    return rows


@pytest.mark.parametrize("key", sorted(DOCS))
def test_doc_exists_and_nonempty(key: str) -> None:
    path = DOCS[key]
    assert path.is_file(), f"missing {path}"
    assert len(path.read_text(encoding="utf-8").strip()) > 200


@pytest.mark.parametrize("key", sorted(DOCS))
def test_no_em_dashes(key: str) -> None:
    assert EM_DASH not in _text(key), f"em dash in {DOCS[key]}"


def test_this_file_has_no_em_dashes() -> None:
    assert EM_DASH not in Path(__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------- prereg version


def test_prereg_v02_exists() -> None:
    assert PREREG.is_file()


@pytest.mark.parametrize("key", ["readme", "code_readme", "runbook"])
def test_docs_point_at_prereg_v02(key: str) -> None:
    text = _text(key)
    assert "prereg/PREREG-v0.2.md" in text or "../prereg/PREREG-v0.2.md" in text
    if key in ("readme", "code_readme"):  # these two link it
        assert re.search(r"\]\((?:\.\./)?prereg/PREREG-v0\.2\.md\)", text), "no markdown link to the v0.2 prereg"


@pytest.mark.parametrize("key", sorted(DOCS))
def test_no_stale_prereg_reference(key: str) -> None:
    assert "PREREG-v0.1" not in _text(key), f"{DOCS[key]} still points at the v0.1 prereg"


# ---------------------------------------------------------------- top-level README


def test_readme_required_sections() -> None:
    text = _text("readme")
    for heading in ("Abstract", "Layout", "Install", "Tests", "Running the study on Explorer",
                    "AI-assistance statement", "Licenses", "Hazards"):
        _section(text, heading)  # raises if absent
    assert "pip install -e" in _section(text, "Install")
    assert "SM_OFFLINE=1" in _section(text, "Tests")
    explorer = _section(text, "Running the study on Explorer")
    assert "jobs/README.md" in explorer and "docs/RUNBOOK.md" in explorer


def test_readme_stage_list_covers_v2_stages() -> None:
    explorer = _section(_text("readme"), "Running the study on Explorer").lower()
    for phrase in ("capability", "monitor scores", "j2 re-score", "v4", "hard gate", "same loader call"):
        assert phrase in explorer, phrase


def test_ai_statement_is_owner_placeholder() -> None:
    body = _section(_text("readme"), "AI-assistance statement").strip()
    assert body == "[owner writes]"


def test_licenses_match_prereg() -> None:
    lic = _section(_text("readme"), "Licenses")
    prereg = PREREG.read_text(encoding="utf-8")
    for name in ("Apache-2.0", "CC BY 4.0"):
        assert name in lic
        assert name in prereg


def test_hazards_cover_prereg_never_published() -> None:
    hz = _section(_text("readme"), "Hazards").lower()
    for phrase in ("ablated weights", "misalignment recipes", "harmful response texts", "results/restricted/", "0700"):
        assert phrase in hz, phrase
    # Harmful batteries named in the contracts are the restricted ones; the harmful ladders join them.
    for battery in ("strongreject", "harmbench", "xstest unsafe", "karne ladders"):
        assert battery in hz, battery


def test_abstract_bases_match_registry_and_prereg() -> None:
    abstract = _section(_text("readme"), "Abstract")
    prereg = PREREG.read_text(encoding="utf-8")
    bases = _bases_yaml()
    primary = [v["hf"] for v in bases.values() if v.get("role") == "primary"]
    assert len(primary) == 4
    assert "four bases" in abstract.lower()
    prereg_bases = _section(prereg, "Study information")
    for hf in primary:
        assert hf in abstract, hf
        assert hf in prereg_bases, hf
    # Fallbacks are not bases of the study and must not be named as such in the abstract.
    for v in bases.values():
        if v.get("role") == "fallback":
            assert v["hf"] not in abstract, v["hf"]


def test_abstract_reflects_v02_design() -> None:
    abstract = _section(_text("readme"), "Abstract").lower()
    for phrase in ("persona-vectors", "risky-financial-advice", "insecure code", "karne", "base 1%-fpr threshold"):
        assert phrase in abstract, phrase
    assert "model-organism datasets" not in abstract  # v0.2 dropped the 2506.11613 organism data


# ---------------------------------------------------------------- RUNBOOK


def test_runbook_fallback_table_matches_registry() -> None:
    fb = _section(_text("runbook"), "Fallback procedure")
    bases = _bases_yaml()
    rows = {}
    for cells in _table_rows(fb):
        if len(cells) == 2 and cells[0] in bases:
            rows[cells[0]] = cells[1]
    primaries = _primaries()
    assert set(rows) == set(primaries)
    for key, entry in primaries.items():
        if entry.get("fallback"):
            assert rows[key].split()[0].strip("`") == entry["fallback"], key
            assert bases[entry["fallback"]]["hf"] in rows[key], key
        else:
            assert rows[key].lower().startswith("none"), key


def test_runbook_budget_numbers_sourced_from_plan() -> None:
    budget = _section(_text("runbook"), "Expected H200-hours")
    plan = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    plan_numbers = set(re.findall(r"\d+(?:\.\d+)?", plan))
    found = []
    for cells in _table_rows(budget)[1:]:  # skip the header row
        nums = re.findall(r"\d+(?:\.\d+)?", cells[-1])
        found.extend(nums)
        for n in nums:
            assert n in plan_numbers, f"{n} H200-hours in RUNBOOK is not in docs/PLAN.md"
    assert found, "budget table has no numbers"
    assert "90" in found


def _runbook_pins() -> dict[str, str]:
    pins = {}
    for cells in _table_rows(_section(_text("runbook"), "Revision pins")):
        if len(cells) >= 2 and SHA_RE.match(cells[-1]):
            pins[cells[0]] = cells[-1]
    return pins


def test_runbook_revision_pins_match_v2_definitions() -> None:
    assert _runbook_pins() == PINNED_REVISIONS


def test_runbook_pins_cover_registry_primaries() -> None:
    pins = _runbook_pins()
    for key, entry in _primaries().items():
        assert entry["hf"] in pins, key


def test_configs_agree_with_runbook_pins() -> None:
    """Where a config carries a revision, it is the one the RUNBOOK documents."""
    pins = _runbook_pins()
    for key, entry in _primaries().items():
        rev = entry.get("revision")
        if rev is not None:
            assert rev == pins[entry["hf"]], f"configs/bases.yaml {key}: {rev}"
    for manifest in sorted((ROOT / "configs").glob("bank_*.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        hf, rev = data.get("base_hf"), data.get("revision")
        if hf in pins and rev is not None:
            assert rev == pins[hf], f"{manifest.name}: {rev}"


def test_runbook_states_same_loader_and_revision() -> None:
    pins = _section(_text("runbook"), "Revision pins").lower()
    assert "same `load_model` call" in pins and "same revision" in pins
    assert "configs/bases.yaml" in pins


def test_runbook_steps_in_spec_order() -> None:
    text = _text("runbook")
    order = ["sm.interp.base_card", "sm.interp.causal", "sm.train.lora_sft", "sm.eval.battery",
             "sm.eval.capability", "sm.eval.judge", "sm.interp.drift", "sm.interp.monitor_scores"]
    positions = [text.index(f"python -m {m}") for m in order]
    assert positions == sorted(positions)
    # The final aggregate run comes after every producer.
    assert text.rindex("python -m sm.analysis.aggregate") > positions[-1]


def test_runbook_j2_rescore_step() -> None:
    text = _text("runbook")
    body = _section(text, "J2 re-score")
    assert "--j2-rejudge-list" in body
    assert "--flagged-runs" in body and "--judges J2" in body
    assert '"pending"' in body and "never \"not confirmed\"" in body
    assert "100%" in body and "base" in body
    # Between the first judge pass and drift.
    start = text.index(body)
    assert text.index("python -m sm.eval.judge") < start < text.index("python -m sm.interp.drift")
    first = _section(text, "Judge, first pass")
    assert "--j2-fraction 0.1" in first and "allenai/wildguard" in first


def test_runbook_capability_and_monitor_scores_outputs() -> None:
    cap = _section(_text("runbook"), "Step 6. Capability")
    for key in ("results/capability/<base>/<run_id>.json", "mmlu_acc", "mmlu_n", "ifeval_strict_acc", "ifeval_n",
                "20 points"):
        assert key in cap, key
    ms = _section(_text("runbook"), "Step 10. Monitor scores")
    assert "results/monitor_scores/<base>/<run_id>/<battery>.jsonl" in ms
    assert "karne_ladders" in ms


def test_runbook_v2_card_and_drift_fields() -> None:
    text = _text("runbook")
    card = _section(text, "Step 2. Base card")
    assert 'declared_best_drift_score: "D2"' in card and "never from the CLI" in card
    assert "monitor_thresholds" in card
    for m in MONITORS:
        assert m in card, m
    drift = _section(text, "Drift (job")
    for key in ("monitor_auroc", "realigned", "stale_recall", "realigned_recall", "base_recall", "unmerged"):
        assert key in drift, key


def test_runbook_v4_hard_gate() -> None:
    v4 = _section(_text("runbook"), "Causal validity V4")
    low = v4.lower()
    assert "hard" in low and "exits non-zero" in low and ("exit 2" in low or "exits 2" in low)
    assert "refuse to run" in low and "not evaluable" in low
    assert "passed: false" in v4


def test_runbook_cuda_and_cu128_note() -> None:
    cuda = _section(_text("runbook"), "CUDA on the H200 nodes")
    assert f"LD_LIBRARY_PATH={COMPAT_LD_PATH}:$LD_LIBRARY_PATH" in cuda
    assert "module load miniconda3/24.11.1" in cuda and "no `cuda` module" in cuda
    assert "10891923" in cuda
    for job in ("jobs/env_fix_cu128.sbatch", "jobs/env_torch_cu130.sbatch", "jobs/env_fallback_cu128.sbatch"):
        assert job in cuda, job
    assert "monitors-cu128" in cuda and "do not run it" in cuda.lower()


def test_runbook_unmerged_adapters_and_decoding() -> None:
    battery = _section(_text("runbook"), "Batteries (job")
    assert "unmerged" in battery
    assert "temperature 1.0" in battery and "top-p 1.0" in battery and "10 samples" in battery
    assert "greedy" in battery


@pytest.mark.parametrize("key", sorted(DOCS))
def test_referenced_job_files_exist(key: str) -> None:
    missing = sorted({p for p in JOBS_PATH_RE.findall(_text(key)) if not (ROOT / p.rstrip(".")).exists()})
    assert not missing, f"{DOCS[key]} names job files that do not exist: {missing}"


# ---------------------------------------------------------------- modules and links


def _module_resolves(dotted: str) -> bool:
    """True when some prefix of `dotted` is a module or package under code/ and any remainder is a name in it."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        base = ROOT / "code" / Path(*parts[:cut])
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                rest = parts[cut:]
                if not rest:
                    return True
                if len(rest) == 1:
                    src = candidate.read_text(encoding="utf-8")
                    return re.search(rf"^(?:def|class)\s+{rest[0]}\b|^{rest[0]}\s*[:=]", src, re.M) is not None
                return False
    return False


@pytest.mark.parametrize("key", sorted(DOCS))
def test_referenced_modules_exist(key: str) -> None:
    refs = sorted(set(MODULE_RE.findall(_text(key))))
    missing = [r for r in refs if not _module_resolves(r)]
    assert not missing, f"{DOCS[key]} references unknown modules or names: {missing}"


@pytest.mark.parametrize("key", sorted(DOCS))
def test_relative_links_resolve(key: str) -> None:
    path = DOCS[key]
    bad = []
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if re.match(r"^[a-z]+:", target):
            continue
        if not (path.parent / target).resolve().exists():
            bad.append(target)
    assert not bad, f"{path} has broken links: {bad}"


def test_code_readme_lists_every_stage_cli() -> None:
    text = _text("code_readme")
    for mod in ("sm.data.build_prompt_sets", "sm.data.build_bank_datasets", "sm.train.lora_sft",
                "sm.eval.battery", "sm.eval.capability", "sm.eval.judge", "sm.interp.base_card",
                "sm.interp.causal", "sm.interp.drift", "sm.interp.monitor_scores", "sm.analysis.aggregate"):
        assert f"`{mod}`" in text, mod
        assert _module_resolves(mod), mod


def test_code_readme_v2_conventions() -> None:
    conv = _section(_text("code_readme"), "Conventions")
    assert "unmerged" in conv and "fp32 before casting" in conv
    assert "pinned revision" in conv and "same `load_model` call" in conv
    assert "temperature 1.0" in conv and "greedy" in conv


# ---------------------------------------------------------------- tests/README


def test_tests_readme_states_laptop_limit() -> None:
    text = _text("tests_readme")
    assert "at most two pytest processes" in text
    assert "SM_OFFLINE=1" in text and "SM_TINY_MODEL" in text


def test_tests_readme_states_device_policy() -> None:
    policy = _section(_text("tests_readme"), "Device policy")
    assert "docs/HARNESS_CONTRACTS.md" in policy
    assert "never use MPS" in policy
    assert 'device="cpu"' in policy and 'dtype="fp32"' in policy
    assert "SM_DEVICE=mps" in policy
    assert "at most two pytest processes" in policy
