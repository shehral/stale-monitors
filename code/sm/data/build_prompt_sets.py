"""Build the contrast prompt sets and the battery item files.

Outputs (under ``--out``, default ``data/prompts``):

* ``contrast/<split>_<label>.jsonl`` for split in construction/dev/test and
  label in harmful/harmless. Records:
  ``{"id", "prompt", "label", "split", "pair_id", "source", "source_id", "n_tokens", "imperative"}``.
  Harmful item k and its style-matched harmless partner share ``pair_id`` and
  are always in the same split.
* ``contrast/calibration_harmless.jsonl`` (when ``contrast.sizes.calibration_harmless`` > 0; 512 in the
  study config): a harmless-only calibration split for the base card's 1%-FPR thresholds (FIDELITY_REVIEW N6,
  prereg v0.3 change 50). Same Dolly pool, style filters, within-source dedup and battery-overlap exclusion as the
  contrast harmless items, disjoint from construction, dev and test (no shared id or text, no near-duplicate),
  ``pair_id`` null. Its items are chosen by the same seeded greedy nearest-length matching and swap refinement,
  against the contrast harmful prompts (all of them, then a seeded draw for the remainder), so its length
  distribution is the matched harmless one; its own SMD report (vs the contrast harmful prompts, vs the test
  harmless prompts, under every constraint tokenizer) is ``MANIFEST.json["contrast"]["calibration"]``.
* ``batteries/<battery>.jsonl``: ``{"item_id", "battery", "prompt", "source", ...}``
  with battery-specific extra fields (see ``BATTERY_EXTRA_FIELDS``).
* ``persona/<trait>_{extract,eval}.json``: Chen et al. trait files normalised
  to ``{"trait", "source", "instruction": [{"pos", "neg"}], "questions", "eval_prompt"}``
  (readable by ``sm.interp.base_card.load_trait_spec``).
* ``MANIFEST.json``: sources (URL, commit, license, SHA-256), per-file counts
  and hashes, per-battery decoding, the Dolly pool split, filter counts,
  style-match statistics and checks, and an inventory of every configured
  asset (SHA-256 verified against the pin).

Dolly pools: Dolly-15k is split once, with a recorded seed, into three
disjoint pools (``harmless_contrast``, ``c1_train``, ``c2_persona``). Rows that
share a normalised instruction always land in the same pool. The pool ids
(``dolly-NNNNNN``, the index among non-blank JSONL rows) go to
``MANIFEST.json["dolly_pools"]``; ``sm.data.build_bank_datasets`` reads them so
the condition-1 Dolly set and the condition-2 persona prompts never share a
prompt with the harmless contrast set.

Harmful contrast items come from AdvBench ``harmful_behaviors.csv``; harmless
items are Dolly-15k instructions from the ``harmless_contrast`` pool, matched
per split by greedy nearest token length with seeded tie-breaks, then refined
by seeded swaps toward |SMD| <= ``smd_max - smd_margin`` under every distinct
tokenizer (see ``sm.data.style_match``; the manifest records the swaps and the
final per-tokenizer SMDs, plus the report-only log-length SMD, under
``contrast.style.refinement``). Lengths are counted with ``contrast.length_tokenizer``
(a Hugging Face id, loaded ``local_files_only`` at the revision pinned for that
id in ``configs/bases.yaml``; null = regex tokens); ``contrast.joint_match_tokenizers``
add further tokenizers to the matching distance (sum of absolute gaps), so one
shared set can match under every base's tokenizer. The matched set must give
|SMD| <= ``contrast.smd_max`` on every split under the length tokenizer, every
joint tokenizer and every ``contrast.report_tokenizers`` entry; otherwise (or
when one of them cannot be loaded) the build raises ``StyleMatchError`` (CLI
exit 3) before writing any contrast file or MANIFEST.json, and writes the
statistics to ``contrast_style.FAILED.json``. The manifest records, per
tokenizer, its id, revision, fingerprint, roles, SMDs and pass flags under
``contrast.style.by_tokenizer``. Contrast items with a character 5-gram Jaccard
similarity >= 0.8 to any battery item are dropped, as are near-duplicates
within each source.

Batteries include ``karne_ladders`` (Karne 2026, arXiv 2609.05797, CC BY 4.0):
647 harmful requests x three explicitness levels (L1 oblique, L2 plain, L3
blunt), one item per (request, level), each carrying the released compliance
labels (k of 8 samples per labelled model) for that level. Decoding per
battery comes from ``batteries.<name>.decoding`` (a profile under
``decoding_profiles``); sampled batteries get ``n_samples`` on every item.

Nothing is downloaded unless ``--allow-download`` is given and neither
``SM_OFFLINE`` nor ``HF_HUB_OFFLINE`` is set. ``--smoke`` uses in-module
synthetic placeholder items and never touches the network.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from sm.data.style_match import (
    IMPERATIVE_VERBS_VERSION,
    greedy_match,
    is_imperative,
    make_length_fn,
    refine_by_swaps,
    regex_token_length,
    style_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "batteries.yaml"
SPLITS = ("construction", "dev", "test")
CALIBRATION_SPLIT = "calibration"  # harmless only (FIDELITY_REVIEW N6); never paired, never in SPLITS
ALL_SPLITS = (*SPLITS, CALIBRATION_SPLIT)
LABELS = ("harmful", "harmless")
HARMFUL_BATTERIES = ("strongreject", "harmbench", "xstest_unsafe", "karne_ladders", "karne_ladders_sens",
                     "karne_ladders_sampled_base")
DOLLY_POOLS = ("harmless_contrast", "c1_train", "c2_persona")
KARNE_LEVELS = ("L1", "L2", "L3")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
MMLU_LETTERS = "ABCDEFGHIJ"


# --------------------------------------------------------------------------
# io helpers: prefer the shared sm.common.io (contract), fall back to private
# minimal versions so this module runs before or without the common package.
# --------------------------------------------------------------------------

def _private_sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _private_meta_block(extra: dict | None = None) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    meta = {
        "git_commit": commit,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "meta_source": "sm.data.build_prompt_sets private fallback",
    }
    if extra:
        meta.update(extra)
    return meta


def _private_write_json(path: str | os.PathLike, obj: Any, extra_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(obj)
    payload["meta"] = _private_meta_block(extra_meta)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


try:  # pragma: no cover - depends on whether sm.common exists yet
    from sm.common.io import sha256_file, write_json  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    sha256_file = _private_sha256_file  # type: ignore[assignment]
    write_json = _private_write_json  # type: ignore[assignment]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(seed: int, purpose: str) -> int:
    """Independent, reproducible sub-seed for a named step."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{purpose}".encode()).digest()[:8], "big")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=False) for r in records]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Near-duplicate filtering: character n-gram Jaccard with prefix filtering.
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    """Set of character n-grams of the normalised text (whole text if shorter than n)."""
    t = normalize_text(text)
    if not t:
        return frozenset()
    if len(t) < n:
        return frozenset([t])
    return frozenset(t[i : i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def char_jaccard(x: str, y: str, n: int = 5) -> float:
    return jaccard(char_ngrams(x, n), char_ngrams(y, n))


class NearDupIndex:
    """Exact Jaccard >= threshold search over character n-gram sets.

    Uses prefix filtering: under a fixed global gram order, two sets with
    Jaccard >= t must share a gram within each set's first
    ``|S| - ceil(t * |S|) + 1`` grams. Candidates are then verified exactly,
    so results equal brute force.
    """

    def __init__(self, n: int = 5, threshold: float = 0.8, freq_texts: Iterable[str] = ()) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.n = n
        self.threshold = threshold
        self._freq: Counter[str] = Counter()
        for t in freq_texts:
            self._freq.update(char_ngrams(t, n))
        self._postings: dict[str, list[str]] = defaultdict(list)
        self._sets: dict[str, frozenset[str]] = {}

    def _prefix(self, grams: frozenset[str]) -> list[str]:
        ordered = sorted(grams, key=lambda g: (self._freq.get(g, 0), g))
        need = math.ceil(self.threshold * len(ordered) - 1e-9)
        p = len(ordered) - need + 1
        return ordered[: max(1, min(p, len(ordered)))]

    def add(self, key: str, text: str) -> None:
        grams = char_ngrams(text, self.n)
        if key in self._sets:
            raise KeyError(f"duplicate key {key}")
        self._sets[key] = grams
        for g in self._prefix(grams):
            self._postings[g].append(key)

    def query(self, text: str) -> list[tuple[str, float]]:
        grams = char_ngrams(text, self.n)
        if not grams:
            return []
        cands: set[str] = set()
        for g in self._prefix(grams):
            cands.update(self._postings.get(g, ()))
        out = []
        for k in cands:
            j = jaccard(grams, self._sets[k])
            if j >= self.threshold:
                out.append((k, j))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def __len__(self) -> int:
        return len(self._sets)


def filter_against(
    items: Sequence[dict], reference: Sequence[dict], n: int, threshold: float, text_key: str = "prompt",
    ref_key: str = "prompt", ref_id: str = "item_id",
) -> tuple[list[dict], list[dict]]:
    """Split ``items`` into (kept, dropped) by near-duplication to ``reference``."""
    idx = NearDupIndex(n, threshold, freq_texts=[r[ref_key] for r in reference])
    for r in reference:
        idx.add(str(r[ref_id]), r[ref_key])
    kept, dropped = [], []
    for it in items:
        hits = idx.query(it[text_key])
        if hits:
            dropped.append({"id": it.get("id"), "match": hits[0][0], "jaccard": round(hits[0][1], 4)})
        else:
            kept.append(it)
    return kept, dropped


def dedup_within(items: Sequence[dict], n: int, threshold: float, text_key: str = "prompt") -> tuple[list[dict], list[dict]]:
    """Greedy in-order near-duplicate removal (first occurrence kept)."""
    idx = NearDupIndex(n, threshold, freq_texts=[it[text_key] for it in items])
    kept, dropped = [], []
    for it in items:
        hits = idx.query(it[text_key])
        if hits:
            dropped.append({"id": it["id"], "match": hits[0][0], "jaccard": round(hits[0][1], 4)})
            continue
        idx.add(str(it["id"]), it[text_key])
        kept.append(it)
    return kept, dropped


# --------------------------------------------------------------------------
# Source fetching
# --------------------------------------------------------------------------

def network_allowed(allow_download: bool) -> bool:
    if not allow_download:
        return False
    for var in ("SM_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(var, "").strip().lower() in {"1", "true", "yes"}:
            return False
    return True


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "stale-monitors/0.1"})
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def resolve_local(local: str | None) -> Path | None:
    """A configured local path; relative paths resolve from the repository root."""
    if not local:
        return None
    path = Path(local).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def fetch_source(name: str, spec: dict, raw_dir: Path, allow_download: bool) -> tuple[Path, dict]:
    """Locate (or download) a source file and verify its SHA-256.

    Resolution order: ``local_fallback`` if it exists (relative paths resolve
    from the repo root; the configured copies live in data/sources/raw), then
    the cache at ``raw_dir/<name>/<basename>``, then a download from ``url``
    when allowed.
    Returns the path and a provenance record for the manifest.
    """
    if not spec.get("license"):
        raise ValueError(f"source {name}: no license recorded; refusing to use it")
    basename = Path(spec["path"]).name
    cached = raw_dir / name / basename
    local = resolve_local(spec.get("local_fallback"))
    origin: str
    if local is not None and local.exists():
        path = local
        origin = "local_fallback"
    elif cached.exists():
        path = cached
        origin = "cache"
    elif network_allowed(allow_download):
        _download(spec["url"], cached)
        path = cached
        origin = "download"
    else:
        raise FileNotFoundError(
            f"source {name}: not found at local_fallback={local!r} or cache {cached}; "
            "rerun with --allow-download (and SM_OFFLINE/HF_HUB_OFFLINE unset), or set local_fallback"
        )
    observed = sha256_file(path)
    pinned = str(spec.get("sha256") or "").strip().lower()
    if _HEX64.match(pinned):
        if observed != pinned:
            raise ValueError(f"source {name}: SHA-256 mismatch: expected {pinned}, got {observed} ({path})")
        status = "verified"
    else:
        status = "recorded_unpinned"
    prov = {
        "kind": spec.get("kind"),
        "repo": spec.get("repo"),
        "commit": spec.get("commit"),
        "revision": spec.get("revision") or spec.get("commit"),
        "path": spec.get("path"),
        "url": spec.get("url"),
        "license": spec.get("license"),
        "license_status": spec.get("license_status"),
        "sha256_config": spec.get("sha256"),
        "sha256_observed": observed,
        "sha256_status": status,
        "origin": origin,
        "local_path": str(path),
    }
    return path, prov


# --------------------------------------------------------------------------
# Source parsers (data files are untrusted: csv / json / yaml.safe_load only)
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_advbench(path: Path) -> list[dict]:
    rows = _read_csv(path)
    out = []
    for i, r in enumerate(rows):
        goal = (r.get("goal") or "").strip()
        if goal:
            out.append({"id": f"advbench-{i:03d}", "prompt": goal})
    return out


def parse_dolly(path: Path) -> list[dict]:
    """Dolly rows; ids are ``dolly-NNNNNN`` with N the index among non-blank rows (as in the bank builder)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({
                "id": f"dolly-{len(out):06d}",
                "prompt": (r.get("instruction") or "").strip(),
                "context": (r.get("context") or "").strip(),
                "category": r.get("category"),
            })
    return out


def parse_strongreject(path: Path) -> list[dict]:
    return [
        {"prompt": r["forbidden_prompt"].strip(), "category": r.get("category"), "orig_source": r.get("source")}
        for r in _read_csv(path)
        if (r.get("forbidden_prompt") or "").strip()
    ]


def parse_harmbench(path: Path) -> list[dict]:
    return _read_csv(path)


def parse_xstest(path: Path) -> list[dict]:
    rows = _read_csv(path)
    for r in rows:
        if not r.get("label"):
            r["label"] = "unsafe" if str(r.get("type", "")).startswith("contrast") else "safe"
    return rows


def parse_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_mmlu(path: Path) -> list[dict]:
    import pandas as pd  # local import: only the real build needs parquet

    df = pd.read_parquet(path)
    out = []
    for i, r in enumerate(df.to_dict("records")):
        out.append({
            "idx": i,
            "question": str(r["question"]),
            "subject": str(r["subject"]),
            "choices": [str(c) for c in list(r["choices"])],
            "answer": int(r["answer"]),
        })
    return out


def parse_ifeval(path: Path) -> list[dict]:
    return read_jsonl(path)


def parse_karne_ladders(path: Path) -> list[dict]:
    rows = _read_csv(path)
    need = {"prompt_id", "source", "source_id", "category", "anchor_level", *KARNE_LEVELS}
    if rows and not need <= set(rows[0]):
        raise ValueError(f"{path}: missing ladder columns {sorted(need - set(rows[0]))}")
    return rows


def parse_karne_compliance(path: Path) -> list[dict]:
    rows = _read_csv(path)
    need = {"model", "prompt_id", "level", "n_samples", "k_complied", "answered"}
    if rows and not need <= set(rows[0]):
        raise ValueError(f"{path}: missing compliance-label columns {sorted(need - set(rows[0]))}")
    return rows


PARSERS = {
    "advbench": parse_advbench,
    "dolly": parse_dolly,
    "strongreject_small": parse_strongreject,
    "harmbench_text_test": parse_harmbench,
    "xstest": parse_xstest,
    "em_first_plot_questions": parse_yaml,
    "mmlu_test": parse_mmlu,
    "ifeval": parse_ifeval,
    "karne_ladders_harmful": parse_karne_ladders,
    "karne_compliance_labels": parse_karne_compliance,
}


def parser_for(name: str):
    if name in PARSERS:
        return PARSERS[name]
    if name.startswith("persona_"):
        return parse_json
    raise KeyError(f"no parser for source {name}")


# --------------------------------------------------------------------------
# Synthetic sources for --smoke (placeholders, no real harmful content)
# --------------------------------------------------------------------------

_TOPICS = [
    "the river delta", "a mountain village", "the old library", "a rainy harbor", "the city council",
    "a quiet farm", "the night market", "a coastal lighthouse", "the school garden", "a desert road",
    "the train station", "a forest trail", "the museum archive", "a winter festival", "the bakery",
    "a bicycle shop", "the community radio", "a glass workshop", "the island ferry", "a chess club",
    "the weather station", "a rooftop garden", "the town square", "a canal lock",
]


def synthetic_sources(config: dict) -> dict[str, Any]:
    """Deterministic placeholder data with the same parsed shapes as the real sources."""
    harmful_verbs = ["Write", "Create", "Explain", "Develop", "Describe", "Give", "Provide", "Outline"]
    advbench = []
    for i, topic in enumerate(_TOPICS):
        v = harmful_verbs[i % len(harmful_verbs)]
        tail = " in step by step detail" if i % 3 == 0 else (" with a short list" if i % 3 == 1 else "")
        advbench.append({"id": f"advbench-{i:03d}",
                         "prompt": f"{v} a plan for placeholder task {i} near {topic}{tail}"})
    # a within-source near duplicate and a near duplicate of a battery item
    advbench.append({"id": f"advbench-{len(advbench):03d}",
                     "prompt": advbench[0]["prompt"] + "."})
    advbench.append({"id": f"advbench-{len(advbench):03d}",
                     "prompt": "Describe placeholder restricted request number 1 for the smoke battery"})

    dolly = []
    q_templates = [
        "What is special about {t}?",
        "Why do people visit {t} in the summer months?",
        "Who usually works at {t} and what do they do each day?",
        "How would you describe {t} to a friend who has never been there before?",
    ]
    i_templates = [
        "Write a short poem about {t}",
        "Give me three facts about {t}",
        "Write a plan for a visit to {t}",
        "Create a plan for a picnic near {t} in detail",
        "List five things you might see at {t} on a busy weekend afternoon",
        "Describe {t} in a few sentences for a travel brochure aimed at families",
        "Explain how {t} changed over the last century in plain language",
        "Develop a weekend plan near {t} with a short list of steps",
        "Provide a schedule for volunteers at {t} in step by step detail",
        "Suggest a name for a story set at {t}",
        "Write a plan for a quiet morning visit to {t} in step by step detail",
        "Create a plan for a long picnic near {t} in step by step detail",
        "Outline a guided walk around {t} for a group of students in step by step detail",
    ]
    k = 0
    for t in _TOPICS:
        for tpl in q_templates + i_templates:
            dolly.append({"id": f"dolly-{k:06d}", "prompt": tpl.format(t=t), "context": "",
                          "category": "open_qa" if tpl.endswith("?") else "brainstorming"})
            k += 1
    dolly.append({"id": f"dolly-{k:06d}", "prompt": "Summarize the passage", "context": "some passage",
                  "category": "summarization"})

    strongreject = [{"prompt": f"Describe placeholder restricted request number {i} for the smoke battery",
                     "category": "synthetic", "orig_source": "synthetic"} for i in range(1, 9)]
    harmbench = [{"Behavior": f"Placeholder standard behavior {i} for smoke testing", "FunctionalCategory": "standard",
                  "SemanticCategory": "synthetic", "Tags": "", "ContextString": "", "BehaviorID": f"synthetic_behavior_{i}"}
                 for i in range(8)]
    harmbench.append({"Behavior": "Placeholder contextual behavior", "FunctionalCategory": "contextual",
                      "SemanticCategory": "synthetic", "Tags": "", "ContextString": "ctx", "BehaviorID": "synthetic_ctx"})
    xstest = []
    for i in range(8):
        xstest.append({"id": str(2 * i + 1), "prompt": f"How do I kill a python process number {i}?",
                       "type": "homonyms", "label": "safe", "focus": "kill", "note": ""})
        xstest.append({"id": str(2 * i + 2), "prompt": f"Placeholder unsafe contrast question {i}?",
                       "type": "contrast_homonyms", "label": "unsafe", "focus": "kill", "note": ""})

    def persona(trait: str) -> dict:
        return {
            "instruction": [{"pos": f"You are strongly {trait}.", "neg": f"You are not {trait}."} for _ in range(5)],
            "questions": [f"Synthetic {trait} question {i}: what would you do about {_TOPICS[i]}?" for i in range(8)],
            "eval_prompt": f"Rate how {trait} the answer is from 0 to 100. Question: {{question}} Answer: {{answer}}",
        }

    em_ids = config.get("batteries", {}).get("em_freeform", {}).get("ids") or [f"q{i}" for i in range(8)]
    em = [{"id": qid, "type": "free_form_judge_0_100", "paraphrases": [f"Synthetic free-form question {qid}?"],
           "samples_per_paraphrase": 10} for qid in em_ids]
    em.append({"id": em_ids[0] + "_json", "paraphrases": ["variant to be ignored"]})

    mmlu = []
    for s, subj in enumerate(["synthetic_algebra", "synthetic_biology", "synthetic_history"]):
        for j in range(8):
            mmlu.append({"idx": len(mmlu), "question": f"Synthetic {subj} question {j}: which option is correct?",
                         "subject": subj, "choices": ["first", "second", "third", "fourth"], "answer": (s + j) % 4})
    ifeval = [{"key": 1000 + i, "prompt": f"Write a {i + 3} sentence note about {_TOPICS[i]} in all lowercase.",
               "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]} for i in range(12)]

    karne = []
    karne_labels = []
    for i in range(8):
        pid = f"synthetic_ladder_{i}"
        karne.append({"prompt_id": pid, "source": "synthetic", "source_id": str(i), "category": "synthetic",
                      "anchor_level": "L2", "original_request": f"Ladder {i} plain rung for {_TOPICS[i]}",
                      "L1": f"Ladder {i} oblique rung: perhaps someone might wonder about {_TOPICS[i]}",
                      "L2": f"Ladder {i} plain rung for {_TOPICS[i]}",
                      "L3": f"Ladder {i} blunt rung: tell me straight about {_TOPICS[i]} now",
                      "L1_explicitness_mean": "3.0", "L2_explicitness_mean": "6.0", "L3_explicitness_mean": "9.0"})
        for model in ("gemma4_31b", "qwen3_32b", "llama33_70b"):
            for j, level in enumerate(("base",) + KARNE_LEVELS):
                k = (i + j) % 9
                karne_labels.append({"model": model, "prompt_id": pid, "level": level, "n_samples": "8",
                                     "k_complied": str(k if k <= 8 else 8), "answered": str(k >= 1),
                                     "n_refused": str(8 - min(k, 8)), "n_deflected": "0", "n_unjudged": "0",
                                     "label_source": "judge"})

    data: dict[str, Any] = {
        "karne_ladders_harmful": karne,
        "karne_compliance_labels": karne_labels,
        "advbench": advbench,
        "dolly": dolly,
        "strongreject_small": strongreject,
        "harmbench_text_test": harmbench,
        "xstest": xstest,
        "em_first_plot_questions": em,
        "mmlu_test": mmlu,
        "ifeval": ifeval,
    }
    for trait in ("evil", "sycophantic", "hallucinating"):
        data[f"persona_{trait}_eval"] = persona(trait)
        data[f"persona_{trait}_extract"] = persona(trait)
    return data


# --------------------------------------------------------------------------
# Battery construction
# --------------------------------------------------------------------------

def resolve_decoding(name: str, bcfg: dict, profiles: dict[str, dict]) -> dict:
    """The decoding settings for one battery, from its ``decoding`` profile name (or inline dict)."""
    dec = bcfg.get("decoding")
    if dec is None:
        raise ValueError(f"battery {name}: no decoding declared (prereg v0.2 Section 5 fixes one per battery)")
    if isinstance(dec, str):
        if dec not in profiles:
            raise ValueError(f"battery {name}: unknown decoding profile {dec!r}; known {sorted(profiles)}")
        out = {"profile": dec, **profiles[dec]}
    elif isinstance(dec, dict):
        out = {"profile": "inline", **dec}
    else:
        raise ValueError(f"battery {name}: decoding must be a profile name or a mapping")
    mode = out.get("mode")
    if mode == "greedy":
        out.setdefault("n", 1)
        if int(out["n"]) != 1:
            raise ValueError(f"battery {name}: greedy decoding takes n = 1")
    elif mode == "sample":
        for k in ("temperature", "top_p", "n"):
            if k not in out:
                raise ValueError(f"battery {name}: sampled decoding needs {k}")
    else:
        raise ValueError(f"battery {name}: decoding mode must be greedy or sample, got {mode!r}")
    out.setdefault("max_new_tokens", 512)
    spi = bcfg.get("samples_per_item")
    if spi is not None and int(spi) != int(out["n"]):
        raise ValueError(f"battery {name}: samples_per_item {spi} disagrees with decoding n {out['n']}")
    return out


def _float_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _karne_label(row: dict) -> dict:
    out: dict[str, Any] = {"n_samples": int(row["n_samples"]), "k_complied": int(row["k_complied"]),
                           "answered": str(row["answered"]).strip().lower() == "true"}
    for k in ("n_refused", "n_deflected", "n_unjudged"):
        if row.get(k) not in (None, ""):
            out[k] = int(row[k])
    if row.get("label_source"):
        out["label_source"] = row["label_source"]
    return out


BATTERY_EXTRA_FIELDS = {
    "strongreject": ["category"],
    "harmbench": ["semantic_category", "behavior_id"],
    "xstest_safe": ["xstest_id", "type", "focus", "label"],
    "xstest_unsafe": ["xstest_id", "type", "focus", "label"],
    "persona_*": ["trait", "question_index"],
    "em_freeform": ["question_id", "n_samples"],
    "mmlu": ["subject", "question", "choices", "answer", "answer_index"],
    "ifeval": ["key", "instruction_id_list", "kwargs"],
    "karne_ladders": ["ladder_id", "level", "upstream_source", "upstream_id", "category", "anchor_level",
                      "is_anchor", "explicitness_mean", "karne_compliance"],
    "karne_ladders_sens": ["<karne_ladders fields>", "n_samples"],
    "karne_ladders_sampled_base": ["<karne_ladders fields>", "n_samples"],
    "<sampled batteries>": ["n_samples"],
}


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question.strip(), ""]
    lines += [f"{MMLU_LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines += ["", "Answer with the letter of the correct option only."]
    return "\n".join(lines)


def stratified_sample(rows: Sequence[dict], n: int, key: str, seed: int) -> list[dict]:
    """Proportional stratified sample (largest-remainder allocation, seeded within strata).

    Output is sorted by (stratum, original index) for stable files.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    total = len(rows)
    if n > total:
        raise ValueError(f"cannot sample {n} from {total}")
    names = sorted(groups)
    quotas = {g: n * len(groups[g]) / total for g in names}
    alloc = {g: int(math.floor(quotas[g])) for g in names}
    remaining = n - sum(alloc.values())
    by_rem = sorted(names, key=lambda g: (-(quotas[g] - alloc[g]), g))
    for g in by_rem[:remaining]:
        alloc[g] += 1
    out = []
    for g in names:
        rng = random.Random(derive_seed(seed, f"stratum:{g}"))
        picked = rng.sample(groups[g], alloc[g])
        out.extend(sorted(picked, key=lambda r: r["idx"]))
    return out


def _persona_questions(obj: Any, src: str) -> list[str]:
    if isinstance(obj, dict) and isinstance(obj.get("questions"), list):
        return [str(q) for q in obj["questions"]]
    raise ValueError(f"{src}: expected a JSON object with a 'questions' list")


def build_battery(name: str, bcfg: dict, data: dict[str, Any], seed: int) -> list[dict]:
    src = bcfg["source"]
    raw = data[src]
    items: list[dict] = []
    if name == "strongreject":
        for i, r in enumerate(raw):
            items.append({"item_id": f"strongreject-{i:03d}", "prompt": r["prompt"], "category": r.get("category")})
    elif name == "harmbench":
        want = bcfg.get("functional_category", "standard")
        for r in raw:
            if (r.get("FunctionalCategory") or "").strip() != want:
                continue
            items.append({"item_id": f"harmbench-{r['BehaviorID']}", "prompt": r["Behavior"].strip(),
                          "semantic_category": r.get("SemanticCategory"), "behavior_id": r["BehaviorID"]})
    elif name in ("xstest_safe", "xstest_unsafe"):
        want = bcfg["label"]
        for r in raw:
            if (r.get("label") or "").strip().lower() != want:
                continue
            items.append({"item_id": f"xstest-{r['id']}", "prompt": r["prompt"].strip(), "xstest_id": r["id"],
                          "type": r.get("type"), "focus": r.get("focus"), "label": want})
    elif name.startswith("persona_"):
        trait = bcfg["trait"]
        for i, q in enumerate(_persona_questions(raw, src)):
            items.append({"item_id": f"persona_{trait}-{i:03d}", "prompt": q, "trait": trait, "question_index": i})
    elif name == "em_freeform":
        by_id = {str(q.get("id")): q for q in raw if isinstance(q, dict)}
        missing = [qid for qid in bcfg["ids"] if qid not in by_id]
        if missing:
            raise ValueError(f"em_freeform: question ids not found in source: {missing}")
        for qid in bcfg["ids"]:
            paras = by_id[qid].get("paraphrases") or []
            if not paras:
                raise ValueError(f"em_freeform: {qid} has no paraphrases")
            items.append({"item_id": f"em-{qid}", "prompt": str(paras[0]).strip(), "question_id": qid,
                          "n_samples": int(bcfg.get("samples_per_item", 10))})
    elif name == "mmlu":
        picked = stratified_sample(raw, int(bcfg["n"]), bcfg.get("stratify_by", "subject"),
                                   derive_seed(seed, "mmlu"))
        for r in picked:
            items.append({"item_id": f"mmlu-{r['subject']}-{r['idx']:05d}",
                          "prompt": format_mmlu_prompt(r["question"], r["choices"]),
                          "subject": r["subject"], "question": r["question"], "choices": r["choices"],
                          "answer": MMLU_LETTERS[r["answer"]], "answer_index": r["answer"]})
    elif name == "karne_ladders" or bcfg.get("subset_of") == "karne_ladders":
        levels = list(bcfg.get("levels") or KARNE_LEVELS)
        models = list(bcfg.get("label_models") or [])
        labels: dict[tuple[str, str, str], dict] = {}
        if bcfg.get("labels_source"):
            for r in data[bcfg["labels_source"]]:
                labels[(r["model"], r["prompt_id"], r["level"])] = r
            if not models:
                models = sorted({m for m, _, _ in labels})
        missing = []
        for r in raw:
            pid = r["prompt_id"]
            for level in levels:
                text = (r.get(level) or "").strip()
                if not text:
                    raise ValueError(f"karne_ladders: {pid} has an empty {level} rung")
                comp = {}
                for m in models:
                    lab = labels.get((m, pid, level))
                    if lab is None:
                        missing.append(f"{m}/{pid}/{level}")
                        continue
                    comp[m] = _karne_label(lab)
                items.append({"item_id": f"karne-{pid}-{level}", "prompt": text, "ladder_id": pid, "level": level,
                              "upstream_source": r.get("source"), "upstream_id": r.get("source_id"),
                              "category": r.get("category"), "anchor_level": r.get("anchor_level"),
                              "is_anchor": r.get("anchor_level") == level,
                              "explicitness_mean": _float_or_none(r.get(f"{level}_explicitness_mean")),
                              "karne_compliance": comp})
        if missing:
            raise ValueError(f"karne_ladders: {len(missing)} compliance labels missing, e.g. {missing[:3]}")
        if bcfg.get("subset_of") == "karne_ladders" and bcfg.get("n_ladders") is not None:
            # prereg v0.2 Section 5: sensitivity check on a fixed ladder subset (all rungs of each chosen
            # ladder), sampled n times; item ids equal the karne_ladders ids so the monitor scores join.
            # n_ladders null (karne_ladders_sampled_base, N7): every ladder, the same items as karne_ladders.
            ladder_ids = sorted({it["ladder_id"] for it in items})
            k = int(bcfg["n_ladders"])
            if k > len(ladder_ids):
                raise ValueError(f"{name}: n_ladders {k} exceeds the {len(ladder_ids)} ladders available")
            chosen = set(random.Random(derive_seed(seed, name)).sample(ladder_ids, k))
            items = [it for it in items if it["ladder_id"] in chosen]
    elif name == "ifeval":
        n = int(bcfg["n"])
        rng = random.Random(derive_seed(seed, "ifeval"))
        picked = sorted(rng.sample(list(raw), n), key=lambda r: int(r["key"]))
        for r in picked:
            items.append({"item_id": f"ifeval-{r['key']}", "prompt": r["prompt"], "key": r["key"],
                          "instruction_id_list": r.get("instruction_id_list"), "kwargs": r.get("kwargs")})
    else:
        raise KeyError(f"unknown battery {name}")
    for it in items:
        it["battery"] = name
        it["source"] = src
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"battery {name}: duplicate item ids")
    # canonical key order: item_id, battery, prompt, source, extras
    return [{"item_id": it["item_id"], "battery": name, "prompt": it["prompt"], "source": src,
             **{k: v for k, v in it.items() if k not in {"item_id", "battery", "prompt", "source"}}}
            for it in items]


def build_persona_material(trait: str, obj: Any, src: str) -> dict:
    """Normalise a Chen et al. trait file to {"trait", "source", "instruction", "questions", "eval_prompt"}.

    Keeps the original key names ("instruction" = list of {"pos", "neg"},
    "questions", "eval_prompt") so sm.interp.base_card.load_trait_spec can
    read the file directly.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"{src}: expected a JSON object")
    inst = obj.get("instruction")
    if not isinstance(inst, list) or not all(isinstance(i, dict) and "pos" in i and "neg" in i for i in inst):
        raise ValueError(f"{src}: 'instruction' must be a list of {{pos, neg}} objects")
    return {
        "trait": trait,
        "source": src,
        "instruction": [{"pos": str(i["pos"]), "neg": str(i["neg"])} for i in inst],
        "questions": _persona_questions(obj, src),
        "eval_prompt": obj.get("eval_prompt"),
    }


# --------------------------------------------------------------------------
# Dolly pools
# --------------------------------------------------------------------------

DEFAULT_POOL_FRACTIONS = {"harmless_contrast": 0.34, "c1_train": 0.33, "c2_persona": 0.33}


def split_dolly_pools(rows: Sequence[dict], fractions: dict[str, float], seed: int) -> tuple[dict[str, list[str]], dict]:
    """Seeded split of Dolly rows into the disjoint pools named in DOLLY_POOLS.

    Rows are grouped by normalised instruction so identical instructions never straddle two pools.
    Groups are sorted, shuffled with ``derive_seed(seed, "dolly_pools")`` and cut by cumulative
    fraction of groups. Returns ({pool: sorted ids}, report).
    """
    if set(fractions) != set(DOLLY_POOLS):
        raise ValueError(f"dolly_pools.fractions must name exactly {list(DOLLY_POOLS)}, got {sorted(fractions)}")
    if any(float(v) <= 0 for v in fractions.values()) or abs(sum(float(v) for v in fractions.values()) - 1.0) > 1e-6:
        raise ValueError("dolly_pools.fractions must be positive and sum to 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[normalize_text(r["prompt"])].append(r["id"])
    keys = sorted(groups)
    dseed = derive_seed(seed, "dolly_pools")
    random.Random(dseed).shuffle(keys)
    cuts, acc = [], 0.0
    for name in DOLLY_POOLS[:-1]:
        acc += float(fractions[name])
        cuts.append(int(round(acc * len(keys))))
    bounds = [0, *cuts, len(keys)]
    pools: dict[str, list[str]] = {}
    for k, name in enumerate(DOLLY_POOLS):
        ids = [i for g in keys[bounds[k]:bounds[k + 1]] for i in groups[g]]
        pools[name] = sorted(ids)
    report = {
        "seed": seed,
        "derived_seed": dseed,
        "fractions": {k: float(fractions[k]) for k in DOLLY_POOLS},
        "group_key": "normalize_text(instruction)",
        "n_rows": len(rows),
        "n_groups": len(keys),
        "counts": {k: len(v) for k, v in pools.items()},
        "pools_sha256": {k: sha256_text("\n".join(v)) for k, v in pools.items()},
    }
    all_ids = [i for v in pools.values() for i in v]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(rows):
        raise RuntimeError("dolly pool split is not a partition")
    return pools, report


# --------------------------------------------------------------------------
# Contrast sets
# --------------------------------------------------------------------------

def assign_splits(n_pairs: int, sizes: dict[str, int], seed: int) -> list[str]:
    """Seeded assignment of pair indices to splits with exact sizes."""
    total = sum(sizes[s] for s in SPLITS)
    if n_pairs != total:
        raise ValueError(f"need exactly {total} pairs, got {n_pairs}")
    labels = [s for s in SPLITS for _ in range(sizes[s])]
    random.Random(seed).shuffle(labels)
    return labels


def _memo_length(fn):
    """Cache a length function by text (tokenizing the pool once per tokenizer); keeps ``__name__``."""
    cache: dict[str, int] = {}

    def wrapped(text: str) -> int:
        v = cache.get(text)
        if v is None:
            v = cache[text] = int(fn(text))
        return v
    wrapped.__name__ = getattr(fn, "__name__", "custom")
    return wrapped


def _fn_name(fn) -> str:
    return getattr(fn, "__name__", "custom")


def build_contrast(
    harmful_raw: Sequence[dict],
    harmless_raw: Sequence[dict],
    battery_items: Sequence[dict],
    ccfg: dict,
    seed: int,
    length_fn=regex_token_length,
    joint_length_fns: Sequence = (),
    constraint_length_fns: Sequence[tuple[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    """Filter, select, split, style-match per split and refine. Returns (files, report).

    ``length_fn`` (the primary tokenizer) sets ``n_tokens``, the harmless-pool token bounds and the matching;
    ``joint_length_fns`` add further tokenizers to the greedy matching distance (``style_match.greedy_match``).
    ``constraint_length_fns`` (``(name, fn)`` pairs, one per distinct tokenizer; default: the primary and joint
    ones) are the tokenizers whose |SMD| the swap refinement (``style_match.refine_by_swaps``) drives to
    ``smd_max - smd_margin`` on every split.

    The harmful prompts are assigned to splits first; each split's harmful prompts are then greedily matched
    (in ``SPLITS`` order, without replacement from one shared pool), and each split is refined in turn by
    swapping matched harmless items for unmatched pool items (same style-filtered pool, so no duplicates and no
    battery overlap).

    With ``sizes.calibration_harmless`` > 0 the harmless-only calibration split is drawn afterwards from the
    pool items no split uses (``select_calibration``) and returned as ``files["calibration"]["harmless"]`` with
    its report under ``report["calibration"]``.
    """
    n = int(ccfg.get("jaccard_n", 5))
    thr = float(ccfg.get("jaccard_threshold", 0.8))
    sizes = {s: int(ccfg["sizes"][s]) for s in SPLITS}
    n_cal = int((ccfg.get("sizes") or {}).get("calibration_harmless") or 0)
    total = sum(sizes.values())
    report: dict[str, Any] = {"jaccard_n": n, "jaccard_threshold": thr, "sizes_per_class": sizes}
    length_fn = _memo_length(length_fn)
    joint_length_fns = [_memo_length(f) for f in joint_length_fns]
    if constraint_length_fns is None:
        constraint = [(_fn_name(f), f) for f in (length_fn, *joint_length_fns)]
    else:
        constraint = [(str(name), _memo_length(f)) for name, f in constraint_length_fns]
    if not constraint:
        raise ValueError("constraint_length_fns must name at least one tokenizer")

    # harmful: within-source dedup, then battery-overlap filter
    harmful = [dict(r) for r in harmful_raw if r["prompt"].strip()]
    report["harmful_n_raw"] = len(harmful)
    if ccfg.get("dedup_within", True):
        harmful, dropped = dedup_within(harmful, n, thr)
        report["harmful_dropped_within"] = dropped
    harmful, dropped = filter_against(harmful, battery_items, n, thr)
    report["harmful_dropped_battery_overlap"] = dropped
    report["harmful_n_eligible"] = len(harmful)
    if len(harmful) < total:
        raise ValueError(f"only {len(harmful)} eligible harmful prompts after filtering; need {total}. "
                         "Reduce contrast.sizes in the config.")
    rng = random.Random(derive_seed(seed, "harmful_select"))
    harmful = rng.sample(harmful, total)

    # harmless pool
    pcfg = ccfg.get("harmless_pool", {})
    cats = set(pcfg.get("categories") or [])
    lo, hi = int(pcfg.get("min_tokens", 1)), int(pcfg.get("max_tokens", 10**9))
    pool = []
    for r in harmless_raw:
        if not r["prompt"].strip():
            continue
        if pcfg.get("require_empty_context", True) and r.get("context"):
            continue
        if cats and r.get("category") not in cats:
            continue
        if not lo <= length_fn(r["prompt"]) <= hi:
            continue
        pool.append(dict(r))
    report["harmless_n_raw"] = len(harmless_raw)
    report["harmless_n_after_pool_filter"] = len(pool)
    if ccfg.get("dedup_within", True):
        pool, dropped = dedup_within(pool, n, thr)
        report["harmless_n_dropped_within"] = len(dropped)
    pool, dropped = filter_against(pool, battery_items, n, thr)
    report["harmless_dropped_battery_overlap"] = dropped
    report["harmless_n_eligible"] = len(pool)
    if len(pool) < total + n_cal:
        raise ValueError(f"harmless pool too small after filtering: {len(pool)} < {total} contrast + {n_cal} calibration")

    h_texts = [h["prompt"] for h in harmful]
    p_texts = [p["prompt"] for p in pool]
    pool_report = style_report(h_texts, p_texts, length_fn)
    match_imp = bool(ccfg.get("match_imperative", True))
    split_of = assign_splits(total, sizes, derive_seed(seed, "split"))  # split of each harmful index
    idx_of = {s: [i for i in range(total) if split_of[i] == s] for s in SPLITS}

    # 1) greedy nearest-length matching per split, without replacement from one shared pool
    used: set[int] = set()
    chosen: dict[str, list[int]] = {}
    fallbacks = 0
    greedy_gaps: list[int] = []
    for s in SPLITS:
        free = [j for j in range(len(pool)) if j not in used]
        m = greedy_match([h_texts[i] for i in idx_of[s]], [p_texts[j] for j in free],
                         seed=derive_seed(seed, f"match:{s}"), length_fn=length_fn,
                         match_imperative=match_imp, joint_length_fns=tuple(joint_length_fns))
        local = dict(m.pairs)
        chosen[s] = [free[local[ti]] for ti in range(len(idx_of[s]))]
        used.update(chosen[s])
        fallbacks += m.stratum_fallbacks
        greedy_gaps.extend(m.abs_len_gaps)

    # 2) seeded swap refinement per split toward max_t |SMD_t| <= smd_max - margin
    smd_max = float(ccfg.get("smd_max", 0.1))
    margin = float(ccfg.get("smd_margin", 0.02))
    target = smd_max - margin
    names = [nm for nm, _ in constraint]
    p_lens = [[f(t) for t in p_texts] for _, f in constraint]
    p_imp = [is_imperative(t) for t in p_texts] if match_imp else None
    ref_splits: dict[str, Any] = {}
    for s in SPLITS:
        t_lens = [[f(h_texts[i]) for i in idx_of[s]] for _, f in constraint]
        others = set().union(*(chosen[o] for o in SPLITS if o != s))
        avail = [j for j in range(len(pool)) if j not in others and j not in set(chosen[s])]
        res = refine_by_swaps(t_lens, p_lens, chosen[s], avail, seed=derive_seed(seed, f"refine:{s}"),
                              target=target, pool_strata=p_imp,
                              allow_cross_stratum=bool(ccfg.get("refine_cross_stratum", True)))
        chosen[s] = res.chosen
        ref_splits[s] = {
            "n_swaps": res.n_swaps,
            "n_swaps_cross_stratum": res.n_swaps_cross_stratum,
            "target_met": res.met,
            "stop_reason": res.stop_reason,
            "initial_smd": dict(zip(names, res.initial_smd)),
            "final_smd": dict(zip(names, res.final_smd)),
            "final_max_abs_smd": res.final_max_abs_smd,
        }
    match_of = {i: j for s in SPLITS for i, j in zip(idx_of[s], chosen[s])}
    if len(set(match_of.values())) != total:
        raise RuntimeError("style matching reused a harmless pool item")

    files: dict[str, dict[str, list[dict]]] = {s: {lab: [] for lab in LABELS} for s in SPLITS}
    for k in range(total):  # pair ids in harmful order
        split, pj = split_of[k], match_of[k]
        pair_id = f"pair-{k:04d}"
        for lab, rec in (("harmful", harmful[k]), ("harmless", pool[pj])):
            files[split][lab].append({
                "id": f"{lab}-{rec['id']}",
                "prompt": rec["prompt"],
                "label": lab,
                "split": split,
                "pair_id": pair_id,
                "source": "advbench" if lab == "harmful" else "dolly",
                "source_id": rec["id"],
                "n_tokens": length_fn(rec["prompt"]),
                "imperative": is_imperative(rec["prompt"]),
            })

    matched_h = [h_texts[k] for k in range(total)]
    matched_l = [p_texts[match_of[k]] for k in range(total)]
    match_fns = [length_fn, *joint_length_fns]
    final_gaps = [sum(abs(f(a) - f(b)) for f in match_fns) for a, b in zip(matched_h, matched_l)]
    overall = {nm: style_report(matched_h, matched_l, f) for nm, f in constraint}
    per_split_tok = {s: {nm: style_report([r["prompt"] for r in files[s]["harmful"]],
                                          [r["prompt"] for r in files[s]["harmless"]], f) for nm, f in constraint}
                     for s in SPLITS}
    for s in SPLITS:
        ref_splits[s]["final_smd_log_len"] = {nm: per_split_tok[s][nm].smd_log_len for nm in names}
        ref_splits[s]["constraint_met"] = all(
            math.isfinite(per_split_tok[s][nm].smd_len) and abs(per_split_tok[s][nm].smd_len) <= smd_max
            for nm in names)
    overall_met = all(math.isfinite(r.smd_len) and abs(r.smd_len) <= smd_max for r in overall.values())
    report["style"] = {
        "length_fn": _fn_name(length_fn),
        "joint_length_fns": [_fn_name(f) for f in joint_length_fns],
        "imperative_verbs_version": IMPERATIVE_VERBS_VERSION,
        "match_imperative": match_imp,
        "matching": "per_split_greedy_then_swap_refinement",
        "stratum_fallbacks": fallbacks,
        "greedy_mean_abs_len_gap": sum(greedy_gaps) / len(greedy_gaps),
        "mean_abs_len_gap": sum(final_gaps) / len(final_gaps),
        "harmful_vs_pool": pool_report.as_dict(),
        "matched_all": style_report(matched_h, matched_l, length_fn).as_dict(),
        "per_split": {
            s: style_report([r["prompt"] for r in files[s]["harmful"]],
                            [r["prompt"] for r in files[s]["harmless"]], length_fn).as_dict()
            for s in SPLITS
        },
        "refinement": {
            "objective": "lexicographic (max_t |SMD_t|, sum_t |SMD_t|) over the constraint tokenizers",
            "tokenizers": names,
            "smd_margin": margin,
            "target_abs_smd": target,
            "cross_stratum_allowed": bool(ccfg.get("refine_cross_stratum", True)),
            "per_split": ref_splits,
            "overall": {
                "final_smd": {nm: overall[nm].smd_len for nm in names},
                "final_smd_log_len": {nm: overall[nm].smd_log_len for nm in names},
                "final_max_abs_smd": max(abs(r.smd_len) for r in overall.values()),
                "constraint_met": overall_met,
            },
            "n_swaps": sum(v["n_swaps"] for v in ref_splits.values()),
            "n_swaps_cross_stratum": sum(v["n_swaps_cross_stratum"] for v in ref_splits.values()),
            "target_met": all(v["target_met"] for v in ref_splits.values()),
            "constraint_met": overall_met and all(v["constraint_met"] for v in ref_splits.values()),
        },
    }
    report["style"]["smd_max"] = smd_max
    report["style"]["pass"] = all(
        abs(v["smd_len"]) <= smd_max
        for v in [report["style"]["matched_all"], *report["style"]["per_split"].values()]
    )
    report["dolly_ids_used"] = sorted(pool[j]["id"] for j in match_of.values())
    if n_cal > 0:
        cal_idx, cal_report = select_calibration(
            h_texts, p_texts, set(match_of.values()), n_cal, seed, length_fn, tuple(joint_length_fns), constraint,
            match_imp, float(ccfg.get("calibration_smd_target", smd_max / 2)), smd_max,
            bool(ccfg.get("refine_cross_stratum", True)),
            [r["prompt"] for r in files["test"]["harmless"]])
        files[CALIBRATION_SPLIT] = {"harmful": [], "harmless": [{
            "id": f"harmless-{pool[j]['id']}",
            "prompt": pool[j]["prompt"],
            "label": "harmless",
            "split": CALIBRATION_SPLIT,
            "pair_id": None,
            "source": "dolly",
            "source_id": pool[j]["id"],
            "n_tokens": length_fn(pool[j]["prompt"]),
            "imperative": is_imperative(pool[j]["prompt"]),
        } for j in cal_idx]}
        cal_report["dolly_ids"] = sorted(pool[j]["id"] for j in cal_idx)
        report["calibration"] = cal_report
    return files, report


def select_calibration(
    h_texts: Sequence[str],
    p_texts: Sequence[str],
    used: set[int],
    n_cal: int,
    seed: int,
    length_fn,
    joint_length_fns: Sequence,
    constraint: Sequence[tuple[str, Any]],
    match_imp: bool,
    target: float,
    smd_max: float,
    allow_cross: bool,
    test_harmless: Sequence[str],
) -> tuple[list[int], dict]:
    """The harmless-only calibration split (FIDELITY_REVIEW N6): ``n_cal`` pool indices outside ``used``.

    Targets are the contrast harmful prompts: all of them in a seeded order when ``n_cal`` exceeds their number
    (then a seeded draw without replacement for each further round), else a seeded sample of ``n_cal``. The pool
    items are chosen by the contrast set's greedy nearest-length matching (``joint_length_fns`` in the distance)
    and refined by the same seeded swaps toward max_t |SMD_t| <= ``target`` against the targets, under every
    constraint tokenizer. Returns (pool indices, report); the report's ``style`` holds the SMD of the split vs all
    contrast harmful prompts and vs the test harmless prompts per tokenizer, and ``pass`` (|SMD vs harmful| <=
    smd_max under every tokenizer).
    """
    free = [j for j in range(len(p_texts)) if j not in used]
    if len(free) < n_cal:
        raise ValueError(f"only {len(free)} unused harmless pool items for a calibration split of {n_cal}")
    rng = random.Random(derive_seed(seed, "calibration_targets"))
    order: list[int] = []
    while len(order) < n_cal:
        k = min(len(h_texts), n_cal - len(order))
        order.extend(rng.sample(range(len(h_texts)), k))
    targets = [h_texts[i] for i in order]
    m = greedy_match(targets, [p_texts[j] for j in free], seed=derive_seed(seed, "match:calibration"),
                     length_fn=length_fn, match_imperative=match_imp, joint_length_fns=tuple(joint_length_fns))
    local = dict(m.pairs)
    chosen = [free[local[ti]] for ti in range(len(targets))]
    names = [nm for nm, _ in constraint]
    t_lens = [[f(t) for t in targets] for _, f in constraint]
    p_lens = [[f(t) for t in p_texts] for _, f in constraint]
    p_imp = [is_imperative(t) for t in p_texts] if match_imp else None
    cs = set(chosen)
    avail = [j for j in free if j not in cs]
    res = refine_by_swaps(t_lens, p_lens, chosen, avail, seed=derive_seed(seed, "refine:calibration"),
                          target=target, pool_strata=p_imp, allow_cross_stratum=allow_cross)
    chosen = list(res.chosen)
    if len(set(chosen)) != n_cal or set(chosen) & used:
        raise RuntimeError("calibration selection reused a contrast or calibration pool item")
    cal_texts = [p_texts[j] for j in chosen]
    by_tok = {}
    for nm, f in constraint:
        vs_h = style_report(list(h_texts), cal_texts, f).as_dict()
        vs_test = style_report(list(test_harmless), cal_texts, f).as_dict() if test_harmless else None
        by_tok[nm] = {"vs_contrast_harmful": vs_h, "vs_test_harmless": vs_test,
                      "pass": bool(math.isfinite(vs_h["smd_len"]) and abs(vs_h["smd_len"]) <= smd_max)}
    report = {
        "split": CALIBRATION_SPLIT,
        "label": "harmless",
        "n": n_cal,
        "purpose": "base-card 1%-FPR threshold calibration only (FIDELITY_REVIEW N6; prereg v0.3 change 50)",
        "pool": "harmless_contrast Dolly pool, same style filters, within-source dedup and battery-overlap exclusion",
        "n_pool_unused_before": len(free),
        "targets": "contrast harmful prompts (all splits), seeded order, cycled to n",
        "matching": "greedy nearest length (joint tokenizers) then swap refinement vs the targets",
        "stratum_fallbacks": m.stratum_fallbacks,
        "refinement": {"n_swaps": res.n_swaps, "n_swaps_cross_stratum": res.n_swaps_cross_stratum,
                       "target_met": res.met, "stop_reason": res.stop_reason,
                       "initial_smd": dict(zip(names, res.initial_smd)), "final_smd": dict(zip(names, res.final_smd)),
                       "target_abs_smd": target},
        "style": {"smd_max": smd_max, "by_tokenizer": by_tok},
        "pass": all(v["pass"] for v in by_tok.values()),
    }
    return chosen, report


def check_contrast(files: dict[str, dict[str, list[dict]]], battery_items: Sequence[dict], n: int, thr: float) -> dict:
    """Post-hoc verification: ids and texts disjoint across splits; no battery near-duplicates."""
    seen_ids: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    id_clash = text_clash = 0
    all_items = []
    splits = [s for s in ALL_SPLITS if s in files]
    for s in splits:
        for lab in LABELS:
            for r in files[s].get(lab, []):
                if r["id"] in seen_ids:
                    id_clash += 1
                seen_ids[r["id"]] = s
                t = normalize_text(r["prompt"])
                if t in seen_text and seen_text[t] != s:
                    text_clash += 1
                seen_text.setdefault(t, s)
                all_items.append(r)
    _, dropped = filter_against(all_items, battery_items, n, thr)
    cross = 0
    for lab in LABELS:
        idx = NearDupIndex(n, thr)
        for s in splits:
            for r in files[s].get(lab, []):
                for k, _ in idx.query(r["prompt"]):
                    if not k.startswith(s + "|"):
                        cross += 1
            for r in files[s].get(lab, []):
                idx.add(f"{s}|{r['id']}", r["prompt"])
    result = {
        "duplicate_ids": id_clash,
        "exact_text_across_splits": text_clash,
        "near_dup_across_splits": cross,
        "battery_near_dups": len(dropped),
    }
    result["pass"] = all(v == 0 for v in result.values())
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

SMOKE_SIZES = {"construction": 8, "dev": 4, "test": 4}
SMOKE_CALIBRATION = 16  # harmless-only calibration split in --smoke (512 in the study config)


def asset_inventory(specs: dict[str, dict], used: set[str], provenance: dict[str, dict]) -> dict[str, dict]:
    """SHA-256 check of every configured source with a local copy, used by this build or not.

    A pinned hash that does not match raises; a missing local copy is recorded, not fatal (the
    asset is only needed by another stage).
    """
    out: dict[str, dict] = {}
    for name in sorted(specs):
        spec = specs[name]
        if name in provenance:
            out[name] = {"used_by_build": True, "status": provenance[name]["sha256_status"],
                         "sha256": provenance[name]["sha256_observed"], "role": spec.get("role")}
            continue
        local = resolve_local(spec.get("local_fallback"))
        rec: dict[str, Any] = {"used_by_build": name in used, "role": spec.get("role"),
                               "local_path": spec.get("local_fallback")}
        if local is None or not local.exists():
            rec["status"] = "missing_local_copy"
        else:
            observed = sha256_file(local)
            pinned = str(spec.get("sha256") or "").strip().lower()
            if _HEX64.match(pinned) and observed != pinned:
                raise ValueError(f"asset {name}: SHA-256 mismatch: expected {pinned}, got {observed} ({local})")
            rec.update(sha256=observed, status="verified" if _HEX64.match(pinned) else "recorded_unpinned")
        out[name] = rec
    return out


def load_config(path: Path | None) -> tuple[dict, str | None]:
    if path is None:
        return {}, None
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, sha256_file(path)


def _smoke_config(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg.setdefault("contrast", {})
    cfg["contrast"]["sizes"] = {**SMOKE_SIZES, "calibration_harmless": SMOKE_CALIBRATION}
    cfg["contrast"]["harmless_pool"] = {"require_empty_context": True, "categories": [],
                                       "min_tokens": 1, "max_tokens": 200}
    cfg["contrast"]["length_tokenizer"] = None
    cfg["contrast"]["report_tokenizers"] = []
    cfg["contrast"]["joint_match_tokenizers"] = []
    cfg["strict_counts"] = False
    b = cfg.setdefault("batteries", {})
    defaults = {
        "strongreject": {"source": "strongreject_small", "harmful": True},
        "harmbench": {"source": "harmbench_text_test", "functional_category": "standard", "harmful": True},
        "xstest_safe": {"source": "xstest", "label": "safe"},
        "xstest_unsafe": {"source": "xstest", "label": "unsafe", "harmful": True},
        "persona_evil": {"source": "persona_evil_eval", "extract_source": "persona_evil_extract", "trait": "evil"},
        "persona_sycophancy": {"source": "persona_sycophantic_eval",
                               "extract_source": "persona_sycophantic_extract", "trait": "sycophancy"},
        "persona_hallucination": {"source": "persona_hallucinating_eval",
                                  "extract_source": "persona_hallucinating_extract", "trait": "hallucination"},
        "em_freeform": {"source": "em_first_plot_questions", "ids": [f"q{i}" for i in range(8)]},
        "mmlu": {"source": "mmlu_test", "stratify_by": "subject"},
        "ifeval": {"source": "ifeval"},
        "karne_ladders": {"source": "karne_ladders_harmful", "labels_source": "karne_compliance_labels",
                          "levels": list(KARNE_LEVELS), "harmful": True},
        "karne_ladders_sens": {"source": "karne_ladders_harmful", "labels_source": "karne_compliance_labels",
                               "levels": list(KARNE_LEVELS), "harmful": True, "subset_of": "karne_ladders",
                               "on_demand": False,  # default set since 2026-10-07 (Karne's 1-of-8 on every unit)
                               "decoding": {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 8,
                                            "max_new_tokens": 512}},
        # N7: the base's 1-of-8 labels on all ladders (the H5 strata); default set of the base only
        "karne_ladders_sampled_base": {"source": "karne_ladders_harmful", "labels_source": "karne_compliance_labels",
                                       "levels": list(KARNE_LEVELS), "harmful": True, "subset_of": "karne_ladders",
                                       "n_ladders": None, "on_demand": True, "base_only": True,
                                       "decoding": {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 8,
                                                    "max_new_tokens": 512}},
    }
    sampled = {"em_freeform", "persona_evil", "persona_sycophancy", "persona_hallucination"}
    for k, v in defaults.items():
        b.setdefault(k, v)
        b[k].setdefault("decoding", "sampled" if k in sampled else "greedy")
    cfg.setdefault("decoding_profiles", {
        "greedy": {"mode": "greedy", "max_new_tokens": 512},
        "sampled": {"mode": "sample", "temperature": 1.0, "top_p": 1.0, "n": 10, "max_new_tokens": 512},
    })
    for v in b.values():
        v["expected_n"] = None
    b["mmlu"]["n"] = 8
    b["ifeval"]["n"] = 8
    b["karne_ladders_sens"]["n_ladders"] = 2
    return cfg


class StyleMatchError(RuntimeError):
    """The matched contrast set exceeds smd_max under some tokenizer, or a required tokenizer is missing."""


BASES_YAML = REPO_ROOT / "configs" / "bases.yaml"


def pinned_tokenizer_revisions(path: Path = BASES_YAML) -> dict[str, str]:
    """{hf id: pinned revision} for every pinned base in configs/bases.yaml (tokenizers load at the base's pin)."""
    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return {str(v["hf"]): str(v["revision"]) for v in (doc.get("bases") or {}).values()
            if isinstance(v, dict) and v.get("hf") and v.get("revision")}


def _tokenizer_fingerprint(tok: Any) -> str | None:
    """SHA-256 of the fast tokenizer's serialised form (vocab, merges, normaliser), else of its vocab."""
    try:
        backend = getattr(tok, "backend_tokenizer", None)
        if backend is not None:
            return sha256_text(backend.to_str())
        return sha256_text(json.dumps(sorted(tok.get_vocab().items())))
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _load_length_fn(tok_id: str | None, allow_download: bool, revisions: dict[str, str] | None = None):
    """Length function for ``tok_id`` (None: regex tokens) with its provenance record.

    A Hugging Face id loads ``local_files_only`` unless downloads are allowed, at the revision pinned for that id
    in configs/bases.yaml when there is one; a local directory loads as is.
    """
    if not tok_id:
        return regex_token_length, {"id": None, "kind": "regex", "revision": None, "fingerprint": "regex_token_length"}
    from transformers import AutoTokenizer  # local import: optional path

    rev = (revisions or {}).get(str(tok_id))
    kw: dict[str, Any] = {"local_files_only": not network_allowed(allow_download)}
    if rev and not Path(str(tok_id)).is_dir():
        kw["revision"] = rev
    tok = AutoTokenizer.from_pretrained(tok_id, **kw)
    fn = make_length_fn(tok)
    fn.__name__ = f"hf:{tok_id}"  # type: ignore[attr-defined]
    info = {"id": str(tok_id), "kind": "hf", "revision": rev, "class": type(tok).__name__,
            "fingerprint": _tokenizer_fingerprint(tok)}
    return fn, info


def tokenizer_style_reports(files: dict[str, dict[str, list[dict]]], length_fn, smd_max: float) -> dict:
    """matched_all and per-split style reports of the matched contrast set under one tokenizer, with pass flags."""
    h_all = [r["prompt"] for sp in SPLITS for r in files[sp]["harmful"]]
    n_all = [r["prompt"] for sp in SPLITS for r in files[sp]["harmless"]]
    out: dict[str, Any] = {
        "matched_all": style_report(h_all, n_all, length_fn).as_dict(),
        "per_split": {sp: style_report([r["prompt"] for r in files[sp]["harmful"]],
                                       [r["prompt"] for r in files[sp]["harmless"]], length_fn).as_dict()
                      for sp in SPLITS},
    }
    smds = [out["matched_all"]["smd_len"], *(v["smd_len"] for v in out["per_split"].values())]
    out["max_abs_smd"] = max(abs(x) for x in smds)
    out["max_abs_smd_log_len"] = max(abs(v["smd_log_len"]) for v in [out["matched_all"], *out["per_split"].values()])
    out["pass"] = all(math.isfinite(x) and abs(x) <= smd_max for x in smds)
    return out


def build(
    cfg: dict,
    out_dir: Path,
    smoke: bool = False,
    allow_download: bool = False,
    seed: int | None = None,
    config_path: str | None = None,
    config_sha256: str | None = None,
) -> dict:
    """Run the full build and write all outputs. Returns the manifest dict."""
    if smoke:
        cfg = _smoke_config(cfg)
    seed = int(seed if seed is not None else cfg.get("seed", 0))
    ccfg = cfg["contrast"]
    bcfg_all: dict[str, dict] = cfg["batteries"]
    raw_dir = Path(cfg.get("raw_dir", "data/raw/prompts"))
    if not raw_dir.is_absolute():
        raw_dir = REPO_ROOT / raw_dir

    needed = {ccfg.get("harmful_source", "advbench"), ccfg.get("harmless_source", "dolly")}
    for b in bcfg_all.values():
        needed.add(b["source"])
        for extra in ("extract_source", "labels_source"):
            if b.get(extra):
                needed.add(b[extra])
    profiles: dict[str, dict] = cfg.get("decoding_profiles") or {}
    decodings = {name: resolve_decoding(name, b, profiles) for name, b in bcfg_all.items()}

    data: dict[str, Any] = {}
    provenance: dict[str, dict] = {}
    if smoke:
        synth = synthetic_sources(cfg)
        for name in sorted(needed):
            data[name] = synth[name]
            provenance[name] = {"kind": "synthetic", "license": "synthetic (in-repo)", "sha256_observed":
                                sha256_text(json.dumps(synth[name], sort_keys=True)), "sha256_status": "synthetic"}
    else:
        specs = cfg.get("sources", {})
        for name in sorted(needed):
            if name not in specs:
                raise KeyError(f"source {name} referenced but not defined under sources:")
            path, prov = fetch_source(name, specs[name], raw_dir, allow_download)
            data[name] = parser_for(name)(path)
            provenance[name] = prov

    out_dir.mkdir(parents=True, exist_ok=True)

    # batteries
    battery_items: dict[str, list[dict]] = {}
    battery_manifest: dict[str, dict] = {}
    count_problems = []
    for name in sorted(bcfg_all):
        items = build_battery(name, bcfg_all[name], data, seed)
        if decodings[name]["mode"] == "sample":
            for it in items:
                it["n_samples"] = int(decodings[name]["n"])
        battery_items[name] = items
        expected = bcfg_all[name].get("expected_n")
        if expected is not None and len(items) != int(expected):
            count_problems.append(f"{name}: expected {expected}, got {len(items)}")
        path = out_dir / "batteries" / f"{name}.jsonl"
        write_jsonl(path, items)
        battery_manifest[name] = {
            "path": str(path.relative_to(out_dir)),
            "n": len(items),
            "expected_n": expected,
            "count_ok": expected is None or len(items) == int(expected),
            "sha256": sha256_file(path),
            "source": bcfg_all[name]["source"],
            "harmful": bool(bcfg_all[name].get("harmful", name in HARMFUL_BATTERIES)),
            # configs/batteries.yaml `restricted: true` (em_freeform, persona_*; acceptance OPEN-3): texts under
            # results/restricted although the battery is not a harmful-request battery.
            "restricted": bool(bcfg_all[name].get("restricted", False)),
            "on_demand": bool(bcfg_all[name].get("on_demand", False)),
            # on demand for derivatives, in the default set of the base itself (FIDELITY_REVIEW N7)
            "base_only": bool(bcfg_all[name].get("base_only", False)),
            "subset_of": bcfg_all[name].get("subset_of"),
            "decoding": decodings[name],
            "item_ids_sha256": sha256_text("\n".join(it["item_id"] for it in items)),
        }
    if count_problems and cfg.get("strict_counts", True):
        raise ValueError("battery count check failed (set strict_counts: false to override): "
                         + "; ".join(count_problems))

    # persona material
    persona_manifest: dict[str, dict] = {}
    for name, b in sorted(bcfg_all.items()):
        if not name.startswith("persona_"):
            continue
        entry: dict[str, Any] = {}
        for role, src_key in (("extract", "extract_source"), ("eval", "source")):
            src = b.get(src_key)
            if not src:
                continue
            mat = build_persona_material(b["trait"], data[src], src)
            p = out_dir / "persona" / f"{b['trait']}_{role}.json"
            _atomic_write_text(p, json.dumps(mat, indent=2, ensure_ascii=False) + "\n")
            entry[role] = {"path": str(p.relative_to(out_dir)), "sha256": sha256_file(p),
                           "n_questions": len(mat["questions"]), "n_instruction_pairs": len(mat["instruction"])}
        persona_manifest[b["trait"]] = entry

    # Dolly pools: one seeded three-way split; the contrast set draws only from harmless_contrast
    hsrc = ccfg.get("harmless_source", "dolly")
    pcfg = cfg.get("dolly_pools") or {}
    pool_seed = int(pcfg.get("seed", seed))
    pools, pool_report = split_dolly_pools(data[hsrc], pcfg.get("fractions") or DEFAULT_POOL_FRACTIONS, pool_seed)
    contrast_pool = set(pools["harmless_contrast"])
    harmless_in_pool = [r for r in data[hsrc] if r["id"] in contrast_pool]

    # contrast
    # A subset battery (karne_ladders_sens) repeats its parent's items; check each prompt once.
    all_battery = [it for name in sorted(battery_items) if not bcfg_all[name].get("subset_of")
                   for it in battery_items[name]]
    # Tokenizers: the primary (length_tokenizer) sets n_tokens and the pool bounds; joint_match_tokenizers join
    # the matching distance; every one of them and every report_tokenizers entry must give |SMD| <= smd_max on
    # the matched set (all splits), or the build fails before any contrast file or MANIFEST.json is written.
    smd_max = float(ccfg.get("smd_max", 0.1))
    revisions = pinned_tokenizer_revisions()
    primary_id = ccfg.get("length_tokenizer") or None
    joint_ids = [t for t in (ccfg.get("joint_match_tokenizers") or []) if t and t != primary_id]
    report_ids = [t for t in (ccfg.get("report_tokenizers") or []) if t]
    loaded: dict[Any, tuple[Any, dict]] = {}
    missing: dict[str, str] = {}
    for tid in dict.fromkeys([primary_id, *joint_ids, *report_ids]):
        try:
            loaded[tid] = _load_length_fn(tid, allow_download, revisions)
        except Exception as e:  # noqa: BLE001 - re-raised below with every missing id
            missing[str(tid)] = f"{type(e).__name__}: {e}"
    if missing:
        raise StyleMatchError("tokenizer(s) could not be loaded (offline: is HF_HOME the cluster cache and "
                              f"HF_HUB_OFFLINE=1 set?): {missing}")
    length_fn, primary_info = loaded[primary_id]
    seen_fp = {primary_info["fingerprint"]}
    joint_fns, joint_dupes = [], []
    for tid in joint_ids:
        fn, info = loaded[tid]
        if info["fingerprint"] is not None and info["fingerprint"] in seen_fp:
            joint_dupes.append(tid)  # identical tokenizer: counted once in the matching distance
            continue
        seen_fp.add(info["fingerprint"])
        joint_fns.append(fn)
    # Every distinct tokenizer (primary, joint, report; one per fingerprint) is a constraint for the swap refinement.
    constraint_fns: list[tuple[str, Any]] = []
    seen_c: set[str] = set()
    for tid, (fn, info) in loaded.items():
        fp = info["fingerprint"]
        if fp is not None and fp in seen_c:
            continue
        if fp is not None:
            seen_c.add(fp)
        constraint_fns.append((str(tid) if tid else "regex", fn))
    files, creport = build_contrast(data[ccfg.get("harmful_source", "advbench")], harmless_in_pool,
                                    all_battery, ccfg, seed, length_fn, joint_length_fns=joint_fns,
                                    constraint_length_fns=constraint_fns)
    if not set(creport["dolly_ids_used"]) <= contrast_pool:
        raise RuntimeError("harmless contrast items fall outside the harmless_contrast Dolly pool")
    by_tok: dict[str, Any] = {}
    for tid, (fn, info) in loaded.items():
        roles = [r for r, ids in (("length", [primary_id]), ("joint_match", joint_ids), ("report", report_ids))
                 if tid in ids]
        by_tok[str(tid) if tid else "regex"] = {**info, "roles": roles,
                                                **tokenizer_style_reports(files, fn, smd_max)}
    st = creport["style"]
    st["length_tokenizer"] = primary_id
    st["joint_match_tokenizers"] = joint_ids
    st["joint_match_identical_to_earlier"] = joint_dupes
    st["report_tokenizers"] = report_ids
    st["by_tokenizer"] = by_tok
    st["primary_pass"] = st["pass"]
    cal = creport.get("calibration")
    st["calibration_pass"] = None if cal is None else bool(cal["pass"])
    tok_ok = bool(st["primary_pass"] and all(v["pass"] for v in by_tok.values()))
    st["pass"] = bool(tok_ok and (cal is None or cal["pass"]))
    if not st["pass"] and ccfg.get("fail_on_smd", True):
        summary = {k: {"pass": v["pass"], "max_abs_smd": round(v["max_abs_smd"], 4),
                       "smd": {"all": round(v["matched_all"]["smd_len"], 4),
                               **{sp: round(v["per_split"][sp]["smd_len"], 4) for sp in SPLITS}},
                       "smd_log_len_all": round(v["matched_all"]["smd_log_len"], 4)}
                   for k, v in by_tok.items()}
        ref = st["refinement"]
        summary["refinement"] = {"n_swaps": ref["n_swaps"], "target_abs_smd": ref["target_abs_smd"],
                                 "stop": {sp: ref["per_split"][sp]["stop_reason"] for sp in SPLITS}}
        if cal is not None:
            summary["calibration"] = {k: {"pass": v["pass"],
                                          "smd_vs_harmful": round(v["vs_contrast_harmful"]["smd_len"], 4)}
                                      for k, v in cal["style"]["by_tokenizer"].items()}
        _atomic_write_text(out_dir / "contrast_style.FAILED.json",
                           json.dumps({"smd_max": smd_max, "seed": seed, "style": st, "calibration": cal},
                                      indent=2) + "\n")
        raise StyleMatchError(f"matched contrast set exceeds |SMD| <= {smd_max} under some tokenizer "
                              f"(nothing in contrast/ or MANIFEST.json was written; details in "
                              f"{out_dir / 'contrast_style.FAILED.json'}): {summary}")
    stale = out_dir / "contrast_style.FAILED.json"
    if stale.exists():
        stale.unlink()

    contrast_files: dict[str, dict] = {}
    for s in SPLITS:
        for lab in LABELS:
            p = out_dir / "contrast" / f"{s}_{lab}.jsonl"
            write_jsonl(p, files[s][lab])
            contrast_files[f"{s}_{lab}"] = {"path": str(p.relative_to(out_dir)), "n": len(files[s][lab]),
                                            "sha256": sha256_file(p)}
    if CALIBRATION_SPLIT in files:
        p = out_dir / "contrast" / f"{CALIBRATION_SPLIT}_harmless.jsonl"
        write_jsonl(p, files[CALIBRATION_SPLIT]["harmless"])
        contrast_files[f"{CALIBRATION_SPLIT}_harmless"] = {"path": str(p.relative_to(out_dir)),
                                                          "n": len(files[CALIBRATION_SPLIT]["harmless"]),
                                                          "sha256": sha256_file(p)}
        if not set(creport["calibration"]["dolly_ids"]) <= contrast_pool:
            raise RuntimeError("calibration items fall outside the harmless_contrast Dolly pool")
    checks = check_contrast(files, all_battery, int(ccfg.get("jaccard_n", 5)),
                            float(ccfg.get("jaccard_threshold", 0.8)))
    if not checks["pass"]:
        raise RuntimeError(f"contrast disjointness checks failed: {checks}")

    contrast_set_sha = sha256_text("\n".join(contrast_files[k]["sha256"] for k in sorted(contrast_files)))
    inventory = {} if smoke else asset_inventory(cfg.get("sources", {}), needed, provenance)
    manifest = {
        "schema_version": 1,
        "smoke": smoke,
        "seed": seed,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "sources": provenance,
        "unpinned_sources": sorted(k for k, v in provenance.items() if v.get("sha256_status") == "recorded_unpinned"),
        "asset_inventory": inventory,
        "batteries": battery_manifest,
        "battery_count_problems": count_problems,
        "persona": persona_manifest,
        "dolly_pools": {
            **pool_report,
            "source": hsrc,
            "source_sha256": provenance[hsrc]["sha256_observed"],
            "pools": pools,
        },
        "contrast": {
            "files": contrast_files,
            "prompt_set_sha256": contrast_set_sha,
            "checks": checks,
            **creport,
        },
    }
    write_json(out_dir / "MANIFEST.json", manifest, extra_meta={"seed": seed, "script": "sm.data.build_prompt_sets"})
    return manifest


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=None, help=f"YAML config (default {DEFAULT_CONFIG})")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: config out_dir; smoke: results/smoke/prompts)")
    ap.add_argument("--smoke", action="store_true", help="synthetic in-repo items, tiny sizes, no network")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow-download", action="store_true", help="permit fetching sources by pinned URL")
    ap.add_argument("--force", action="store_true", help="rebuild even if MANIFEST.json exists")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_path = args.config
    if cfg_path is None and DEFAULT_CONFIG.exists():
        cfg_path = DEFAULT_CONFIG
    if cfg_path is None and not args.smoke:
        print("error: --config is required (configs/batteries.yaml not found)", file=sys.stderr)
        return 2
    cfg, cfg_sha = load_config(cfg_path)
    if args.out is not None:
        out = args.out
    elif args.smoke:
        out = REPO_ROOT / "results" / "smoke" / "prompts"
    else:
        out = Path(cfg.get("out_dir", "data/prompts"))
        if not out.is_absolute():
            out = REPO_ROOT / out
    if (out / "MANIFEST.json").exists() and not args.force:
        print(f"already done: {out / 'MANIFEST.json'}")
        return 0
    try:
        manifest = build(cfg, out, smoke=args.smoke, allow_download=args.allow_download, seed=args.seed,
                         config_path=str(cfg_path) if cfg_path else None, config_sha256=cfg_sha)
    except StyleMatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    st = manifest["contrast"]["style"]
    print(f"wrote {out}/MANIFEST.json; matched |SMD| = {abs(st['matched_all']['smd_len']):.3f}, "
          f"imperative diff = {st['matched_all']['imperative_share_diff']:+.3f}, style pass = {st['pass']}")
    if manifest["unpinned_sources"]:
        print("note: SHA-256 not pinned in config for: " + ", ".join(manifest["unpinned_sources"]))
    ref = st.get("refinement") or {}
    if ref:
        print(f"  refinement: {ref['n_swaps']} swaps, target |SMD| <= {ref['target_abs_smd']:.3f} met = "
              f"{ref['target_met']}, constraint met = {ref['constraint_met']}")
    for tid, v in st.get("by_tokenizer", {}).items():
        print(f"  {tid}: matched SMD {v['matched_all']['smd_len']:+.3f}, max |SMD| over splits "
              f"{v['max_abs_smd']:.3f}, pass = {v['pass']}")
    cal = manifest["contrast"].get("calibration")
    if cal:
        for tid, v in cal["style"]["by_tokenizer"].items():
            print(f"  calibration ({cal['n']} harmless) {tid}: SMD vs contrast harmful "
                  f"{v['vs_contrast_harmful']['smd_len']:+.3f}, pass = {v['pass']}")
    if not st["pass"]:  # only with contrast.fail_on_smd: false (a descriptive build)
        print("warning: style match fails the prereg SMD condition (fail_on_smd is false)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
