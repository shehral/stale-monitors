"""Capability scores for one run: MMLU letter accuracy (format-invariant parse) and IFEval strict prompt-level accuracy.

Output (v2 harness artifact), consumed by sm.analysis.aggregate for the prereg Section 7 exclusion
(a derivative whose capability drops more than 20 points below the base is excluded):
    <out>/<base>/<run_id>.json
    {"mmlu_acc": float, "mmlu_n": int, "ifeval_strict_acc": float, "ifeval_n": int, "meta": {...}}

Inputs. The greedy battery outputs of sm.eval.battery
    <batteries_root>/<base>/<run_id>/mmlu.jsonl and ifeval.jsonl   (+ their .meta.json sidecars)
and the item files (MMLU answer letters; IFEval instruction ids and kwargs) located through
<items_dir>/MANIFEST.json. The items are the prereg subsets built by sm.data.build_prompt_sets: MMLU
500 items stratified by subject (letter-choice prompt "...Answer with the letter of the correct option
only."), IFEval 200 items. Scoring refuses battery outputs whose sidecar records a different item-file
hash, or non-greedy decoding.
With --generate, missing battery outputs are first produced by sm.eval.battery (same loader, pinned
revision, adapter unmerged, greedy); without it a missing output is an error (exit 5), so a CPU scoring
job never loads a model.

MMLU: ``mmlu_acc`` is the format-invariant letter parse (mmlu_letter): an explicit "answer is X" /
"Answer: X" phrase first, else a leading option letter ("B", "B.", "(B)", "**B**"), else the first standalone
capital A-D. It is the score the prereg Section 7 capability exclusion uses (sm.analysis.aggregate), so a
persona or format-shifted derivative that answers "Ha, it's B!" or "The answer is B" is not counted as a
capability drop. Decided before the first derivative is measured (to be listed in the deviations log as
a clarification of "capability subset drops more than 20 points"). The strict score (the first
whitespace-delimited token must be exactly the gold letter after stripping markdown and punctuation) is
kept in meta as ``mmlu_exact_acc``, a diagnostic. A leading empty <think></think> block is dropped first.

IFEval: strict prompt-level accuracy (every instruction of a prompt followed, response non-empty), as
in test_instruction_following_strict of google-research instruction_following_eval. Checker used, in
order: the official registry if importable (``instruction_following_eval`` or lm-eval-harness
``lm_eval.tasks.ifeval``); else the port below of the official checkers (Apache-2.0) for all 25
instruction types in IFEval. Deviations of the port (recorded in meta["ifeval_deviations"]):
  * sentence counting uses the regex splitter split_into_sentences (instructions_util) instead of
    nltk punkt; capital-word counting uses a regex Treebank approximation instead of nltk.word_tokenize;
  * the English-language part of change_case:english_capital / english_lowercase uses langdetect when
    importable (DetectorFactory.seed = 0), else only the case condition is checked;
  * language:response_language needs langdetect; without it, prompts containing that instruction are
    excluded from ifeval_n and listed in meta["ifeval_excluded"].
No official checker is cloned under third_party/ (docs/CODE_REUSE.md: "not cloned").

CLI:
    python -m sm.eval.capability --base <key> --run-id <id> [--results-root results]
        [--batteries-root results/batteries] [--items-dir data/prompts] [--out results/capability]
        [--generate [--adapter DIR] [--revision SHA]] [--smoke] [--force]
--smoke builds 8 synthetic MMLU and 8 synthetic IFEval items, generates on the tiny model through
sm.eval.battery --smoke (CPU, fp32), and scores them.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

MMLU_BATTERY = "mmlu"
IFEVAL_BATTERY = "ifeval"
LETTERS = ("A", "B", "C", "D")
EXPECTED_N = {MMLU_BATTERY: 500, IFEVAL_BATTERY: 200}
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISSING = 5

# --------------------------------------------------------------------------- MMLU

_THINK_BLOCK = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)
_EXACT_TOKEN = re.compile(r"[\*_`\(\[\"']*([ABCD])[\*_`\)\]\"'\.:,]*")
_LEADING = re.compile(r"^\W*\(?\s*([ABCD])\s*\)?(?:[\s\.\):,]|$)")
# "answer is X" / "Answer: X": the phrase is case-insensitive; the letter is a capital, or a lower-case letter that
# ends the text or is followed by punctuation ("Answer: b"), never an article ("the answer is a bit ...").
_PHRASE = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)?\s*(?:(?i:option)\s*)?\(?\s*([ABCD]\b|[abcd](?=\s*$|[\.\):,!]))")
_STANDALONE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def strip_empty_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "", count=1)


def mmlu_exact_letter(completion: str) -> str | None:
    """The letter if the completion's first token is exactly one option letter, else None."""
    text = strip_empty_think(completion).strip()
    if not text:
        return None
    m = _EXACT_TOKEN.fullmatch(text.split()[0])
    return m.group(1) if m else None


def mmlu_letter(completion: str) -> str | None:
    """Format-invariant parse (the scored mmlu_acc): an explicit answer phrase, else a leading option letter,
    else the first standalone capital A-D; None when the completion commits to no letter."""
    text = strip_empty_think(completion).strip()
    for pat in (_PHRASE, _LEADING, _STANDALONE):
        m = pat.search(text)
        if m:
            return m.group(1).upper()
    return None


def score_mmlu(records: Sequence[dict], items: dict[str, dict]) -> dict[str, Any]:
    """Letter accuracy over records that have a gold answer; per-subject counts in the result.

    mmlu_acc uses the format-invariant parse (mmlu_letter); mmlu_exact_acc the strict first-token parse.
    """
    n = correct = exact = no_letter = 0
    by_subject: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    missing = 0
    for r in records:
        it = items.get(str(r["item_id"]))
        gold = str(it.get("answer", "")).strip().upper() if it else ""
        if gold not in LETTERS:
            missing += 1
            continue
        n += 1
        got = mmlu_letter(r.get("completion", ""))
        ok = got == gold
        correct += ok
        exact += mmlu_exact_letter(r.get("completion", "")) == gold
        no_letter += got is None
        subj = by_subject[str(it.get("subject", "unknown"))]
        subj[0] += ok
        subj[1] += 1
    return {
        "mmlu_acc": correct / n if n else None,
        "mmlu_n": n,
        "mmlu_exact_acc": exact / n if n else None,
        "mmlu_no_letter": no_letter,
        "mmlu_records_without_gold": missing,
        "mmlu_by_subject": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(by_subject.items())},
    }


# --------------------------------------------------------------------------- IFEval checker port
# Port of google-research/instruction_following_eval instructions.py / instructions_util.py (Apache-2.0).

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Regex sentence splitter (instructions_util.split_into_sentences)."""
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>").replace("?", "?<stop>").replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    """nltk RegexpTokenizer(r"\\w+") word count (instructions_util.count_words)."""
    return len(re.findall(r"\w+", text))


def _word_tokenize(text: str) -> list[str]:
    try:
        from nltk.tokenize import TreebankWordTokenizer  # type: ignore

        return TreebankWordTokenizer().tokenize(text)
    except Exception:  # noqa: BLE001  (nltk missing): Treebank-like regex
        return re.findall(r"\w+(?=n't\b)|n't\b|'\w+|\w+|[^\w\s]", text, flags=re.IGNORECASE)


def _langdetect() -> Any | None:
    try:
        import langdetect  # type: ignore

        langdetect.DetectorFactory.seed = 0
        return langdetect
    except Exception:  # noqa: BLE001
        return None


def _rel(actual: int, relation: str, threshold: int) -> bool:
    if relation == "less than":
        return actual < threshold
    if relation == "at least":
        return actual >= threshold
    raise ValueError(f"unknown relation {relation!r}")


def _is_english(value: str) -> bool:
    ld = _langdetect()
    if ld is None:
        return True  # deviation: case condition only (recorded)
    try:
        return ld.detect(value) == "en"
    except Exception:  # noqa: BLE001  (LangDetectException: official returns True)
        return True


def _response_language(value: str, kw: dict, prompt: str) -> bool:
    ld = _langdetect()
    if ld is None:
        raise LookupError("langdetect not importable")
    try:
        return ld.detect(value) == kw["language"]
    except Exception:  # noqa: BLE001
        return True


def _postscript(value: str, kw: dict, prompt: str) -> bool:
    marker = str(kw["postscript_marker"]).strip()
    low = value.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, low, flags=re.MULTILINE))


def _bullets(value: str, kw: dict, prompt: str) -> bool:
    a = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    b = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    return len(a) + len(b) == int(kw["num_bullets"])


def _constrained(value: str, kw: dict, prompt: str) -> bool:
    v = value.strip()
    return any(opt in v for opt in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def _highlights(value: str, kw: dict, prompt: str) -> bool:
    num = 0
    for h in re.findall(r"\*[^\n\*]*\*", value):
        if h.strip("*").strip():
            num += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", value):
        if h.removeprefix("**").removesuffix("**").strip():
            num += 1
    return num >= int(kw["num_highlights"])


def _sections(value: str, kw: dict, prompt: str) -> bool:
    spliter = str(kw["section_spliter"]).strip()
    sections = re.split(r"\s?" + spliter + r"\s?\d+\s?", value)
    return len(sections) - 1 >= int(kw["num_sections"])


def _paragraphs(value: str, kw: dict, prompt: str) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i == 0 or i == len(paragraphs) - 1:
                num -= 1
            else:
                return False
    return num == int(kw["num_paragraphs"])


def _json_format(value: str, kw: dict, prompt: str) -> bool:
    v = (value.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
         .removeprefix("```").removesuffix("```").strip())
    try:
        json.loads(v)
    except ValueError:
        return False
    return True


def _nth_paragraph_first_word(value: str, kw: dict, prompt: str) -> bool:
    paragraphs = re.split(r"\n\n", value)
    num = len(paragraphs)
    for p in paragraphs:
        if not p.strip():
            num -= 1
    nth = int(kw["nth_paragraph"])
    if nth <= num:
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
    else:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        first += ch.lower()
    return num == int(kw["num_paragraphs"]) and first == str(kw["first_word"]).lower()


def _two_responses(value: str, kw: dict, prompt: str) -> bool:
    valid = []
    parts = value.split("******")
    for i, r in enumerate(parts):
        if not r.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(r)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _title(value: str, kw: dict, prompt: str) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", value))


def _capital_word_frequency(value: str, kw: dict, prompt: str) -> bool:
    n = sum(1 for w in _word_tokenize(value) if w.isupper())
    return _rel(n, kw["capital_relation"], int(kw["capital_frequency"]))


IFEVAL_CHECKERS: dict[str, Callable[[str, dict, str], bool]] = {
    "keywords:existence": lambda v, kw, p: all(re.search(k, v, flags=re.IGNORECASE) for k in kw["keywords"]),
    "keywords:frequency": lambda v, kw, p: _rel(
        len(re.findall(str(kw["keyword"]).strip(), v, flags=re.IGNORECASE)), kw["relation"], int(kw["frequency"])),
    "keywords:forbidden_words": lambda v, kw, p: not any(
        re.search(r"\b" + w + r"\b", v, flags=re.IGNORECASE) for w in kw["forbidden_words"]),
    "keywords:letter_frequency": lambda v, kw, p: _rel(
        collections.Counter(v.lower())[str(kw["letter"]).strip().lower()], kw["let_relation"], int(kw["let_frequency"])),
    "language:response_language": _response_language,
    "length_constraints:number_sentences": lambda v, kw, p: _rel(
        len(split_into_sentences(v)), kw["relation"], int(kw["num_sentences"])),
    "length_constraints:number_paragraphs": _paragraphs,
    "length_constraints:number_words": lambda v, kw, p: _rel(count_words(v), kw["relation"], int(kw["num_words"])),
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": lambda v, kw, p: len(re.findall(r"\[.*?\]", v)) >= int(kw["num_placeholders"]),
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _bullets,
    "detectable_format:constrained_response": _constrained,
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:multiple_sections": _sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": lambda v, kw, p: v.strip().lower().startswith(str(kw["prompt_to_repeat"]).strip().lower()),
    "startend:end_checker": lambda v, kw, p: v.strip().strip('"').lower().endswith(str(kw["end_phrase"]).strip().lower()),
    "startend:quotation": lambda v, kw, p: len(v.strip()) > 1 and v.strip()[0] == '"' and v.strip()[-1] == '"',
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": lambda v, kw, p: v.isupper() and _is_english(v),
    "change_case:english_lowercase": lambda v, kw, p: v.islower() and _is_english(v),
    "punctuation:no_comma": lambda v, kw, p: not re.search(r"\,", v),
}


class PortedRegistry:
    """Marker for the in-module port (``ifeval_follows`` dispatches on it)."""

    INSTRUCTION_DICT = IFEVAL_CHECKERS


def port_deviations() -> list[str]:
    devs = [
        "length_constraints:number_sentences counts sentences with the regex split_into_sentences, not nltk punkt",
        "change_case:capital_word_frequency tokenises with "
        + ("nltk TreebankWordTokenizer (no sentence split)" if _has_nltk() else "a Treebank-like regex (nltk missing)"),
    ]
    if _langdetect() is None:
        devs.append("change_case:english_capital/english_lowercase check case only (langdetect missing)")
        devs.append("language:response_language prompts excluded (langdetect missing)")
    else:
        devs.append("langdetect DetectorFactory.seed = 0 for deterministic language detection")
    return devs


def _has_nltk() -> bool:
    try:
        import nltk  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def ifeval_registry() -> tuple[Any, str]:
    """(registry, name): the official checker registry if importable, else the in-module port."""
    for mod, name in (
        ("instruction_following_eval.instructions_registry", "google-research instruction_following_eval"),
        ("lm_eval.tasks.ifeval.instructions_registry", "lm-eval-harness lm_eval.tasks.ifeval"),
    ):
        try:
            import importlib

            return importlib.import_module(mod), name
        except Exception:  # noqa: BLE001 (ImportError, or a broken optional dependency such as langdetect)
            continue
    return PortedRegistry, "sm.eval.capability port of instruction_following_eval (Apache-2.0)"


def _clean_kwargs(kwargs: Sequence[dict | None], i: int) -> dict:
    if i >= len(kwargs):
        return {}
    return {k: v for k, v in (kwargs[i] or {}).items() if v is not None}


def ifeval_follows(registry: Any, prompt: str, response: str, instruction_ids: Sequence[str],
                   kwargs: Sequence[dict | None]) -> tuple[bool, list[bool]]:
    """Strict check (test_instruction_following_strict): (all followed, per-instruction results).

    Raises LookupError when an instruction cannot be checked (unknown id, or a missing dependency).
    """
    per: list[bool] = []
    for i, iid in enumerate(instruction_ids):
        kw = _clean_kwargs(kwargs, i)
        if registry is PortedRegistry:
            fn = IFEVAL_CHECKERS.get(iid)
            if fn is None:
                raise LookupError(f"no checker for {iid}")
            ok = bool(response.strip()) and bool(fn(response, kw, prompt))
        else:
            cls = registry.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=prompt)
            ok = bool(response.strip()) and bool(inst.check_following(response))
        per.append(ok)
    return all(per), per


def score_ifeval(records: Sequence[dict], items: dict[str, dict], registry: Any, name: str) -> dict[str, Any]:
    n = ok = 0
    inst_n = inst_ok = 0
    by_type: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    excluded: dict[str, list[str]] = collections.defaultdict(list)
    for r in records:
        it = items.get(str(r["item_id"]))
        if not it or not it.get("instruction_id_list"):
            continue
        ids = list(it["instruction_id_list"])
        try:
            follows, per = ifeval_follows(registry, it.get("prompt", ""), r.get("completion", ""), ids, it.get("kwargs") or [])
        except LookupError as e:
            excluded[str(e)].append(str(r["item_id"]))
            continue
        n += 1
        ok += follows
        for iid, f in zip(ids, per):
            inst_n += 1
            inst_ok += f
            by_type[iid][0] += f
            by_type[iid][1] += 1
    return {
        "ifeval_strict_acc": ok / n if n else None,
        "ifeval_n": n,
        "ifeval_instruction_strict_acc": inst_ok / inst_n if inst_n else None,
        "ifeval_by_type": {k: {"followed": v[0], "n": v[1]} for k, v in sorted(by_type.items())},
        "ifeval_excluded": {k: {"n": len(v), "item_ids": v} for k, v in excluded.items()},
        "ifeval_checker": name,
        "ifeval_deviations": port_deviations() if registry is PortedRegistry else [],
    }


# --------------------------------------------------------------------------- run scoring


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CapabilityError(RuntimeError):
    pass


def _item_file(items_dir: Path, battery: str) -> Path:
    from sm.eval.battery import load_manifest, resolve_item_file

    manifest = load_manifest(items_dir) if items_dir.exists() else None
    path, expected = resolve_item_file(items_dir, battery, manifest)
    if expected and expected != _sha256(path):
        raise CapabilityError(f"{battery}: item file {path} does not match its manifest hash")
    return path


def _battery_rows(battery_dir: Path, battery: str, item_path: Path) -> tuple[list[dict], Path, dict]:
    from sm.eval.battery import meta_path_for

    out = battery_dir / f"{battery}.jsonl"
    if not out.is_file():
        raise FileNotFoundError(f"missing battery output {out}")
    side_meta: dict = {}
    side = meta_path_for(out)
    if side.is_file():
        side_meta = (json.loads(side.read_text(encoding="utf-8")).get("meta") or {})
        rec_sha = side_meta.get("item_file_sha256")
        if rec_sha and rec_sha != _sha256(item_path):
            raise CapabilityError(f"{out} was generated from a different {battery} item file "
                                  f"(recorded {rec_sha}); regenerate the battery")
        dec = side_meta.get("decoding") or {}
        if dec and not dec.get("greedy", dec.get("mode") == "greedy"):
            raise CapabilityError(f"{out}: capability batteries must be decoded greedily, sidecar says {dec}")
    rows = [r for r in _read_jsonl(out) if int(r.get("sample_idx", 0)) == 0]
    return rows, out, side_meta


def score_run(
    battery_dir: Path,
    items_dir: Path,
    registry_fn: Callable[[], tuple[Any, str]] = ifeval_registry,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(top-level scores, meta details) for one run directory. Raises FileNotFoundError / CapabilityError."""
    inputs: dict[str, str] = {}
    details: dict[str, Any] = {}
    battery_meta: dict[str, Any] = {}

    mmlu_items_path = _item_file(items_dir, MMLU_BATTERY)
    rows, out, side = _battery_rows(battery_dir, MMLU_BATTERY, mmlu_items_path)
    mmlu_items = {str(r["item_id"]): r for r in _read_jsonl(mmlu_items_path)}
    m = score_mmlu(rows, mmlu_items)
    inputs.update({str(out): _sha256(out), str(mmlu_items_path): _sha256(mmlu_items_path)})
    battery_meta[MMLU_BATTERY] = _sidecar_summary(side)

    ife_items_path = _item_file(items_dir, IFEVAL_BATTERY)
    rows, out, side = _battery_rows(battery_dir, IFEVAL_BATTERY, ife_items_path)
    ife_items = {str(r["item_id"]): r for r in _read_jsonl(ife_items_path)}
    registry, name = registry_fn()
    f = score_ifeval(rows, ife_items, registry, name)
    inputs.update({str(out): _sha256(out), str(ife_items_path): _sha256(ife_items_path)})
    battery_meta[IFEVAL_BATTERY] = _sidecar_summary(side)

    scores = {"mmlu_acc": m.pop("mmlu_acc"), "mmlu_n": m.pop("mmlu_n"),
              "ifeval_strict_acc": f.pop("ifeval_strict_acc"), "ifeval_n": f.pop("ifeval_n")}
    details.update(m)
    details.update(f)
    details["n_items_in_files"] = {MMLU_BATTERY: len(mmlu_items), IFEVAL_BATTERY: len(ife_items)}
    details["input_sha256"] = inputs
    details["battery_meta"] = battery_meta
    return scores, details


def _sidecar_summary(side: dict) -> dict:
    model = side.get("model") or {}
    keep = ("revision", "resolved_revision", "revision_requested", "revision_source", "adapter", "adapter_sha256",
            "adapter_mode", "adapter_merged", "dtype", "compute_dtype", "hf", "model_class")
    return {
        "model": {k: model.get(k) for k in keep if k in model},
        "decoding": side.get("decoding"),
        "backend": side.get("backend"),
        "item_file_sha256": side.get("item_file_sha256"),
        "git_commit": side.get("git_commit"),
    }


# --------------------------------------------------------------------------- smoke inputs


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_smoke_items(items_dir: Path, n: int = 8) -> Path:
    """Synthetic MMLU and IFEval item files (+ MANIFEST.json) in the prompt-builder format."""
    subjects = ("anatomy", "astronomy")
    mmlu = []
    for i in range(n):
        q = f"Synthetic {subjects[i % 2]} question {i}: which option is correct?"
        choices = [f"option {j}" for j in range(4)]
        prompt = "\n".join([q, ""] + [f"{LETTERS[j]}. {c}" for j, c in enumerate(choices)]
                           + ["", "Answer with the letter of the correct option only."])
        mmlu.append({"item_id": f"mmlu-smoke-{i:03d}", "battery": MMLU_BATTERY, "prompt": prompt, "subject": subjects[i % 2],
                     "question": q, "choices": choices, "answer": LETTERS[i % 4], "answer_index": i % 4})
    kinds = [
        (["punctuation:no_comma"], [{}]),
        (["change_case:english_lowercase"], [{}]),
        (["length_constraints:number_words"], [{"relation": "less than", "num_words": 40}]),
        (["keywords:existence"], [{"keywords": ["hello"]}]),
        (["detectable_format:title"], [{}]),
        (["startend:quotation"], [{}]),
        (["keywords:letter_frequency"], [{"letter": "a", "let_frequency": 1, "let_relation": "at least"}]),
        (["punctuation:no_comma", "length_constraints:number_sentences"], [{}, {"relation": "less than", "num_sentences": 5}]),
    ]
    ife = []
    for i in range(n):
        ids, kws = kinds[i % len(kinds)]
        ife.append({"item_id": f"ifeval-smoke-{i:03d}", "battery": IFEVAL_BATTERY,
                    "prompt": f"Write a short note number {i} about the sea. Follow: {', '.join(ids)}.",
                    "key": 9000 + i, "instruction_id_list": ids, "kwargs": kws})
    _write_jsonl(items_dir / "mmlu.jsonl", mmlu)
    _write_jsonl(items_dir / "ifeval.jsonl", ife)
    manifest = {"batteries": {b: {"path": f"{b}.jsonl", "sha256": _sha256(items_dir / f"{b}.jsonl")}
                              for b in (MMLU_BATTERY, IFEVAL_BATTERY)}}
    (items_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (items_dir / "batteries_smoke.yaml").write_text(
        "batteries:\n  mmlu:\n    decoding: greedy\n  ifeval:\n    decoding: greedy\n", encoding="utf-8")
    return items_dir


# --------------------------------------------------------------------------- CLI


def _write(path: Path, obj: dict) -> None:
    try:
        from sm.common.io import write_json  # type: ignore

        write_json(path, obj)
    except ImportError:  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))


def _generate(base: str, run_id: str, results_root: Path, items_dir: Path, args: argparse.Namespace,
              smoke: bool, batteries_config: str | None) -> int:
    from sm.eval import battery

    argv = ["--base", base, "--run-id", run_id, "--out", str(results_root), "--items-dir", str(items_dir),
            "--batteries", MMLU_BATTERY, IFEVAL_BATTERY, "--seed", str(args.seed)]
    if smoke:
        argv.append("--smoke")
    for flag, val in (("--adapter", args.adapter), ("--revision", args.revision), ("--bases-yaml", args.bases_yaml),
                      ("--batteries-config", batteries_config), ("--device", args.device), ("--dtype", args.dtype),
                      ("--batch-size", args.batch_size)):
        if val is not None:
            argv += [flag, str(val)]
    if args.allow_download:
        argv.append("--allow-download")
    return battery.main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sm.eval.capability", description="MMLU and IFEval capability scores for one run.")
    ap.add_argument("--config", help="YAML with any of the flags below (flags win)")
    ap.add_argument("--base", help="base key (directory name under the batteries root)")
    ap.add_argument("--run-id", default=None, help="run id (default base)")
    ap.add_argument("--results-root", default=None, help="default results (batteries under <root>/batteries)")
    ap.add_argument("--batteries-root", default=None, help="default <results-root>/batteries")
    ap.add_argument("--items-dir", default=None, help="default data/prompts")
    ap.add_argument("--out", default=None, help="default <results-root>/capability")
    ap.add_argument("--generate", action="store_true", help="run sm.eval.battery for missing mmlu/ifeval outputs first")
    ap.add_argument("--adapter", default=None, help="--generate: LoRA adapter (evaluated unmerged)")
    ap.add_argument("--revision", default=None, help="--generate: revision (default the configs/bases.yaml pin)")
    ap.add_argument("--bases-yaml", default=None)
    ap.add_argument("--batteries-config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny model, synthetic items, CPU fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-download", action="store_true", help="--generate only; never downloads otherwise")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg: dict[str, Any] = {}
    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        cfg = cfg.get("capability", cfg) if isinstance(cfg, dict) else {}

    def opt(name: str, default: Any = None) -> Any:
        v = getattr(args, name)
        return v if v is not None else cfg.get(name, default)

    from sm.eval.battery import path_slug

    if args.smoke:
        out_root = Path(opt("out") or "results/smoke/capability")
        label = opt("base") or "smoke"
    else:
        results_root = Path(opt("results_root") or "results")
        out_root = Path(opt("out") or results_root / "capability")
        label = opt("base")
    run_id = opt("run_id") or "base"
    if not label:
        print("error: --base is required", file=sys.stderr)
        return EXIT_USAGE
    out_path = out_root / path_slug(label) / f"{run_id}.json"
    if out_path.exists() and not args.force:
        print(f"already done: {out_path}")
        return EXIT_OK

    batteries_config = opt("batteries_config")
    if args.smoke:
        from sm.eval.battery import smoke_base

        work = out_root / "_smoke_inputs"
        items_dir = build_smoke_items(work / "items")
        results_root = work / "results"
        model_base = args.base or smoke_base(work)
        batteries_root = results_root / "batteries"
        batteries_config = str(items_dir / "batteries_smoke.yaml")
        rc = _generate(model_base, run_id, results_root, items_dir, args, True, batteries_config)
        if rc != 0:
            return rc
    else:
        model_base = label
        items_dir = Path(opt("items_dir") or "data/prompts")
        batteries_root = Path(opt("batteries_root") or results_root / "batteries")
    bdir = batteries_root / path_slug(model_base) / run_id
    restricted_dir = None  # mmlu and ifeval are never restricted

    if not args.smoke and args.generate:
        missing = [b for b in (MMLU_BATTERY, IFEVAL_BATTERY) if not (bdir / f"{b}.jsonl").is_file()]
        if missing:
            if batteries_root.name != "batteries":
                print(f"error: --generate needs a batteries root named 'batteries' (got {batteries_root})", file=sys.stderr)
                return EXIT_USAGE
            rc = _generate(model_base, run_id, batteries_root.parent, items_dir, args, False, batteries_config)
            if rc != 0:
                return rc
    try:
        scores, details = score_run(bdir, items_dir)
    except FileNotFoundError as e:
        print(f"error: {e} (run sm.eval.battery for mmlu and ifeval, or pass --generate)", file=sys.stderr)
        return EXIT_MISSING
    except CapabilityError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_MISSING
    if scores["mmlu_n"] == 0 or scores["ifeval_n"] == 0:
        print(f"error: no scorable items (mmlu_n={scores['mmlu_n']}, ifeval_n={scores['ifeval_n']})", file=sys.stderr)
        return EXIT_MISSING
    expected = {} if args.smoke else {b: n for b, n in EXPECTED_N.items()
                                       if details["n_items_in_files"].get(b) != n}
    meta = {
        "stage": "capability",
        "base": label,
        "model_base": model_base,
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "battery_dir": str(bdir),
        "items_dir": str(items_dir),
        "restricted_dir": restricted_dir,
        "mmlu_metric": "letter accuracy, format-invariant parse (answer phrase, else leading letter, else first standalone capital A-D); the prereg Section 7 exclusion metric",
        "mmlu_exact_metric": "mmlu_exact_acc: strict first-token letter match (diagnostic only)",
        "ifeval_metric": "strict prompt-level accuracy (test_instruction_following_strict)",
        "item_count_differs_from_prereg": {b: {"expected": EXPECTED_N[b], "found": details["n_items_in_files"][b]}
                                           for b in expected},
        **details,
    }
    _write(out_path, {**scores, "meta": meta})
    print(f"wrote {out_path}: mmlu_acc={scores['mmlu_acc']} (n={scores['mmlu_n']}) "
          f"ifeval_strict_acc={scores['ifeval_strict_acc']} (n={scores['ifeval_n']}; {details['ifeval_checker']})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
