"""Build the J-lens fit corpus from public WikiText-103 (docs/JLENS_PLAN.md, section "Provenance").

Outputs (default names, under ``data/jlens/``):

* ``wikitext103_<n>_seed<seed>.jsonl``: UTF-8 JSONL, one JSON string per line (the format
  ``sm.interp.jlens.read_prompts`` reads). U+2028, U+2029 and U+0085 are written as JSON escapes, so a
  line-splitting reader cannot break a passage.
* ``wikitext103_<n>_seed<seed>.meta.json``: dataset id and revision, the SHA-256 of every source file, seed,
  selection rule, tokenizer pin, counts, token statistics, the selected row indices, licence and attribution,
  and the corpus SHA-256.

Source. WikiText-103 raw (Merity, Xiong, Bradbury and Socher 2016, arXiv:1609.07843), train split, from the
Hugging Face dataset ``Salesforce/wikitext`` (config ``wikitext-103-raw-v1``) at a pinned revision. The split's
parquet files are downloaded through ``huggingface_hub`` at that revision, and their SHA-256 values are checked
against the Hub's LFS metadata. Nothing else is read: no other corpus and no local copy of one.

Selection rule. It is deterministic and uses no library random-number generator, so it does not depend on the
numpy or Python version.

1. Rows: the ``text`` column of the split's parquet files, concatenated in file-name order. The row index ``r``
   counts from 0 across the files; each row is one line of WikiText-103.
2. Candidates: rows whose text, stripped of surrounding whitespace, is non-empty and is not a section heading.
   A heading is stripped text that starts and ends with "=", such as "= = Gameplay = =". The passage is the
   stripped text, otherwise unchanged: WikiText's own tokenisation (" @-@ ", " @,@ ", spaces around
   punctuation) is kept.
3. Order: candidates sorted by the SHA-256 digest of the ASCII string ``f"{seed}:{r}"``, ties broken by ``r``.
4. Walk that order. Drop a passage whose exact text was already kept. Keep a passage when the base's pinned
   tokenizer encodes it to at least ``min_tokens`` ids. Ids are counted the way the official jlens fit encodes a
   prompt (``jlens.hf.HFLensModel.encode``: ``tokenizer(text)`` with the default special tokens, and
   ``add_bos_token`` forced on when the tokenizer has a BOS token), without truncation. Stop at ``n`` passages.
5. The file lists the passages in selection order.

The fit truncates every prompt to ``max_seq_len`` tokens (128), so with ``min_tokens`` = 128 every prompt fills
the window.

Licence. WikiText-103 is Wikipedia text under CC BY-SA 3.0 (the dataset card also lists the GFDL). The corpus
keeps CC BY-SA 3.0, and the sidecar carries the attribution and the list of changes.

Usage (any machine with Hub access; the train split is about 315 MB of parquet, downloaded once into the HF
cache; the tokenizer is about 23 MB):

    python -m sm.data.build_jlens_corpus             # 1,000 passages, seed 0, the qwen3.5-9b tokenizer pin
    python -m sm.data.build_jlens_corpus --seed 1    # another draw: data/jlens/wikitext103_1000_seed1.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "sm-jlens-corpus-v1"

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
# Hub main on 2026-10-10 (last modified 2024-01-04).
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

DEFAULT_BASE = "qwen3.5-9b"
DEFAULT_BASES_YAML = REPO_ROOT / "configs" / "bases.yaml"
DEFAULT_N = 1000
DEFAULT_MIN_TOKENS = 128
DEFAULT_SEED = 0
DEFAULT_BATCH = 256

WIKITEXT_CITATION = (
    "Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher. Pointer Sentinel Mixture Models. "
    "arXiv:1609.07843, 2016 (ICLR 2017)."
)
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"

SELECTION_RULE = (
    "Rows: the 'text' column of the split's parquet files at the pinned revision, concatenated in file-name "
    "order; row index r counts from 0 across the files. Candidates: rows whose stripped text is non-empty and "
    "does not both start and end with '=' (section headings). Passage: the stripped text, otherwise unchanged. "
    "Order: candidates sorted by SHA-256 of the ASCII string f'{seed}:{r}', ties by r. Walk the order, drop a "
    "passage whose exact text was already kept, keep it when the pinned tokenizer encodes it to at least "
    "min_tokens ids (tokenizer(text) with default special tokens, add_bos_token forced on when the tokenizer "
    "has a BOS token, as jlens.hf.HFLensModel.encode does; no truncation); stop at n. Output order = selection "
    "order."
)

# Characters that str.splitlines() treats as line breaks and json.dumps(ensure_ascii=False) leaves raw.
_LINE_SEPARATOR_ESCAPES = {" ": "\\u2028", " ": "\\u2029", "\x85": "\\u0085"}


# ----------------------------------------------------------------------------------------- selection (pure)


def is_passage(text: str | None) -> bool:
    """True for a WikiText line that holds body text: non-empty after stripping and not a section heading."""
    s = (text or "").strip()
    return bool(s) and not (s.startswith("=") and s.endswith("="))


def order_key(seed: int, row: int) -> bytes:
    """Sort key of row ``row`` under ``seed``: SHA-256 of the ASCII string ``f"{seed}:{row}"``."""
    return hashlib.sha256(f"{int(seed)}:{int(row)}".encode("ascii")).digest()


def candidate_order(rows: Sequence[str | None], seed: int) -> list[int]:
    """Row indices of every candidate passage, in the seeded selection order."""
    cands = [r for r, t in enumerate(rows) if is_passage(t)]
    cands.sort(key=lambda r: (order_key(seed, r), r))
    return cands


def select_passages(rows: Sequence[str | None], count_tokens: Callable[[list[str]], list[int]], *, n: int,
                    min_tokens: int, seed: int, batch: int = DEFAULT_BATCH) -> dict[str, Any]:
    """Apply the selection rule (module docstring). ``count_tokens`` maps a list of passages to their token counts.

    Returns {"rows", "passages", "n_tokens", "counts"}: the kept row indices, texts and token counts in selection
    order, and the bookkeeping counts. Raises ValueError when fewer than ``n`` passages qualify.
    """
    if n < 1 or min_tokens < 1 or batch < 1:
        raise ValueError("n, min_tokens and batch must be positive")
    order = candidate_order(rows, seed)
    kept_rows: list[int] = []
    kept: list[str] = []
    kept_counts: list[int] = []
    seen: set[str] = set()
    examined = short = duplicate = 0
    i = 0
    while len(kept) < n and i < len(order):
        chunk = order[i:i + batch]
        i += len(chunk)
        texts = [str(rows[r]).strip() for r in chunk]
        counts = count_tokens(texts)
        if len(counts) != len(texts):
            raise ValueError("count_tokens returned the wrong number of counts")
        for r, t, c in zip(chunk, texts, counts):
            if len(kept) >= n:
                break
            examined += 1
            if t in seen:
                duplicate += 1
                continue
            if int(c) < min_tokens:
                short += 1
                continue
            seen.add(t)
            kept_rows.append(r)
            kept.append(t)
            kept_counts.append(int(c))
    if len(kept) < n:
        raise ValueError(f"only {len(kept)} of {len(order)} candidates qualify (need {n} with >= {min_tokens} tokens)")
    return {"rows": kept_rows, "passages": kept, "n_tokens": kept_counts,
            "counts": {"rows_total": len(rows), "candidates": len(order), "examined": examined,
                       "rejected_short": short, "rejected_duplicate": duplicate, "kept": len(kept)}}


def corpus_bytes(passages: Sequence[str]) -> bytes:
    """The JSONL file content: one JSON string per line, UTF-8, line-separator characters escaped."""
    lines = []
    for p in passages:
        line = json.dumps(p, ensure_ascii=False)
        for ch, esc in _LINE_SEPARATOR_ESCAPES.items():
            line = line.replace(ch, esc)
        lines.append(line + "\n")
    return "".join(lines).encode("utf-8")


def token_stats(counts: Sequence[int]) -> dict[str, Any]:
    """Summary of per-passage token counts (full passages, before the fit's truncation)."""
    xs = sorted(int(c) for c in counts)
    dec = statistics.quantiles(xs, n=10, method="inclusive") if len(xs) > 1 else [float(xs[0])] * 9
    return {"n": len(xs), "min": xs[0], "p10": dec[0], "median": statistics.median(xs),
            "mean": round(statistics.fmean(xs), 2), "p90": dec[8], "max": xs[-1], "total": sum(xs),
            "quantile_method": "statistics.quantiles(n=10, method='inclusive')"}


# ----------------------------------------------------------------------------------------- io helpers


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _git_head() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "code/sm/data"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"git_head": head, "code_sm_data_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "code_sm_data_dirty": None}


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("huggingface_hub", "pyarrow", "transformers", "tokenizers"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


# ----------------------------------------------------------------------------------------- network steps


def fetch_split(dataset: str, config: str, split: str, revision: str) -> tuple[str, Any, list[dict[str, Any]]]:
    """Download the split's parquet files at ``revision`` and check each against the Hub's LFS SHA-256.

    Returns (resolved revision, the card's licence field, [{"name", "bytes", "sha256", "lfs_sha256_match",
    "path"}]) with files in name order. ``path`` is the local cache path (not written to the sidecar).
    """
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(dataset, revision=revision, files_metadata=True)
    resolved = info.sha
    prefix = f"{config}/{split}-"
    sibs = sorted((s for s in info.siblings if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet")),
                  key=lambda s: s.rfilename)
    if not sibs:
        raise FileNotFoundError(f"no {prefix}*.parquet in {dataset}@{resolved}")
    files = []
    for s in sibs:
        local = hf_hub_download(dataset, s.rfilename, repo_type="dataset", revision=resolved)
        sha = _sha256_file(local)
        expected = getattr(getattr(s, "lfs", None), "sha256", None)
        if expected is None:
            raise RuntimeError(f"{s.rfilename}: the Hub lists no LFS SHA-256 to check against")
        if sha != expected:
            raise RuntimeError(f"{s.rfilename}: SHA-256 {sha} differs from the Hub's {expected}")
        files.append({"name": s.rfilename, "bytes": os.path.getsize(local), "sha256": sha, "lfs_sha256_match": True,
                      "path": local})
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return resolved, card.get("license"), files


def read_rows(files: list[dict[str, Any]]) -> list[str | None]:
    """The ``text`` column of the parquet files, concatenated in the given order; records each file's row count."""
    import pyarrow.parquet as pq

    rows: list[str | None] = []
    for f in files:
        col = pq.read_table(f["path"], columns=["text"]).column("text").to_pylist()
        f["rows"] = len(col)
        rows.extend(col)
    return rows


def load_token_counter(base: str, bases_yaml: Path) -> tuple[Callable[[list[str]], list[int]], dict[str, Any]]:
    """Token counter with the base's pinned tokenizer, encoding as ``jlens.hf.HFLensModel.encode`` does."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from sm.common.registry import resolve_base

    info = resolve_base(base, bases_yaml=bases_yaml)
    hf, rev = info["hf"], info.get("revision")
    if not rev:
        raise SystemExit(f"{base} has no pinned revision in {_rel(bases_yaml)}")
    tok = AutoTokenizer.from_pretrained(hf, revision=rev)
    forced = False
    if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True  # official from_hf(force_bos=True)
        forced = True

    def count(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tok(list(texts))["input_ids"]]

    tj = hf_hub_download(hf, "tokenizer.json", revision=rev)
    meta = {"base": base, "hf": hf, "revision": rev, "class": type(tok).__name__, "vocab_size": len(tok),
            "bos_token_id": getattr(tok, "bos_token_id", None), "add_bos_token_forced": forced,
            "tokenizer_json_sha256": _sha256_file(tj),
            "counting": "len(tokenizer(text)['input_ids']), default special tokens, no truncation"}
    return count, meta


# ----------------------------------------------------------------------------------------- build


def attribution(dataset: str, config: str, split: str, revision: str, n: int) -> str:
    return (
        f"This corpus contains {n:,} passages of text from WikiText-103 (Merity, Xiong, Bradbury and Socher, "
        "\"Pointer Sentinel Mixture Models\", arXiv:1609.07843, 2016), which was extracted from Wikipedia's Good "
        "and Featured articles, written by Wikipedia contributors. Source: the "
        f"Hugging Face dataset {dataset}, config {config}, {split} split, revision {revision}. Licensed under the "
        f"Creative Commons Attribution-ShareAlike 3.0 Unported licence (CC BY-SA 3.0, {LICENCE_URL})."
    )


def build(*, out: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED, min_tokens: int = DEFAULT_MIN_TOKENS,
          base: str = DEFAULT_BASE, bases_yaml: Path = DEFAULT_BASES_YAML, dataset: str = DATASET_ID,
          config: str = DATASET_CONFIG, split: str = DATASET_SPLIT, revision: str = DATASET_REVISION,
          batch: int = DEFAULT_BATCH, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build the corpus and its sidecar; returns the sidecar dict."""
    from sm.interp.jlens import read_prompts

    resolved, card_licence, files = fetch_split(dataset, config, split, revision)
    rows = read_rows(files)
    count, tok_meta = load_token_counter(base, bases_yaml)
    sel = select_passages(rows, count, n=n, min_tokens=min_tokens, seed=seed, batch=batch)

    data = corpus_bytes(sel["passages"])
    _atomic_write_bytes(out, data)
    sha = hashlib.sha256(data).hexdigest()
    if read_prompts(out) != sel["passages"] or _sha256_file(out) != sha:
        raise RuntimeError(f"{out}: the written corpus does not read back as the selected passages")

    meta_path = out.with_name(out.name[: -len(".jsonl")] + ".meta.json") if out.name.endswith(".jsonl") \
        else out.with_name(out.name + ".meta.json")
    meta = {
        "format": FORMAT,
        "purpose": "J-lens fit corpus (docs/JLENS_PLAN.md); read by sm.interp.jlens.read_prompts",
        "output": {"path": _rel(out), "sha256": sha, "bytes": len(data), "n_passages": len(sel["passages"]),
                   "encoding": "UTF-8 JSONL, one JSON string per line; U+2028, U+2029 and U+0085 escaped"},
        "dataset": {"id": dataset, "config": config, "split": split, "revision": resolved,
                    "revision_requested": revision,
                    "url": f"https://huggingface.co/datasets/{dataset}/tree/{resolved}/{config}",
                    "card_license": card_licence, "citation": WIKITEXT_CITATION,
                    "files": [{k: f[k] for k in ("name", "bytes", "rows", "sha256", "lfs_sha256_match")}
                              for f in files],
                    "rows_total": len(rows)},
        "tokenizer": tok_meta,
        "selection": {"seed": seed, "n": n, "min_tokens": min_tokens, "rule": SELECTION_RULE,
                      "order_key": "sha256(f'{seed}:{row}'.encode('ascii')).digest(), ascending; ties by row",
                      **sel["counts"], "row_indices": sel["rows"]},
        "token_stats": {**token_stats(sel["n_tokens"]), "fit_max_seq_len": 128,
                        "note": "full-passage counts; the fit truncates every prompt to max_seq_len"},
        "licence": {"spdx": "CC-BY-SA-3.0", "name": "Creative Commons Attribution-ShareAlike 3.0 Unported",
                    "url": LICENCE_URL, "dataset_card_licenses": card_licence,
                    "attribution": attribution(dataset, config, split, resolved, len(sel["passages"])),
                    "changes": ("Selected lines only (rule above); surrounding whitespace stripped; text otherwise "
                                "unchanged, including WikiText's own tokenisation; written one JSON string per line."),
                    "share_alike": "The corpus file is distributed under CC BY-SA 3.0, the licence of its source "
                                   "text."},
        "independence": ("Built only from the dataset files listed under 'dataset.files' (downloaded from the Hub "
                         "at the pinned revision, SHA-256 checked against the Hub's LFS metadata) and the pinned "
                         "tokenizer. No other corpus is read."),
        "builder": {"module": "sm.data.build_jlens_corpus", "file": _rel(Path(__file__)),
                    "sha256": _sha256_file(Path(__file__)), **_git_head(),
                    "argv": list(argv) if argv is not None else None,
                    "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "versions": _versions()},
    }
    _atomic_write_bytes(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the J-lens fit corpus from public WikiText-103.")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="passages to keep (default 1000)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="selection seed (default 0)")
    ap.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS, help="minimum passage length (default 128)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="registry key whose pinned tokenizer counts tokens")
    ap.add_argument("--bases-yaml", type=Path, default=DEFAULT_BASES_YAML)
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--config", default=DATASET_CONFIG)
    ap.add_argument("--split", default=DATASET_SPLIT)
    ap.add_argument("--revision", default=DATASET_REVISION, help="dataset revision (default: the pinned commit)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="passages per tokenizer call")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSONL (default data/jlens/wikitext103_<n>_seed<seed>.jsonl)")
    a = ap.parse_args(argv)
    out = a.out or REPO_ROOT / "data" / "jlens" / f"wikitext103_{a.n}_seed{a.seed}.jsonl"
    meta = build(out=out, n=a.n, seed=a.seed, min_tokens=a.min_tokens, base=a.base, bases_yaml=a.bases_yaml,
                 dataset=a.dataset, config=a.config, split=a.split, revision=a.revision, batch=a.batch,
                 argv=sys.argv[1:] if argv is None else argv)
    s, ts = meta["selection"], meta["token_stats"]
    print(f"[jlens-corpus] {meta['output']['path']}: {meta['output']['n_passages']} passages, "
          f"sha256 {meta['output']['sha256']}")
    print(f"[jlens-corpus] {meta['dataset']['id']} {meta['dataset']['config']} {meta['dataset']['split']} "
          f"@ {meta['dataset']['revision']}; seed {s['seed']}; {s['examined']} examined, {s['rejected_short']} short, "
          f"{s['rejected_duplicate']} duplicate")
    print(f"[jlens-corpus] tokens per passage: min {ts['min']}, median {ts['median']}, mean {ts['mean']}, "
          f"max {ts['max']}, total {ts['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
