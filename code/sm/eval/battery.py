"""Generate completions for every battery item for a base or base+adapter.

Output (per battery, per HARNESS_CONTRACTS.md plus the v2 sampling field): JSONL with one record per
item and sample
    {"item_id", "battery", "prompt", "completion", "completion_sha256", "truncated", "sample_idx"}
written to
    <out>/batteries/<base>/<run_id>/<battery>.jsonl                (ordinary batteries)
    <out>/restricted/batteries/<base>/<run_id>/<battery>.jsonl     (restricted batteries, dirs mode 0700, files 0600)
plus a sidecar <battery>.meta.json (no completion texts) holding the run meta.

Restricted batteries (hazard rule, prereg Section 8; HARNESS_CONTRACTS.md "Hazard routing"): every battery that
configs/batteries.yaml marks `harmful: true` (refusal-rate and ladder batteries) or `restricted: true` (free-form
batteries whose completions a condition 3 or 4 derivative can make harmful: em_freeform, persona_*), plus the
built-in name rule (is_restricted: strongreject, harmbench, xstest_unsafe, karne_ladders*, em_*, persona_*), so a
missing or edited config never moves texts out of results/restricted. A config cannot un-restrict a battery. An
output a previous version wrote outside results/restricted for a battery that is restricted now is moved into the
restricted location on the next run (WARNING in the log; the sidecar records `restricted_migrated_from`).

"item_id" is always the id from the item file. Greedy batteries write one row per item with
sample_idx 0; sampled batteries write n rows per item with sample_idx 0..n-1, so (item_id, sample_idx)
is the row key (item_id repeats inside a sampled battery).

Idempotent per battery file: an existing final JSONL is skipped unless --force, and regenerated when
its item file hash or its decoding settings differ from the ones recorded in its sidecar. Generation
is checkpointed to <battery>.jsonl.partial in chunks so a killed job resumes where it stopped; the
final file is written atomically. A sampled item is always regenerated as a whole (its n samples
come from one seeded call), so a resume cannot mix seeds.

Model (prereg v0.2, v2 harness artifacts): the base is loaded with the revision pinned in
configs/bases.yaml (or --revision / the config's revision), through the same
sm.common.models.load_model call the smoke stage uses. A measured (non-smoke) run of a Hub model with
no pinned revision is refused unless --allow-unpinned. An adapter is evaluated UNMERGED (PEFT model,
LoRA applied in the forward pass at the bf16 compute dtype).

Backends: "hf" (default; sm.common.models.generate / generate_detailed) and "vllm" (optional, refused
for measured batteries unless --allow-vllm-measured, recorded as a deviation). Under vllm both the base
and the derivative are exported through the same text-only load_model path into --merge-dir, the
adapter merged in fp32 and the result cast to the serving dtype (default bf16), so base and derivative
are served by the same model class and dtype; --vllm-lora serves the adapter through LoRARequest on the
original checkpoint instead (and the base from the same checkpoint). A merged export cast to bf16 rounds
LoRA updates below half a bf16 ULP away, so a measured derivative battery refuses it: use --vllm-lora or
--vllm-dtype fp32 (the export records the measured rounding share in meta.export.rounding). Prompts are rendered once with the chat template (thinking off) and passed
to vLLM as token ids built with sm.common.models.encode_text, so no second BOS is added. If vllm is not
importable the run falls back to "hf" with a warning (or exits 3 under --strict-backend). Every battery is
text-only: a multimodal checkpoint (the Qwen3.5 VL bases under --vllm-lora) is served with
limit_mm_per_prompt={"image": 0, "video": 0} so the vision tower is neither profiled nor warmed up (review A14);
the sidecar records meta.model.multimodal_checkpoint, meta.model.limit_mm_per_prompt and meta.model.vllm_kwargs.

Sidecar provenance for a derivative: meta.adapter_config_sha256 (adapter_config.json) and meta.adapter_sha256
(adapter_model.safetensors, the file train_meta.adapter_sha256 hashes), plus meta.restricted and
meta.restricted_reason ("config:harmful", "config:restricted" or "name_rule").

Decoding per battery comes from configs/batteries.yaml (see resolve_decoding for the accepted keys);
batteries absent there use the prereg v0.2 Section 5 rule: greedy for refusal, over-refusal, ladder and
capability batteries; temperature 1.0, top-p 1.0, n = 10 for the emergent-misalignment and persona
batteries. Sampled items are seeded per item from sha256(seed:item_id).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RESTRICTED_BATTERIES: tuple[str, ...] = ("strongreject", "harmbench", "xstest_unsafe", "karne_ladders")  # karne_ladders_* too (prefix)
# Not harmful-request batteries, but derivatives of conditions 3 and 4 can write harmful text in them (OPEN-3, safe
# default): em_freeform (and any em_* / emergent_* battery) and every persona_* battery are restricted too.
RESTRICTED_FREEFORM_BATTERIES: tuple[str, ...] = ("em", "emergent", "persona")  # names and their "_" / "-" prefixes
# A14: text-only serving of a multimodal checkpoint (Qwen3.5 VL): no image or video inputs, so vLLM neither
# profiles nor warms up the vision tower.
TEXT_ONLY_MM_LIMIT: dict[str, int] = {"image": 0, "video": 0}
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 16
SMOKE_N_ITEMS = 8
SMOKE_MAX_NEW_TOKENS = 16
SMOKE_BATTERIES: tuple[str, ...] = ("smoke", "smoke_restricted")
# Prereg v0.2 Section 5: sampled decoding for the free-form evaluations (fallback rule only;
# configs/batteries.yaml is authoritative when it declares a battery's decoding).
SAMPLED_BATTERY_PREFIXES: tuple[str, ...] = ("em_", "emergent", "persona_")
DEFAULT_N_SAMPLES = 10
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 1.0
DEFAULT_BATTERIES_CONFIG = "configs/batteries.yaml"
# Karne 2609.05797's measurement (v0.3, fidelity review M7, N7): the karne_ladders_sens battery (every unit, fixed
# 100-ladder subset) and the karne_ladders_sampled_base battery (the base only, all ladders: the H5 strata) must be decoded exactly
# as Karne's compliance labels were ("answered = at least 1 of 8 samples at T = 1.0"): sampled, temperature 1.0,
# top-p 1.0, 8 samples per rung, thinking off. A CLI or config override of these values is refused (smoke runs
# may shorten max_new_tokens only). Karne's max_tokens is not stated in the released README; 512 new tokens is ours.
KARNE_BATTERIES: tuple[str, ...] = ("karne_ladders_sens", "karne_ladders_sampled_base")
BASE_RUN_ID = "base"
KARNE_DECODING: dict[str, float] = {"temperature": 1.0, "top_p": 1.0, "n_samples": 8}


def check_karne_decoding(battery: str, dec: "Decoding") -> None:
    """Refuse a decoding for Karne's measurement that would not reproduce Karne's labels."""
    if battery not in KARNE_BATTERIES:
        return
    got = {"mode": dec.mode, "temperature": dec.temperature, "top_p": dec.top_p, "n_samples": dec.n_samples}
    want = {"mode": "sample", **KARNE_DECODING}
    if got != want:
        raise SystemExit(f"{battery}: decoding {got} is not Karne's measurement {want} (prereg v0.3 H5); fix configs/batteries.yaml "
                         "or drop the CLI override")
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"  # train_meta.adapter_sha256 hashes this file (sm.train.lora_sft)
DEFAULT_BASES_YAML = "configs/bases.yaml"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BACKEND_UNAVAILABLE = 3

# Prompt fields accepted in item files, in priority order.
_PROMPT_KEYS = ("prompt", "question", "instruction", "forbidden_prompt", "behavior", "goal", "text")
_ID_KEYS = ("item_id", "id", "uid", "key")

# A greedy generate function maps a list of chat message lists to (completion, truncated) pairs.
# A sampled one has ``wants_rows = True`` and maps a list of Items (all samples of each item,
# contiguous, sample_idx ascending) to (completion, truncated) pairs in the same order.
GenerateFn = Callable[..., list[tuple[str, bool]]]


# ---------------------------------------------------------------------------
# Small io helpers (prefer sm.common.io; private fallbacks keep this module usable alone)
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_file(path: str | Path) -> str:
    try:
        from sm.common.io import sha256_file  # type: ignore

        return sha256_file(path)
    except ImportError:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()


def _fallback_meta(extra: dict | None) -> dict:
    meta: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "meta_source": "sm.eval.battery fallback (sm.common.io not importable)",
    }
    for mod in ("torch", "transformers"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            meta[mod] = None
    if extra:
        meta.update(extra)
    return meta


def _atomic_write_bytes(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_meta_json(path: Path, obj: dict, extra_meta: dict, mode: int) -> None:
    """Write obj with a meta block; uses sm.common.io.write_json when available."""
    try:
        from sm.common.io import write_json  # type: ignore
    except ImportError:
        write_json = None
    if write_json is not None:
        write_json(path, obj, extra_meta=extra_meta)
        os.chmod(path, mode)
        return
    payload = dict(obj)
    payload["meta"] = _fallback_meta(extra_meta)
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"), mode)


def _set_seed(seed: int) -> None:
    try:
        from sm.common.io import set_seed  # type: ignore

        set_seed(seed)
        return
    except ImportError:
        pass
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def find_config(path: str | Path | None, default: str) -> Path | None:
    """Resolve a config path: explicit path (must exist), else ``default`` in the cwd, else in the repo root."""
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config not found: {p}")
        return p
    for cand in (Path(default), REPO_ROOT / default):
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# Paths and restriction
# ---------------------------------------------------------------------------


def is_restricted(battery: str, extra: Iterable[str] = ()) -> bool:
    """Built-in name rule: True for batteries whose texts must live under results/restricted.

    Harmful-request batteries (RESTRICTED_BATTERIES and their prefixes, e.g. karne_ladders_sens) and the free-form
    batteries a misaligned derivative can fill with harmful text (em_*, persona_*), plus ``extra`` names.
    """
    name = battery.lower()
    names = set(RESTRICTED_BATTERIES) | set(RESTRICTED_FREEFORM_BATTERIES) | {e.lower() for e in extra}
    return any(name == n or name.startswith(n + "_") or name.startswith(n + "-") for n in names)


def restricted_battery_names(bcfg: dict) -> set[str]:
    """Batteries configs/batteries.yaml marks `harmful: true` or `restricted: true`."""
    entries = bcfg.get("batteries") if isinstance(bcfg.get("batteries"), dict) else {}
    return {str(name) for name, e in entries.items() if isinstance(e, dict) and (e.get("harmful") or e.get("restricted"))}


def restriction_reason(battery: str, entry: dict | None, extra: Iterable[str] = ()) -> str | None:
    """Why a battery is restricted ("config:harmful", "config:restricted", "name_rule"), or None when it is not.

    The config can add restriction but never remove it: a battery the name rule restricts stays restricted.
    """
    entry = entry or {}
    if entry.get("harmful"):
        return "config:harmful"
    if entry.get("restricted"):
        return "config:restricted"
    if is_restricted(battery, extra):
        return "name_rule"
    return None


def path_slug(name: str) -> str:
    """Filesystem-safe slug for a base key, HF id or local path."""
    p = Path(name)
    if p.is_absolute() or (len(p.parts) > 2) or p.exists():
        name = p.name or name
    slug = name.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-.")
    return slug or "base"


def default_run_id(adapter_path: str | None) -> str:
    if not adapter_path:
        return "base"
    return path_slug(Path(adapter_path).resolve().name)


def battery_output_path(
    out_root: str | Path,
    base: str,
    run_id: str,
    battery: str,
    restricted: bool,
    restricted_root: str | Path | None = None,
) -> Path:
    """Final JSONL path for one battery. Restricted batteries go under restricted_root."""
    out_root = Path(out_root)
    if restricted:
        root = Path(restricted_root) if restricted_root is not None else out_root / "restricted"
        return root / "batteries" / path_slug(base) / path_slug(run_id) / f"{battery}.jsonl"
    return out_root / "batteries" / path_slug(base) / path_slug(run_id) / f"{battery}.jsonl"


def meta_path_for(out_path: Path) -> Path:
    return out_path.with_name(out_path.name[: -len(".jsonl")] + ".meta.json")


def partial_path_for(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".partial")


LEGACY_SUFFIX = ".legacy-unrestricted"


def migrate_unrestricted_outputs(
    legacy_out: Path, restricted_out: Path, restricted_root: Path, log: Callable[[str], None] = print
) -> list[str]:
    """Move a restricted battery's files from a legacy unrestricted location into the restricted one.

    Older code wrote em_freeform and persona_* outputs under <out>/batteries/. The final JSONL, its sidecar and a
    .partial checkpoint are moved (0600, directories 0700); when the restricted location already holds a file of
    the same name, the legacy file is kept next to it with the suffix ``.legacy-unrestricted`` (matched by no
    reader glob) instead of being overwritten. The moved sidecar gets ``restricted: true`` and
    ``restricted_migrated_from``. Returns the moved source paths (empty when there was nothing to move).
    """
    legacy_out, restricted_out = Path(legacy_out), Path(restricted_out)
    if legacy_out.resolve() == restricted_out.resolve():
        return []
    pairs = [(legacy_out, restricted_out), (meta_path_for(legacy_out), meta_path_for(restricted_out)),
             (partial_path_for(legacy_out), partial_path_for(restricted_out))]
    moved: list[str] = []
    for src, dst in pairs:
        if not src.exists():
            continue
        ensure_dir(dst.parent, Path(restricted_root))
        target = dst if not dst.exists() else dst.with_name(dst.name + LEGACY_SUFFIX)
        os.replace(src, target)
        os.chmod(target, 0o600)
        moved.append(str(src))
        log(f"[battery] WARNING: moved {src} (restricted battery written outside results/restricted by an older "
            f"version) to {target}")
        if target.name.endswith(".meta.json"):
            try:
                side = json.loads(target.read_text(encoding="utf-8"))
                side["restricted"] = True
                side["restricted_migrated_from"] = str(src)
                _atomic_write_bytes(target, (json.dumps(side, indent=2, sort_keys=True) + "\n").encode("utf-8"), 0o600)
            except (OSError, json.JSONDecodeError) as exc:
                log(f"[battery] WARNING: could not mark the migrated sidecar {target}: {exc}")
    return moved


def ensure_dir(path: Path, restricted_root: Path | None = None) -> None:
    """Create path. If restricted_root is given, path must lie under it and every
    directory from restricted_root down to path is forced to mode 0700."""
    path.mkdir(parents=True, exist_ok=True)
    if restricted_root is None:
        return
    root = restricted_root.resolve()
    target = path.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"{target} is not under restricted root {root}")
    os.chmod(root, 0o700)
    cur = root
    for part in target.relative_to(root).parts:
        cur = cur / part
        os.chmod(cur, 0o700)


# ---------------------------------------------------------------------------
# Item files
# ---------------------------------------------------------------------------


@dataclass
class Item:
    item_id: str
    prompt: str
    messages: list[dict]
    n_samples: int = 1  # samples requested by the item file (informational; see resolve_n_samples)
    sample_idx: int = 0


def row_key(item_id: str, sample_idx: int) -> tuple[str, int]:
    return (str(item_id), int(sample_idx))


def is_sampled_battery(battery: str) -> bool:
    """Prereg v0.2 Section 5 fallback rule (used only when configs/batteries.yaml says nothing)."""
    b = battery.lower()
    return b == "em" or b.startswith(SAMPLED_BATTERY_PREFIXES)


def _id_seed(seed: int, item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{item_id}".encode()).digest()[:4], "little")


def expand_samples(items: Sequence[Item], n: int | Callable[[Item], int]) -> list[Item]:
    """One row per sample: same item_id, sample_idx 0..n-1 (n an int or a per-item function)."""
    out: list[Item] = []
    for it in items:
        k_max = int(n(it) if callable(n) else n)
        if k_max < 1:
            raise ValueError(f"item {it.item_id}: n_samples must be >= 1, got {k_max}")
        out.extend(Item(it.item_id, it.prompt, it.messages, it.n_samples, k) for k in range(k_max))
    return out


def _recorded_meta(out_path: Path) -> dict:
    try:
        side = json.loads(meta_path_for(out_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return side.get("meta") or {}


def _recorded_item_sha(out_path: Path) -> str | None:
    return _recorded_meta(out_path).get("item_file_sha256")


def _item_from_record(rec: dict, index: int, battery: str) -> Item:
    item_id = None
    for k in _ID_KEYS:
        if rec.get(k) is not None:
            item_id = str(rec[k])
            break
    if item_id is None:
        item_id = f"{battery}-{index:05d}"
    n = int(rec.get("n_samples") or 1)
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs:
        messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        users = [m["content"] for m in messages if m["role"] == "user"]
        if not users:
            raise ValueError(f"{battery} item {item_id}: messages contain no user turn")
        return Item(item_id, users[-1], messages, n)
    for k in _PROMPT_KEYS:
        if isinstance(rec.get(k), str) and rec[k].strip():
            return Item(item_id, rec[k], [{"role": "user", "content": rec[k]}], n)
    raise ValueError(f"{battery} item {item_id}: no prompt field (tried messages, {', '.join(_PROMPT_KEYS)})")


def load_items(path: str | Path, battery: str) -> list[Item]:
    """Read a battery item file (JSONL; JSON list also accepted) into normalized Items."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        records = json.loads(text)
        if isinstance(records, dict):
            records = records.get("items", [])
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    items = [_item_from_record(r, i, battery) for i, r in enumerate(records)]
    seen: set[str] = set()
    for it in items:
        if it.item_id in seen:
            raise ValueError(f"{path}: duplicate item_id {it.item_id!r}")
        seen.add(it.item_id)
    return items


def _manifest_entry(manifest: dict | None, battery: str) -> dict | None:
    if not manifest:
        return None
    for container in (manifest.get("batteries"), manifest):
        if isinstance(container, dict) and isinstance(container.get(battery), dict):
            return container[battery]
    return None


def load_manifest(items_dir: Path) -> dict | None:
    mf = items_dir / "MANIFEST.json"
    if mf.exists():
        return json.loads(mf.read_text(encoding="utf-8"))
    return None


def resolve_item_file(items_dir: Path, battery: str, manifest: dict | None) -> tuple[Path, str | None]:
    """Return (path, expected_sha256 or None) for a battery's item file."""
    entry = _manifest_entry(manifest, battery)
    if entry:
        p = entry.get("path") or entry.get("file")
        if p:
            cand = Path(p)
            if not cand.is_absolute():
                cand = items_dir / cand if (items_dir / cand).exists() else cand
            return cand, entry.get("sha256")
    for ext in (".jsonl", ".json"):
        cand = items_dir / f"{battery}{ext}"
        if cand.exists():
            return cand, entry.get("sha256") if entry else None
    for sub in ("batteries", "battery"):
        cand = items_dir / sub / f"{battery}.jsonl"
        if cand.exists():
            return cand, entry.get("sha256") if entry else None
    raise FileNotFoundError(f"no item file for battery {battery!r} under {items_dir}")


def _in_default_set(entry: Any, run_id: str | None) -> bool:
    """A manifest battery runs by default unless it is on demand; an on-demand battery marked ``base_only`` runs by
    default for the base itself (run_id "base") and stays on demand for every other run (FIDELITY_REVIEW N7)."""
    if not isinstance(entry, dict) or not entry.get("on_demand"):
        return True
    return bool(entry.get("base_only")) and run_id == BASE_RUN_ID


def discover_batteries(items_dir: Path, manifest: dict | None, run_id: str | None = None) -> list[str]:
    """Default battery set: every manifest battery except on-demand ones (``on_demand: true``; name them with
    --batteries). karne_ladders_sens is in the default set since 2026-10-07 (configs/batteries.yaml on_demand: false:
    Karne's 1-of-8 statistic is measured on every unit, prereg v0.3 change 37). karne_ladders_sampled_base
    (``on_demand: true, base_only: true``; prereg v0.3 change 54) is in the default set of the base only (``run_id``
    "base"): its all-ladder 1-of-8 labels are the H5 strata of every unit, and derivatives never need it."""
    if manifest and isinstance(manifest.get("batteries"), dict):
        return sorted(k for k, v in manifest["batteries"].items() if _in_default_set(v, run_id))
    found = []
    for sub in (items_dir / "batteries", items_dir / "battery"):
        if sub.is_dir():
            found += [p.stem for p in sorted(sub.glob("*.jsonl"))]
    return sorted(set(found))


def smoke_items(battery: str, n: int = SMOKE_N_ITEMS) -> list[Item]:
    """Benign placeholder items for --smoke when no item file is available."""
    out = []
    for i in range(n):
        q = f"Say hello and count to {i + 1}."
        out.append(Item(f"{battery}-smoke-{i:02d}", q, [{"role": "user", "content": q}]))
    return out


# ---------------------------------------------------------------------------
# Per-battery decoding (configs/batteries.yaml)
# ---------------------------------------------------------------------------


@dataclass
class Decoding:
    mode: str  # "greedy" | "sample"
    temperature: float | None = None
    top_p: float | None = None
    n_samples: int = 1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    source: str = "builtin"

    @property
    def greedy(self) -> bool:
        return self.mode == "greedy"

    def record(self, batch_size: int, seed: int) -> dict:
        d = asdict(self)
        d.update({"greedy": self.greedy, "batch_size": int(batch_size), "thinking": False})
        if self.greedy:
            d.update({"temperature": None, "top_p": None, "n_samples": 1})
        else:
            d.update({
                "top_k": None, "min_p": None, "repetition_penalty": 1.0, "seed": int(seed),
                "seed_rule": "per item: sha256(f'{seed}:{item_id}')[:4] little-endian; all n samples of an "
                             "item in one seeded call",
                "row_key": "(item_id, sample_idx)",
            })
        return d


_SAMPLE_WORDS = {"sample", "sampled", "sampling", "temperature"}
_GREEDY_WORDS = {"greedy", "argmax"}


def load_batteries_config(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def battery_config_entry(bcfg: dict, battery: str) -> dict | None:
    entries = bcfg.get("batteries") if isinstance(bcfg.get("batteries"), dict) else {}
    e = entries.get(battery)
    return e if isinstance(e, dict) else None


def _decoding_spec(bcfg: dict, entry: dict | None) -> dict | None:
    """The decoding mapping for one battery, with named profiles resolved (None if undeclared).

    Profiles live under ``decoding_profiles`` in configs/batteries.yaml (the key sm.data.build_prompt_sets reads);
    a top-level ``decoding`` mapping is still accepted. Before 2026-10-07 only ``decoding`` was read, so the
    ``karne_sampled`` profile of karne_ladders_sens did not resolve (fidelity fix M7, found while pinning Karne's
    decoding).
    """
    profiles: dict = {}
    for key in ("decoding", "decoding_profiles"):
        if isinstance(bcfg.get(key), dict):
            profiles.update(bcfg[key])
    if entry is None:
        return None
    spec = entry.get("decoding")
    if spec is None and entry.get("samples_per_item") is not None:
        spec = {"mode": "sample", "n": entry["samples_per_item"]}
    if spec is None:
        return None
    if isinstance(spec, str):
        name = spec.strip().lower()
        if isinstance(profiles.get(spec), dict):
            prof = dict(profiles[spec])
            prof.setdefault("mode", name)
            return prof
        return {"mode": name}
    if isinstance(spec, dict):
        prof_name = spec.get("profile")
        base = dict(profiles.get(prof_name) or {}) if prof_name else {}
        base.update({k: v for k, v in spec.items() if k != "profile"})
        if "mode" not in base and prof_name:
            base["mode"] = str(prof_name)
        return base
    raise ValueError(f"unsupported decoding spec {spec!r}")


def _spec_mode(spec: dict) -> str:
    raw = str(spec.get("mode") or spec.get("strategy") or spec.get("type") or "").strip().lower()
    if raw in _GREEDY_WORDS:
        return "greedy"
    if raw in _SAMPLE_WORDS:
        return "sample"
    if not raw:
        if spec.get("do_sample") is True or float(spec.get("temperature") or 0.0) > 0.0:
            return "sample"
        return "greedy"
    raise ValueError(f"unknown decoding mode {raw!r} (use greedy or sample)")


def resolve_decoding(
    battery: str,
    bcfg: dict,
    *,
    cli_n_samples: int | None = None,
    cli_max_new_tokens: int | None = None,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    config_label: str = "configs/batteries.yaml",
) -> Decoding:
    """Decoding for one battery.

    configs/batteries.yaml forms accepted (per battery entry under ``batteries:``):
        decoding: greedy                          # or: sample
        decoding: {mode: sample, temperature: 1.0, top_p: 1.0, n: 10, max_new_tokens: 512}
        decoding: <profile>                       # a named mapping under a top-level ``decoding:`` block
        samples_per_item: 10                      # legacy: implies sample mode with n = 10
    Missing fields: temperature 1.0, top_p 1.0, n 10 for sample mode; max_new_tokens 512.
    Batteries with no declaration fall back to the prereg v0.2 Section 5 rule (is_sampled_battery).
    Precedence: explicit CLI flags (recorded) > config > rule defaults.
    """
    spec = _decoding_spec(bcfg, battery_config_entry(bcfg, battery))
    if spec is not None:
        mode = _spec_mode(spec)
        source = f"{config_label}:batteries.{battery}.decoding"
    else:
        spec = {}
        mode = "sample" if is_sampled_battery(battery) else "greedy"
        source = "builtin rule (prereg v0.2 Section 5)"
    mnt = spec.get("max_new_tokens", default_max_new_tokens)
    if cli_max_new_tokens is not None:
        mnt = cli_max_new_tokens
    if mode == "greedy":
        return Decoding("greedy", None, None, 1, int(mnt), source)
    n = spec.get("n", spec.get("n_samples", spec.get("samples", DEFAULT_N_SAMPLES)))
    if cli_n_samples is not None:
        n = cli_n_samples
        source += " (n_samples overridden on the CLI)"
    return Decoding(
        "sample",
        float(spec.get("temperature", SAMPLE_TEMPERATURE)),
        float(spec.get("top_p", SAMPLE_TOP_P)),
        int(n),
        int(mnt),
        source,
    )


# ---------------------------------------------------------------------------
# Core: run one battery with an injected generate function
# ---------------------------------------------------------------------------


def _read_partial(partial: Path, valid: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    done: dict[tuple[str, int], dict] = {}
    if not partial.exists():
        return done
    for line in partial.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line from a killed job
        if not isinstance(rec, dict) or "completion" not in rec or rec.get("item_id") is None:
            continue
        key = row_key(rec["item_id"], rec.get("sample_idx", 0))
        if key in valid:
            done[key] = rec
    return done


def make_record(battery: str, item: Item, completion: str, truncated: bool) -> dict:
    return {
        "item_id": item.item_id,
        "battery": battery,
        "prompt": item.prompt,
        "completion": completion,
        "completion_sha256": sha256_text(completion),
        "truncated": bool(truncated),
        "sample_idx": int(item.sample_idx),
    }


@dataclass
class BatteryResult:
    battery: str
    out_path: Path
    status: str  # "done" | "skipped"
    n_items: int = 0
    n_truncated: int = 0
    n_generated: int = 0
    extra: dict = field(default_factory=dict)


def _units(rows: Sequence[Item], grouped: bool) -> list[list[Item]]:
    """Greedy: one unit per row. Sampled: one unit per item (its samples, contiguous)."""
    if not grouped:
        return [[r] for r in rows]
    units: dict[str, list[Item]] = {}
    for r in rows:
        units.setdefault(r.item_id, []).append(r)
    return [sorted(u, key=lambda r: r.sample_idx) for u in units.values()]


def _decoding_changed(meta: dict, recorded: dict) -> bool:
    want, have = meta.get("decoding"), recorded.get("decoding")
    if not isinstance(want, dict) or not isinstance(have, dict):
        return False
    keys = ("mode", "temperature", "top_p", "n_samples", "max_new_tokens")
    return any(want.get(k) != have.get(k) for k in keys if k in want)


def run_battery(
    battery: str,
    items: Sequence[Item],
    generate_fn: GenerateFn,
    out_path: Path,
    meta: dict,
    restricted_root: Path | None = None,
    chunk_size: int = 64,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> BatteryResult:
    """Generate completions for rows and write the battery JSONL plus meta sidecar.

    ``items`` are rows: for a sampled battery pass ``expand_samples(...)`` output and a generate
    function with ``wants_rows = True``. restricted_root: when not None the output lives under it;
    directories are forced to 0700 and files to 0600. A finished file whose sidecar records a different
    item_file_sha256 or decoding than ``meta`` is regenerated.
    """
    out_path = Path(out_path)
    if out_path.exists() and not force:
        recorded = _recorded_meta(out_path)
        want = meta.get("item_file_sha256")
        have = recorded.get("item_file_sha256")
        if want and have != want:
            log(f"[battery] {battery}: item file changed (recorded {have}, now {want}); regenerating")
            force = True
        elif _decoding_changed(meta, recorded):
            log(f"[battery] {battery}: decoding changed since the recorded run; regenerating")
            force = True
        else:
            log(f"[battery] {battery}: already done ({out_path})")
            return BatteryResult(battery, out_path, "skipped")
    file_mode = 0o600 if restricted_root is not None else 0o644
    ensure_dir(out_path.parent, restricted_root)
    partial = partial_path_for(out_path)
    if force and partial.exists():
        partial.unlink()
    keys = [row_key(it.item_id, it.sample_idx) for it in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{battery}: duplicate (item_id, sample_idx) rows")
    grouped = bool(getattr(generate_fn, "wants_rows", False))
    done = _read_partial(partial, set(keys))
    units = _units(items, grouped)
    todo_units = []
    for u in units:
        ks = [row_key(r.item_id, r.sample_idx) for r in u]
        if all(k in done for k in ks):
            continue
        for k in ks:  # a sampled item is regenerated whole so its samples share one seeded call
            done.pop(k, None)
        todo_units.append(u)
    # Longest prompts first: padding waste is lower and an OOM shows up in the first chunk.
    order = sorted(range(len(todo_units)), key=lambda i: (-len(todo_units[i][0].prompt), i))
    todo_units = [todo_units[i] for i in order]
    if done:
        log(f"[battery] {battery}: resuming, {len(done)} of {len(items)} rows already in {partial.name}")
    n_generated = 0
    if todo_units:
        if done and grouped:
            # Rewrite the partial with only the kept rows (incomplete items were dropped).
            kept = "".join(json.dumps(done[k], ensure_ascii=False) + "\n" for k in keys if k in done)
            _atomic_write_bytes(partial, kept.encode("utf-8"), file_mode)
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_APPEND, file_mode)
        os.chmod(partial, file_mode)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            chunks: list[list[Item]] = []
            cur: list[Item] = []
            for u in todo_units:
                cur.extend(u)
                if len(cur) >= max(1, chunk_size):
                    chunks.append(cur)
                    cur = []
            if cur:
                chunks.append(cur)
            for chunk in chunks:
                if grouped:
                    results = generate_fn(chunk)
                else:
                    results = generate_fn([it.messages for it in chunk])
                if len(results) != len(chunk):
                    raise RuntimeError(f"generate returned {len(results)} results for {len(chunk)} rows")
                for it, (text, trunc) in zip(chunk, results):
                    rec = make_record(battery, it, text, trunc)
                    done[row_key(it.item_id, it.sample_idx)] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_generated += len(chunk)
                log(f"[battery] {battery}: {len(done)}/{len(items)}")
    records = []
    for it, k in zip(items, keys):
        rec = done[k]
        records.append(make_record(battery, it, rec["completion"], rec["truncated"]))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    n_trunc = sum(1 for r in records if r["truncated"])
    summary = {
        "battery": battery,
        "n_items": len({r["item_id"] for r in records}),
        "n_rows": len(records),
        "n_truncated": n_trunc,
        "truncated_rate": (n_trunc / len(records)) if records else 0.0,
        "output_file": out_path.name,
        "output_sha256": _sha256_bytes(data),
        "restricted": restricted_root is not None,
    }
    # Meta first, then the JSONL: the JSONL is the completion marker for idempotence.
    _write_meta_json(meta_path_for(out_path), summary, extra_meta=meta, mode=file_mode)
    _atomic_write_bytes(out_path, data, file_mode)
    if partial.exists():
        partial.unlink()
    log(f"[battery] {battery}: wrote {len(records)} rows ({n_trunc} truncated) to {out_path}")
    return BatteryResult(battery, out_path, "done", len(records), n_trunc, n_generated)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _count_tokens(tokenizer: Any, text: str) -> int:
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    return len(ids)


def sampled_generate_fn(gen: Any, seed: int, temperature: float = SAMPLE_TEMPERATURE, top_p: float = SAMPLE_TOP_P) -> GenerateFn:
    """Generate function for sampled batteries.

    Takes rows (all samples of each item, contiguous) and asks the generator for n samples per item in one
    call seeded with sha256(seed:item_id), so a row's text depends only on (model, seed, item, n).
    """

    def fn(rows: list[Item]) -> list[tuple[str, bool]]:
        groups: list[list[Item]] = []
        for r in rows:
            if groups and groups[-1][0].item_id == r.item_id:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            if [r.sample_idx for r in g] != list(range(len(g))):
                raise ValueError(f"item {g[0].item_id}: rows must carry sample_idx 0..n-1 in order")
        reqs = [(g[0].messages, len(g), _id_seed(seed, g[0].item_id)) for g in groups]
        outs = gen.sample_groups(reqs, float(temperature), float(top_p))
        flat: list[tuple[str, bool]] = []
        for g, o in zip(groups, outs):
            if len(o) != len(g):
                raise RuntimeError(f"item {g[0].item_id}: asked for {len(g)} samples, got {len(o)}")
            flat.extend(o)
        return flat

    fn.wants_rows = True  # type: ignore[attr-defined]
    return fn


class _Renderer:
    """Renders a message list once with the chat template (thinking off); caches by content."""

    def __init__(self, models_mod: Any, tokenizer: Any) -> None:
        self._models = models_mod
        self.tokenizer = tokenizer
        self.render_meta: dict = {}
        self._cache: dict[str, str] = {}

    def render(self, messages: list[dict]) -> str:
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        text = self._models.render_chat(self.tokenizer, messages, thinking=False, add_generation_prompt=True)
        last = getattr(self._models, "LAST_RENDER_META", None)
        if isinstance(last, dict):
            self.render_meta = dict(last)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = text
        return text


class HFGenerator(_Renderer):
    """Generation through sm.common.models (left padding, explicit position ids).

    Greedy: ``generate(..., greedy=True)``. truncated: if ``generate`` accepts ``return_finish=True`` it is
    used directly; otherwise a completion is flagged truncated when its re-encoded token count reaches
    max_new_tokens (EOS is stripped on decode, so a completion that stopped on EOS has at most
    max_new_tokens - 1 content tokens).
    Sampled: ``generate_detailed(..., greedy=False, temperature, top_p, top_k=0)``, the n samples of one
    item in one batch after torch.manual_seed(item seed).
    """

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int, batch_size: int) -> None:
        from sm.common import models as sm_models  # type: ignore

        super().__init__(sm_models, tokenizer)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        try:
            params = inspect.signature(sm_models.generate).parameters
        except (TypeError, ValueError):
            params = {}
        self._native_finish = "return_finish" in params

    def __call__(self, batch_messages: list[list[dict]]) -> list[tuple[str, bool]]:
        prompts = [self.render(m) for m in batch_messages]
        if self._native_finish:
            out = self._models.generate(
                self.model, self.tokenizer, prompts,
                max_new_tokens=self.max_new_tokens, batch_size=self.batch_size, greedy=True, return_finish=True,
            )
            return [(str(t), bool(f)) for t, f in out]
        texts = self._models.generate(
            self.model, self.tokenizer, prompts,
            max_new_tokens=self.max_new_tokens, batch_size=self.batch_size, greedy=True,
        )
        return [(t, _count_tokens(self.tokenizer, t) >= self.max_new_tokens) for t in texts]

    def sample_groups(self, reqs: list[tuple[list[dict], int, int]], temperature: float, top_p: float) -> list[list[tuple[str, bool]]]:
        import torch

        detailed = getattr(self._models, "generate_detailed", None)
        if detailed is None:
            raise RuntimeError("sampled batteries need sm.common.models.generate_detailed")
        out: list[list[tuple[str, bool]]] = []
        for messages, n, seed in reqs:
            text = self.render(messages)
            torch.manual_seed(int(seed))
            rows = detailed(
                self.model, self.tokenizer, [text] * int(n), max_new_tokens=self.max_new_tokens,
                batch_size=max(1, int(n)), greedy=False, temperature=float(temperature), top_p=float(top_p),
                top_k=0, min_p=None, typical_p=1.0, repetition_penalty=1.0,
            )
            out.append([(r["completion"], bool(r["truncated"])) for r in rows])
        return out

    def method(self) -> str:
        return "native_finish_flag" if self._native_finish else "reencoded_token_count>=max_new_tokens"


class VLLMGenerator(_Renderer):
    """Generation through vLLM from token ids; truncated = finish_reason == "length"."""

    def __init__(
        self,
        model_path: str,
        tokenizer: Any,
        max_new_tokens: int,
        seed: int,
        revision: str | None = None,
        lora_path: str | None = None,
        max_lora_rank: int = 128,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        llm: Any = None,
        sampling_params_cls: Any = None,
        lora_request: Any = None,
        limit_mm_per_prompt: dict[str, int] | None = None,
    ) -> None:
        from sm.common import models as sm_models  # type: ignore

        super().__init__(sm_models, tokenizer)
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.llm_kwargs: dict[str, Any] = {}
        # A14: {"image": 0, "video": 0} for a multimodal checkpoint served text-only; None for a text-only one
        # (vLLM rejects multimodal settings on a text-only model).
        self.limit_mm_per_prompt = dict(limit_mm_per_prompt) if limit_mm_per_prompt is not None else None
        self.limit_mm_note: str | None = None
        if llm is not None:  # injected (tests)
            self.llm, self._sampling_params_cls, self.lora_request = llm, sampling_params_cls, lora_request
            if self.limit_mm_per_prompt is not None:
                self.llm_kwargs["limit_mm_per_prompt"] = dict(self.limit_mm_per_prompt)
            return
        from vllm import LLM, SamplingParams  # type: ignore

        self._sampling_params_cls = SamplingParams
        kwargs: dict[str, Any] = dict(
            model=model_path,
            dtype=dtype,
            seed=seed,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            # Ignore the checkpoint's generation_config.json defaults (top_k, temperature ...):
            # decoding comes only from configs/batteries.yaml.
            generation_config="vllm",
        )
        if revision:
            kwargs["revision"] = revision
        self.lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest  # type: ignore

            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = max_lora_rank
            self.lora_request = LoRARequest("adapter", 1, lora_path)
        if self.limit_mm_per_prompt is not None:
            kwargs["limit_mm_per_prompt"] = dict(self.limit_mm_per_prompt)
        try:
            self.llm = LLM(**kwargs)
        except (ValueError, TypeError) as exc:
            # Detection said multimodal but this vLLM build treats the checkpoint as text-only (or has no such
            # argument): serve without the limit and record why.
            if "limit_mm_per_prompt" not in kwargs or not re.search(r"multimodal|multi-modal|limit_mm", str(exc), re.I):
                raise
            kwargs.pop("limit_mm_per_prompt")
            self.limit_mm_per_prompt = None
            self.limit_mm_note = f"limit_mm_per_prompt rejected by vLLM ({type(exc).__name__}: {exc}); served without it"
            print(f"[battery] WARNING: {self.limit_mm_note}", file=sys.stderr)
            self.llm = LLM(**kwargs)
        self.llm_kwargs = {k: v for k, v in kwargs.items() if k != "model"}

    def token_prompts(self, batch_messages: list[list[dict]]) -> list[dict]:
        """Token-id prompts: render once, encode with encode_text (no second BOS on rendered chat text)."""
        return [{"prompt_token_ids": self._models.encode_text(self.tokenizer, self.render(m))} for m in batch_messages]

    def _lora_kwargs(self) -> dict:
        return {"lora_request": self.lora_request} if self.lora_request is not None else {}

    def __call__(self, batch_messages: list[list[dict]]) -> list[tuple[str, bool]]:
        prompts = self.token_prompts(batch_messages)
        params = self._sampling_params_cls(temperature=0.0, max_tokens=self.max_new_tokens, seed=self.seed)
        outs = self.llm.generate(prompts, params, **self._lora_kwargs())
        return [(o.outputs[0].text, o.outputs[0].finish_reason == "length") for o in outs]

    def sample_groups(self, reqs: list[tuple[list[dict], int, int]], temperature: float, top_p: float) -> list[list[tuple[str, bool]]]:
        prompts = self.token_prompts([m for m, _, _ in reqs])
        params = [
            self._sampling_params_cls(n=int(n), temperature=float(temperature), top_p=float(top_p),
                                      max_tokens=self.max_new_tokens, seed=int(s))
            for _, n, s in reqs
        ]
        outs = self.llm.generate(prompts, params, **self._lora_kwargs())
        res = []
        for o in outs:
            comps = sorted(o.outputs, key=lambda c: getattr(c, "index", 0))
            res.append([(c.text, c.finish_reason == "length") for c in comps])
        return res

    def method(self) -> str:
        return "vllm_finish_reason==length"


_MM_CONFIG_KEYS = ("vision_config", "visual", "vision_tower", "audio_config", "image_token_id", "video_token_id")


def is_multimodal_checkpoint(model_path: str, revision: str | None = None) -> bool:
    """True when a checkpoint's config describes a multimodal model (vision or audio tower).

    Reads <model_path>/config.json for a local directory, else the cached Hub config (offline, at ``revision``).
    An unreadable config counts as text-only (the caller records the result).
    """
    cfg: dict | None = None
    local = Path(model_path) / "config.json"
    if local.is_file():
        try:
            cfg = json.loads(local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = None
    else:
        try:
            from transformers import AutoConfig  # type: ignore

            kw = {"revision": revision} if revision else {}
            cfg = AutoConfig.from_pretrained(model_path, **kw).to_dict()
        except Exception:  # noqa: BLE001  (not cached, unknown architecture, no transformers)
            cfg = None
    if not isinstance(cfg, dict):
        return False
    if any(cfg.get(k) is not None for k in _MM_CONFIG_KEYS):
        return True
    archs = " ".join(str(a) for a in (cfg.get("architectures") or []))
    return bool(re.search(r"ConditionalGeneration|VL|Vision", archs))


def vllm_available() -> bool:
    try:
        import vllm  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001  (ImportError, or a broken CUDA build raising at import)
        return False
    return True


def resolve_base_info(base: str, bases_yaml: Path | None) -> dict:
    """Registry entry for a base (with configs/bases.yaml pins); pass-through dict if unknown."""
    try:
        from sm.common.registry import resolve_base  # type: ignore

        return resolve_base(base, bases_yaml=str(bases_yaml) if bases_yaml else None)
    except Exception:  # noqa: BLE001  (registry missing or key unknown: treat as HF id / path)
        p = Path(base)
        return {"hf": str(p.resolve()) if p.is_dir() else base, "revision": None, "key": None,
                "thinking_switch": None, "family": None}


def resolve_revision(base: str, cli_revision: str | None, bases_yaml: Path | None) -> tuple[str, str | None, str, bool]:
    """(hf id or local path, revision, revision source, exempt_from_pin) for a base."""
    info = resolve_base_info(base, bases_yaml)
    hf = str(info.get("hf") or base)
    # Local directories carry no revision; the registry's test model is exempt from pinning.
    local = Path(hf).is_dir() or info.get("role") == "test"
    if cli_revision:
        return hf, str(cli_revision), "cli_or_config", local
    rev = info.get("revision")
    if rev:
        return hf, str(rev), f"{bases_yaml or 'registry'}", local
    return hf, None, "unpinned", local


_DTYPE_NAMES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16",
                "half": "float16", "fp32": "float32", "float32": "float32", "float": "float32"}


def torch_dtype(name: Any):
    """torch dtype for a short or long dtype name."""
    import torch

    if isinstance(name, torch.dtype):
        return name
    key = _DTYPE_NAMES.get(str(name).lower())
    if key is None:
        raise ValueError(f"unknown dtype {name!r}; use one of {sorted(_DTYPE_NAMES)}")
    return getattr(torch, key)


def _adapter_rank(adapter_path: str) -> int | None:
    cfg = Path(adapter_path) / "adapter_config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text())["r"])
        except (KeyError, ValueError, TypeError):
            return None
    return None


def merge_lora_fp32(peft_model: Any, stats: dict | None = None) -> tuple[Any, int]:
    """Merge every LoRA layer into its base weight in fp32, cast back to the weight dtype, drop the LoRA modules.

    W <- dtype(float32(W) + transpose(float32(B) @ float32(A)) * scaling), one layer at a time, so the full model
    never has to exist in fp32. For a bf16 checkpoint this equals a full fp32 merge followed by a bf16 cast, and
    the cast still rounds away every update smaller than half a bf16 ULP of W: only an fp32 weight dtype keeps
    the trained model. With ``stats`` the rounding is measured: n_delta_nonzero (entries with a non-zero
    update), n_rounded_away (of those, entries whose stored weight did not change), share_rounded_away.
    Returns (plain transformers model, number of merged layers). DoRA layers are refused.
    """
    import torch
    from peft.tuners.lora import LoraLayer  # type: ignore
    from peft.utils import transpose  # type: ignore

    n = 0
    if stats is not None:
        stats.setdefault("n_delta_nonzero", 0)
        stats.setdefault("n_rounded_away", 0)
    with torch.no_grad():
        for name, mod in peft_model.named_modules():
            if not isinstance(mod, LoraLayer):
                continue
            base = mod.get_base_layer()
            for an in list(mod.lora_A.keys()):
                if getattr(mod, "use_dora", {}).get(an):
                    raise NotImplementedError(f"{name}: DoRA adapters are not supported by the fp32 merge")
                a = mod.lora_A[an].weight.detach().float()
                b = mod.lora_B[an].weight.detach().float()
                delta = transpose(b @ a, getattr(mod, "fan_in_fan_out", False)) * float(mod.scaling[an])
                w = base.weight
                old = w.data
                new = (old.float() + delta.to(w.device)).to(w.dtype)
                if stats is not None:
                    nz = delta.to(w.device) != 0
                    stats["n_delta_nonzero"] = int(stats.get("n_delta_nonzero", 0)) + int(nz.sum().item())
                    stats["n_rounded_away"] = int(stats.get("n_rounded_away", 0)) + int((nz & (new == old)).sum().item())
                w.data = new
                n += 1
    if stats is not None:
        nzt = stats["n_delta_nonzero"]
        stats["share_rounded_away"] = (stats["n_rounded_away"] / nzt) if nzt else None
    unload = getattr(peft_model, "unload", None) or peft_model.base_model.unload
    return unload(), n


def merge_adapter_to_dir(
    base: str,
    adapter_path: str | None,
    merge_root: Path,
    revision: str | None,
    out_dtype: str = "bf16",
    bases_yaml: str | None = None,
    device: str = "auto",
) -> tuple[Path, Any, dict]:
    """Export the text decoder (base, or base+adapter) to a fresh temp dir under merge_root.

    The base is loaded through load_model (pinned revision, text-only class) in ``out_dtype``, the adapter is
    merged layer by layer in fp32 (merge_lora_fp32; a bf16 merge would round LoRA updates before the final
    cast) and the result is cast to ``out_dtype`` and saved. The base goes through the same export
    (adapter_path None) so base and derivative are served by the same text-only class in the same dtype.
    Returns (dir, tokenizer, meta).
    """
    import gc

    from sm.common.models import load_model  # type: ignore

    merge_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="merged-" if adapter_path else "export-", dir=str(merge_root)))
    kwargs: dict[str, Any] = {"adapter_path": adapter_path, "revision": revision, "dtype": out_dtype, "device": device,
                              "merge_adapter": False}
    if bases_yaml:
        kwargs["bases_yaml"] = bases_yaml
    model, tokenizer, load_meta = load_model(base, **kwargs)
    n_merged = 0
    rounding: dict = {}
    if adapter_path:
        model, n_merged = merge_lora_fp32(model, rounding)
        if n_merged == 0:
            raise RuntimeError(f"adapter {adapter_path} has no LoRA layers to merge")
    target = torch_dtype(out_dtype)
    model = model.to(target)
    model.save_pretrained(str(tmp))
    tokenizer.save_pretrained(str(tmp))
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    meta = {
        "export_dir": str(tmp),
        "merge_dtype": "float32" if adapter_path else None,
        "merge_method": "per-layer fp32 (W + B A * scaling), then cast" if adapter_path else None,
        "n_lora_layers_merged": n_merged,
        "saved_dtype": str(target).replace("torch.", ""),
        "adapter_merged": bool(adapter_path),
        "rounding": rounding or None,
        "load_meta": load_meta,
    }
    return tmp, tokenizer, meta


# ---------------------------------------------------------------------------
# Tiny offline model for --smoke (private: contracts define no builder)
# ---------------------------------------------------------------------------


def build_tiny_local_model(dest: Path) -> Path:
    """Build a tiny random Qwen2 causal LM and a word-level tokenizer from configs (no network)."""
    if (dest / "config.json").exists() and (dest / "tokenizer.json").exists():
        return dest
    from tokenizers import Tokenizer, decoders, models as tk_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    import torch

    torch.manual_seed(0)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    words = (
        "user assistant system : say hello and count to the a of is what how why please "
        "1 2 3 4 5 6 7 8 9 10 . , ? ! yes no A B C D answer with letter correct option only write in lowercase"
    ).split()
    for w in words:
        vocab.setdefault(w, len(vocab))
    for i in range(64):
        vocab.setdefault(f"w{i}", len(vocab))
    tok = Tokenizer(tk_models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="##")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    fast.padding_side = "left"
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant : {% endif %}"
    )
    cfg = Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=1, pad_token_id=0, bos_token_id=None, tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    fast.save_pretrained(str(dest))
    return dest


def smoke_base(out_root: Path) -> str:
    """The base --smoke uses when none is given: SM_TINY_MODEL, else a locally built tiny model offline."""
    env_tiny = os.environ.get("SM_TINY_MODEL")
    if os.environ.get("SM_OFFLINE") == "1" and not env_tiny:
        return str(build_tiny_local_model(Path(out_root) / ".smoke_model"))
    return "tiny"  # registry key; honours SM_TINY_MODEL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sm.eval.battery", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="YAML config; CLI flags override its keys")
    p.add_argument("--base", help="base key (configs/bases.yaml), HF id, or local path")
    p.add_argument("--adapter", help="LoRA adapter directory (omit for the base itself); evaluated unmerged")
    p.add_argument("--revision", help="model revision (default: the pin in configs/bases.yaml)")
    p.add_argument("--bases-yaml", help=f"base registry pins (default {DEFAULT_BASES_YAML})")
    p.add_argument("--batteries-config", help=f"per-battery decoding and harm flags (default {DEFAULT_BATTERIES_CONFIG})")
    p.add_argument("--allow-unpinned", action="store_true", help="permit a measured run of a Hub model with no pinned revision")
    p.add_argument("--run-id", help="run id for the output path (default: 'base' or the adapter dir name)")
    p.add_argument("--batteries", nargs="+", help="battery names (default: all in the manifest)")
    p.add_argument("--items-dir", help="directory with battery item files and MANIFEST.json (default data/prompts)")
    p.add_argument("--out", help="results root (default results)")
    p.add_argument("--restricted-root", help="root for harmful batteries (default <out>/restricted)")
    p.add_argument("--restricted-extra", nargs="*", help="extra battery names to treat as restricted")
    p.add_argument("--max-new-tokens", type=int, help="override every battery's max_new_tokens (recorded)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--chunk-size", type=int, help="rows per checkpoint chunk (default 4 x batch size; vllm 512)")
    p.add_argument("--limit", type=int, help="only the first N items per battery (debugging)")
    p.add_argument("--backend", choices=("hf", "vllm"))
    p.add_argument("--strict-backend", action="store_true", help="exit 3 instead of falling back when vllm is missing")
    p.add_argument("--allow-vllm-measured", action="store_true",
                   help="permit vllm for non-smoke batteries (recorded as a deviation in the meta)")
    p.add_argument("--n-samples", type=int, help="override samples per question for sampled batteries (recorded)")
    p.add_argument("--merge-dir", help="vllm: export base / merged derivative into a temp copy under this dir (scratch)")
    p.add_argument("--vllm-lora", action="store_true", help="vllm: serve the adapter via LoRARequest instead of an fp32 merge")
    p.add_argument("--vllm-dtype", help="vllm: dtype the exported copy is cast to and served in (default bf16)")
    p.add_argument("--keep-merged", action="store_true", help="vllm: keep the exported temp copy")
    p.add_argument("--max-model-len", type=int, help="vllm max_model_len (default 4096)")
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--tensor-parallel-size", type=int)
    p.add_argument("--device", help="device for HF loading (default auto)")
    p.add_argument("--dtype", help="compute dtype for HF loading (default bf16; --smoke uses fp32)")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true", help="tiny model, 8 items per battery, 16 new tokens, CPU fp32")
    p.add_argument("--force", action="store_true", help="regenerate even if the output exists")
    p.add_argument("--allow-download", action="store_true", help="permit Hub downloads (default offline)")
    return p


_DEFAULTS: dict[str, Any] = {
    "base": None,
    "adapter": None,
    "revision": None,
    "bases_yaml": None,
    "batteries_config": None,
    "allow_unpinned": False,
    "run_id": None,
    "batteries": None,
    "items_dir": "data/prompts",
    "out": "results",
    "restricted_root": None,
    "restricted_extra": [],
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "batch_size": DEFAULT_BATCH_SIZE,
    "chunk_size": None,
    "limit": None,
    "backend": "hf",
    "strict_backend": False,
    "allow_vllm_measured": False,
    "n_samples": None,
    "merge_dir": None,
    "vllm_lora": False,
    "vllm_dtype": "bf16",
    "keep_merged": False,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 1,
    "device": "auto",
    "dtype": "bf16",
    "seed": 0,
    "smoke": False,
    "force": False,
    "allow_download": False,
}


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        import yaml

        loaded = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        loaded = loaded.get("battery", loaded) if isinstance(loaded, dict) else {}
        cfg = {k.replace("-", "_"): v for k, v in loaded.items()}
    settings = dict(_DEFAULTS)
    explicit: set[str] = set()
    for k, v in cfg.items():
        if k in settings and v is not None:
            settings[k] = v
            explicit.add(k)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if isinstance(v, bool):
            settings[k] = bool(settings.get(k)) or v
            if v:
                explicit.add(k)
        elif v is not None:
            settings[k] = v
            explicit.add(k)
    settings["explicit"] = sorted(explicit)
    if isinstance(settings["batteries"], str):
        settings["batteries"] = [settings["batteries"]]
    settings["config_path"] = args.config
    settings["config_sha256"] = _sha256_file(args.config) if args.config else None
    return settings


@dataclass
class Target:
    battery: str
    items: list[Item]
    item_meta: dict
    restricted: bool
    out_path: Path
    decoding: Decoding
    restricted_reason: str | None = None


def _plan_batteries(s: dict[str, Any], bcfg: dict, bcfg_label: str) -> list[Target]:
    """Targets (battery, items, item meta, restriction, output path, decoding) to consider."""
    items_dir = Path(s["items_dir"])
    manifest = load_manifest(items_dir) if items_dir.exists() else None
    names = s["batteries"]
    explicit = set(s["explicit"])
    cli_mnt = int(s["max_new_tokens"]) if ("max_new_tokens" in explicit or s["smoke"]) else None
    cli_n = int(s["n_samples"]) if s.get("n_samples") is not None else None
    raw: list[tuple[str, list[Item], dict]] = []
    if s["smoke"] and not names:
        for b in SMOKE_BATTERIES:
            raw.append((b, smoke_items(b), {"item_file": None, "item_file_sha256": None, "synthetic": True}))
    else:
        if not names:
            names = discover_batteries(items_dir, manifest, s.get("run_id"))
            if not names:
                raise SystemExit(f"no batteries given and none found under {items_dir}")
        for b in names:
            try:
                path, expected = resolve_item_file(items_dir, b, manifest)
            except FileNotFoundError:
                if s["smoke"]:
                    raw.append((b, smoke_items(b), {"item_file": None, "item_file_sha256": None, "synthetic": True}))
                    continue
                raise
            digest = _sha256_file(path)
            if expected and expected != digest:
                raise SystemExit(f"{b}: item file {path} sha256 {digest} does not match manifest {expected}")
            items = load_items(path, b)
            if s["smoke"]:
                items = items[:SMOKE_N_ITEMS]
            if s["limit"]:
                items = items[: int(s["limit"])]
            raw.append((b, items, {"item_file": str(path), "item_file_sha256": digest, "synthetic": False}))
    out_root = Path(s["out"])
    restricted_root = Path(s["restricted_root"]) if s["restricted_root"] else out_root / "restricted"
    targets = []
    for b, items, item_meta in raw:
        entry = battery_config_entry(bcfg, b) or {}
        reason = restriction_reason(b, entry, s["restricted_extra"] or ())
        restricted = reason is not None
        dec = resolve_decoding(b, bcfg, cli_n_samples=cli_n, cli_max_new_tokens=cli_mnt,
                               default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS, config_label=bcfg_label)
        if not item_meta.get("synthetic"):
            check_karne_decoding(b, dec)
        out_path = battery_output_path(out_root, s["base"], s["run_id"], b, restricted, restricted_root)
        if restricted:
            legacy = battery_output_path(out_root, s["base"], s["run_id"], b, False)
            migrate_unrestricted_outputs(legacy, out_path, restricted_root)
        targets.append(Target(b, items, item_meta, restricted, out_path, dec, reason))
    return targets


def _rows_for(t: Target) -> list[Item]:
    if t.decoding.greedy:
        return [Item(it.item_id, it.prompt, it.messages, it.n_samples, 0) for it in t.items]
    return expand_samples(t.items, t.decoding.n_samples)


def _needs_run(t: Target, force: bool) -> bool:
    if force or not t.out_path.exists():
        return True
    rec = _recorded_meta(t.out_path)
    if t.item_meta.get("item_file_sha256") and rec.get("item_file_sha256") != t.item_meta["item_file_sha256"]:
        return True
    return _decoding_changed({"decoding": t.decoding.record(0, 0)}, rec)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    s = resolve_settings(args)
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    if not s["allow_download"]:
        for k in saved:
            os.environ.setdefault(k, "1")
    try:
        return _run(args, s)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(args: argparse.Namespace, s: dict[str, Any]) -> int:
    explicit = set(s["explicit"])
    if s["smoke"]:
        # Device policy (HARNESS_CONTRACTS): every smoke entry point runs on CPU in fp32.
        if "max_new_tokens" not in explicit:
            s["max_new_tokens"] = SMOKE_MAX_NEW_TOKENS
        if "batch_size" not in explicit:
            s["batch_size"] = 4
        if "device" not in explicit:
            s["device"] = "cpu"
        if "dtype" not in explicit:
            s["dtype"] = "fp32"
        if not s["base"]:
            s["base"] = smoke_base(Path(s["out"]))
        s["restricted_extra"] = list(s["restricted_extra"] or []) + ["smoke_restricted"]
    if not s["base"]:
        print("error: --base is required (or base: in --config)", file=sys.stderr)
        return EXIT_USAGE

    s["run_id"] = s["run_id"] or default_run_id(s["adapter"])
    run_id = s["run_id"]
    out_root = Path(s["out"])
    restricted_root = Path(s["restricted_root"]) if s["restricted_root"] else out_root / "restricted"
    base_key = s["base"]
    try:
        bases_yaml = find_config(s["bases_yaml"], DEFAULT_BASES_YAML)
        bcfg_path = find_config(s["batteries_config"], DEFAULT_BATTERIES_CONFIG)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    bcfg = load_batteries_config(bcfg_path)
    bcfg_label = str(bcfg_path) if bcfg_path else "configs/batteries.yaml (absent)"

    hf, revision, rev_source, is_local = resolve_revision(base_key, s["revision"], bases_yaml)
    if not s["smoke"] and not is_local and not revision and not s["allow_unpinned"]:
        print(
            f"error: {base_key} ({hf}) has no pinned revision in {bases_yaml or 'the registry'}; measured runs must "
            "load the pinned revision (pin it in configs/bases.yaml, pass --revision, or --allow-unpinned to log a deviation)",
            file=sys.stderr,
        )
        return EXIT_USAGE

    targets = _plan_batteries(s, bcfg, bcfg_label)
    pending = [t for t in targets if _needs_run(t, bool(s["force"]))]
    if not pending:
        print("[battery] already done")
        return EXIT_OK

    _set_seed(int(s["seed"]))
    backend = s["backend"]
    backend_note = None
    if backend == "vllm" and not s["smoke"] and not s["allow_vllm_measured"]:
        print(
            "[battery] vllm is refused for measured batteries (serving differs from the HF path used for the base "
            "and the drift stage); use --backend hf, or --allow-vllm-measured to record a deviation",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if backend == "vllm" and not vllm_available():
        if s["strict_backend"]:
            print("[battery] vllm is not importable and --strict-backend is set; exiting", file=sys.stderr)
            return EXIT_BACKEND_UNAVAILABLE
        backend_note = "vllm requested but not importable; fell back to hf"
        print(f"[battery] WARNING: {backend_note}", file=sys.stderr)
        backend = "hf"
    if backend == "vllm" and not s["merge_dir"] and not s["vllm_lora"]:
        print("[battery] vllm needs --merge-dir (scratch) for the text-only export, or --vllm-lora", file=sys.stderr)
        return EXIT_USAGE
    if (backend == "vllm" and not s["smoke"] and s["adapter"] and not s["vllm_lora"]
            and str(torch_dtype(str(s["vllm_dtype"]))) != "torch.float32"):
        # Review finding 6 (vLLM path): the export casts W + B A back to the serving dtype, so a bf16 export rounds
        # away every LoRA update below half a bf16 ULP of W and the battery would measure a different model from
        # the unmerged one that drift and monitor_scores read. Measured batteries keep B A separate (--vllm-lora,
        # LoRARequest) or serve the merged export in fp32 (--vllm-dtype fp32, then the base export too).
        print(
            "[battery] a merged vLLM export in " + str(s["vllm_dtype"]) + " rounds LoRA updates away; for measured "
            "batteries use --vllm-lora (adapter applied separately) or --vllm-dtype fp32 (base and derivative), or --backend hf",
            file=sys.stderr,
        )
        return EXIT_USAGE

    merged_dir: Path | None = None
    model_meta: dict[str, Any]
    try:
        if backend == "vllm":
            lora_path = None
            vllm_dtype = str(s["vllm_dtype"])
            if s["vllm_lora"]:
                from transformers import AutoTokenizer

                model_path = hf
                lora_path = s["adapter"]
                tok_kwargs = {"revision": revision} if revision and not is_local else {}
                tokenizer = AutoTokenizer.from_pretrained(hf, **tok_kwargs)
                info = resolve_base_info(base_key, bases_yaml)
                tokenizer.sm_thinking_switch = info.get("thinking_switch") if info.get("key") else "auto"
                tokenizer.sm_family = info.get("family")
                serve_rev = revision
                export_meta = None
                adapter_mode = "vllm_lora_request" if lora_path else "none"
            else:
                # Base and derivative both go through the same text-only export; adapter merged in fp32.
                merged_dir, tokenizer, export_meta = merge_adapter_to_dir(
                    base_key, s["adapter"], Path(s["merge_dir"]), revision, out_dtype=vllm_dtype,
                    bases_yaml=str(bases_yaml) if bases_yaml else None, device=str(s["device"]),
                )
                model_path, serve_rev = str(merged_dir), None
                adapter_mode = f"merged_fp32_then_{export_meta['saved_dtype']}" if s["adapter"] else "none"
            serve_dtype = str(torch_dtype(vllm_dtype)).replace("torch.", "")
            # A14: batteries are text-only; a multimodal checkpoint (the Qwen3.5 VL bases served directly under
            # --vllm-lora) gets limit_mm_per_prompt {"image": 0, "video": 0}. The text-only export needs none.
            multimodal = is_multimodal_checkpoint(model_path, serve_rev)
            gen: Any = VLLMGenerator(
                model_path, tokenizer, int(s["max_new_tokens"]), int(s["seed"]), revision=serve_rev,
                lora_path=lora_path, max_lora_rank=(_adapter_rank(lora_path) or 128) if lora_path else 128,
                max_model_len=int(s["max_model_len"]), gpu_memory_utilization=float(s["gpu_memory_utilization"]),
                tensor_parallel_size=int(s["tensor_parallel_size"]), dtype=serve_dtype,
                limit_mm_per_prompt=dict(TEXT_ONLY_MM_LIMIT) if multimodal else None,
            )
            model_meta = {"hf": hf, "revision": revision, "revision_source": rev_source, "adapter": s["adapter"],
                          "adapter_mode": adapter_mode, "exported_text_only": merged_dir is not None,
                          "export": export_meta, "dtype": serve_dtype, "prompt_input": "token_ids (encode_text)",
                          "vllm_kwargs": gen.llm_kwargs,
                          "multimodal_checkpoint": multimodal,
                          "limit_mm_per_prompt": getattr(gen, "limit_mm_per_prompt", None),
                          "limit_mm_note": getattr(gen, "limit_mm_note", None),
                          "deviation": None if s["smoke"] else "vllm used for measured batteries (--allow-vllm-measured)"}
            default_chunk = 512
        else:
            from sm.common.models import load_model  # type: ignore

            load_kwargs: dict[str, Any] = {
                "adapter_path": s["adapter"], "revision": revision, "dtype": s["dtype"], "device": s["device"],
                "attn": "sdpa", "merge_adapter": False,
            }
            if bases_yaml:
                load_kwargs["bases_yaml"] = str(bases_yaml)
            model, tokenizer, model_meta = load_model(base_key, **load_kwargs)
            if hasattr(model, "eval"):
                model.eval()
            model_meta = dict(model_meta)
            model_meta.update({
                "revision_requested": revision, "revision_source": rev_source,
                "adapter_mode": "unmerged_peft" if s["adapter"] else "none",
                "compute_dtype": str(s["dtype"]),
            })
            gen = HFGenerator(model, tokenizer, int(s["max_new_tokens"]), int(s["batch_size"]))
            default_chunk = 4 * int(s["batch_size"])
        chunk = int(s["chunk_size"] or default_chunk)
        # Render one prompt up front so the chat-template switch is known for the meta.
        first = next((t.items[0] for t in pending if t.items), None)
        if first is not None:
            gen.render(first.messages)

        adapter_cfg = Path(s["adapter"]) / "adapter_config.json" if s["adapter"] else None
        # OPEN-7: the weights hash ties the battery to train_meta.adapter_sha256 (same file, same digest).
        adapter_weights = Path(s["adapter"]) / ADAPTER_WEIGHTS_FILE if s["adapter"] else None
        run_meta_common = {
            "stage": "battery",
            "base": base_key,
            "run_id": run_id,
            "model": model_meta,
            "revision": revision,
            "revision_source": rev_source,
            "bases_yaml": str(bases_yaml) if bases_yaml else None,
            "bases_yaml_sha256": _sha256_file(bases_yaml) if bases_yaml else None,
            "batteries_config": str(bcfg_path) if bcfg_path else None,
            "batteries_config_sha256": _sha256_file(bcfg_path) if bcfg_path else None,
            "adapter_path": s["adapter"],
            "adapter_config_sha256": _sha256_file(adapter_cfg) if adapter_cfg and adapter_cfg.exists() else None,
            "adapter_sha256": _sha256_file(adapter_weights) if adapter_weights and adapter_weights.exists() else None,
            "adapter_weights_file": ADAPTER_WEIGHTS_FILE if adapter_weights and adapter_weights.exists() else None,
            "backend": backend,
            "backend_note": backend_note,
            "truncation_method": gen.method(),
            "seed": int(s["seed"]),
            "smoke": bool(s["smoke"]),
            "config_path": s["config_path"],
            "config_sha256": s["config_sha256"],
        }
        for t in pending:
            meta = dict(run_meta_common)
            meta.update(t.item_meta)
            meta["battery"] = t.battery
            meta["restricted"] = bool(t.restricted)
            meta["restricted_reason"] = t.restricted_reason
            meta["chat_render"] = gen.render_meta or None
            meta["decoding"] = t.decoding.record(int(s["batch_size"]), int(s["seed"]))
            disagree = sorted({it.item_id for it in t.items if it.n_samples > 1 and it.n_samples != t.decoding.n_samples})
            if disagree and not t.decoding.greedy:
                meta["item_n_samples_ignored"] = {"n_items": len(disagree), "used": t.decoding.n_samples}
            gen.max_new_tokens = t.decoding.max_new_tokens
            rows = _rows_for(t)
            fn: Any = gen if t.decoding.greedy else sampled_generate_fn(gen, int(s["seed"]), t.decoding.temperature, t.decoding.top_p)
            run_battery(
                t.battery, rows, fn, t.out_path, meta,
                restricted_root=restricted_root if t.restricted else None,
                chunk_size=chunk, force=bool(s["force"]) or t.out_path.exists(),
            )
    finally:
        if merged_dir is not None and not s["keep_merged"]:
            shutil.rmtree(merged_dir, ignore_errors=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
