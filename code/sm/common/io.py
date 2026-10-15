"""IO helpers: meta block, atomic JSON/JSONL writes, digests, seeding.

Contract (docs/HARNESS_CONTRACTS.md):
    meta_block(extra=None) -> dict
    write_json(path, obj, extra_meta=None) -> None   (atomic; adds "meta")
    sha256_file(path) -> str
    set_seed(seed) -> None
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

_STATE: dict[str, Any] = {"seed": None}

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(repo: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Commit SHA and dirty flag of the harness repo (or ``repo``); values are None outside git."""
    cwd = Path(repo) if repo is not None else _REPO_ROOT
    commit = _git(["rev-parse", "HEAD"], cwd)
    dirty: bool | None = None
    if commit is not None:
        status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
        dirty = bool(status) if status is not None else None
    return {"git_commit": commit, "git_dirty": dirty}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def meta_block(extra: dict | None = None) -> dict:
    """Provenance block: git commit, timestamp, host, library versions, seed if set, plus ``extra``.

    Callers put config hash, model id and revision, and input-file SHA-256s in ``extra``
    (see ``config_hash`` and ``sha256_file``).
    """
    meta: dict[str, Any] = {}
    meta.update(git_info())
    meta.update(
        {
            "timestamp": utc_now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "numpy": _pkg_version("numpy"),
            "sm": _pkg_version("stale-monitors"),
        }
    )
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        meta["slurm_job_id"] = slurm
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task:
            meta["slurm_array_task_id"] = task
    if _STATE["seed"] is not None:
        meta["seed"] = _STATE["seed"]
    if extra:
        meta.update(extra)
    return meta


def _default(o: Any) -> Any:
    """JSON fallback for numpy scalars/arrays, paths, sets, and torch tensors."""
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "detach") and hasattr(o, "tolist"):
        return o.detach().cpu().tolist()
    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dumps(obj: Any, indent: int | None = 2) -> str:
    """JSON text with numpy support. NaN/inf are written as JSON NaN/Infinity (Python and pandas read them)."""
    return json.dumps(obj, indent=indent, default=_default, ensure_ascii=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path, obj, extra_meta: dict | None = None) -> None:
    """Atomically write ``obj`` as JSON with a ``meta`` block.

    A dict keeps its keys and gets ``meta`` = meta_block() updated with any existing ``obj["meta"]`` and then
    ``extra_meta``. A non-dict is wrapped as ``{"data": obj, "meta": ...}``. The write goes to a temp file
    in the same directory, is fsynced, then renamed over the target, so readers never see a partial file.
    """
    p = Path(path)
    meta = meta_block()
    if isinstance(obj, dict):
        existing = obj.get("meta")
        if isinstance(existing, dict):
            meta.update(existing)
        if extra_meta:
            meta.update(extra_meta)
        out = {k: v for k, v in obj.items() if k != "meta"}
        out["meta"] = meta
    else:
        if extra_meta:
            meta.update(extra_meta)
        out = {"data": obj, "meta": meta}
    _atomic_write_text(p, dumps(out) + "\n")


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, records: Iterable[dict]) -> int:
    """Atomically write records as JSONL (one compact object per line). Returns the record count.

    JSONL files carry no inline meta; write a sidecar with ``write_json(path + '.meta.json', {...})``.
    """
    lines = [dumps(r, indent=None) for r in records]
    _atomic_write_text(Path(path), "".join(line + "\n" for line in lines))
    return len(lines)


def read_jsonl(path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(cfg: Any) -> str:
    """SHA-256 of the canonical JSON (sorted keys, compact) of a config object."""
    return sha256_text(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=_default))


def load_yaml(path) -> Any:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU, CUDA, MPS); the seed is recorded in later meta blocks."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    _STATE["seed"] = seed


def get_seed() -> int | None:
    return _STATE["seed"]

