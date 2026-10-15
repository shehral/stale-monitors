"""J1 identity behind L*, persona filtering and V4 (ACCEPTANCE.md OPEN-5).

The base card (``sm.interp.base_card``) and the V4 file (``sm.interp.causal``) used to record only the name
of the judge callable. They now record

    judge_identity = {"judge": "J1", "model": str, "revision": str, "policy_sha256": str, ...}

where ``policy_sha256`` is the SHA-256 of the policy text the decision used: the harmful policy for L* and
V4 (``configs/judge_policies/harmful.txt``). The base card also lists the persona policy behind the persona
filtering under ``policies`` and ``uses``. Two ways to fill it:

- ``runtime_identity``: at run time, from what the in-process J1 callables of ``sm.eval.judge`` use: the
  served model and revision that ``jobs/common.sh sm_start_judge`` exports (``SM_J1_MODEL``,
  ``SM_J1_REVISION``) and the policy files under ``SM_JUDGE_POLICY_DIR`` (default
  ``configs/judge_policies``). ``source: "runtime"``.
- ``derived_identity``: after the fact, deterministically from ``configs/judge.yaml`` and
  ``configs/judge_policies/*`` at the artifact's recorded ``meta.git_commit`` (``git show <commit>:<path>``),
  for artifacts written before this field existed (the base card of commit 5acbc0a). ``source: "derived"``,
  ``derived_from: <commit>``. It assumes what the job log can confirm: the J1 server ran at the
  ``configs/judge.yaml`` pin (no ``J1 override from the environment`` line) and ``SM_JUDGE_POLICY_DIR`` was
  unset; ``assumptions`` lists both.

``backfill`` applies ``derived_identity`` to an existing JSON in place (base card: keys ``judge`` and
``judge_identity``; causal: ``judge_identity`` and ``judge_revision``; every other key, ``meta`` included,
is kept). Stdlib plus pyyaml; no model or network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from sm import ops as OPS

JUDGE_YAML = "configs/judge.yaml"
BASES_YAML = "configs/bases.yaml"
DEFAULT_POLICY_DIR = "configs/judge_policies"

# The in-process J1 callables of sm.eval.judge and the policy each one scores under.
CALLABLE_POLICY: dict[str, str] = {
    "sm.eval.judge:judge_refusal_labels": "harmful",
    "sm.eval.judge:judge_refusal": "harmful",
    "sm.eval.judge:score_trait": "persona",
}
SMOKE_CALLABLES = {"fake:refusal", "fake:trait", "fake:chen", "keyword", "keyword_v1"}


class BackfillError(RuntimeError):
    """The artifact cannot be backfilled deterministically (smoke output, fallback judge, unknown commit, ...)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def judge_kind(spec: str | None) -> str:
    """'J1' for an sm.eval.judge callable, 'fake' for the smoke/keyword judges, else 'other'."""
    if not spec:
        return "none"
    if spec in SMOKE_CALLABLES or spec.endswith(":fake_refusal_judge") or spec.endswith(":fake_trait_scorer") \
            or spec.endswith(":fake_chen_scorer"):
        return "fake"
    if spec.endswith(":keyword_refusal_judge"):
        return "fake"
    if spec.startswith("sm.eval.judge:"):
        return "J1"
    return "other"


def _yaml(data: bytes | str | None) -> dict[str, Any]:
    if data is None:
        return {}
    import yaml

    obj = yaml.safe_load(data.decode("utf-8") if isinstance(data, bytes) else data)
    return obj if isinstance(obj, dict) else {}


def _uses_to_policies(uses: Mapping[str, str | None]) -> dict[str, str | None]:
    """{use: callable spec} -> {use: policy name or None}."""
    return {u: (CALLABLE_POLICY.get(spec) if spec else None) for u, spec in uses.items()}


def _assemble(
    model: str | None,
    revision: str | None,
    uses: Mapping[str, str | None],
    policy_sha: Mapping[str, str | None],
    policy_dir: str,
    pin: Mapping[str, Any],
    primary_use: str,
    source: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    use_policy = _uses_to_policies(uses)
    primary_policy = use_policy.get(primary_use)
    pinned = bool(pin.get("model") and pin.get("revision") and model == pin.get("model") and revision == pin.get("revision"))
    out = {
        "judge": "J1",
        "model": model,
        "revision": revision,
        "policy_sha256": policy_sha.get(primary_policy) if primary_policy else None,
        "policy": primary_policy,
        "primary_use": primary_use,
        "uses": {u: {"callable": uses[u], "policy": p, "policy_sha256": policy_sha.get(p) if p else None} for u, p in use_policy.items()},
        "policies": {p: {"sha256": policy_sha.get(p), "path": f"{policy_dir}/{p}.txt"} for p in sorted({p for p in use_policy.values() if p})},
        "pin": {"model": pin.get("model"), "revision": pin.get("revision"), "source": JUDGE_YAML},
        "revision_is_pinned": pinned,
        "source": source,
        "derived_from": None,
    }
    out.update(dict(extra or {}))
    return out


def _non_j1_identity(uses: Mapping[str, str | None], primary_use: str, kind: str) -> dict[str, Any]:
    return {
        "judge": kind,
        "model": None,
        "revision": None,
        "policy_sha256": None,
        "policy": None,
        "primary_use": primary_use,
        "uses": {u: {"callable": s, "policy": None, "policy_sha256": None} for u, s in uses.items()},
        "policies": {},
        "pin": None,
        "revision_is_pinned": False,
        "source": "runtime",
        "derived_from": None,
        "note": (
            "smoke or keyword judge: no J1 model, revision or policy behind these labels"
            if kind == "fake"
            else "not an sm.eval.judge J1 callable: its model, revision and policy are not known here"
        ),
    }


def runtime_identity(uses: Mapping[str, str | None], primary_use: str, repo: str | Path | None = None) -> dict[str, Any]:
    """The J1 identity in effect for this process (what sm.eval.judge._env_j1_judge will use).

    ``uses`` maps a decision ("L_star", "persona_filtering", "V4") to the judge callable spec that made it
    (None when the decision did not run). model: SM_J1_MODEL, else J1_MODEL, else configs/judge.yaml j1.model;
    revision: SM_J1_REVISION, else "unpinned" (as the callable records it); policy texts: SM_JUDGE_POLICY_DIR,
    else configs/judge_policies. A smoke or keyword judge gives judge "fake" with null identity fields.
    """
    kinds = {judge_kind(s) for s in uses.values() if s}
    if "J1" not in kinds:
        return _non_j1_identity(uses, primary_use, "other" if "other" in kinds else "fake")
    root = Path(repo) if repo is not None else OPS.REPO_ROOT
    jy_path = Path(os.environ.get("JUDGE_CONFIG") or (root / JUDGE_YAML))
    jy = _yaml(jy_path.read_bytes()) if jy_path.is_file() else {}
    j1 = dict(jy.get("j1") or {})
    model = os.environ.get("SM_J1_MODEL") or os.environ.get("J1_MODEL") or j1.get("model")
    revision = os.environ.get("SM_J1_REVISION") or "unpinned"
    env_dir = os.environ.get("SM_JUDGE_POLICY_DIR")
    pdir = Path(env_dir) if env_dir else root / DEFAULT_POLICY_DIR
    policy_sha: dict[str, str | None] = {}
    for p in {CALLABLE_POLICY.get(s) for s in uses.values() if s} - {None}:
        f = pdir / f"{p}.txt"
        policy_sha[p] = sha256_bytes(f.read_bytes()) if f.is_file() else None
    try:
        shown_dir = str(pdir.resolve().relative_to(root.resolve()))
    except ValueError:
        shown_dir = str(pdir)
    return _assemble(
        model, revision, uses, policy_sha, shown_dir, j1, primary_use, "runtime",
        {"policy_dir_from_env": bool(env_dir), "model_from_env": bool(os.environ.get("SM_J1_MODEL") or os.environ.get("J1_MODEL"))},
    )


def derived_identity(commit: str, uses: Mapping[str, str | None], primary_use: str, repo: str | Path | None = None) -> dict[str, Any]:
    """The J1 identity at ``commit``: configs/judge.yaml j1.{model, revision} and the SHA-256 of
    ``<policy_dir>/<policy>.txt`` read with ``git show <commit>:<path>``. Deterministic in (commit, uses)."""
    if not commit or OPS.git(["cat-file", "-e", f"{commit}^{{commit}}"], repo) is None:
        raise BackfillError(f"commit {commit!r} does not resolve in this checkout (git cat-file -e)")
    kinds = {judge_kind(s) for s in uses.values() if s}
    # J1 must make a decision (L*, V4); a non-J1 persona grader (sm.interp.persona_grader, docs/PERSONA_FIDELITY_RERUN.md)
    # is recorded with no policy, as runtime_identity does; fake or keyword judges have nothing to derive.
    if "J1" not in kinds or "fake" in kinds:
        raise BackfillError(f"judge callables {dict(uses)} include no sm.eval.judge J1 callable, or a fake one; nothing to derive")
    raw = OPS.git_show_bytes(commit, JUDGE_YAML, repo)
    if raw is None:
        raise BackfillError(f"{JUDGE_YAML} does not exist at {commit}")
    jy = _yaml(raw)
    j1 = dict(jy.get("j1") or {})
    if not j1.get("model") or not j1.get("revision"):
        raise BackfillError(f"{JUDGE_YAML} at {commit} has no j1 model/revision")
    pdir = str(jy.get("policy_dir") or DEFAULT_POLICY_DIR).rstrip("/")
    policy_sha: dict[str, str | None] = {}
    for p in sorted({CALLABLE_POLICY.get(s) for s in uses.values() if s} - {None}):
        data = OPS.git_show_bytes(commit, f"{pdir}/{p}.txt", repo)
        if data is None:
            raise BackfillError(f"{pdir}/{p}.txt does not exist at {commit}")
        policy_sha[p] = sha256_bytes(data)
    bj = dict((_yaml(OPS.git_show_bytes(commit, BASES_YAML, repo)).get("judges") or {}).get("J1") or {})
    bases_agree = (not bj) or (str(bj.get("hf")) == str(j1["model"]) and str(bj.get("revision")) == str(j1["revision"]))
    out = _assemble(
        str(j1["model"]), str(j1["revision"]), uses, policy_sha, pdir, j1, primary_use, "derived",
        {
            "derived_from": commit,
            "bases_yaml_agrees": bases_agree,
            "assumptions": [
                "the J1 server ran at the configs/judge.yaml pin (the job log has no 'J1 override from the environment' line)",
                "SM_JUDGE_POLICY_DIR was unset, so the policies came from configs/judge_policies at this commit",
            ],
        },
    )
    return out


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    try:
        from sm.common.io import dumps  # same serialiser as the original writer (NaN kept as NaN)

        text = dumps(obj) + "\n"
    except ImportError:  # pragma: no cover
        text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_COMPARE = ("model", "revision", "policy_sha256", "policies", "derived_from")


def backfill(
    path: str | Path,
    uses: Mapping[str, str | None],
    primary_use: str,
    keys: tuple[str, ...],
    extra_top: Mapping[str, str] | None = None,
    tool: str = "",
    force: bool = False,
    dry_run: bool = False,
    repo: str | Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Derive the J1 identity at the artifact's ``meta.git_commit`` and write it under each of ``keys``.

    ``extra_top`` maps further top-level keys to identity fields (causal: {"judge_revision": "revision"}).
    Returns (status, identity) with status "written", "unchanged" (already derived with the same values),
    "has_runtime" (the artifact already carries a runtime identity; nothing written) or "dry_run".
    Raises BackfillError when the artifact is a smoke output or the derivation is impossible, and when a
    different identity is already recorded (unless ``force``).
    """
    p = Path(path)
    raw = p.read_bytes()
    obj = json.loads(raw)
    meta = obj.get("meta") or {}
    if obj.get("smoke") or meta.get("smoke"):
        raise BackfillError(f"{p} is a --smoke output; it has no J1 identity to derive")
    commit = meta.get("git_commit")
    if not commit:
        raise BackfillError(f"{p} records no meta.git_commit")
    ident = derived_identity(str(commit), uses, primary_use, repo)
    ident["backfill"] = {"original_sha256": sha256_bytes(raw), "tool": tool, "at": _utc_now()}
    have = obj.get(keys[0])
    if isinstance(have, dict) and have:
        if have.get("source") == "runtime" and not force:
            return "has_runtime", have
        if all(have.get(k) == ident.get(k) for k in _COMPARE):
            return "unchanged", have
        if not force:
            raise BackfillError(f"{p} already records a different judge identity under {keys[0]!r}; pass --force to replace it")
    if dry_run:
        return "dry_run", ident
    new: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "meta":
            continue
        new[k] = v
    for k in keys:
        new[k] = ident
    for k, field in (extra_top or {}).items():
        new[k] = ident.get(field)
    new["meta"] = meta
    _atomic_write_json(p, new)
    return "written", ident


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
