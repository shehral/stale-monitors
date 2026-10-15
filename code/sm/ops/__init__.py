"""Operational provenance for the harness (no model, GPU or network access).

- ``sm.ops`` (this module): git state of the harness checkout: the commit, objects at a commit
  (``git rev-parse <commit>:<path>``, ``git show <commit>:<path>``), and the analysis-code state that
  ``sm.analysis.aggregate`` records next to ``git_commit`` (ACCEPTANCE.md OPEN-10 and Section 2.4).
- ``sm.ops.judge_identity``: the J1 identity {model, revision, policy_sha256} behind L*, persona filtering
  and V4, recorded at run time by ``sm.interp.base_card`` and ``sm.interp.causal`` and derived at a recorded
  commit by their ``--backfill-judge`` subcommands (OPEN-5).
- ``sm.ops.smoke_artifact``: the machine-readable smoke result ``results/smoke/<base>/<jobid>.json`` and the
  laptop half of G1, ``results/smoke/local/<git_sha>.json`` (OPEN-8).

Stdlib only; every git helper returns None outside a git checkout instead of raising.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

# code/sm/ops/__init__.py -> the repository root (the directory that holds code/, configs/, jobs/).
REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = "code/sm/analysis"
ACCEPT_PY = "code/sm/accept.py"
RULES_PY = "code/sm/analysis/rules.py"


def _run_git(args: Sequence[str], repo: str | Path | None = None, binary: bool = False) -> Any:
    cwd = Path(repo) if repo is not None else REPO_ROOT
    try:
        res = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout if binary else res.stdout.decode("utf-8", "replace").strip()


def git(args: Sequence[str], repo: str | Path | None = None) -> str | None:
    """stdout of ``git -C <repo> <args>`` stripped, or None on any failure."""
    return _run_git(args, repo)


def repo_root(repo: str | Path | None = None) -> Path | None:
    """Top level of the git work tree that holds ``repo`` (default: this checkout), or None."""
    top = git(["rev-parse", "--show-toplevel"], repo)
    return Path(top) if top else None


def git_commit(repo: str | Path | None = None) -> str | None:
    """HEAD commit (40 hex) of the checkout, or None."""
    return git(["rev-parse", "HEAD"], repo)


def git_object(rev: str, path: str, repo: str | Path | None = None) -> str | None:
    """Object id of ``<rev>:<path>`` (a tree for a directory, a blob for a file), or None when absent."""
    return git(["rev-parse", "--verify", "--quiet", f"{rev}:{path}"], repo)


def git_show_bytes(rev: str, path: str, repo: str | Path | None = None) -> bytes | None:
    """Raw bytes of ``<rev>:<path>`` (``git show``), or None when the commit or path does not exist."""
    out = _run_git(["show", f"{rev}:{path}"], repo, binary=True)
    return out if isinstance(out, bytes) else None


def paths_clean(paths: Sequence[str], repo: str | Path | None = None) -> bool | None:
    """True when ``git status --porcelain --untracked-files=all -- <paths>`` prints nothing; None outside git."""
    out = git(["status", "--porcelain", "--untracked-files=all", "--", *paths], repo)
    if out is None:
        return None
    return out == ""


def analysis_code_state(repo: str | Path | None = None) -> dict[str, Any]:
    """The analysis-code state recorded next to ``git_commit`` (ACCEPTANCE.md Section 2.4, OPEN-10, OPEN-11).

    analysis_tree   ``git rev-parse HEAD:code/sm/analysis`` (tree id)
    accept_blob     ``git rev-parse HEAD:code/sm/accept.py`` (blob id), None while accept.py is not committed
    rules_blob      ``git rev-parse HEAD:code/sm/analysis/rules.py`` (inside analysis_tree; listed for the freeze row)
    code_tree       ``git rev-parse HEAD:code`` (the tree C3 compares between a smoke row and a stage row)
    analysis_dirty  True when ``git status --porcelain --untracked-files=all -- code/sm/analysis code/sm/accept.py``
                    prints anything (only these paths, so a synced docs edit does not count)
    analysis_sha    ``<analysis_tree>:<accept_blob>`` when analysis_dirty is False and both ids exist, else None
    """
    tree = git_object("HEAD", ANALYSIS_DIR, repo)
    blob = git_object("HEAD", ACCEPT_PY, repo)
    clean = paths_clean([ANALYSIS_DIR, ACCEPT_PY], repo)
    return {
        "analysis_tree": tree,
        "accept_blob": blob,
        "rules_blob": git_object("HEAD", RULES_PY, repo),
        "code_tree": git_object("HEAD", "code", repo),
        "analysis_dirty": None if clean is None else (not clean),
        "analysis_sha": f"{tree}:{blob}" if (clean and tree and blob) else None,
    }
