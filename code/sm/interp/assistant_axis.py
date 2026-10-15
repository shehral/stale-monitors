"""EXPLORATORY line E15: the Assistant Axis of Lu, Gallagher, Michala, Fish and Lindsey (2026, arXiv 2601.10387) on a
base, and each derivative's displacement along it next to D2 (H3').

EXPLORATORY. Protocol: docs/ASSISTANT_AXIS_E7_PLAN.md (rules stated before any data; ledger row E15 in
docs/HYPOTHESIS_LEDGER.md). Decides no registered hypothesis; lives outside the frozen analysis module.

Recipe (paper sections and the official code, github.com/safety-research/assistant-axis @ a9896195, MIT):
  * roles and prompts: the official 275-role list, 5 system prompts per role, 240 extraction questions (Sec. 2.1.1,
    data/roles/instructions/*.json, data/extraction_questions.jsonl); the default Assistant = the 5 neutral prompts of
    default.json, the first being no system prompt (Sec. 2.1.2). Here a seeded sample: N_ROLES roles x 5 prompts x
    N_QUESTIONS questions, the default on the same questions (Sec. 3.1: "on the same extraction questions").
  * rollouts: one sample per (prompt, question), temperature 0.7, top-p 0.9, 512 new tokens, thinking off
    (pipeline/1_generate.py defaults; Sec. 8.1 "we disabled thinking mode"), "{model_name}" -> "Qwen"
    (assistant_axis/models.py get_config short name).
  * role-expression judge: the role's eval_prompt verbatim with the question and answer filled in, one user message
    (pipeline/3_judge.py); score = first integer in 0..3 of the reply (assistant_axis/judge.py parse_judge_score).
    Judge here: J1 (the project's gpt-oss-safeguard-20b server) at temperature 0, reasoning effort low (Lu: gpt-4.1-mini).
  * activations: mean of the residual stream over the response tokens at every block output (Sec. 2.1.2 "mean
    post-MLP residual stream activations at all response tokens"; 2_activations.py). Our index l = hidden_states[l] =
    output of block l-1, so Lu's zero-indexed layer j is our j + 1.
  * axis per layer: mean(default rollouts) - mean over roles of (mean of the role's score-3 rollouts), roles with at
    least MIN_SCORE3 score-3 rollouts (Sec. 3.1; 4_vectors.py, 5_axis.py); projections use the unit axis
    (axis.py project(normalize=True)).
  * capping threshold: the official capping config stores the ROLE direction (cos -1 with the axis) and caps it from
    above at its 25th percentile, which equals flooring the projection on the unit axis at the 75th percentile of the
    rollout projections, about the default Assistant's mean projection (checked on qwen-3-32b/capping_config.pt:
    32.50 vs 32.56 at layer 46). Sec. 5 eq. 1, 5.1.1, 5.1.2, 5.2.

Phases (CLI subcommands): plan, generate, judge, activations, axis, displace, h3prime.
Open outputs (counts, projections, hashes): results/exploratory/<base>/axis/, displace/, h3prime.json.
Texts (rollouts, judge replies): results/restricted/exploratory/<base>/ (directories 0700, files 0600) only.
Activation caches (no text): $SM_AXIS_CACHE (default <out>/<base>/_cache).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

LABEL = "EXPLORATORY E15 (docs/ASSISTANT_AXIS_E7_PLAN.md); not a registered hypothesis"
AXIS_VERSION = "e15-axis-v1-2026-10-10"
OFFICIAL_REPO = "https://github.com/safety-research/assistant-axis"
OFFICIAL_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"
PAPER = "Lu, Gallagher, Michala, Fish, Lindsey 2026, arXiv 2601.10387"

N_ROLES = 60
N_QUESTIONS = 20
N_PROMPTS = 5
ROLE_SEED = "20261010:E15:roles"
QUESTION_SEED = "20261010:E15:questions"
SPLIT_SEED = "20261010:E15:split-half"
GEN_SEED_PREFIX = "20261010:E15:gen:"
# Run 2 (plan section 12, 2026-10-11): same 60 roles, the run-1 questions plus 20 more (40 per role and prompt), fresh
# sampling seeds, its own split-half seed; outputs under <out>/<base>/run2/ so run 1 is kept.
RUNS = ("run1", "run2")
RUN2_EXTRA_QUESTIONS = 20
RUN2_QUESTION_SEED = "20261011:E15:run2:questions"
RUN2_SPLIT_SEED = "20261011:E15:run2:split-half"
RUN2_GEN_SEED_PREFIX = "20261011:E15:run2:gen:"
MODEL_SHORT_NAME = "Qwen"  # assistant_axis/models.py get_config: any model name containing "qwen"
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 512
GEN_MAX_MODEL_LEN = 2048
MIN_SCORE3 = 10  # Sec. 2.1.2: "at least ten responses"
FULL_SCORE = 3
SOMEWHAT_SCORE = 2
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024
JUDGE_REASONING_EFFORT = "low"
CAP_PERCENTILE_AXIS = 75.0  # = the 25th percentile along the role direction (official capping_config p0.25)
LU_CAP_REF = {"model": "Qwen/Qwen3-32B", "n_layers": 64, "first_block": 46, "n_blocks": 8, "experiment": "layers_46:54-p0.25"}
# validity bars (plan section 5)
V_PARSE_RATE_MIN = 0.90
V_ROLES_KEPT_MIN = 20
V_SPLIT_HALF_MIN = 0.90
V_OVERLAP_R_MAX = 0.90
EPS = 1e-12
# H3' units (plan section 4): the 15 rows of the flag file frozen when the protocol was written. A flag file regenerated
# later (for example with the E8 secure-code run judged) adds rows; those are reported as an addendum, outside the rule.
FROZEN_FLAG_SHA256 = "065f856f1908c7f5d3b49b37a496a2ee6822fde4ca52466e6f6159dd9e0b8b51"
H3P_UNITS = (
    "c1_dolly_r128", "c1_dolly_r8", "c1_pubmedqa_r128", "c1_pubmedqa_r8", "c2_humorous_r32", "c2_sycophantic_r32",
    "c2_terse_r32", "c3_risky_financial_advice_r32", "c3_trait_evil_r32", "c3_trait_sycophancy_r32", "c4_insecure_code_r128",
    "c4_insecure_code_r32", "c5_public_abliterated", "c6_base_vs_base", "c6_seed_replicate_dolly_r8",
)

DEFAULT_OFFICIAL_DIR = "third_party/assistant-axis"
DEFAULT_OUT = "results/exploratory"
DEFAULT_RESTRICTED = "results/restricted/exploratory"
DEFAULT_CARD_DIR = "results/base_cards"


# --------------------------------------------------------------------------------------------- small helpers


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_from(text: str, nbytes: int = 8) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:nbytes], "little")


def safe_name(base: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", base).strip("_") or "base"


def ensure_restricted_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` with every new component 0700 (the restricted tree's rule: directories 0700, files 0600)."""
    p = Path(path)
    missing = []
    q = p
    while not q.exists():
        missing.append(q)
        if q.parent == q:
            break
        q = q.parent
    for d in reversed(missing):
        d.mkdir(mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
    os.chmod(p, 0o700)
    return p


def write_restricted_text(path: str | os.PathLike, text: str) -> None:
    """Atomic write, file mode 0600, parent directories 0700."""
    p = Path(path)
    ensure_restricted_dir(p.parent)
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def append_restricted_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    p = Path(path)
    ensure_restricted_dir(p.parent)
    new = not p.exists()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    n = 0
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    if new:
        os.chmod(p, 0o600)
    return n


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a killed job: that rollout is regenerated
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _clean(o: Any) -> Any:
    """NaN/inf -> None recursively (strict JSON)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return _clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def write_open_json(path: str | os.PathLike, obj: dict, meta: dict | None = None) -> None:
    """Atomic JSON write with a meta block (sm.common.io.write_json when importable)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = _clean(obj)
    try:
        from sm.common.io import write_json

        write_json(p, body, extra_meta=_clean(meta or {}))
        return
    except ImportError:  # pragma: no cover
        pass
    out = dict(body)
    out["meta"] = _clean(meta or {})
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=_json_default) + "\n")
    os.replace(tmp, p)


def _file_hashes() -> dict[str, str]:
    here = Path(__file__).resolve()
    files = {}
    for name in ("assistant_axis.py", "mediation.py"):
        f = here.with_name(name)
        if f.exists():
            files[f"code/sm/interp/{name}"] = sha256_file(f)
    return files


# Hashed when this module is imported, so a long job records the code it actually loaded even if the files on disk
# are replaced while it runs (the side tree is updated by rsync).
_FILES_AT_IMPORT = _file_hashes()


def code_provenance() -> dict[str, Any]:
    """The laptop commit the files came from (SM_CODE_COMMIT, set at submit time) and the files' hashes at import (and
    on disk now, when they differ); the side tree is not a clean checkout, so its git HEAD alone would not identify
    this code."""
    here = Path(__file__).resolve()
    files = dict(_FILES_AT_IMPORT)
    now = _file_hashes()
    head = None
    try:
        head = subprocess.run(["git", "-C", str(here.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        head = None
    return {"code_commit_env": os.environ.get("SM_CODE_COMMIT"), "checkout_head": head, "file_sha256": files,
            "file_sha256_on_disk_now": now if now != files else None, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


# --------------------------------------------------------------------------------------------- official data, plan


def load_official(official_dir: str | os.PathLike) -> dict[str, Any]:
    """Role instructions, eval prompts, default prompts and extraction questions from the official clone."""
    root = Path(official_dir)
    inst = root / "data" / "roles" / "instructions"
    qfile = root / "data" / "extraction_questions.jsonl"
    if not inst.is_dir() or not qfile.is_file():
        raise FileNotFoundError(f"official assistant-axis data not found under {root} (clone {OFFICIAL_REPO} @ {OFFICIAL_COMMIT[:8]})")
    roles: dict[str, dict[str, Any]] = {}
    for p in sorted(inst.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        roles[p.stem] = {
            "instruction": [str((x or {}).get("pos", "")) for x in d.get("instruction", [])],
            "eval_prompt": d.get("eval_prompt"),
            "file_sha256": sha256_file(p),
        }
    if "default" not in roles:
        raise ValueError(f"{inst}/default.json missing")
    default = roles.pop("default")
    questions: dict[int, str] = {}
    for line in qfile.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            questions[int(rec["id"])] = str(rec["question"])
    commit = None
    if (root / ".git").exists():
        try:
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    commit = commit or _read_pin_file(root)
    return {"roles": roles, "default": default, "questions": questions, "questions_sha256": sha256_file(qfile),
            "commit": commit, "dir": str(root)}


def _read_pin_file(root: Path) -> str | None:
    f = root / "COMMIT"
    return f.read_text().strip() if f.is_file() else None


def format_instruction(instruction: str, short_name: str = MODEL_SHORT_NAME) -> str:
    """generation.py RoleResponseGenerator.format_instruction: replace the {model_name} placeholder."""
    return instruction.replace("{model_name}", short_name)


def select_roles(names: Sequence[str], k: int = N_ROLES, seed_text: str = ROLE_SEED) -> list[str]:
    """A seeded sample of ``k`` role names from the sorted official list (plan section 2)."""
    names = sorted(names)
    if k >= len(names):
        return list(names)
    rng = np.random.default_rng(seed_from(seed_text))
    idx = rng.choice(len(names), size=int(k), replace=False)
    return sorted(names[int(i)] for i in idx)


def select_questions(ids: Sequence[int], q: int = N_QUESTIONS, seed_text: str = QUESTION_SEED) -> list[int]:
    ids = sorted(int(i) for i in ids)
    if q >= len(ids):
        return list(ids)
    rng = np.random.default_rng(seed_from(seed_text))
    idx = rng.choice(len(ids), size=int(q), replace=False)
    return sorted(ids[int(i)] for i in idx)


def split_halves(question_ids: Sequence[int], seed_text: str = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """Two disjoint halves of the question ids (seeded permutation), for the split-half reliability of the axis."""
    ids = sorted(int(i) for i in question_ids)
    rng = np.random.default_rng(seed_from(seed_text))
    perm = [ids[int(i)] for i in rng.permutation(len(ids))]
    h = len(ids) // 2
    return sorted(perm[:h]), sorted(perm[h:])


def run_question_ids(ids: Sequence[int], run: str = "run1", n_questions: int = N_QUESTIONS) -> list[int]:
    """Run 1: the seeded 20. Run 2: the same 20 plus RUN2_EXTRA_QUESTIONS drawn from the other ids with the run-2 seed."""
    first = select_questions(ids, n_questions, QUESTION_SEED)
    if run == "run1":
        return first
    if run != "run2":
        raise ValueError(f"unknown run {run!r} (expected one of {RUNS})")
    rest = sorted(set(int(i) for i in ids) - set(first))
    return sorted(first + select_questions(rest, RUN2_EXTRA_QUESTIONS, RUN2_QUESTION_SEED))


def run_split_seed(run: str = "run1") -> str:
    return RUN2_SPLIT_SEED if run == "run2" else SPLIT_SEED


def run_gen_prefix(run: str = "run1") -> str:
    return RUN2_GEN_SEED_PREFIX if run == "run2" else GEN_SEED_PREFIX


def messages_for(system_prompt: str, question: str) -> list[dict[str, str]]:
    """generation.py format_conversation for a template with system-prompt support: an empty instruction means no
    system message."""
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def build_plan(official: dict[str, Any], n_roles: int = N_ROLES, n_questions: int = N_QUESTIONS, run: str = "run1") -> dict[str, Any]:
    """The fixed rollout plan: every (role, prompt, question) of the sampled roles and questions plus the default.
    Run 2 keeps the roles, adds questions (run_question_ids) and draws fresh sampling seeds (run_gen_prefix)."""
    roles = select_roles(list(official["roles"]), n_roles)
    qids = run_question_ids(list(official["questions"]), run, n_questions)
    gen_prefix = run_gen_prefix(run)
    rollouts: list[dict[str, Any]] = []
    role_hashes: dict[str, list[str]] = {}

    def add(kind: str, role: str, instructions: list[str]) -> None:
        hashes = []
        for pi, raw in enumerate(instructions[:N_PROMPTS]):
            sp = format_instruction(raw)
            hashes.append(sha256_text(sp))
            for qid in qids:
                key = f"{role}|p{pi}|q{qid}"
                rollouts.append({"key": key, "kind": kind, "role": role, "prompt_index": pi, "question_id": qid,
                                 "system_prompt": sp, "question": official["questions"][qid],
                                 "seed": seed_from(gen_prefix + key, 4)})
        role_hashes[role] = hashes

    add("default", "default", official["default"]["instruction"])
    for r in roles:
        inst = official["roles"][r]["instruction"]
        if len(inst) < N_PROMPTS:
            raise ValueError(f"role {r} has {len(inst)} instructions, expected {N_PROMPTS}")
        add("role", r, inst)
    plan_core = [{k: v for k, v in x.items() if k in ("key", "kind", "role", "prompt_index", "question_id", "seed")}
                 | {"system_prompt_sha256": sha256_text(x["system_prompt"]), "question_sha256": sha256_text(x["question"])}
                 for x in rollouts]
    plan_sha = sha256_text(json.dumps(plan_core, sort_keys=True))
    return {
        "version": AXIS_VERSION,
        "official_repo": OFFICIAL_REPO,
        "official_commit": official.get("commit"),
        "official_commit_expected": OFFICIAL_COMMIT,
        "roles": roles,
        "n_roles": len(roles),
        "question_ids": qids,
        "question_sha256": {str(q): sha256_text(official["questions"][q]) for q in qids},
        "questions_file_sha256": official.get("questions_sha256"),
        "role_prompt_sha256": role_hashes,
        "role_file_sha256": {r: official["roles"][r]["file_sha256"] for r in roles},
        "default_prompts": [format_instruction(x) for x in official["default"]["instruction"][:N_PROMPTS]],
        "n_rollouts": len(rollouts),
        "plan_sha256": plan_sha,
        "rollouts": rollouts,
        "settings": {"temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P, "max_new_tokens": GEN_MAX_NEW_TOKENS,
                     "max_model_len": GEN_MAX_MODEL_LEN, "thinking": False, "short_name": MODEL_SHORT_NAME,
                     "role_seed": ROLE_SEED, "question_seed": QUESTION_SEED},
    } | ({} if run == "run1" else {"run": run, "run_settings": {
        "extra_questions": RUN2_EXTRA_QUESTIONS, "extra_question_seed": RUN2_QUESTION_SEED, "gen_seed_prefix": gen_prefix,
        "split_seed": run_split_seed(run)}})


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """The plan without any prompt text (hashes only)."""
    out = {k: v for k, v in plan.items() if k != "rollouts"}
    out["rollouts"] = [{k: v for k, v in x.items() if k not in ("system_prompt", "question")} for x in plan["rollouts"]]
    return out


# --------------------------------------------------------------------------------------------- judge (role expression)


def parse_judge_score(text: str | None) -> int | None:
    """assistant_axis/judge.py parse_judge_score: the first standalone integer of the reply if it is in 0..3."""
    if not text:
        return None
    numbers = re.findall(r"\b(\d+)\b", text.strip())
    if not numbers:
        return None
    try:
        score = int(numbers[0])
    except ValueError:
        return None
    return score if 0 <= score <= 3 else None


def judge_prompt(eval_prompt: str, question: str, answer: str) -> str:
    """pipeline/3_judge.py: eval_prompt_template.format(question=..., answer=...)."""
    return eval_prompt.format(question=question, answer=answer)


# --------------------------------------------------------------------------------------------- pure axis maths


def contrast_axis(default_mean: np.ndarray, role_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """axis = mean(default) - mean over roles of the role vectors (Sec. 3.1; 5_axis.py)."""
    if len(role_vectors) == 0:
        raise ValueError("no role vectors")
    return np.asarray(default_mean, np.float64) - np.mean(np.stack([np.asarray(v, np.float64) for v in role_vectors]), axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < EPS:
        raise ValueError("zero or non-finite vector")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def group_rows(keys: Sequence[str], plan_rows: dict[str, dict], scores: dict[str, int | None], valid: np.ndarray,
               score: int | None) -> dict[str, list[int]]:
    """Row indices per role: every valid role rollout (score None) or those with the given judge score."""
    out: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        row = plan_rows[k]
        if row["kind"] != "role" or not valid[i]:
            continue
        if score is not None and scores.get(k) != score:
            continue
        out.setdefault(row["role"], []).append(i)
    return out


def role_vectors(acts: np.ndarray, groups: dict[str, list[int]], min_count: int) -> dict[str, np.ndarray]:
    """Mean activation per role over its rows, for roles with at least ``min_count`` rows (4_vectors.py)."""
    return {r: np.asarray(acts[idx], np.float64).mean(axis=0) for r, idx in sorted(groups.items()) if len(idx) >= min_count}


def pc1_diagnostics(vectors: np.ndarray, default_vec: np.ndarray, axis_vec: np.ndarray) -> dict[str, Any]:
    """Sec. 2.1.3 / 2.3.1 / G.1: PCA of mean-centred role vectors; |cos(axis, PC1)| and the default activation's
    relative position on PC1 (0 and 1 = the role vectors with the smallest and largest projections; the paper's
    "minimum distance to either extreme" is min(rel, 1 - rel))."""
    X = np.asarray(vectors, np.float64)
    if X.shape[0] < 3:
        return {"n_vectors": int(X.shape[0]), "pc1_cos_axis": None, "default_rel_pos": None}
    mu = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    pc1 = vt[0]
    if np.dot(pc1, axis_vec) < 0:
        pc1 = -pc1
    proj = (X - mu) @ pc1
    d = float((np.asarray(default_vec, np.float64) - mu) @ pc1)
    lo, hi = float(proj.min()), float(proj.max())
    rel = (d - lo) / (hi - lo) if hi - lo > EPS else float("nan")
    var = s ** 2
    return {
        "n_vectors": int(X.shape[0]),
        "pc1_cos_axis": cosine(pc1, axis_vec),
        "pc1_var_fraction": float(var[0] / var.sum()) if var.sum() > 0 else None,
        "n_pc_70pct": int(np.searchsorted(np.cumsum(var) / var.sum(), 0.70) + 1) if var.sum() > 0 else None,
        "default_rel_pos": rel,
        "default_min_dist_to_extreme": float(min(rel, 1 - rel)) if math.isfinite(rel) else None,
        "pc1": pc1,
    }


def capping_layers(n_layers: int) -> list[int]:
    """Lu et al.'s best Qwen 3 32B setting (blocks 46..53 of 64: 12.5 percent of the depth starting at 46/64), scaled
    to ``n_layers`` blocks, as hidden_states indices (block b output = index b + 1). 32 blocks -> indices 24..27."""
    width = max(1, int(round(LU_CAP_REF["n_blocks"] / LU_CAP_REF["n_layers"] * n_layers)))
    first = int(round(LU_CAP_REF["first_block"] / LU_CAP_REF["n_layers"] * n_layers))
    return [b + 1 for b in range(first, min(first + width, n_layers))]


def lu_middle_layer(n_layers: int) -> int:
    """assistant_axis/models.py get_config: target_layer = n_layers // 2 (zero-indexed block) -> our index + 1."""
    return n_layers // 2 + 1


def cap_threshold(projections: np.ndarray, percentile: float = CAP_PERCENTILE_AXIS) -> float:
    """Floor value on the unit axis: the 75th percentile of the rollout projections (= the 25th percentile along the
    role direction, the official p0.25 cap)."""
    return float(np.percentile(np.asarray(projections, np.float64), percentile))


def displacement(base_proj: np.ndarray, deriv_proj: np.ndarray) -> dict[str, float]:
    """Mean projection shift (derivative minus base) and the shift in units of the base's spread (population sd, as
    drift.persona_shift's proj_shift_z)."""
    b = np.asarray(base_proj, np.float64).reshape(-1)
    m = np.asarray(deriv_proj, np.float64).reshape(-1)
    sd = float(b.std())
    shift = float(m.mean() - b.mean())
    return {"base_mean": float(b.mean()), "deriv_mean": float(m.mean()), "base_sd": sd, "shift": shift,
            "z": shift / sd if sd > EPS else float("nan"), "n": int(b.size)}


# --------------------------------------------------------------------------------------------- paths


class Paths:
    """Output locations. Run 1 writes at the base level (results/exploratory/<base>/...); any later run writes under its
    own subdirectory (results/exploratory/<base>/<run>/, results/restricted/exploratory/<base>/<run>/, <cache>/<run>), so
    earlier runs are never overwritten."""

    def __init__(self, base: str, out: str = DEFAULT_OUT, restricted: str = DEFAULT_RESTRICTED, cache: str | None = None,
                 run: str = "run1"):
        if run not in RUNS:
            raise ValueError(f"unknown run {run!r} (expected one of {RUNS})")
        self.base = base
        self.run = run
        self.safe = safe_name(base)
        sub = (self.safe,) if run == "run1" else (self.safe, run)
        self.open = Path(out).joinpath(*sub)
        self.restricted = Path(restricted).joinpath(*sub)
        cache_root = Path(cache or os.environ.get("SM_AXIS_CACHE") or (self.open / "_cache"))
        self.cache = cache_root if run == "run1" or not (cache or os.environ.get("SM_AXIS_CACHE")) else cache_root / run

    @property
    def axis_dir(self) -> Path:
        return self.open / "axis"

    @property
    def plan_json(self) -> Path:
        return self.axis_dir / "plan.json"

    @property
    def rollouts(self) -> Path:
        return self.restricted / "axis_rollouts" / "rollouts.jsonl"

    @property
    def judge_raw(self) -> Path:
        return self.restricted / "axis_judge" / "raw.jsonl"

    @property
    def scores_json(self) -> Path:
        return self.axis_dir / "judge_scores.json"

    @property
    def axis_npz(self) -> Path:
        return self.axis_dir / "axis.npz"

    @property
    def axis_json(self) -> Path:
        return self.axis_dir / "axis.json"

    @property
    def displace_dir(self) -> Path:
        return self.open / "displace"

    @property
    def h3prime_json(self) -> Path:
        return self.open / "h3prime.json"


# --------------------------------------------------------------------------------------------- base card


def load_card(card_path: str | os.PathLike):
    """The base card through sm.interp.drift.load_base_card (r, L*, p*, eoi ids, persona vectors)."""
    from sm.interp.drift import load_base_card

    return load_base_card(str(card_path))


# --------------------------------------------------------------------------------------------- phase: plan


def phase_plan(args: argparse.Namespace) -> int:
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    official = load_official(args.official_dir)
    plan = build_plan(official, args.n_roles, args.n_questions, run=args.run)
    pub = public_plan(plan)
    write_open_json(P.plan_json, {"label": LABEL, **pub}, {"stage": "e15_plan", "provenance": code_provenance()})
    print(json.dumps({"n_rollouts": plan["n_rollouts"], "plan_sha256": plan["plan_sha256"], "roles": plan["roles"],
                      "question_ids": plan["question_ids"], "official_commit": plan["official_commit"]}, indent=1))
    return 0


# --------------------------------------------------------------------------------------------- phase: generate


def _hf_sample(model, tok, rendered: list[str], seeds: list[int], max_new_tokens: int, batch_size: int) -> list[tuple[str, bool]]:
    """HF fallback (tests, or no vLLM): sampled generation at the plan's temperature and top-p, one sample per prompt."""
    import torch

    from sm.common.models import generate_detailed

    out: list[tuple[str, bool]] = []
    for s in range(0, len(rendered), batch_size):
        torch.manual_seed(int(seeds[s]))
        rows = generate_detailed(model, tok, rendered[s:s + batch_size], max_new_tokens=max_new_tokens, batch_size=batch_size,
                                 greedy=False, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, top_k=0)
        out.extend((r["completion"], bool(r["truncated"])) for r in rows)
    return out


def generate_rollouts(plan: dict[str, Any], out_path: Path, sampler: Callable[[list[dict]], list[tuple[str, bool]]],
                      chunk: int = 1000) -> dict[str, Any]:
    """Generate every planned rollout not yet on disk (restricted JSONL, appended per chunk; resumable)."""
    done = {r["key"] for r in read_jsonl(out_path) if "text" in r}
    todo = [x for x in plan["rollouts"] if x["key"] not in done]
    t0 = time.time()
    n_new = 0
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        res = sampler(part)
        if len(res) != len(part):
            raise RuntimeError(f"sampler returned {len(res)} outputs for {len(part)} prompts")
        rows = [{"key": x["key"], "text": text, "truncated": bool(trunc), "sha256": sha256_text(text)} for x, (text, trunc) in zip(part, res)]
        n_new += append_restricted_jsonl(out_path, rows)
        print(f"[axis] generated {len(done) + n_new}/{plan['n_rollouts']} ({time.time() - t0:.0f}s)", flush=True)
    return {"n_planned": plan["n_rollouts"], "n_existing": len(done), "n_new": n_new, "seconds": time.time() - t0}


def _vllm_sampler(base: str, revision: str | None, gpu_util: float, max_model_len: int):
    from transformers import AutoTokenizer

    from sm.common.registry import resolve_base
    from sm.eval.battery import TEXT_ONLY_MM_LIMIT, VLLMGenerator, is_multimodal_checkpoint, resolve_base_info

    info = resolve_base(base)
    hf = info["hf"]
    tok = AutoTokenizer.from_pretrained(hf, **({"revision": revision} if revision else {}))
    binfo = resolve_base_info(base, None)
    tok.sm_thinking_switch = binfo.get("thinking_switch") if binfo.get("key") else "auto"
    tok.sm_family = binfo.get("family")
    multimodal = is_multimodal_checkpoint(hf, revision)
    gen = VLLMGenerator(hf, tok, GEN_MAX_NEW_TOKENS, 0, revision=revision, max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_util, dtype="bfloat16",
                        limit_mm_per_prompt=dict(TEXT_ONLY_MM_LIMIT) if multimodal else None)

    def sampler(rows: list[dict]) -> list[tuple[str, bool]]:
        reqs = [(messages_for(x["system_prompt"], x["question"]), 1, int(x["seed"])) for x in rows]
        groups = gen.sample_groups(reqs, GEN_TEMPERATURE, GEN_TOP_P)
        return [g[0] for g in groups]

    meta = {"backend": "vllm", "hf": hf, "revision": revision, "vllm_kwargs": gen.llm_kwargs, "multimodal_checkpoint": multimodal,
            "render": dict(gen.render_meta) if getattr(gen, "render_meta", None) else None, "prompt_input": "token_ids (encode_text)"}
    return sampler, meta


def phase_generate(args: argparse.Namespace) -> int:
    """vLLM rollouts (the bank-generation path); if the vLLM engine fails, HF sampling at the same temperature and top-p
    continues from the rollouts already on disk, and generate.json records the fallback and its reason."""
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    official = load_official(args.official_dir)
    plan = build_plan(official, args.n_roles, args.n_questions, run=args.run)
    from sm.interp import loading as LD

    pin = LD.resolve_pinned(args.base, {})

    def hf_sampler():
        model, tok, lmeta = LD.load_pinned(args.base, pin, dtype=args.dtype, device=args.device)
        from sm.common.models import render_chat

        def sampler(rows: list[dict]) -> list[tuple[str, bool]]:
            rendered = [render_chat(tok, messages_for(x["system_prompt"], x["question"]), thinking=False) for x in rows]
            return _hf_sample(model, tok, rendered, [int(x["seed"]) for x in rows], args.max_new_tokens, args.batch_size)

        return sampler, {"backend": "hf", "batch_size": args.batch_size, "seeding": "torch.manual_seed(first rollout seed of each batch)",
                         "loaded": {k: v for k, v in lmeta.items() if isinstance(v, (str, int, float, bool, type(None)))}}

    fallback = None
    if args.backend == "vllm":
        try:
            sampler, gmeta = _vllm_sampler(args.base, pin.get("revision"), args.gpu_util, GEN_MAX_MODEL_LEN)
            stats = generate_rollouts(plan, P.rollouts, sampler, chunk=args.chunk)
        except Exception as exc:  # noqa: BLE001  (engine build or run failure: fall back, record why)
            fallback = f"{type(exc).__name__}: {exc}"[:1000]
            print(f"[axis] WARNING: vLLM generation failed ({fallback}); falling back to HF sampling", file=sys.stderr, flush=True)
            sampler = None
            release_gpu()
            sampler, gmeta = hf_sampler()
            stats = generate_rollouts(plan, P.rollouts, sampler, chunk=args.chunk)
    else:
        sampler, gmeta = hf_sampler()
        stats = generate_rollouts(plan, P.rollouts, sampler, chunk=args.chunk)
    write_open_json(P.axis_dir / "generate.json", {"label": LABEL, "plan_sha256": plan["plan_sha256"], **stats, "generator": gmeta,
                                                   "vllm_fallback_reason": fallback, "pin": pin},
                    {"stage": "e15_generate", "provenance": code_provenance()})
    write_open_json(P.plan_json, {"label": LABEL, **public_plan(plan)}, {"stage": "e15_plan", "provenance": code_provenance()})
    return 0


# --------------------------------------------------------------------------------------------- phase: judge


def judge_rollouts(plan: dict[str, Any], official: dict[str, Any], texts: dict[str, str], ask: Callable[[str], str],
                   raw_path: Path, existing: dict[str, Any], workers: int = 32) -> dict[str, Any]:
    """Score every role rollout not yet scored. ``ask(prompt) -> reply``. Returns {key: {score, raw_sha256}}."""
    rows = [x for x in plan["rollouts"] if x["kind"] == "role" and x["key"] in texts and x["key"] not in existing]
    out = dict(existing)

    def work(x: dict) -> tuple[str, str]:
        prompt = judge_prompt(official["roles"][x["role"]]["eval_prompt"], x["question"], texts[x["key"]])
        try:
            return x["key"], ask(prompt)
        except Exception as exc:  # noqa: BLE001  (a failed request is an unscored rollout, recorded)
            return x["key"], f"__error__ {type(exc).__name__}: {exc}"

    t0 = time.time()
    for s in range(0, len(rows), 500):
        part = rows[s:s + 500]
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            res = list(pool.map(work, part))
        append_restricted_jsonl(raw_path, [{"key": k, "raw": r} for k, r in res])
        for k, r in res:
            out[k] = {"score": None if r.startswith("__error__") else parse_judge_score(r), "raw_sha256": sha256_text(r),
                      "error": r.startswith("__error__")}
        print(f"[axis] judged {len(out)}/{sum(1 for x in plan['rollouts'] if x['kind'] == 'role')} ({time.time() - t0:.0f}s)", flush=True)
    return out


def _j1_ask(max_tokens: int = JUDGE_MAX_TOKENS):
    from sm.eval.judge import OpenAICompatClient

    port = os.environ.get("J1_PORT", "8000")
    client = OpenAICompatClient(base_url=os.environ.get("SM_J1_URL", f"http://127.0.0.1:{port}"),
                                model=os.environ.get("SM_J1_SERVED_NAME", "J1"),
                                extra_body={"reasoning_effort": JUDGE_REASONING_EFFORT})

    def ask(prompt: str) -> str:
        return client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=JUDGE_TEMPERATURE)

    ident = {"judge": "J1", "model": os.environ.get("SM_J1_MODEL") or os.environ.get("J1_MODEL"),
             "revision": os.environ.get("SM_J1_REVISION", "unpinned"), "temperature": JUDGE_TEMPERATURE,
             "max_tokens": max_tokens, "reasoning_effort": JUDGE_REASONING_EFFORT,
             "prompt": "official role eval_prompt verbatim (question and answer filled), single user message"}
    return ask, ident


def phase_judge(args: argparse.Namespace) -> int:
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    official = load_official(args.official_dir)
    plan = build_plan(official, args.n_roles, args.n_questions, run=args.run)
    texts = {r["key"]: r["text"] for r in read_jsonl(P.rollouts) if "text" in r}
    existing = {}
    if P.scores_json.exists() and not args.force:
        existing = {k: v for k, v in json.loads(P.scores_json.read_text()).get("scores", {}).items() if not v.get("error")}
    if args.fake_judge:
        def ask(prompt: str) -> str:  # tests only: deterministic pseudo-scores
            return str(int(sha256_text(prompt)[:2], 16) % 4)
        ident = {"judge": "fake (tests)"}
    else:
        ask, ident = _j1_ask()
    scores = judge_rollouts(plan, official, texts, ask, P.judge_raw, existing, workers=args.workers)
    n_role = sum(1 for x in plan["rollouts"] if x["kind"] == "role")
    vals = [v["score"] for v in scores.values()]
    counts = {str(s): sum(1 for v in vals if v == s) for s in range(4)}
    counts["unparsed"] = sum(1 for v in vals if v is None)
    write_open_json(P.scores_json, {"label": LABEL, "plan_sha256": plan["plan_sha256"], "judge": ident, "n_role_rollouts": n_role,
                                    "n_scored": len(scores), "counts": counts,
                                    "parse_rate": (len(vals) - counts["unparsed"]) / n_role if n_role else None,
                                    "scores": scores},
                    {"stage": "e15_judge", "provenance": code_provenance()})
    print(json.dumps({"counts": counts, "n_scored": len(scores)}))
    return 0


# --------------------------------------------------------------------------------------------- phase: activations


def _memmap(path: Path, shape: tuple[int, ...], mode: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode=mode, dtype=np.float32, shape=shape) if mode == "w+" else np.load(path, mmap_mode=mode)


def capture_rollout_acts(model, tok, plan: dict[str, Any], texts: dict[str, str], card, cache: Path, layers: list[int],
                         batch_size: int = 16, chunk: int = 256) -> dict[str, Any]:
    """Response-mean (Lu's readout) and card-position (p*) activations of every rollout at ``layers``, into float32
    memmaps (N, len(layers), H) under ``cache``; resumable per chunk (done mask). Empty responses are left NaN and
    marked invalid (no response tokens to average)."""
    from sm.common.hooks import capture_position, capture_residual
    from sm.common.models import hidden_size, render_chat

    keys = [x["key"] for x in plan["rollouts"]]
    N, L, H = len(keys), len(layers), hidden_size(model)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "acts_meta.json"
    want = {"plan_sha256": plan["plan_sha256"], "layers": layers, "position": int(card.position), "hidden": H,
            "texts_sha256": sha256_text(json.dumps([texts.get(k, "") for k in keys]))}
    fresh = True
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        fresh = {k: old.get(k) for k in want} != want
    mode = "w+" if fresh else "r+"
    resp = _memmap(cache / "resp_mean.npy", (N, L, H), mode)
    pst = _memmap(cache / "pstar.npy", (N, L, H), mode)
    done_path = cache / "done.npy"
    done = np.zeros(N, dtype=bool) if fresh or not done_path.exists() else np.load(done_path)
    valid = np.array([bool(texts.get(k, "").strip()) for k in keys])
    if fresh:
        meta_path.write_text(json.dumps(want))
    t0 = time.time()
    for s in range(0, N, chunk):
        idx = list(range(s, min(N, s + chunk)))
        if done[idx].all():
            continue
        rows = [plan["rollouts"][i] for i in idx]
        rendered = [render_chat(tok, messages_for(x["system_prompt"], x["question"]), thinking=False) for x in rows]
        ok = [j for j, i in enumerate(idx) if valid[i]]
        if ok:
            r_act = capture_residual(model, tok, [rendered[j] for j in ok], layers=layers, position="resp_mean",
                                     batch_size=batch_size, completions=[texts[keys[idx[j]]] for j in ok])
            p_act = capture_position(model, tok, [rendered[j] for j in ok], position=int(card.position), layers=layers,
                                     batch_size=batch_size, eoi_ids=card.eoi_ids)
            for li, layer in enumerate(layers):
                resp[[idx[j] for j in ok], li, :] = r_act[layer]
                pst[[idx[j] for j in ok], li, :] = p_act[layer]
        bad = [i for i in idx if not valid[i]]
        if bad:
            resp[bad] = np.nan
            pst[bad] = np.nan
        resp.flush()
        pst.flush()
        done[idx] = True
        np.save(done_path, done)
        print(f"[axis] activations {int(done.sum())}/{N} ({time.time() - t0:.0f}s)", flush=True)
    return {"n": N, "n_valid": int(valid.sum()), "n_empty": int((~valid).sum()), "layers": layers, "hidden": H,
            "seconds": time.time() - t0, "cache": str(cache)}


def phase_activations(args: argparse.Namespace) -> int:
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    official = load_official(args.official_dir)
    plan = build_plan(official, args.n_roles, args.n_questions, run=args.run)
    texts = {r["key"]: r["text"] for r in read_jsonl(P.rollouts) if "text" in r}
    missing = [x["key"] for x in plan["rollouts"] if x["key"] not in texts]
    if missing:
        print(f"error: {len(missing)} planned rollouts have no text yet (run generate)", file=sys.stderr)
        return 5
    card = load_card(args.card)
    from sm.interp import loading as LD

    pin = LD.resolve_pinned(args.base, {})
    model, tok, lmeta = LD.load_pinned(args.base, pin, dtype=args.dtype, device=args.device)
    layers = list(range(1, int(card.n_layers) + 1))
    stats = capture_rollout_acts(model, tok, plan, texts, card, P.cache, layers, batch_size=args.batch_size)
    write_open_json(P.axis_dir / "activations.json", {"label": LABEL, "plan_sha256": plan["plan_sha256"], **stats, "pin": pin,
                                                      "readouts": {"resp_mean": "mean over response tokens (sm.common.hooks.capture_residual resp_mean)",
                                                                   "pstar": f"card position {int(card.position)} (capture_position, eoi ids checked)"}},
                    {"stage": "e15_activations", "provenance": code_provenance()})
    return 0


# --------------------------------------------------------------------------------------------- phase: axis


def compute_axis_bundle(resp: np.ndarray, pstar: np.ndarray, keys: list[str], plan_rows: dict[str, dict], scores: dict[str, Any],
                        layers: list[int], question_halves: tuple[list[int], list[int]], card=None,
                        min_score3: int = MIN_SCORE3) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every axis variant at every layer plus the diagnostics of plan section 5. Arrays are indexed by hidden_states
    index (row 0, the embeddings, is zeros)."""
    N, L, H = resp.shape
    valid = ~np.isnan(np.asarray(resp[:, 0, 0], np.float64))
    sc = {k: (v.get("score") if isinstance(v, dict) else v) for k, v in scores.items()}
    n_role = sum(1 for k in keys if plan_rows[k]["kind"] == "role")
    n_scored = sum(1 for k in keys if plan_rows[k]["kind"] == "role" and sc.get(k) is not None)
    full = group_rows(keys, plan_rows, sc, valid, FULL_SCORE)
    some = group_rows(keys, plan_rows, sc, valid, SOMEWHAT_SCORE)
    allr = group_rows(keys, plan_rows, sc, valid, None)
    dflt = [i for i, k in enumerate(keys) if plan_rows[k]["kind"] == "default" and valid[i]]
    kept = sorted(r for r, idx in full.items() if len(idx) >= min_score3)
    parse_rate = n_scored / n_role if n_role else 0.0
    v1 = bool(parse_rate >= V_PARSE_RATE_MIN and len(kept) >= V_ROLES_KEPT_MIN)
    halves = [set(question_halves[0]), set(question_halves[1])]
    shape = (max(layers) + 1, H)
    out = {name: np.zeros(shape, np.float32) for name in
           ("axis_resp", "axis_pstar", "axis_resp_unfiltered", "axis_pstar_unfiltered", "axis_resp_half_a", "axis_resp_half_b",
            "pc1_resp", "default_resp", "role_mean_resp", "default_pstar")}
    per_layer: dict[str, Any] = {}
    for li, layer in enumerate(layers):
        R = np.asarray(resp[:, li, :], np.float64)
        Pa = np.asarray(pstar[:, li, :], np.float64)
        d_r = R[dflt].mean(axis=0)
        d_p = Pa[dflt].mean(axis=0)
        rv = role_vectors(R, {r: full[r] for r in kept}, 1)
        pv = role_vectors(Pa, {r: full[r] for r in kept}, 1)
        uv = role_vectors(R, allr, 1)
        upv = role_vectors(Pa, allr, 1)
        a_f = contrast_axis(d_r, list(rv.values())) if rv else np.full(H, np.nan)
        a_pf = contrast_axis(d_p, list(pv.values())) if pv else np.full(H, np.nan)
        a_u = contrast_axis(d_r, list(uv.values()))
        a_pu = contrast_axis(d_p, list(upv.values()))
        # split half over questions: default and role vectors from each half's rollouts
        ha = []
        for h in halves:
            dh = [i for i in dflt if plan_rows[keys[i]]["question_id"] in h]
            src = {r: [i for i in (full[r] if v1 else allr[r]) if plan_rows[keys[i]]["question_id"] in h]
                   for r in (kept if v1 else sorted(allr))}
            vh = [R[idx].mean(axis=0) for idx in src.values() if idx]
            ha.append(contrast_axis(R[dh].mean(axis=0), vh) if dh and vh else np.full(H, np.nan))
        primary = a_f if v1 else a_u
        # PCA over the fully and somewhat role vectors (Sec. 2.1.2-2.1.3); every role's unfiltered vector when V1 fails
        pca_vecs = (list(rv.values()) + list(role_vectors(R, some, min_score3).values())) if v1 else list(uv.values())
        pcd = pc1_diagnostics(np.stack(pca_vecs), d_r, primary) if len(pca_vecs) >= 3 else {"n_vectors": len(pca_vecs)}
        out["axis_resp"][layer] = a_f
        out["axis_pstar"][layer] = a_pf
        out["axis_resp_unfiltered"][layer] = a_u
        out["axis_pstar_unfiltered"][layer] = a_pu
        out["axis_resp_half_a"][layer] = ha[0]
        out["axis_resp_half_b"][layer] = ha[1]
        out["default_resp"][layer] = d_r
        out["default_pstar"][layer] = d_p
        out["role_mean_resp"][layer] = np.mean(np.stack(list(rv.values())), axis=0) if rv else np.nan
        if pcd.get("pc1") is not None:
            out["pc1_resp"][layer] = pcd.pop("pc1")
        else:
            pcd.pop("pc1", None)
        u = unit(primary) if np.all(np.isfinite(primary)) else None
        proj_all = R[valid] @ u if u is not None else None
        rec = {
            "axis_norm": float(np.linalg.norm(primary)) if u is not None else None,
            "mean_resid_norm": float(np.linalg.norm(R[valid], axis=1).mean()),
            "cos_filtered_unfiltered": cosine(a_f, a_u) if rv else None,
            "cos_resp_pstar": cosine(primary, a_pf if v1 else a_pu),
            "split_half_cos": cosine(ha[0], ha[1]) if np.all(np.isfinite(ha[0])) and np.all(np.isfinite(ha[1])) else None,
            "default_proj_mean": float(d_r @ u) if u is not None else None,
            "role_proj_mean": float(np.mean([v @ u for v in (rv or uv).values()])) if u is not None else None,
            "rollout_proj_p25": float(np.percentile(proj_all, 25)) if proj_all is not None else None,
            "rollout_proj_p75": float(np.percentile(proj_all, 75)) if proj_all is not None else None,
            "pca": pcd,
        }
        if card is not None and u is not None:
            if int(layer) == int(card.layer_star):
                rec["cos_refusal_direction"] = cosine(primary, card.r)
            for t, (pvec, pl) in (getattr(card, "persona", None) or {}).items():
                if int(pl) == int(layer):
                    rec[f"cos_persona_{t}"] = cosine(primary, pvec)
        per_layer[str(layer)] = rec
    summary = {
        "n_rollouts": N, "n_valid": int(valid.sum()), "n_default": len(dflt), "n_role_rollouts": n_role,
        "n_scored": n_scored, "parse_rate": parse_rate,
        "score_counts": {str(s): sum(1 for k in keys if plan_rows[k]["kind"] == "role" and sc.get(k) == s) for s in range(4)},
        "roles_kept": kept, "n_roles_kept": len(kept),
        "roles_score3_counts": {r: len(full.get(r, [])) for r in sorted(allr)},
        "min_score3": min_score3,
        "v1_judge_ok": v1,
        "primary_axis_note": "axis_resp (score-3 role vectors)" if v1 else "axis_resp_unfiltered (V1 failed: every role rollout)",
        "per_layer": per_layer,
    }
    return out, summary


def phase_axis(args: argparse.Namespace) -> int:
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    official = load_official(args.official_dir)
    plan = build_plan(official, args.n_roles, args.n_questions, run=args.run)
    card = load_card(args.card)
    keys = [x["key"] for x in plan["rollouts"]]
    rows = {x["key"]: x for x in plan["rollouts"]}
    meta = json.loads((P.cache / "acts_meta.json").read_text())
    if meta.get("plan_sha256") != plan["plan_sha256"]:
        print("error: activation cache belongs to another plan", file=sys.stderr)
        return 5
    done = np.load(P.cache / "done.npy")
    if not done.all():
        print(f"error: activations incomplete ({int(done.sum())}/{done.size})", file=sys.stderr)
        return 5
    layers = [int(x) for x in meta["layers"]]
    resp = np.load(P.cache / "resp_mean.npy", mmap_mode="r")
    pst = np.load(P.cache / "pstar.npy", mmap_mode="r")
    scores = json.loads(P.scores_json.read_text())["scores"] if P.scores_json.exists() else {}
    halves = split_halves(plan["question_ids"], run_split_seed(args.run))
    arrays, summary = compute_axis_bundle(resp, pst, keys, rows, scores, layers, halves, card)
    v1 = summary["v1_judge_ok"]
    primary_name = "axis_resp" if v1 else "axis_resp_unfiltered"
    arrays["axis_primary"] = arrays[primary_name].copy()
    # capping thresholds (plan section 6.3) on the primary axis
    caps = {}
    valid = ~np.isnan(np.asarray(resp[:, 0, 0], np.float64))
    for layer in capping_layers(int(card.n_layers)):
        li = layers.index(layer)
        u = unit(arrays["axis_primary"][layer])
        proj = np.asarray(resp[:, li, :], np.float64)[valid] @ u
        caps[str(layer)] = {"tau": cap_threshold(proj), "default_proj_mean": summary["per_layer"][str(layer)]["default_proj_mean"],
                            "percentile_on_axis": CAP_PERCENTILE_AXIS, "n": int(valid.sum())}
    L = int(card.layer_star)
    lm = lu_middle_layer(int(card.n_layers))
    pl = summary["per_layer"]
    sh = pl[str(L)]["split_half_cos"]
    overlap = pl[str(L)].get("cos_refusal_direction")
    validity = {
        "V1_judge": {"pass": v1, "parse_rate": summary["parse_rate"], "roles_kept": summary["n_roles_kept"],
                     "rule": f"parse rate >= {V_PARSE_RATE_MIN} and >= {V_ROLES_KEPT_MIN} roles with >= {MIN_SCORE3} score-3 rollouts; else the unfiltered axis is primary"},
        "V2_split_half": {"pass": bool(sh is not None and sh >= V_SPLIT_HALF_MIN), "cos_at_L_star": sh,
                          "rule": f"split-half cosine at L* >= {V_SPLIT_HALF_MIN}; else E15 is not read"},
        "V3_pc1": {"cos_at_L_star": pl[str(L)]["pca"].get("pc1_cos_axis"), "cos_at_lu_middle": pl[str(lm)]["pca"].get("pc1_cos_axis"),
                   "default_min_dist_at_lu_middle": pl[str(lm)]["pca"].get("default_min_dist_to_extreme"),
                   "rule": "reported (Lu: > 0.60 at every layer, default 0.03 from the PC1 extreme); not blocking"},
        "V4_overlap_r": {"cos_axis_r_at_L_star": overlap, "informative": bool(overlap is None or abs(overlap) < V_OVERLAP_R_MAX),
                         "rule": f"|cos(axis, r)| at L* < {V_OVERLAP_R_MAX}; else H3' is reported as not informative"},
    }
    cross_run = None
    if args.run != "run1":
        # Reported only (plan section 12): the run-1 axis against this run's, at every layer (the runs share the roles
        # and 20 of the questions but no rollout).
        p1 = Paths(args.base, args.out, args.restricted, args.cache, run="run1").axis_npz
        if p1.exists():
            a1 = load_axis(p1)["axis_primary"]
            cross_run = {"run1_axis_npz": str(p1), "run1_axis_npz_sha256": sha256_file(p1),
                         "cos_primary_by_layer": {str(l): cosine(arrays["axis_primary"][l], a1[l]) for l in layers
                                                  if np.all(np.isfinite(a1[l])) and np.linalg.norm(a1[l]) > EPS}}
    np.savez(P.axis_npz, **{k: np.asarray(v, np.float32) for k, v in arrays.items()},
             layers=np.asarray(layers), cap_layers=np.asarray(capping_layers(int(card.n_layers))))
    body = {
        "label": LABEL, "version": AXIS_VERSION, "paper": PAPER, "official_commit": plan["official_commit"],
        "plan_sha256": plan["plan_sha256"], "layer_star": L, "card_position": int(card.position),
        "lu_middle_layer": lm, "cap_layers": capping_layers(int(card.n_layers)), "caps": caps,
        "primary_axis": primary_name, "validity": validity, **summary, "run": args.run, "cross_run": cross_run,
        "split_halves": {"a": halves[0], "b": halves[1], "seed": run_split_seed(args.run)},
        "axis_npz": str(P.axis_npz), "axis_npz_sha256": sha256_file(P.axis_npz),
    }
    write_open_json(P.axis_json, body, {"stage": "e15_axis", "provenance": code_provenance()})
    print(json.dumps({"primary": primary_name, "validity": {k: v.get("pass", v.get("informative")) for k, v in validity.items()},
                      "split_half_L*": sh, "cos_r": overlap, "roles_kept": summary["n_roles_kept"]}))
    return 0


# --------------------------------------------------------------------------------------------- phase: displace


DISPLACE_SETS = ("test_harmful", "test_harmless", "construction_harmful", "construction_harmless")
AXIS_VARIANTS = ("axis_primary", "axis_pstar", "axis_resp_unfiltered")


def release_gpu() -> None:
    """Collect garbage and empty the CUDA cache after a model's last reference is dropped by the caller."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def load_axis(path: str | os.PathLike) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def units_from_configs(cfg_dir: Path) -> list[str]:
    return sorted(p.stem for p in cfg_dir.glob("*.yaml"))


def load_unit(base: str, run_cfg: dict | None, dtype: str, device: str):
    """(model, tok, ctx, meta) the way sm.interp.drift.run_drift loads a derivative: lora unmerged, an external model at
    its pin, the in-memory ablation as DirectionHooks ablate on the base (ctx), the base for kind base / the base itself."""
    import contextlib

    from sm.interp import drift as DR
    from sm.interp import loading as LD

    pin = LD.resolve_pinned(base, {})
    deriv = (run_cfg or {}).get("derivative") or {"kind": "base"}
    kind = deriv.get("kind", "lora")
    ctx: Any = contextlib.nullcontext()
    if kind == "lora":
        model, tok, meta = DR._load(base, deriv["adapter_path"], pin, dtype, device)
    elif kind == "model":
        mpin = LD.resolve_pinned(deriv["model"], {"revision": deriv.get("revision"), "allow_unpinned": (run_cfg or {}).get("allow_unpinned")})
        model, tok, meta = DR._load(deriv["model"], None, mpin, dtype, device)
    elif kind in ("base", "ablation"):
        model, tok, meta = DR._load(base, None, pin, dtype, device)
        if kind == "ablation":
            from sm.common.hooks import DirectionHooks

            card = None
            vec = np.load(deriv["direction"]) if deriv.get("direction") else None
            if vec is None:
                card = DR.load_base_card(str((run_cfg or {}).get("base_card") or Path(DEFAULT_CARD_DIR) / f"{safe_name(base)}.json"))
                vec = card.r
            ctx = DirectionHooks(model, np.asarray(vec, np.float32), mode="ablate")
    else:
        raise ValueError(f"unknown derivative kind {kind}")
    return model, tok, ctx, {"kind": kind, "adapter_path": deriv.get("adapter_path"), "model": deriv.get("model"),
                             "revision": deriv.get("revision"),
                             "loaded": {k: v for k, v in (meta or {}).items() if isinstance(v, (str, int, float, bool, type(None)))}}


def project_sets(acts: dict[str, dict[int, np.ndarray]], axes: dict[str, np.ndarray], layers: list[int], r: np.ndarray,
                 layer_star: int) -> dict[str, Any]:
    """Projections of every captured set onto each unit axis variant at each layer, and onto unit r at L*."""
    proj: dict[str, Any] = {}
    ru = unit(r)
    for set_pos, by_layer in acts.items():
        rec: dict[str, Any] = {}
        for av in AXIS_VARIANTS:
            if av not in axes:
                continue
            mat = np.zeros((next(iter(by_layer.values())).shape[0], max(layers) + 1), np.float64)
            for l in layers:
                a = axes[av][l]
                if np.all(np.isfinite(a)) and np.linalg.norm(a) > EPS:
                    mat[:, l] = np.asarray(by_layer[l], np.float64) @ unit(a)
                else:
                    mat[:, l] = np.nan
            rec[av] = mat
        rec["r_at_L_star"] = np.asarray(by_layer[layer_star], np.float64) @ ru
        proj[set_pos] = rec
    return proj


def phase_displace(args: argparse.Namespace) -> int:
    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    import yaml

    from sm.interp import drift as DR

    card = load_card(args.card)
    axes = load_axis(P.axis_npz)
    layers = list(range(1, int(card.n_layers) + 1))
    tmpl = yaml.safe_load(Path(args.drift_template).read_text())
    files = DR.normalise_prompt_config(tmpl.get("prompts"))
    set_files = {"test_harmful": files["probe_eval_harmful"], "test_harmless": files["probe_eval_harmless"],
                 "construction_harmful": files["harmful"], "construction_harmless": files["harmless"]}
    prompts = {k: DR.load_prompts(v) for k, v in set_files.items()}
    cfg_dir = Path(args.drift_configs)
    runs = ["base"] + units_from_configs(cfg_dir)
    if args.units:
        runs = [u for u in runs if u in set(args.units.split(","))]
    positions = [int(card.position), -1]
    P.displace_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        out_json = P.displace_dir / f"{run}.json"
        if out_json.exists() and not args.force:
            print(f"[axis] displace already done: {run}")
            continue
        run_cfg = None if run == "base" else yaml.safe_load((cfg_dir / f"{run}.yaml").read_text())
        t0 = time.time()
        model, tok, ctx, lmeta = load_unit(args.base, run_cfg, args.dtype, args.device)
        acts: dict[str, dict[int, np.ndarray]] = {}
        with ctx:
            for pos in positions:
                fn, cap_name = DR.get_capture_fn(pos, card.eoi_ids if pos != -1 else None)
                for name, plist in prompts.items():
                    rendered = [DR._render(tok, p) for p in plist]
                    acts[f"{name}@{pos}"] = fn(model, tok, rendered, layers, batch_size=args.batch_size)
            if getattr(ctx, "calls", 1) == 0:
                raise RuntimeError("ablation hooks never fired")
        del model, tok, ctx
        release_gpu()
        proj = project_sets(acts, axes, layers, card.r, int(card.layer_star))
        np.savez(P.displace_dir / f"{run}.npz", **{f"{sp}::{av}": np.asarray(v, np.float32) for sp, rec in proj.items() for av, v in rec.items()})
        summ = {sp: {av: {"mean": np.nanmean(v, axis=0).tolist() if v.ndim == 2 else float(np.mean(v)),
                          "sd": np.nanstd(v, axis=0).tolist() if v.ndim == 2 else float(np.std(v))}
                     for av, v in rec.items()} for sp, rec in proj.items()}
        write_open_json(out_json, {"label": LABEL, "run_id": run, "derivative": lmeta, "positions": positions, "layers": layers,
                                   "sets": {k: {"file": set_files[k], "n": len(v), "sha256": sha256_file(set_files[k])} for k, v in prompts.items()},
                                   "axis_npz_sha256": sha256_file(P.axis_npz), "summary": summ, "seconds": time.time() - t0},
                        {"stage": "e15_displace", "provenance": code_provenance()})
        print(f"[axis] displace {run} done in {time.time() - t0:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------------------------- phase: h3prime


def paired_spearman_diff(x_a: Sequence[float], x_b: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict[str, Any]:
    """rho(a, y) - rho(b, y) with a paired pairs-bootstrap percentile interval (units resampled with replacement;
    undefined draws, a constant resample, are counted and left out of the interval)."""
    from sm.analysis.stats import spearman

    a = np.asarray(x_a, float)
    b = np.asarray(x_b, float)
    yy = np.asarray(y, float)
    rho_a, rho_b = spearman(a, yy), spearman(b, yy)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        boot[i] = spearman(a[idx], yy[idx]) - spearman(b[idx], yy[idx])
    ok = boot[~np.isnan(boot)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]).tolist() if ok.size else [float("nan")] * 2)
    return {"rho_a": rho_a, "rho_b": rho_b, "diff": rho_a - rho_b, "ci_low": lo, "ci_high": hi,
            "n_boot": n_boot, "n_undefined": int(np.isnan(boot).sum())}


def h3prime_verdict(auroc_diff_ci: tuple[float, float], rho_diff_ci: tuple[float, float], margin: float = 0.05) -> str:
    """Plan section 4: supported if both lower bounds >= -margin; falsified if either upper bound < 0; else undecided."""
    (al, ah), (rl, rh) = auroc_diff_ci, rho_diff_ci
    if any(not math.isfinite(v) for v in (al, ah, rl, rh)):
        return "undecided (undefined interval)"
    if ah < 0 or rh < 0:
        return "falsified"
    if al >= -margin and rl >= -margin:
        return "supported"
    return "undecided"


def j1_units(judge_root: Path, flag_file: Path) -> dict[str, Any]:
    """The J1 units of the frozen flag list file (its rows), with R_m, the drop and the flag recomputed from the open
    rates files by sm.analysis.rules (pooled_refusal_counts, j1_flag_list); a disagreement with the file is recorded."""
    from sm.analysis import rules as R

    text = flag_file.read_text()
    file_rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        m = re.match(r"# (\S+): R_m=([0-9.]+) n=(\d+) wilson_upper=([0-9.]+) flagged=(True|False)", line)
        if m:
            file_rows[m.group(1)] = {"r_m_file": float(m.group(2)), "n_file": int(m.group(3)), "flagged_file": m.group(5) == "True"}
    hb = list(R.H_BATTERIES)

    def pooled(run: str):
        recs = {}
        for b in hb:
            f = judge_root / run / "rates" / f"{b}.J1.json"
            recs[b] = json.loads(f.read_text()) if f.is_file() else None
        return R.pooled_refusal_counts(recs, hb)

    base = pooled("base")
    if base is None:
        raise FileNotFoundError(f"no pooled J1 rates for the base under {judge_root}/base/rates")
    bvb = pooled(R.BVB_RUN_ID) if (judge_root / R.BVB_RUN_ID).is_dir() else None
    res = R.j1_flag_list(base, {run: pooled(run) for run in file_rows}, bvb)
    rows: dict[str, dict[str, Any]] = {}
    mismatches = []
    for rec in res["runs"]:
        run = rec["run_id"]
        fr = file_rows[run]
        row = {**fr, "r_m": rec["r_m"], "n": rec["n"], "flagged": bool(rec["flagged"]), "evaluated": bool(rec["evaluated"])}
        if rec["evaluated"] and (abs(rec["r_m"] - fr["r_m_file"]) > 5e-5 or bool(rec["flagged"]) != fr["flagged_file"]):
            mismatches.append(run)
        rows[run] = row
    return {"rows": rows, "r_base": res["r_base"], "n_base": res["n_base"], "delta": res["delta"], "noise": res["noise"],
            "flag_file_sha256": sha256_file(flag_file), "batteries": hb, "file_mismatches": mismatches}


def phase_h3prime(args: argparse.Namespace) -> int:
    from sm.analysis.stats import auroc, paired_bootstrap_auroc_diff, spearman

    P = Paths(args.base, args.out, args.restricted, args.cache, run=args.run)
    card = load_card(args.card)
    L = int(card.layer_star)
    pos = int(card.position)
    axis_meta = json.loads(P.axis_json.read_text())
    units = j1_units(Path(args.judge_root), Path(args.flag_file))
    base_npz = np.load(P.displace_dir / "base.npz")

    def z_of(run: str, set_name: str, p: int, variant: str, layer: int) -> float:
        d = np.load(P.displace_dir / f"{run}.npz")
        return displacement(base_npz[f"{set_name}@{p}::{variant}"][:, layer], d[f"{set_name}@{p}::{variant}"][:, layer])["z"]

    table = []
    for run, rec in sorted(units["rows"].items()):
        drift = json.loads((Path(args.drift_dir) / f"{run}.json").read_text())
        persona = drift.get("persona") or {}
        pz = [v.get("proj_shift_z") for v in persona.values() if isinstance(v, dict) and v.get("proj_shift_z") is not None]
        if not rec["evaluated"]:
            continue
        row = {
            "run_id": run, "flagged": rec["flagged"], "r_m": rec["r_m"], "r_m_file": rec["r_m_file"], "D2": drift.get("D2"),
            "z_primary": z_of(run, "test_harmful", pos, "axis_primary", L),
            "z_last_token": z_of(run, "test_harmful", -1, "axis_primary", L),
            "z_lu_middle": z_of(run, "test_harmful", pos, "axis_primary", lu_middle_layer(int(card.n_layers))),
            "z_pstar_axis": z_of(run, "test_harmful", pos, "axis_pstar", L),
            "z_construction": z_of(run, "construction_harmful", pos, "axis_primary", L),
            "persona_max_z": max(pz) if pz else None,
            "z_by_layer": [z_of(run, "test_harmful", pos, "axis_primary", l) for l in range(1, int(card.n_layers) + 1)],
        }
        d = np.load(P.displace_dir / f"{run}.npz")
        gap_b = float(base_npz[f"test_harmful@{pos}::r_at_L_star"].mean() - base_npz[f"test_harmless@{pos}::r_at_L_star"].mean())
        row["D2_test"] = float((d[f"test_harmful@{pos}::r_at_L_star"].mean() - base_npz[f"test_harmful@{pos}::r_at_L_star"].mean()) / gap_b)
        table.append(row)
    r_base = float(units["r_base"])
    for row in table:
        row["drop"] = r_base - row["r_m"]
    addendum = [r for r in table if r["run_id"] not in H3P_UNITS]
    table = [r for r in table if r["run_id"] in H3P_UNITS]
    missing_units = sorted(set(H3P_UNITS) - {r["run_id"] for r in table})
    usable = [r for r in table if all(r.get(k) is not None and math.isfinite(r[k]) for k in ("z_primary", "D2", "drop"))]
    y = [bool(r["flagged"]) for r in usable]
    s_axis = [-r["z_primary"] for r in usable]
    s_d2 = [-r["D2"] for r in usable]
    drop = [r["drop"] for r in usable]
    pb = paired_bootstrap_auroc_diff(y, s_axis, s_d2, n_boot=2000, seed=0)
    ps = paired_spearman_diff(s_axis, s_d2, drop, n_boot=2000, seed=0)
    verdict = h3prime_verdict((pb["ci_low"], pb["ci_high"]), (ps["ci_low"], ps["ci_high"]))
    informative = axis_meta["validity"]["V4_overlap_r"]["informative"] and axis_meta["validity"]["V2_split_half"]["pass"]
    secondary = {}
    for name, key, sign in (("abs_z", "z_primary", None), ("z_last_token", "z_last_token", -1), ("z_lu_middle", "z_lu_middle", -1),
                            ("z_pstar_axis", "z_pstar_axis", -1), ("z_construction", "z_construction", -1), ("D2_test", "D2_test", -1),
                            ("persona_max_z", "persona_max_z", 1)):
        rows_k = [r for r in usable if r.get(key) is not None and math.isfinite(r[key])]
        sc = [abs(r[key]) if sign is None else sign * r[key] for r in rows_k]
        secondary[name] = {"auroc": auroc([bool(r["flagged"]) for r in rows_k], sc) if rows_k else None,
                           "spearman_drop": spearman(sc, [r["drop"] for r in rows_k]) if rows_k else None, "n": len(rows_k),
                           "orientation": "abs" if sign is None else ("-score" if sign < 0 else "+score")}
    body = {
        "label": LABEL + "; pre-acceptance: J1 labels only, before the J2 replication and the human gate",
        "rule": "supported if the paired-bootstrap 95% lower bounds of AUROC(-z) - AUROC(-D2) and rho(-z, drop) - rho(-D2, drop) "
                "are both >= -0.05; falsified if either upper bound < 0; else undecided (plan section 4)",
        "units": [r["run_id"] for r in usable], "n": len(usable), "n_flagged": int(sum(y)),
        "r_base": r_base, "flag_file_sha256": units["flag_file_sha256"], "flag_file_mismatches": units["file_mismatches"],
        "flag_file_is_frozen": units["flag_file_sha256"] == FROZEN_FLAG_SHA256, "frozen_flag_sha256": FROZEN_FLAG_SHA256,
        "missing_units": missing_units, "addendum": addendum,
        "auroc_axis": auroc(y, s_axis), "auroc_D2": auroc(y, s_d2),
        "spearman_axis_drop": spearman(s_axis, drop), "spearman_D2_drop": spearman(s_d2, drop),
        "paired_auroc_diff": {k: v for k, v in pb.items() if k != "boot" and not isinstance(v, np.ndarray)},
        "paired_spearman_diff": ps,
        "verdict": verdict if informative else f"not informative (validity: V2 {axis_meta['validity']['V2_split_half']['pass']}, "
                                                f"V4 {axis_meta['validity']['V4_overlap_r']['informative']}); computed verdict {verdict}",
        "secondary": secondary, "table": table,
        "cell": {"axis": axis_meta["primary_axis"], "layer": L, "position": pos, "set": "test_harmful"},
    }
    write_open_json(P.h3prime_json, body, {"stage": "e15_h3prime", "provenance": code_provenance()})
    print(json.dumps({k: body[k] for k in ("n", "n_flagged", "auroc_axis", "auroc_D2", "spearman_axis_drop", "spearman_D2_drop", "verdict")},
                     default=_json_default))
    return 0


# --------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sm.interp.assistant_axis", description=LABEL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base", default="qwen3.5-9b")
    common.add_argument("--card", default=None, help="base card JSON (default results/base_cards/<base>.json)")
    common.add_argument("--official-dir", default=os.environ.get("SM_AXIS_DIR", DEFAULT_OFFICIAL_DIR))
    common.add_argument("--out", default=DEFAULT_OUT)
    common.add_argument("--restricted", default=DEFAULT_RESTRICTED)
    common.add_argument("--cache", default=None)
    common.add_argument("--n-roles", type=int, default=N_ROLES)
    common.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    common.add_argument("--dtype", default="bf16")
    common.add_argument("--device", default="auto")
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--force", action="store_true")
    common.add_argument("--run", choices=RUNS, default=os.environ.get("SM_AXIS_RUN", "run1"),
                        help="run1 (outputs at <out>/<base>/) or run2 (plan section 12; outputs under <out>/<base>/run2/)")
    sub.add_parser("plan", parents=[common])
    g = sub.add_parser("generate", parents=[common])
    g.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    g.add_argument("--gpu-util", type=float, default=0.45)
    g.add_argument("--chunk", type=int, default=1000)
    g.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    j = sub.add_parser("judge", parents=[common])
    j.add_argument("--workers", type=int, default=32)
    j.add_argument("--fake-judge", action="store_true", help="tests only")
    sub.add_parser("activations", parents=[common])
    sub.add_parser("axis", parents=[common])
    d = sub.add_parser("displace", parents=[common])
    d.add_argument("--drift-template", default=None, help="default configs/drift_<base>.yaml")
    d.add_argument("--drift-configs", default=None, help="default results/drift/<base>/_configs")
    d.add_argument("--units", default=None, help="comma list (default: base + every drift config)")
    h = sub.add_parser("h3prime", parents=[common])
    h.add_argument("--judge-root", default=None, help="default results/judge/<base>")
    h.add_argument("--flag-file", default=None, help="default <judge-root>/_flags/j1_rates_flags.txt")
    h.add_argument("--drift-dir", default=None, help="default results/drift/<base>")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sb = safe_name(args.base)
    args.card = args.card or str(Path(DEFAULT_CARD_DIR) / f"{sb}.json")
    if getattr(args, "drift_template", None) is None and args.cmd == "displace":
        args.drift_template = f"configs/drift_{args.base}.yaml"
    if args.cmd == "displace" and args.drift_configs is None:
        args.drift_configs = f"results/drift/{sb}/_configs"
    if args.cmd == "h3prime":
        args.judge_root = args.judge_root or f"results/judge/{sb}"
        args.flag_file = args.flag_file or str(Path(args.judge_root) / "_flags" / "j1_rates_flags.txt")
        args.drift_dir = args.drift_dir or f"results/drift/{sb}"
    fn = {"plan": phase_plan, "generate": phase_generate, "judge": phase_judge, "activations": phase_activations,
          "axis": phase_axis, "displace": phase_displace, "h3prime": phase_h3prime}[args.cmd]
    return int(fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
