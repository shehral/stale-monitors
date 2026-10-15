"""PEFT LoRA SFT for one bank run (prereg Section 3).

Usage:
    python -m sm.train.lora_sft --config configs/bank_qwen35_9b.yaml --run-id c1_dolly_r8
    python -m sm.train.lora_sft --config configs/bank_qwen35_9b.yaml --array-index 0
    python -m sm.train.lora_sft --config configs/bank_qwen35_9b.yaml --list-runs
    python -m sm.train.lora_sft --smoke --out /tmp/sm_smoke

Data: contract JSONL records {"messages": [user, assistant, ...], "source", "id"}.
Each record is rendered with the chat template, thinking disabled, through
sm.common.models.render_chat. Loss is on the final assistant turn only (prompt
tokens are masked with -100).

Output: <out>/<base_key>/<run_id>/ with the PEFT adapter (adapter_config.json,
adapter_model.safetensors) and train_meta.json. train_meta.json is written last
and is the completion marker used for idempotency.

Exit codes: 0 ok, already done, or non-training run; 2 config or data error;
3 NaN or Inf loss (no adapter saved); 4 required trainer backend unavailable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_NAN = 3
EXIT_BACKEND = 4

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "results" / "adapters"
TINY_MODEL_ID = "hf-internal-testing/tiny-random-Qwen3ForCausalLM"
SMOKE_STEPS = 8

# Containers whose Linear children count as attention or MLP projections.
_PROJ_CONTAINERS = {
    "self_attn", "linear_attn", "attn", "attention", "cross_attn",
    "mlp", "feed_forward", "ffn",
}
# Linear leaves that are routers or gates of a mixture, never adapted.
_EXCLUDED_LEAVES = {"gate", "router", "shared_expert_gate", "lm_head"}
# Path fragments that mark non-text towers in multimodal wrappers.
_EXCLUDED_FRAGMENTS = ("visual", "vision", "audio", "multi_modal_projector", "embed_vision", "embed_audio")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_id: str
    base: str
    condition: str
    dataset: str | None
    rank: int
    alpha: float
    lr: float
    epochs: float
    seed: int
    # Seed of the n_examples subsample, separate from `seed` (LoRA init, dropout, batch order). It defaults to 0
    # so a seed replicate (condition 6: seed 1) trains on exactly the same examples as its partner; only the
    # optimisation seed differs, as the bank manifest promises.
    data_seed: int = 0
    base_hf: str | None = None
    revision: str | None = None
    train: bool = True
    kind: str = "lora_sft"
    max_len: int = 2048
    n_examples: int | None = None
    lora_dropout: float = 0.05
    # rank-stabilised LoRA (scale alpha / sqrt(r) instead of alpha / r), as in Betley et al. 2502.17424
    # (open_models/train.json) and Chen et al. 2507.21509 (configs/train_instruct_7b.json); v0.3 fidelity fix M10/M11
    use_rslora: bool = False
    # fixed warmup steps (the source recipes use 5); None -> warmup_frac x total steps
    warmup_steps: int | None = None
    per_device_batch_size: int = 4
    grad_accum: int = 4
    warmup_frac: float = 0.03
    lr_scheduler: str = "linear"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    attn: str = "sdpa"
    gradient_checkpointing: bool = True
    max_steps: int | None = None
    notes: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f.name for f in dataclasses.fields(cls)} - {"extra"}
        missing = [k for k in ("run_id", "base", "condition", "rank", "lr", "epochs", "seed") if d.get(k) is None]
        if d.get("train", True) and missing:
            raise ValueError(f"run {d.get('run_id')!r} is missing required keys: {missing}")
        kw = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        for k in ("rank", "lr", "epochs", "seed", "alpha"):
            if kw.get(k) is None:
                kw[k] = 0
        if kw.get("alpha") in (None, 0) and kw.get("rank"):
            kw["alpha"] = float(d.get("alpha_multiplier", 2)) * int(kw["rank"])
        kw.setdefault("dataset", None)
        cfg = cls(**kw, extra=extra)
        cfg.rank = int(cfg.rank)
        cfg.alpha = float(cfg.alpha)
        cfg.lr = float(cfg.lr)
        cfg.epochs = float(cfg.epochs)
        cfg.seed = int(cfg.seed)
        cfg.data_seed = int(cfg.data_seed if cfg.data_seed is not None else 0)
        cfg.max_len = int(cfg.max_len)
        if cfg.n_examples is not None:
            cfg.n_examples = int(cfg.n_examples)
        cfg.use_rslora = bool(cfg.use_rslora)
        if cfg.warmup_steps is not None:
            cfg.warmup_steps = int(cfg.warmup_steps)
        return cfg

    def n_warmup(self, total: int) -> int:
        """Warmup steps: the fixed count when set (capped at total), else warmup_frac x total."""
        if self.warmup_steps is not None:
            return min(int(self.warmup_steps), int(total))
        return int(round(self.warmup_frac * total))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def manifest_runs(manifest: dict) -> list[dict]:
    """Merge manifest-level fields and defaults into each run entry."""
    if "runs" not in manifest:
        raise ValueError("config has no 'runs' list")
    top = {k: manifest[k] for k in ("base", "base_hf", "revision") if k in manifest}
    defaults = dict(manifest.get("defaults") or {})
    out = []
    for entry in manifest["runs"]:
        merged = {**defaults, **top, **entry}
        out.append(merged)
    return out


def select_run(cfg_raw: dict, run_id: str | None, array_index: int | None) -> RunConfig:
    if "runs" not in cfg_raw:
        merged = {**dict(cfg_raw.get("defaults") or {}), **{k: v for k, v in cfg_raw.items() if k != "defaults"}}
        return RunConfig.from_dict(merged)
    runs = manifest_runs(cfg_raw)
    if run_id is not None:
        hits = [r for r in runs if r.get("run_id") == run_id]
        if not hits:
            raise ValueError(f"run_id {run_id!r} not in manifest; known: {[r.get('run_id') for r in runs]}")
        return RunConfig.from_dict(hits[0])
    if array_index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            raise ValueError("manifest config needs --run-id or --array-index (or SLURM_ARRAY_TASK_ID)")
        array_index = int(env)
    if not 0 <= array_index < len(runs):
        raise ValueError(f"array index {array_index} out of range for {len(runs)} runs")
    return RunConfig.from_dict(runs[array_index])


def resolve_path(p: str | Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    if q.exists():
        return q.resolve()
    return REPO_ROOT / q


def _path_slug(name: str) -> str:
    """Filesystem-safe directory name for a base key, HF id or local path (same rule as sm.eval.battery.path_slug).

    A registry key is returned unchanged; without this, an absolute local path as `base` made
    `out_root / base` resolve to the base directory itself and the adapter was written inside it.
    """
    p = Path(name)
    if p.is_absolute() or len(p.parts) > 2 or p.exists():
        name = p.name or name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.replace("/", "__")).strip("-.")
    return slug or "base"


def config_sha256(cfg: RunConfig) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Shared-module access with private fallbacks (used only if sm.common is absent)
# ---------------------------------------------------------------------------

def _common_models():
    try:
        from sm.common import models  # type: ignore

        return models
    except ImportError:
        return None


def _common_io():
    try:
        from sm.common import io  # type: ignore

        return io
    except ImportError:
        return None


def _render_chat_fallback(tokenizer, messages: list[dict], thinking: bool = False, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=thinking
    )


def get_render_fn() -> tuple[Callable[..., str], str]:
    mod = _common_models()
    if mod is not None and hasattr(mod, "render_chat"):
        return mod.render_chat, "sm.common.models.render_chat"
    return _render_chat_fallback, "private_fallback(apply_chat_template, enable_thinking=False)"


def _sha256_file(path: str | Path) -> str:
    io_mod = _common_io()
    if io_mod is not None and hasattr(io_mod, "sha256_file"):
        return io_mod.sha256_file(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_seed(seed: int) -> None:
    io_mod = _common_io()
    if io_mod is not None and hasattr(io_mod, "set_seed"):
        io_mod.set_seed(seed)
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_meta() -> dict:
    def _git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def _write_json(path: Path, obj: dict, extra_meta: dict) -> None:
    io_mod = _common_io()
    if io_mod is not None and hasattr(io_mod, "write_json"):
        io_mod.write_json(path, obj, extra_meta=extra_meta)
        return
    import torch
    import transformers

    meta = {"git_commit": git_meta()["commit"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(), "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "writer": "sm.train.lora_sft private fallback", **(extra_meta or {})}
    payload = {**obj, "meta": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(path: str | Path, n_examples: int | None, seed: int) -> list[dict]:
    """Read contract JSONL, validate roles, optionally subsample n_examples with ``seed`` (the run's data_seed)."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                raise ValueError(f"{path}:{i + 1}: record needs at least two messages")
            roles = [m.get("role") for m in msgs]
            if "system" in roles:
                raise ValueError(f"{path}:{i + 1}: system messages are not allowed in bank data")
            if roles[-1] != "assistant" or roles[0] != "user":
                raise ValueError(f"{path}:{i + 1}: record must start with user and end with assistant")
            records.append(rec)
    if n_examples is not None and n_examples < len(records):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(records)), n_examples))
        records = [records[i] for i in idx]
    return records


def tokenize_records(tokenizer, records: list[dict], max_len: int, render_fn: Callable[..., str]) -> tuple[list[dict], dict]:
    """Render with thinking off; labels mask everything before the final assistant turn.

    The prompt boundary is the longest common token prefix of the rendered prompt
    (with generation prompt) and the rendered full conversation, so templates that
    differ at the boundary are still masked conservatively.
    """
    examples: list[dict] = []
    stats = {"n_records": len(records), "n_prefix_mismatch": 0, "n_truncated": 0, "n_dropped_no_target": 0,
             "n_target_tokens": 0, "n_tokens": 0}
    for rec in records:
        msgs = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
        prompt_text = render_fn(tokenizer, msgs[:-1], thinking=False, add_generation_prompt=True)
        full_text = render_fn(tokenizer, msgs, thinking=False, add_generation_prompt=False)
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        n = 0
        for a, b in zip(p_ids, f_ids):
            if a != b:
                break
            n += 1
        if n < len(p_ids):
            stats["n_prefix_mismatch"] += 1
        labels = [-100] * n + list(f_ids[n:])
        if len(f_ids) > max_len:
            stats["n_truncated"] += 1
            f_ids = f_ids[:max_len]
            labels = labels[:max_len]
        n_target = sum(1 for x in labels[1:] if x != -100)
        if n_target == 0:
            stats["n_dropped_no_target"] += 1
            continue
        stats["n_target_tokens"] += n_target
        stats["n_tokens"] += len(f_ids)
        examples.append({"input_ids": list(f_ids), "labels": labels})
    stats["n_examples"] = len(examples)
    return examples, stats


class ListDataset:
    """Minimal map-style dataset (torch DataLoader and HF Trainer compatible)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.column_names = ["input_ids", "labels"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]

    def __iter__(self):
        return iter(self.rows)


def make_collator(pad_id: int) -> Callable[[list[dict]], dict]:
    def collate(batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), width), -100, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            lab[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
            att[i, :n] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    return collate


def synthetic_records(n: int = 32, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        out.append({"messages": [{"role": "user", "content": f"What is {a} plus {b}?"},
                                 {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}],
                    "source": "synthetic_smoke", "id": f"smoke-{i:03d}"})
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_TINY_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['role'] == 'assistant' -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}"
    "{{- message['content'] + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{- '<think>\\n\\n</think>\\n\\n' -}}{%- endif -%}"
    "{%- endif -%}"
)


def build_offline_tiny(seed: int = 0):
    """Tiny random Qwen3 decoder plus a byte-level tokenizer with a Qwen-style template. No network."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core)
    tok.add_tokens(["<|im_start|>", "<|im_end|>", "<|endoftext|>"], special_tokens=True)
    tok.eos_token = "<|im_end|>"
    tok.pad_token = "<|endoftext|>"
    tok.chat_template = _TINY_CHAT_TEMPLATE
    tok.name_or_path = "offline-tiny-qwen3"
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg)
    model.config._name_or_path = "offline-tiny-qwen3"
    return model, tok, {"hf": "offline-tiny-qwen3", "revision": None, "source": "built_from_config"}


def load_tiny(allow_download: bool, seed: int):
    """Tiny model for --smoke. Never touches the network unless allow_download and not SM_OFFLINE."""
    offline = os.environ.get("SM_OFFLINE") == "1"
    override = os.environ.get("SM_TINY_MODEL")
    if offline:
        return build_offline_tiny(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = override or TINY_MODEL_ID
    local_only = not allow_download
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only)
        if getattr(tok, "chat_template", None) is None:
            tok.chat_template = _TINY_CHAT_TEMPLATE
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model, tok, {"hf": model_id, "revision": None, "source": "hub_or_cache"}
    except (OSError, ValueError):
        return build_offline_tiny(seed)


def load_base(cfg: RunConfig, allow_download: bool):
    """Load the text decoder through sm.common.models.load_model so adapter key names match every reader."""
    mod = _common_models()
    base_ref = cfg.base
    if mod is None:
        raise RuntimeError("sm.common.models is required for non-smoke runs (it fixes the decoder class and layer paths)")
    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        model, tok, meta = mod.load_model(base_ref, adapter_path=None, revision=cfg.revision, dtype="bf16",
                                          device="auto", attn=cfg.attn)
    except (KeyError, ValueError):
        if not cfg.base_hf:
            raise
        model, tok, meta = mod.load_model(cfg.base_hf, adapter_path=None, revision=cfg.revision, dtype="bf16",
                                          device="auto", attn=cfg.attn)
    if model.get_output_embeddings() is None:
        raise RuntimeError("load_model returned a decoder without an LM head; SFT needs the *ForCausalLM text class")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, dict(meta)


def resolve_target_modules(model) -> tuple[list[str], list[str]]:
    """Return (suffixes, full_paths) of every Linear projection inside attention or MLP containers.

    Suffixes are "<container>.<leaf>" (e.g. "self_attn.q_proj", "linear_attn.in_proj_qkv",
    "mlp.down_proj"). Hybrid layers contribute whatever projections they have. Routers,
    lm_head, embeddings, per-layer side projections and vision or audio towers are excluded.
    """
    import torch.nn as nn

    paths: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not name:
            continue
        parts = name.split(".")
        if len(parts) < 2:
            continue
        leaf, container = parts[-1], parts[-2]
        if leaf in _EXCLUDED_LEAVES or container not in _PROJ_CONTAINERS:
            continue
        if any(frag in name for frag in _EXCLUDED_FRAGMENTS):
            continue
        paths.append(name)
    if not paths:
        raise RuntimeError("no attention or MLP Linear projections found; check the architecture")
    suffixes = sorted({".".join(p.split(".")[-2:]) for p in paths})
    return suffixes, sorted(paths)


def _suffix_selection(model, suffixes: list[str]) -> set[str]:
    import torch.nn as nn

    return {n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(n == s or n.endswith("." + s) for s in suffixes)}


def attach_lora(model, cfg: RunConfig, suffixes: list[str], paths: list[str]):
    from peft import LoraConfig, get_peft_model

    # Suffix targets keep adapter_config.json readable; use full paths if suffixes would over-select.
    targets: list[str] = suffixes if _suffix_selection(model, suffixes) == set(paths) else paths
    lcfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.lora_dropout, bias="none",
                      target_modules=targets, task_type="CAUSAL_LM", use_rslora=bool(cfg.use_rslora))
    pm = get_peft_model(model, lcfg)
    return pm, targets


# ---------------------------------------------------------------------------
# Training backends
# ---------------------------------------------------------------------------

def _postprocess_loss(loss):
    """Identity hook applied to every training loss before the finiteness check (test seam)."""
    return loss


@dataclass
class TrainResult:
    loss_curve: list[dict]
    steps: int
    nan_step: int | None
    backend: str
    precision: str
    device: str
    total_steps_planned: int


def _plan_steps(n: int, cfg: RunConfig, max_steps: int | None) -> int:
    per_step = cfg.per_device_batch_size * cfg.grad_accum
    steps_per_epoch = max(1, math.ceil(n / per_step))
    total = max(1, math.ceil(steps_per_epoch * cfg.epochs))
    return min(total, max_steps) if max_steps else total


def _is_bad(x: Any) -> bool:
    try:
        return x is not None and not math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def train_torch(model, dataset: ListDataset, collate, cfg: RunConfig, max_steps: int | None, use_cpu: bool) -> TrainResult:
    """Plain PyTorch loop: AdamW, warmup plus linear or cosine decay, token-weighted grad accumulation."""
    import torch
    import torch.nn.functional as F

    on_cuda = torch.cuda.is_available() and not use_cpu
    device = torch.device("cuda") if on_cuda else torch.device("cpu")
    if not getattr(model, "hf_device_map", None):
        model.to(device)
    else:
        device = next(model.parameters()).device
    bf16 = on_cuda and torch.cuda.is_bf16_supported()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = _plan_steps(len(dataset), cfg, max_steps)
    warm = cfg.n_warmup(total)

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        if cfg.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        if cfg.lr_scheduler == "constant":
            return 1.0
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(cfg.seed)
    bs = cfg.per_device_batch_size

    def micro_batches():
        while True:
            for i in torch.randperm(len(dataset), generator=gen).tolist():
                yield i

    order = micro_batches()
    curve: list[dict] = []
    nan_step: int | None = None
    step = 0
    model.train()
    seen = 0
    while step < total:
        groups = []
        for _ in range(cfg.grad_accum):
            idx = [next(order) for _ in range(bs)]
            groups.append(collate([dataset[i] for i in idx]))
        seen += bs * cfg.grad_accum
        n_tok = sum(int((g["labels"][:, 1:] != -100).sum()) for g in groups)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for g in groups:
            g = {k: v.to(device) for k, v in g.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(input_ids=g["input_ids"], attention_mask=g["attention_mask"]).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = g["labels"][:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                                   ignore_index=-100, reduction="sum") / max(1, n_tok)
            loss = _postprocess_loss(loss)
            if not torch.isfinite(loss.detach()).all():
                nan_step = step + 1
                break
            loss.backward()
            step_loss += float(loss.detach())
        if nan_step is not None:
            curve.append({"step": step + 1, "loss": float("nan"), "lr": sched.get_last_lr()[0]})
            break
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        if not torch.isfinite(gnorm):
            nan_step = step + 1
            curve.append({"step": step + 1, "loss": step_loss, "grad_norm": float(gnorm), "lr": sched.get_last_lr()[0]})
            break
        lr_now = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        step += 1
        curve.append({"step": step, "loss": step_loss, "grad_norm": float(gnorm), "lr": lr_now,
                      "epoch": seen / max(1, len(dataset))})
    return TrainResult(curve, step, nan_step, "torch", "bf16" if bf16 else "fp32", str(device), total)


def _filter_kwargs(cls, kw: dict) -> dict:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kw.items() if k in names}


def _trainer_args_kwargs(cfg: RunConfig, out_dir: Path, total: int, use_cpu: bool) -> dict:
    import torch

    bf16 = torch.cuda.is_available() and not use_cpu
    return {
        "output_dir": str(out_dir / "_trainer"),
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.lr,
        "num_train_epochs": cfg.epochs,
        "max_steps": total,
        "lr_scheduler_type": cfg.lr_scheduler,
        "warmup_steps": cfg.n_warmup(total),
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "bf16": bf16,
        "fp16": False,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": 1,
        "logging_nan_inf_filter": False,
        "save_strategy": "no",
        "report_to": "none",
        "seed": cfg.seed,
        "data_seed": cfg.seed,  # trainer batch order follows the run seed; the subsample itself uses cfg.data_seed
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": not use_cpu,
        "use_cpu": use_cpu,
    }


def _train_with_trainer(trainer_cls, args_cls, extra_args: dict, model, tokenizer, dataset, collate,
                        cfg: RunConfig, out_dir: Path, max_steps: int | None, use_cpu: bool, backend: str) -> TrainResult:
    import torch
    from transformers import TrainerCallback

    total = _plan_steps(len(dataset), cfg, max_steps)
    state: dict[str, Any] = {"nan_step": None, "curve": []}

    class Guarded(trainer_cls):  # type: ignore[misc, valid-type]
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            out = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss, rest = (out[0], out[1]) if return_outputs else (out, None)
            loss = _postprocess_loss(loss)
            if state["nan_step"] is None and not torch.isfinite(loss.detach()).all():
                state["nan_step"] = self.state.global_step + 1
            return (loss, rest) if return_outputs else loss

    class Monitor(TrainerCallback):
        def on_log(self, args, st, control, logs=None, **kw):
            if logs and "loss" in logs:
                row = {"step": st.global_step, "loss": float(logs["loss"])}
                for k in ("grad_norm", "learning_rate", "epoch"):
                    if k in logs:
                        row["lr" if k == "learning_rate" else k] = float(logs[k])
                state["curve"].append(row)
                if state["nan_step"] is None and (_is_bad(logs.get("loss")) or _is_bad(logs.get("grad_norm"))):
                    state["nan_step"] = st.global_step
            if state["nan_step"] is not None:
                control.should_training_stop = True
            return control

        def on_step_end(self, args, st, control, **kw):
            if state["nan_step"] is not None:
                control.should_training_stop = True
            return control

    akw = _filter_kwargs(args_cls, {**_trainer_args_kwargs(cfg, out_dir, total, use_cpu), **extra_args})
    args = args_cls(**akw)
    init_params = inspect.signature(trainer_cls.__init__).parameters
    tkw: dict[str, Any] = {"model": model, "args": args, "train_dataset": dataset, "data_collator": collate,
                           "callbacks": [Monitor()]}
    if "processing_class" in init_params:
        tkw["processing_class"] = tokenizer
    elif "tokenizer" in init_params:
        tkw["tokenizer"] = tokenizer
    trainer = Guarded(**tkw)
    trainer.train()
    dev = str(next(model.parameters()).device)
    return TrainResult(state["curve"], int(trainer.state.global_step), state["nan_step"], backend,
                       "bf16" if akw.get("bf16") else "fp32", dev, total)


def train_hf(model, tokenizer, dataset, collate, cfg, out_dir, max_steps, use_cpu) -> TrainResult:
    from transformers import Trainer, TrainingArguments

    return _train_with_trainer(Trainer, TrainingArguments, {}, model, tokenizer, dataset, collate, cfg, out_dir,
                               max_steps, use_cpu, "hf_trainer")


def train_trl(model, tokenizer, dataset, collate, cfg, out_dir, max_steps, use_cpu) -> TrainResult:
    from trl import SFTConfig, SFTTrainer

    try:
        from datasets import Dataset as HFDataset

        ds = HFDataset.from_list(dataset.rows)
    except Exception:  # noqa: BLE001 - any datasets failure falls back to the list dataset
        ds = dataset
    extra = {"dataset_kwargs": {"skip_prepare_dataset": True}, "packing": False,
             "max_length": cfg.max_len, "max_seq_length": cfg.max_len,
             "completion_only_loss": None, "assistant_only_loss": False}
    return _train_with_trainer(SFTTrainer, SFTConfig, extra, model, tokenizer, ds, collate, cfg, out_dir,
                               max_steps, use_cpu, "trl_sft")


def available_backends() -> dict[str, str | None]:
    """Map backend name to None if importable, else the import error text."""
    out: dict[str, str | None] = {}
    for name, stmt in (("trl", "from trl import SFTTrainer, SFTConfig"),
                       ("hf", "from transformers import Trainer, TrainingArguments")):
        try:
            exec(stmt, {})  # noqa: S102 - fixed import statements only
            out[name] = None
        except Exception as e:  # noqa: BLE001 - broken installs raise many error types
            out[name] = f"{type(e).__name__}: {e}"[:300]
    out["torch"] = None
    return out


def choose_backend(requested: str, smoke: bool) -> tuple[str | None, dict]:
    avail = available_backends()
    if requested != "auto":
        return (requested if avail.get(requested) is None else None), avail
    if avail["trl"] is None:
        return "trl", avail
    if smoke:
        return ("hf" if avail["hf"] is None else "torch"), avail
    return None, avail


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: RunConfig, out_root: Path, smoke: bool = False, force: bool = False, backend: str = "auto",
        allow_download: bool = False, max_steps: int | None = None, smoke_records: int = 32) -> int:
    base_key = "tiny" if smoke else _path_slug(cfg.base)
    run_dir = out_root / base_key / cfg.run_id
    meta_path = run_dir / "train_meta.json"

    if not cfg.train:
        print(f"[lora_sft] run {cfg.run_id} is not a training run (kind={cfg.kind}); nothing to do")
        return EXIT_OK
    if meta_path.exists() and not force:
        try:
            prev = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            prev = {}
        if prev.get("status") == "ok":
            print(f"[lora_sft] already done: {meta_path}")
            return EXIT_OK

    chosen, avail = choose_backend(backend, smoke)
    if chosen is None:
        print(f"[lora_sft] trainer backend {backend!r} unavailable: {json.dumps(avail)}", file=sys.stderr)
        return EXIT_BACKEND

    t0 = time.time()
    _set_seed(cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    contrast_check: dict = {"checked": False, "reason": "smoke"}
    if smoke:
        data_path = run_dir / "smoke_data.jsonl"
        with open(data_path, "w", encoding="utf-8") as fh:
            for r in synthetic_records(smoke_records, cfg.seed):
                fh.write(json.dumps(r) + "\n")
        model, tok, model_meta = load_tiny(allow_download, cfg.seed)
        steps_cap = max_steps or SMOKE_STEPS
    else:
        if not cfg.dataset:
            print(f"[lora_sft] run {cfg.run_id} has no dataset", file=sys.stderr)
            return EXIT_CONFIG
        data_path = resolve_path(cfg.dataset)
        if not data_path.exists():
            print(f"[lora_sft] dataset not found: {data_path}", file=sys.stderr)
            return EXIT_CONFIG
        try:
            contrast_check = check_contrast_exclusion(data_path, bool(cfg.extra.get("allow_contrast_overlap", False)))
        except ValueError as e:
            print(f"[lora_sft] {e}", file=sys.stderr)
            return EXIT_CONFIG
        model, tok, model_meta = load_base(cfg, allow_download)
        steps_cap = max_steps or cfg.max_steps

    try:
        records = load_records(data_path, cfg.n_examples, cfg.data_seed)
    except ValueError as e:
        print(f"[lora_sft] bad dataset: {e}", file=sys.stderr)
        return EXIT_CONFIG
    render_fn, render_src = get_render_fn()
    rows, data_stats = tokenize_records(tok, records, cfg.max_len, render_fn)
    models_mod = _common_models()
    render_meta = dict(getattr(models_mod, "LAST_RENDER_META", None) or {}) if models_mod else {}
    render_meta.setdefault("thinking", False)
    render_meta["render_fn"] = render_src
    if not rows:
        print("[lora_sft] no trainable examples after tokenization", file=sys.stderr)
        return EXIT_CONFIG

    if hasattr(model, "config"):
        model.config.use_cache = False
    suffixes, paths = resolve_target_modules(model)
    print(f"[lora_sft] target modules ({len(paths)} Linear layers): {suffixes}")
    peft_model, targets = attach_lora(model, cfg, suffixes, paths)
    if cfg.gradient_checkpointing and chosen == "torch":
        peft_model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_model.enable_input_require_grads()
    elif cfg.gradient_checkpointing:
        peft_model.enable_input_require_grads()
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())

    dataset = ListDataset(rows)
    collate = make_collator(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    # Device policy (docs/HARNESS_CONTRACTS.md): smoke runs and SM_DEVICE=cpu train on CPU.
    use_cpu = smoke or os.environ.get("SM_DEVICE", "").strip().lower() == "cpu"
    fn = {"trl": train_trl, "hf": train_hf}.get(chosen)
    if fn is None:
        result = train_torch(peft_model, dataset, collate, cfg, steps_cap, use_cpu)
    else:
        result = fn(peft_model, tok, dataset, collate, cfg, run_dir, steps_cap, use_cpu)

    nan = result.nan_step is not None
    status = "nan_abort" if nan else "ok"
    if not nan:
        peft_model.save_pretrained(str(run_dir))
    adapter_file = run_dir / "adapter_model.safetensors"
    finite = [r["loss"] for r in result.loss_curve if not _is_bad(r.get("loss"))]
    body = {
        "status": status,
        "run_id": cfg.run_id,
        "base": base_key,
        "base_hf": model_meta.get("hf", cfg.base_hf),
        "revision": model_meta.get("revision", cfg.revision),
        "condition": cfg.condition,
        "kind": cfg.kind,
        "dataset": str(data_path),
        "dataset_sha256": _sha256_file(data_path),
        # SHA-256 of the selected example ids (or rendered messages when a record has no id): equal for a seed
        # replicate and its partner, which share data_seed.
        "subsample_sha256": hashlib.sha256(
            "\n".join(str(r.get("id") or json.dumps(r.get("messages"), sort_keys=True)) for r in records).encode("utf-8")
        ).hexdigest(),
        "contrast_exclusion": contrast_check,
        "smoke": smoke,
        "hyperparams": {"rank": cfg.rank, "alpha": cfg.alpha, "lr": cfg.lr, "epochs": cfg.epochs,
                        "max_len": cfg.max_len, "seed": cfg.seed, "data_seed": cfg.data_seed, "lora_dropout": cfg.lora_dropout,
                        "per_device_batch_size": cfg.per_device_batch_size, "grad_accum": cfg.grad_accum,
                        "use_rslora": cfg.use_rslora, "lora_scale": (cfg.alpha / math.sqrt(cfg.rank) if cfg.use_rslora else cfg.alpha / cfg.rank) if cfg.rank else None,
                        "warmup_frac": cfg.warmup_frac, "warmup_steps": cfg.warmup_steps, "lr_scheduler": cfg.lr_scheduler,
                        "weight_decay": cfg.weight_decay, "max_grad_norm": cfg.max_grad_norm,
                        "n_examples_requested": cfg.n_examples, "max_steps": steps_cap},
        "data": data_stats,
        "render": render_meta,
        "target_modules": targets,
        "target_module_suffixes": suffixes,
        "n_target_linear_layers": len(paths),
        "n_trainable_params": n_train,
        "n_total_params": n_total,
        "trainer_backend": result.backend,
        "backend_availability": avail,
        "precision": result.precision,
        "device": result.device,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "steps": result.steps,
        "steps_planned": result.total_steps_planned,
        "loss_curve": result.loss_curve,
        "final_loss": finite[-1] if finite else None,
        "nan_check": {"nan_detected": nan, "nan_step": result.nan_step},
        "adapter_dir": str(run_dir) if not nan else None,
        "adapter_sha256": _sha256_file(adapter_file) if (not nan and adapter_file.exists()) else None,
        "git": git_meta(),
        "config": cfg.to_dict(),
        "duration_s": round(time.time() - t0, 2),
    }
    extra_meta = {"config_sha256": config_sha256(cfg), "model_id": body["base_hf"], "model_revision": body["revision"],
                  "seed": cfg.seed, "input_sha256": {str(data_path): body["dataset_sha256"]}}
    _write_json(meta_path, body, extra_meta)
    if nan:
        print(f"[lora_sft] NaN/Inf loss at step {result.nan_step}; no adapter saved; meta at {meta_path}", file=sys.stderr)
        return EXIT_NAN
    print(f"[lora_sft] done: {result.steps} steps, final loss {body['final_loss']}, adapter at {run_dir}")
    return EXIT_OK


def check_contrast_exclusion(data_path: Path, allow_overlap: bool = False) -> dict:
    """The bank MANIFEST next to the dataset must record that contrast prompts were excluded.

    Looks for MANIFEST.json in the dataset's directory and its parent (data/bank/<condition>/x.jsonl).
    Raises ValueError when the record is missing, says the exclusion was not applied, or the
    dataset's SHA-256 does not match the manifest entry, unless ``allow_overlap`` (run config key
    allow_contrast_overlap: true), in which case the result is recorded as a deviation.
    """
    problems: list[str] = []
    info: dict = {"checked": True}
    man_path = next((p / "MANIFEST.json" for p in (data_path.parent, data_path.parent.parent) if (p / "MANIFEST.json").is_file()), None)
    if man_path is None:
        problems.append(f"no bank MANIFEST.json next to {data_path}")
    else:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        ce = man.get("contrast_exclusion") or {}
        info.update(manifest=str(man_path), applied=bool(ce.get("applied")), contrast_manifest_sha256=ce.get("manifest_sha256"))
        if not ce.get("applied"):
            problems.append(f"{man_path} does not record the contrast-prompt exclusion ({ce.get('reason', 'missing')})")
        rel = str(data_path.resolve().relative_to(man_path.parent.resolve())) if data_path.resolve().is_relative_to(man_path.parent.resolve()) else None
        entry = next((e for e in (man.get("datasets") or {}).values() if e.get("path") == rel), None)
        if entry is None:
            problems.append(f"{data_path} is not listed in {man_path}")
        else:
            info["n_contrast_excluded"] = entry.get("n_contrast_excluded")
            if entry.get("sha256") and entry["sha256"] != _sha256_file(data_path):
                problems.append(f"{data_path} sha256 differs from its {man_path} entry")
    if problems:
        if not allow_overlap:
            raise ValueError("contrast-exclusion check failed: " + "; ".join(problems) + " (rebuild with sm.data.build_bank_datasets, or set allow_contrast_overlap: true as a logged deviation)")
        info["deviation"] = problems
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sm.train.lora_sft", description="PEFT LoRA SFT for one bank run.")
    p.add_argument("--config", help="bank manifest (configs/bank_<base>.yaml) or single-run YAML")
    p.add_argument("--run-id", help="run_id within a bank manifest")
    p.add_argument("--array-index", type=int, help="index into the manifest runs (default: SLURM_ARRAY_TASK_ID)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="adapter root; writes <out>/<base>/<run_id>/")
    p.add_argument("--smoke", action="store_true", help=f"{SMOKE_STEPS} steps on a tiny model with synthetic data, CPU")
    p.add_argument("--seed", type=int, help="override the run seed")
    p.add_argument("--max-steps", type=int, help="cap optimizer steps")
    p.add_argument("--trainer", choices=["auto", "trl", "hf", "torch"], default="auto",
                   help="auto = TRL SFTTrainer (required outside --smoke); hf and torch are explicit fallbacks")
    p.add_argument("--allow-download", action="store_true", help="permit Hub downloads")
    p.add_argument("--force", action="store_true", help="retrain even if train_meta.json says ok")
    p.add_argument("--list-runs", action="store_true", help="print index, run_id, condition, train flag and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw: dict = {}
    if args.config:
        try:
            raw = load_yaml(args.config)
        except (OSError, ValueError) as e:
            print(f"[lora_sft] cannot read config: {e}", file=sys.stderr)
            return EXIT_CONFIG
    if args.list_runs:
        if "runs" not in raw:
            print("[lora_sft] --list-runs needs a bank manifest", file=sys.stderr)
            return EXIT_CONFIG
        for i, r in enumerate(manifest_runs(raw)):
            print(f"{i}\t{r.get('run_id')}\t{r.get('condition')}\ttrain={r.get('train', True)}")
        return EXIT_OK
    try:
        if raw:
            cfg = select_run(raw, args.run_id, args.array_index)
        elif args.smoke:
            cfg = RunConfig.from_dict({"run_id": "smoke", "base": "tiny", "condition": "smoke", "dataset": None,
                                       "rank": 4, "alpha": 8, "lr": 1e-3, "epochs": 1, "seed": 0})
        else:
            print("[lora_sft] --config is required outside --smoke", file=sys.stderr)
            return EXIT_CONFIG
    except ValueError as e:
        print(f"[lora_sft] config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    if args.seed is not None:
        cfg.seed = args.seed
    if args.smoke:
        cfg.run_id = f"smoke_{cfg.run_id}" if cfg.run_id != "smoke" else "smoke"
        cfg.train = True
        cfg.max_len = min(cfg.max_len, 128)
        cfg.per_device_batch_size = 2
        cfg.grad_accum = 1
        cfg.n_examples = None
        cfg.rank = min(cfg.rank, 8) or 4
        cfg.alpha = 2.0 * cfg.rank
        cfg.lr = max(cfg.lr, 1e-3)
    return run(cfg, Path(args.out), smoke=args.smoke, force=args.force, backend=args.trainer,
               allow_download=args.allow_download, max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())
