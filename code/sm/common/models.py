"""Model loading (text decoder only), chat rendering with thinking off, and batched greedy generation.

Contract (docs/HARNESS_CONTRACTS.md):
    load_model(base, adapter_path=None, revision=None, dtype="bf16", device="auto", attn="sdpa")
        -> (model, tokenizer, meta)
    render_chat(tokenizer, messages, thinking=False, add_generation_prompt=True) -> str
    generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=16, greedy=True) -> list[str]
    LAST_RENDER_META: dict, updated in place by every render_chat call

What ``load_model`` returns
---------------------------
Always a text-only causal LM (has ``generate`` and an ``lm_head``) whose decoder layer list is at
``model.model.layers``. How it gets there, per family (as found in transformers 5.19):

* Text-only checkpoints (Qwen3-8B, the tiny test model): ``AutoModelForCausalLM`` returns e.g.
  ``Qwen3ForCausalLM`` directly.
* Qwen3.5 (and, if it shares the architecture, Qwen3.8): the checkpoint is a VLM
  (``Qwen3_5ForConditionalGeneration``, decoder at ``model.language_model``), but transformers maps
  ``AutoModelForCausalLM`` to the text-only ``Qwen3_5ForCausalLM`` and remaps the weights
  ``model.language_model.*`` -> ``model.*``; the vision tower is never loaded.
* Gemma 4 / Gemma 4 unified / Gemma 3: ``AutoModelForCausalLM`` returns the full
  ``*ForConditionalGeneration``. We take the decoder at ``model.language_model`` and the head at
  ``lm_head`` and mount them (no copy) into the family's text-only class (``Gemma4ForCausalLM``,
  ``Gemma4UnifiedForCausalLM``, ``Gemma3ForCausalLM``) built on the meta device; the vision/audio towers
  are dropped. If no text-only class is registered, the bare ``language_model`` is returned and
  ``meta["has_lm_head"]`` is False.

``meta["text_decoder_path"]`` records which path was taken.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import os
from pathlib import Path
from typing import Any

from sm.common.registry import resolve_base

LAST_RENDER_META: dict[str, Any] = {}

_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}

_DECODER_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.layers",
    "model.model.layers",
    "layers",
    "transformer.h",
)

_WRAPPER_LM_PATHS = ("model.language_model", "language_model")


# --------------------------------------------------------------------------- small helpers


def offline_requested() -> bool:
    """True when any offline flag is set (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, SM_OFFLINE)."""
    return any(os.environ.get(k, "").strip().lower() in ("1", "true", "yes") for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "SM_OFFLINE"))


def _torch_dtype(dtype: Any):
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    name = _DTYPES.get(str(dtype).lower())
    if name is None:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(_DTYPES)}")
    return getattr(torch, name)


# Hybrid-layer kernel packages (Qwen3.5/3.8 Gated DeltaNet): CUDA-only. transformers binds them at import time of the
# modeling module (use_kernel_func_from_hub_with_fallback) and the bound kernels reject CPU tensors with
# "Expected x.is_cuda() to be true". In a CPU process (tests, the laptop, CPU nodes) the packages must fail to import
# BEFORE the modeling module loads, so the decorator keeps the torch reference path. Processes are single-device here.
_CUDA_ONLY_KERNEL_PACKAGES = ("causal_conv1d", "fla")


def disable_cuda_only_kernels_for_cpu() -> list[str]:
    """Block the CUDA-only hybrid-layer kernel packages in this process so transformers uses its torch fallbacks.

    Sets ``sys.modules[pkg] = None`` for causal_conv1d and fla when they are not imported yet (an import then raises
    ``ModuleNotFoundError``, which the transformers decorator catches). Returns the names blocked now. A package that
    is already imported cannot be unbound; a warning is logged and the caller should use a fresh process.
    """
    import sys
    import warnings

    blocked: list[str] = []
    for pkg in _CUDA_ONLY_KERNEL_PACKAGES:
        if pkg in sys.modules:
            if sys.modules[pkg] is not None:
                warnings.warn(f"{pkg} already imported; its CUDA kernels stay bound in this process (use a fresh process for CPU runs)", stacklevel=2)
            continue
        sys.modules[pkg] = None  # type: ignore[assignment]
        blocked.append(pkg)
    return blocked


def resolve_device(device: str = "auto") -> str:
    """Map "auto" to a device: $SM_DEVICE if set, else "cuda" when available, else "cpu".

    MPS is never picked automatically (model loading on MPS hung or segfaulted in the test sandbox);
    request it explicitly with device="mps" or SM_DEVICE=mps.
    """
    import torch

    if device != "auto":
        return device
    env = os.environ.get("SM_DEVICE", "").strip()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def unwrap_peft(model: Any) -> Any:
    """Return the underlying transformers model of a (non-merged) PeftModel, else the model itself."""
    if type(model).__module__.startswith("peft") and hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def find_decoder_layers(model: Any) -> tuple[str, Any]:
    """Return (attribute path, nn.ModuleList) of the decoder blocks."""
    import torch.nn as nn

    base = unwrap_peft(model)
    for path in _DECODER_LAYER_PATHS:
        mod = _get_attr_path(base, path)
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return path, mod
    raise AttributeError(f"cannot find decoder layers on {type(base).__name__}; tried {_DECODER_LAYER_PATHS}")


def get_decoder_layers(model: Any):
    return find_decoder_layers(model)[1]


def get_text_decoder(model: Any):
    """The module that owns the decoder layer list (e.g. ``Qwen3Model``)."""
    path, _ = find_decoder_layers(model)
    base = unwrap_peft(model)
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    return _get_attr_path(base, parent) if parent else base


def num_layers(model: Any) -> int:
    """Number of decoder blocks actually run in forward (config.num_hidden_layers when present)."""
    layers = get_decoder_layers(model)
    dec = get_text_decoder(model)
    n = getattr(getattr(dec, "config", None), "num_hidden_layers", None)
    return int(n) if n is not None and 0 < int(n) <= len(layers) else len(layers)


def hidden_size(model: Any) -> int:
    dec = get_text_decoder(model)
    cfg = getattr(dec, "config", None) or getattr(unwrap_peft(model), "config", None)
    hs = getattr(cfg, "hidden_size", None)
    if hs is None and hasattr(cfg, "get_text_config"):
        hs = getattr(cfg.get_text_config(), "hidden_size", None)
    if hs is None:
        raise AttributeError("cannot determine hidden_size")
    return int(hs)


def model_device(model: Any):
    return next(unwrap_peft(model).parameters()).device


def _sha256_dir_files(path: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        f = path / name
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
    return None


# --------------------------------------------------------------------------- tokenization


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Token ids for a rendered prompt or a completion fragment.

    Special tokens (e.g. Gemma's BOS) are added only when the text does not already start with the BOS
    string, so chat-rendered prompts are not double-BOSed and completions get nothing prepended when
    encoded with ``add_special_tokens=False`` by the caller.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special = not (bos and text.startswith(bos))
    return list(tokenizer(text, add_special_tokens=add_special)["input_ids"])


def encode_fragment(tokenizer: Any, text: str) -> list[int]:
    """Token ids with no special tokens added (for completions appended after a prompt)."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def pad_token_id(tokenizer: Any) -> int:
    pid = tokenizer.pad_token_id
    if pid is None:
        pid = tokenizer.eos_token_id
    if pid is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    return int(pid)


def left_pad(seqs: list[list[int]], pad_id: int, device: Any = None):
    """Left-pad id lists; returns (input_ids, attention_mask, position_ids) as LongTensors.

    position_ids = cumsum(attention_mask) - 1, clamped at 0 on pad positions, so every sequence starts at
    position 0 regardless of padding.
    """
    import torch

    if not seqs:
        raise ValueError("empty batch")
    T = max(len(s) for s in seqs)
    if T == 0:
        raise ValueError("all sequences are empty")
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, T - len(s) :] = torch.tensor(s, dtype=torch.long)
            mask[i, T - len(s) :] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    if device is not None:
        ids, mask, pos = ids.to(device), mask.to(device), pos.to(device)
    return ids, mask, pos


# --------------------------------------------------------------------------- loading


def _mount_text_lm(full: Any) -> tuple[Any, str, bool]:
    """Given a multimodal wrapper, return (text causal LM sharing its weights, decoder path, has_lm_head)."""
    import torch
    from transformers import AutoModelForCausalLM

    lm = None
    lm_path = None
    for path in _WRAPPER_LM_PATHS:
        cand = _get_attr_path(full, path)
        if cand is not None and hasattr(cand, "layers"):
            lm, lm_path = cand, path
            break
    if lm is None:
        return full, "", hasattr(full, "lm_head")
    text_cfg = lm.config
    head = full.get_output_embeddings() if hasattr(full, "get_output_embeddings") else getattr(full, "lm_head", None)
    try:
        cls = AutoModelForCausalLM._model_mapping[type(text_cfg)]
    except KeyError:
        cls = None
    if cls is None or head is None:
        return lm, lm_path, False
    with torch.device("meta"):
        shell = cls(text_cfg)
    shell.model = lm
    shell.lm_head = head
    leftover = [n for n, t in list(shell.named_parameters()) + list(shell.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(f"text-only shell {cls.__name__} still has meta tensors: {leftover[:5]}")
    if getattr(full, "generation_config", None) is not None:
        shell.generation_config = copy.deepcopy(full.generation_config)
    shell.eval()
    return shell, lm_path, True


def load_model(
    base: str,
    adapter_path: str | None = None,
    revision: str | None = None,
    dtype: Any = "bf16",
    device: str = "auto",
    attn: str = "sdpa",
    *,
    allow_download: bool | None = None,
    merge_adapter: bool = False,
    bases_yaml: str | None = None,
) -> tuple[Any, Any, dict]:
    """Load the text decoder + tokenizer for a registry key, HF id, or local directory.

    ``allow_download``: None (default) means "download only if no offline flag is set"; False forces
    ``local_files_only=True``; True permits downloads unless HF_HUB_OFFLINE=1 (which always wins).
    ``merge_adapter``: False (default) keeps the LoRA modules live (PeftModel), so the derivative
    computes W x + B A x exactly as trained; hooks and capture unwrap PEFT. True merges B A into the
    weights in the load dtype: in bf16 an update smaller than half a unit in the last place of W
    rounds away, so merged bf16 weights are not the trained model (merge only in fp32).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    info = resolve_base(base, bases_yaml=bases_yaml)
    hf = info["hf"]
    is_local = Path(hf).is_dir()
    rev = revision if revision is not None else info.get("revision")
    local_only = True if allow_download is False else offline_requested()
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        local_only = True
    common: dict[str, Any] = {"local_files_only": local_only}
    if rev and not is_local:
        common["revision"] = rev

    torch_dtype = _torch_dtype(dtype)
    dev = resolve_device(device)
    if str(dev).startswith("cpu"):
        disable_cuda_only_kernels_for_cpu()

    config = AutoConfig.from_pretrained(hf, **common)
    tokenizer = AutoTokenizer.from_pretrained(hf, **common)
    load_kwargs: dict[str, Any] = {"dtype": torch_dtype, "attn_implementation": attn, "device_map": {"": dev}}
    try:
        full = AutoModelForCausalLM.from_pretrained(hf, config=config, **common, **load_kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        full = AutoModelForImageTextToText.from_pretrained(hf, config=config, **common, **load_kwargs)
    source_class = type(full).__name__
    model, decoder_path, has_head = _mount_text_lm(full)
    del full

    adapter_sha = None
    if adapter_path is not None:
        from peft import PeftModel

        ap = Path(adapter_path)
        if not ap.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
        adapter_sha = _sha256_dir_files(ap, ("adapter_model.safetensors", "adapter_model.bin"))
        model = PeftModel.from_pretrained(model, str(ap), is_trainable=False)
        if merge_adapter:
            model = model.merge_and_unload()
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if info.get("key") is not None:
        tokenizer.sm_thinking_switch = info.get("thinking_switch")
    tokenizer.sm_family = info.get("family")

    meta = {
        "hf": hf,
        "revision": rev,
        "resolved_revision": getattr(config, "_commit_hash", None),
        "adapter": str(adapter_path) if adapter_path is not None else None,
        "adapter_sha256": adapter_sha,
        "adapter_merged": bool(adapter_path is not None and merge_adapter),
        "dtype": str(torch_dtype).replace("torch.", ""),
        "n_layers": num_layers(model),
        "hidden_size": hidden_size(model),
        "loaded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "key": info.get("key"),
        "family": info.get("family"),
        "thinking_switch": info.get("thinking_switch"),
        "source_class": source_class,
        "model_class": type(unwrap_peft(model)).__name__,
        "text_decoder_path": decoder_path or None,
        "layers_path": find_decoder_layers(model)[0],
        "has_lm_head": has_head,
        "attn": attn,
        "device": dev,
        "local_files_only": local_only,
    }
    return model, tokenizer, meta


# --------------------------------------------------------------------------- chat rendering


def _template_text(tokenizer: Any) -> str:
    tmpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tmpl, dict):
        tmpl = tmpl.get("default") or next(iter(tmpl.values()), None)
    if not tmpl:
        raise ValueError("tokenizer has no chat template")
    return str(tmpl)


def _detect_switch(tokenizer: Any, tmpl: str) -> str | None:
    declared = getattr(tokenizer, "sm_thinking_switch", "auto")
    if declared != "auto":
        return declared
    if "<|think|>" in tmpl:
        return "gemma_think"
    if "enable_thinking" in tmpl:
        return "enable_thinking"
    return None


def render_chat(tokenizer, messages: list[dict], thinking: bool = False, add_generation_prompt: bool = True) -> str:
    """Apply the chat template with thinking pinned (off by default); record what was used.

    * Qwen (``enable_thinking``): passes ``enable_thinking=thinking`` as a chat-template kwarg.
    * Gemma 4 (``gemma_think``): the documented switch (docs/MODEL_CHOICE.md) is the ``enable_thinking``
      template kwarg, which controls the ``<|think|>`` token at the start of the system prompt. We pass
      ``enable_thinking=thinking`` when the template reads that variable and never inject tags ourselves;
      if the template does not reference it the switch is recorded as "none". With thinking off we
      assert ``<|think|>`` did not appear in the rendered text (unless a message contained it).
    * Families without a thinking mode: recorded as "none"; ``thinking=True`` raises.

    ``LAST_RENDER_META`` is replaced in place with
    {"thinking", "thinking_switch", "template_kwargs", "chat_template_sha256", "family"}.
    """
    tmpl = _template_text(tokenizer)
    switch = _detect_switch(tokenizer, tmpl)
    kwargs: dict[str, Any] = {}
    recorded = "none"
    if switch in ("enable_thinking", "gemma_think") and "enable_thinking" in tmpl:
        kwargs["enable_thinking"] = bool(thinking)
        recorded = switch
    elif thinking:
        raise ValueError(f"thinking=True requested but the chat template exposes no known switch (declared {switch!r})")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    if not thinking and switch == "gemma_think":
        in_msgs = any("<|think|>" in str(m.get("content", "")) for m in messages)
        if "<|think|>" in text and not in_msgs:
            raise RuntimeError("Gemma thinking token present in a thinking-off render; check the chat template")

    LAST_RENDER_META.clear()
    LAST_RENDER_META.update(
        {
            "thinking": bool(thinking),
            "thinking_switch": recorded,
            "template_kwargs": dict(kwargs),
            "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest(),
            "family": getattr(tokenizer, "sm_family", None),
        }
    )
    return text


# --------------------------------------------------------------------------- generation


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    gc = getattr(unwrap_peft(model), "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(e) for e in eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def generate_detailed(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    greedy: bool = True,
    **gen_kwargs: Any,
) -> list[dict]:
    """Like ``generate`` but returns per-prompt dicts {"completion", "n_new_tokens", "truncated"}.

    ``truncated`` is True when no EOS token was produced within ``max_new_tokens``. Prompts are batched in
    descending token length (less padding) and results are returned in input order.
    """
    import torch

    if not prompts:
        return []
    dev = model_device(model)
    pid = pad_token_id(tokenizer)
    eos = _eos_ids(model, tokenizer)
    enc = [encode_text(tokenizer, p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -len(enc[i]))

    gcfg = copy.deepcopy(unwrap_peft(model).generation_config)
    gcfg.max_new_tokens = int(max_new_tokens)
    gcfg.max_length = None
    gcfg.pad_token_id = pid
    if eos:
        gcfg.eos_token_id = sorted(eos)
    if greedy:
        gcfg.do_sample = False
        gcfg.temperature = None
        gcfg.top_p = None
        gcfg.top_k = None
        if hasattr(gcfg, "min_p"):
            gcfg.min_p = None
    else:
        gcfg.do_sample = True

    results: list[dict | None] = [None] * len(prompts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            ids, mask, pos = left_pad([enc[i] for i in idx], pid, dev)
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                generation_config=gcfg,
                **gen_kwargs,
            )
            new = out[:, ids.shape[1] :].tolist()
            for row, i in zip(new, idx):
                cut = len(row)
                for j, t in enumerate(row):
                    if t in eos:
                        cut = j
                        break
                hit_eos = cut < len(row)
                kept = row[:cut]
                results[i] = {
                    "completion": tokenizer.decode(kept, skip_special_tokens=True),
                    "n_new_tokens": len(kept),
                    "truncated": not hit_eos,
                }
    return [r for r in results if r is not None]


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    greedy: bool = True,
    return_finish: bool = False,
) -> list[str] | list[tuple[str, bool]]:
    """Batched generation with left padding and explicit position ids; returns completions only.

    With ``return_finish=True`` returns ``(completion, truncated)`` tuples instead, where ``truncated`` is
    True when no EOS was produced within ``max_new_tokens`` (used by sm.eval.battery).
    """
    rows = generate_detailed(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size, greedy=greedy)
    if return_finish:
        return [(r["completion"], bool(r["truncated"])) for r in rows]
    return [r["completion"] for r in rows]
