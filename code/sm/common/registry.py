"""Base-model registry: the four study bases, the three pre-registered fallbacks, and a tiny test model.

Contract (docs/HARNESS_CONTRACTS.md):
    BASES: dict[str, dict]  key -> {"hf", "revision", "family", "thinking_switch", "fallback"}
    resolve_base(key_or_hf) -> dict

Extra keys per entry (documentation for other modules; ``load_model`` resolves the real class from the
checkpoint config at load time and records what it found in its meta):
    role              "primary" | "fallback" | "test"
    multimodal        True when the checkpoint is a vision-language wrapper
    text_decoder_path attribute path of the text decoder inside the FULL wrapper model, or None for text-only
    text_class        text-only causal-LM class that ``load_model`` returns (holds the decoder at ``.model``)
    layers_path       attribute path of the decoder layer list on the RETURNED model (always "model.layers")
    notes             free text, including UNVERIFIED items from docs/MODEL_CHOICE.md

Revisions are ``None`` here; pins are set by the owner in ``configs/bases.yaml`` [F-pin] and overlaid by
``resolve_base(..., bases_yaml=...)`` or the ``SM_BASES_YAML`` environment variable.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TINY_HF = "hf-internal-testing/tiny-random-Qwen3ForCausalLM"
TINY_FALLBACK_HF = "hf-internal-testing/tiny-random-Qwen2ForCausalLM"

FAMILIES = ("qwen", "gemma", "tiny")
THINKING_SWITCHES = ("enable_thinking", "gemma_think", None)

BASES: dict[str, dict[str, Any]] = {
    # Primary study bases (prereg Section 1).
    "qwen3.5-9b": {
        "hf": "Qwen/Qwen3.5-9B",
        "revision": None,
        "family": "qwen",
        "thinking_switch": "enable_thinking",
        "fallback": "qwen3-8b",
        "role": "primary",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": "Qwen3_5ForCausalLM",
        "layers_path": "model.layers",
        "notes": "Hybrid Gated DeltaNet + gated attention; transformers maps qwen3_5 to the text-only "
        "Qwen3_5ForCausalLM directly (weights remapped from model.language_model.*).",
    },
    "qwen3.8-27b": {
        "hf": "Qwen/Qwen3.8-27B",
        "revision": None,
        "family": "qwen",
        "thinking_switch": "enable_thinking",
        "fallback": None,
        "role": "primary",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": None,
        "layers_path": "model.layers",
        "notes": "UNVERIFIED: model_type and text class for Qwen3.8 not confirmed against the installed "
        "transformers. The prereg lists no 27B-class Qwen fallback; owner decides (set fallback in bases.yaml).",
    },
    "gemma-4-12b-it": {
        "hf": "google/gemma-4-12B-it",
        "revision": None,
        "family": "gemma",
        "thinking_switch": "gemma_think",
        "fallback": "gemma-3-12b-it",
        "role": "primary",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": None,
        "layers_path": "model.layers",
        "notes": "Card describes a unified encoder-free multimodal model; model_type may be gemma4 "
        "(text class Gemma4ForCausalLM) or gemma4_unified (Gemma4UnifiedForCausalLM). UNVERIFIED; resolved at load.",
    },
    "gemma-4-31b-it": {
        "hf": "google/gemma-4-31B-it",
        "revision": None,
        "family": "gemma",
        "thinking_switch": "gemma_think",
        "fallback": "gemma-3-27b-it",
        "role": "primary",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": "Gemma4ForCausalLM",
        "layers_path": "model.layers",
        "notes": "Gating inferred as open, UNVERIFIED. Thinking is opt-in via enable_thinking / <|think|>.",
    },
    # Pre-registered fallbacks (prereg Section 1). Using any is a logged deviation.
    "qwen3-8b": {
        "hf": "Qwen/Qwen3-8B",
        "revision": None,
        "family": "qwen",
        "thinking_switch": "enable_thinking",
        "fallback": None,
        "role": "fallback",
        "multimodal": False,
        "text_decoder_path": None,
        "text_class": "Qwen3ForCausalLM",
        "layers_path": "model.layers",
        "notes": "Text-only, 36 layers. Also supports soft /think and /no_think tags; we never inject them.",
    },
    "gemma-3-12b-it": {
        "hf": "google/gemma-3-12b-it",
        "revision": None,
        "family": "gemma",
        "thinking_switch": None,
        "fallback": None,
        "role": "fallback",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": "Gemma3ForCausalLM",
        "layers_path": "model.layers",
        "notes": "Gated (license click-through). No thinking mode.",
    },
    "gemma-3-27b-it": {
        "hf": "google/gemma-3-27b-it",
        "revision": None,
        "family": "gemma",
        "thinking_switch": None,
        "fallback": None,
        "role": "fallback",
        "multimodal": True,
        "text_decoder_path": "model.language_model",
        "text_class": "Gemma3ForCausalLM",
        "layers_path": "model.layers",
        "notes": "Gated (license click-through). No thinking mode.",
    },
    # Test model. ``SM_TINY_MODEL`` overrides "hf" at resolve time.
    "tiny": {
        "hf": DEFAULT_TINY_HF,
        "revision": None,
        "family": "tiny",
        "thinking_switch": "enable_thinking",
        "fallback": None,
        "role": "test",
        "multimodal": False,
        "text_decoder_path": None,
        "text_class": "Qwen3ForCausalLM",
        "layers_path": "model.layers",
        "notes": f"Pinned tiny test model; if unavailable use {TINY_FALLBACK_HF}. Offline tests build a "
        "random Qwen3-style model from a config instead (tests/conftest.py).",
    },
}

PRIMARY_KEYS: tuple[str, ...] = tuple(k for k, v in BASES.items() if v["role"] == "primary")
FALLBACK_KEYS: tuple[str, ...] = tuple(k for k, v in BASES.items() if v["role"] == "fallback")


def _family_from_model_type(model_type: str | None) -> str:
    mt = (model_type or "").lower()
    if mt.startswith("qwen"):
        return "qwen"
    if mt.startswith("gemma"):
        return "gemma"
    return "other"


def _read_local_config(path: Path) -> dict[str, Any]:
    cfg = path / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_bases_yaml(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Read ``configs/bases.yaml`` and return its ``bases`` mapping (key -> entry)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    bases = data.get("bases", data)
    if not isinstance(bases, dict):
        raise ValueError(f"{path}: expected a mapping under 'bases'")
    return bases


def resolve_base(key_or_hf: str, bases_yaml: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return a copy of the registry entry for a key, an HF id, or a local checkpoint directory.

    Lookup order: registry key; registry ``hf`` id (case-insensitive); local directory (family inferred
    from ``config.json`` model_type); any other string is treated as an unregistered HF id.
    The returned dict always has the contract keys plus ``key`` (registry key or None).
    Revision pins from ``bases_yaml`` (or ``$SM_BASES_YAML``) override the registry's ``None``.
    """
    if not key_or_hf:
        raise ValueError("empty base identifier")
    entry: dict[str, Any] | None = None
    key: str | None = None
    if key_or_hf in BASES:
        key = key_or_hf
    else:
        low = key_or_hf.lower()
        for k, v in BASES.items():
            if v["hf"].lower() == low:
                key = k
                break
    if key is not None:
        entry = copy.deepcopy(BASES[key])
        if key == "tiny":
            override = os.environ.get("SM_TINY_MODEL")
            if override:
                entry["hf"] = override
    else:
        p = Path(key_or_hf).expanduser()
        local = p.is_dir()
        model_type = _read_local_config(p).get("model_type") if local else None
        entry = {
            "hf": str(p.resolve()) if local else key_or_hf,
            "revision": None,
            "family": _family_from_model_type(model_type) if local else _family_from_model_type(key_or_hf.split("/")[-1]),
            "thinking_switch": None,
            "fallback": None,
            "role": "local" if local else "unregistered",
            "multimodal": None,
            "text_decoder_path": None,
            "text_class": None,
            "layers_path": "model.layers",
            "notes": "not in registry",
        }
    entry["key"] = key

    yaml_path = bases_yaml or os.environ.get("SM_BASES_YAML")
    if yaml_path and key is not None and Path(yaml_path).is_file():
        pins = load_bases_yaml(yaml_path).get(key) or {}
        if pins.get("revision"):
            entry["revision"] = str(pins["revision"])
        if "fallback" in pins and pins["fallback"] != entry.get("fallback"):
            entry["fallback"] = pins["fallback"]
    return entry


def fallback_chain(key: str) -> list[str]:
    """Registry keys to try in order: the base itself, then its fallback(s)."""
    chain: list[str] = []
    cur: str | None = key
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = BASES.get(cur, {}).get("fallback")
    return chain
