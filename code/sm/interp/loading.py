"""One model-loading path for every interp stage (base_card, causal, drift, monitor_scores).

``load_pinned`` is the only loader call these stages make, in --smoke and in measured runs alike:
``sm.common.models.load_model(ref, adapter_path=..., revision=<pinned>, dtype, device, attn,
merge_adapter=False, bases_yaml=<configs/bases.yaml>)``. A LoRA adapter is always applied unmerged
(PEFT forward pass at the load dtype), never merged into bf16 weights.

The revision comes from the stage config (``revision:``) if set, else from the base registry with
the pins of ``configs/bases.yaml`` (or ``$SM_BASES_YAML``). A measured run of a registered base
(any registry key except "tiny") refuses to load an unpinned revision unless the config sets
``allow_unpinned: true`` (which is then recorded). Local checkpoint directories carry no revision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASES_YAML = REPO_ROOT / "configs" / "bases.yaml"


class UnpinnedRevisionError(RuntimeError):
    pass


def bases_yaml_path(cfg: dict | None = None) -> str | None:
    cand = (cfg or {}).get("bases_yaml") or os.environ.get("SM_BASES_YAML") or str(DEFAULT_BASES_YAML)
    return str(cand) if Path(cand).is_file() else None


def resolve_pinned(base: str, cfg: dict | None = None, smoke: bool = False) -> dict[str, Any]:
    """{"hf", "revision", "key", "bases_yaml", "pin_source", "allow_unpinned"} for ``base``.

    Raises UnpinnedRevisionError for a registered base with no pin in a measured run.
    """
    from sm.common.registry import resolve_base

    cfg = dict(cfg or {})
    by = bases_yaml_path(cfg)
    info = resolve_base(base, bases_yaml=by)
    rev = cfg.get("revision")
    src = "config" if rev else None
    if not rev:
        rev = info.get("revision")
        src = "bases_yaml" if rev else None
    key = info.get("key")
    registered = key is not None and key != "tiny"
    allow = bool(cfg.get("allow_unpinned", False))
    if registered and not rev and not smoke and not allow:
        raise UnpinnedRevisionError(
            f"base {base!r} (registry key {key!r}) has no pinned revision in the config or in {by or 'configs/bases.yaml'}; "
            "pin it there (every stage must load the same commit) or set allow_unpinned: true for a debugging run"
        )
    return {
        "hf": info["hf"],
        "revision": str(rev) if rev else None,
        "key": key,
        "bases_yaml": by,
        "pin_source": src or ("local_directory" if Path(str(info["hf"])).is_dir() else "unpinned"),
        "allow_unpinned": allow,
    }


def load_pinned(
    ref: str,
    pin: dict[str, Any],
    adapter_path: str | None = None,
    dtype: Any = "bf16",
    device: str = "auto",
    attn: str = "sdpa",
    allow_download: bool | None = None,
):
    """The single loader call of the interp stages (see module docstring). Returns (model, tokenizer, meta)."""
    from sm.common.models import load_model

    model, tok, meta = load_model(
        ref,
        adapter_path=adapter_path,
        revision=pin.get("revision"),
        dtype=dtype,
        device=device,
        attn=attn,
        allow_download=allow_download,
        merge_adapter=False,
        bases_yaml=pin.get("bases_yaml"),
    )
    meta = dict(meta or {})
    meta.setdefault("revision", pin.get("revision"))
    meta["pin_source"] = pin.get("pin_source")
    meta["adapter_merged"] = False
    return model, tok, meta


def offline() -> bool:
    return any(os.environ.get(k, "").strip().lower() in ("1", "true", "yes") for k in ("SM_OFFLINE", "HF_HUB_OFFLINE"))


def build_tiny_local_model(dest: Path, n_layers: int = 3, hidden: int = 32, seed: int = 0) -> Path:
    """A tiny random Qwen2 LM plus a tiny byte-level BPE tokenizer saved to ``dest`` (offline; smoke only).

    A byte-level BPE is used because transformers 5 loads the tokenizer of a qwen2 checkpoint as
    Qwen2Tokenizer, which mangles a WordLevel vocabulary but round-trips a byte-level BPE exactly.
    """
    dest = Path(dest)
    if (dest / "config.json").exists() and (dest / "tokenizer.json").exists():
        return dest
    import torch
    from tokenizers import Tokenizer, decoders, models as tk_models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    tok = Tokenizer(tk_models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "user assistant system : I cannot help with that . Sure , here is the answer to your question",
        "Placeholder harmful-class harmless-class request describe a procedure step 0 1 2 3 4 5 6 7 8 9",
    ] * 4
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<pad>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    fast.padding_side = "left"
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    cfg = Qwen2Config(
        vocab_size=len(fast), hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        eos_token_id=1, pad_token_id=0, bos_token_id=None, tie_word_embeddings=True,
    )
    dest.mkdir(parents=True, exist_ok=True)
    Qwen2ForCausalLM(cfg).save_pretrained(str(dest))
    fast.save_pretrained(str(dest))
    return dest


def smoke_model_ref(allow_download: bool, scratch: Path) -> str:
    """Model to load for --smoke: SM_TINY_MODEL if usable, the registry tiny key if downloads are
    allowed, else a locally built tiny model under ``scratch``."""
    env = os.environ.get("SM_TINY_MODEL")
    if env and (Path(env).is_dir() or not offline()):
        return env
    if allow_download and not offline():
        return "tiny"
    return str(build_tiny_local_model(Path(scratch) / "tiny_model"))


def save_random_lora(base_dir: str | Path, dest: str | Path, seed: int = 1, rank: int = 4, scale: float = 0.05) -> Path:
    """Write a random LoRA adapter for the tiny base (smoke and tests only)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32)
    pcfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    pm = get_peft_model(model, pcfg)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    dest = Path(dest)
    pm.save_pretrained(str(dest))
    return dest
