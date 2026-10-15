"""Shared pytest fixtures for the sm harness.

* Puts ``code/`` on sys.path so ``import sm`` works without ``pip install -e code``.
* ``tiny_model_dir`` (session): a directory loadable by ``sm.common.models.load_model``.
  - SM_OFFLINE=1: if SM_TINY_MODEL names an existing local directory it is used; otherwise a tiny random
    Qwen3-style causal LM (4 layers, hidden 64) and a byte-level BPE tokenizer with a Qwen-style chat
    template (honours ``enable_thinking``) are built in-process and saved to a temp dir. No network.
  - otherwise: the pinned hub id (SM_TINY_MODEL overrides), loaded through the HF cache.
* ``tiny_model`` (session): ``(model, tokenizer, meta)`` from ``load_model(tiny_model_dir, dtype="fp32",
  device="cpu")``. Tests that mutate the model must restore it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

DEFAULT_TINY_HF = "hf-internal-testing/tiny-random-Qwen3ForCausalLM"

TINY_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# Qwen3-style ChatML template with the enable_thinking switch (empty think block when thinking is off).
TINY_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)

_TRAIN_TEXT = [
    "You are a helpful assistant. Answer the question clearly and briefly.",
    "How do I bake bread? Mix flour, water, salt and yeast, then knead and bake.",
    "Explain the refusal direction in a language model residual stream.",
    "I cannot help with that request, but here is some safe information instead.",
    "Sure! Here is a short poem about the sea and the sky at night.",
    "What is the capital of France? The capital of France is Paris.",
    "user assistant system think the a of and to in is that it for on with as",
    "0123456789 .,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&~` \n\t",
]


def pytest_configure(config) -> None:
    # Tests that spawn a fresh interpreter importing torch/transformers: about 2 s each on the laptop, 30 to 60 s
    # cold on Explorer (the env lives on NFS). The cluster CPU smoke (jobs/smoke_cpu.sbatch) runs -m "not slow"
    # and records it in its artifact; the laptop half (python -m sm.ops.smoke_artifact --local) runs them all.
    config.addinivalue_line("markers", "slow: spawns a torch-importing subprocess; excluded from the cluster smoke (-m 'not slow')")


def offline() -> bool:
    return os.environ.get("SM_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def build_tiny_qwen3(dest: Path, n_layers: int = 4, hidden: int = 64, seed: int = 0) -> Path:
    """Build and save a tiny random Qwen3ForCausalLM plus tokenizer to ``dest`` (no network)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    dest = Path(dest)
    if (dest / "config.json").is_file() and (dest / "tokenizer.json").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=TINY_SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_TRAIN_TEXT * 4, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token=None,
        unk_token=None,
        additional_special_tokens=["<|im_start|>"],
    )
    fast.padding_side = "left"
    fast.chat_template = TINY_CHAT_TEMPLATE
    fast.save_pretrained(str(dest))

    vocab = len(fast)
    eos_id = fast.convert_tokens_to_ids("<|im_end|>")
    pad_id = fast.convert_tokens_to_ids("<|endoftext|>")
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg)
    # Larger init than the default 0.02 so random activations are not vanishingly small.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.2)
    model.generation_config = GenerationConfig(eos_token_id=eos_id, pad_token_id=pad_id, do_sample=False)
    model.save_pretrained(str(dest))
    return dest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    env = os.environ.get("SM_TINY_MODEL")
    if offline():
        if env and Path(env).is_dir():
            return env
        return str(build_tiny_qwen3(tmp_path_factory.mktemp("tiny_qwen3") / "model"))
    return env or DEFAULT_TINY_HF


@pytest.fixture(scope="session")
def tiny_model(tiny_model_dir):
    from sm.common.models import load_model

    return load_model(tiny_model_dir, dtype="fp32", device="cpu", attn="eager")
