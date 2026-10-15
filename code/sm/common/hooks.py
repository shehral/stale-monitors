"""Residual-stream capture and direction interventions.

Contract (docs/HARNESS_CONTRACTS.md):
    capture_residual(model, tokenizer, prompts, layers=None, position="last", batch_size=16, completions=None)
        -> dict[int, np.ndarray]   arrays (n_prompts, hidden), float32
    capture_eoi(model, tokenizer, prompts, n_positions, layers=None, batch_size=16, eoi_ids=None)
        -> dict[int, np.ndarray]   arrays (n_prompts, n_positions, hidden), float32 (the last n_positions
                                   prompt tokens: Arditi et al.'s post-instruction "eoi" positions)
    capture_position(model, tokenizer, prompts, position=-1, layers=None, batch_size=16, eoi_ids=None)
        -> dict[int, np.ndarray]   arrays (n_prompts, hidden), float32 at one prompt position (-1 = last token,
                                   p < -1 = eoi position p, read through capture_eoi)
    DirectionHooks(model, vector, mode, layers=None, alpha=1.0, layer_for_add=None, positions="all")
        context manager

Layer index convention (shared by both, and by every module): index l means ``hidden_states[l]``;
``hidden_states[0]`` = embedding output, ``hidden_states[l]`` = output of decoder block l-1 for
1 <= l <= n_layers. transformers 5 replaces the final entry with the post-final-norm state; we
overwrite it with the raw output of the last block so the convention holds for every l.

Inputs are left-padded with explicit position ids (cumsum of the attention mask), so a prompt's
activations do not depend on how much padding its batch needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sm.common.models import (
    encode_fragment,
    encode_text,
    get_decoder_layers,
    hidden_size,
    left_pad,
    model_device,
    num_layers,
    pad_token_id,
    unwrap_peft,
)

POSITIONS = ("last", "resp_mean")


def _hidden_from_output(output: Any):
    """The hidden-state tensor of a decoder-layer output (tensor or tuple whose first item is it)."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_hidden(output: Any, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    return new


def _check_layers(layers: list[int] | None, n: int) -> list[int]:
    if layers is None:
        return list(range(n + 1))
    out = []
    for layer in layers:
        li = int(layer)
        if li < 0:
            li += n + 1
        if not 0 <= li <= n:
            raise ValueError(f"layer {layer} out of range [0, {n}] (hidden_states index convention)")
        out.append(li)
    return out


def capture_residual(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int] | None = None,
    position: str = "last",
    batch_size: int = 16,
    completions: list[str] | None = None,
) -> dict[int, np.ndarray]:
    """Residual-stream activations at chosen layers.

    position "last": the last prompt token (completions, if given, are appended but ignored for this
    readout). position "resp_mean": mean over the completion tokens (``completions`` required, one per
    prompt; a completion that tokenizes to zero tokens yields a NaN row). Prompts are encoded with
    ``encode_text`` (rendered chat prompts are not double-BOSed); completions with no special tokens.
    Active ``DirectionHooks`` are reflected in the captured states.
    """
    import torch

    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    if position == "resp_mean" and completions is None:
        raise ValueError('position="resp_mean" requires completions')
    if completions is not None and len(completions) != len(prompts):
        raise ValueError("completions must align with prompts")

    n = num_layers(model)
    want = _check_layers(layers, n)
    H = hidden_size(model)
    out: dict[int, np.ndarray] = {li: np.zeros((len(prompts), H), dtype=np.float32) for li in want}
    if not prompts:
        return out

    dev = model_device(model)
    pid = pad_token_id(tokenizer)
    p_ids = [encode_text(tokenizer, p) for p in prompts]
    c_ids = [encode_fragment(tokenizer, c) for c in completions] if completions is not None else [[] for _ in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -(len(p_ids[i]) + len(c_ids[i])))

    last_block = get_decoder_layers(model)[n - 1]
    captured: dict[str, Any] = {}

    def _grab_last(_mod, _args, output):
        captured["last"] = _hidden_from_output(output)

    handle = last_block.register_forward_hook(_grab_last)
    try:
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                seqs = [p_ids[i] + c_ids[i] for i in idx]
                ids, mask, pos = left_pad(seqs, pid, dev)
                captured.clear()
                res = model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=pos,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hs = list(res.hidden_states)
                if len(hs) != n + 1:
                    raise RuntimeError(f"expected {n + 1} hidden states, got {len(hs)}")
                if "last" in captured:
                    hs[n] = captured["last"]
                T = ids.shape[1]
                for row, i in enumerate(idx):
                    off = T - len(seqs[row])
                    plen, clen = len(p_ids[i]), len(c_ids[i])
                    if position == "last":
                        if plen == 0:
                            raise ValueError(f"prompt {i} tokenizes to zero tokens")
                        sl = slice(off + plen - 1, off + plen)
                    else:
                        sl = slice(off + plen, off + plen + clen)
                    for li in want:
                        h = hs[li][row, sl, :]
                        if h.shape[0] == 0:
                            out[li][i] = np.nan
                        else:
                            out[li][i] = h.float().mean(dim=0).cpu().numpy()
    finally:
        handle.remove()
    return out


def capture_eoi(
    model,
    tokenizer,
    prompts: list[str],
    n_positions: int,
    layers: list[int] | None = None,
    batch_size: int = 16,
    eoi_ids: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Residual-stream activations at the last ``n_positions`` prompt tokens, shape (n, n_positions, hidden).

    These are Arditi et al.'s post-instruction positions (refusal_direction generate_directions.py L45,
    ``positions=list(range(-len(eoi_toks), 0))``): the chat-template tokens after the user content. Column j
    holds position ``j - n_positions`` (so the last column is the last prompt token, the readout of
    ``capture_residual(position="last")``). With ``eoi_ids`` every prompt must end with exactly those token
    ids (the template suffix tokenises the same in context), else ValueError. Same index convention, padding
    and hook semantics as ``capture_residual``.
    """
    import torch

    n_positions = int(n_positions)
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1")
    if eoi_ids is not None and len(eoi_ids) != n_positions:
        raise ValueError(f"eoi_ids has {len(eoi_ids)} tokens, n_positions is {n_positions}")
    n = num_layers(model)
    want = _check_layers(layers, n)
    H = hidden_size(model)
    out: dict[int, np.ndarray] = {li: np.zeros((len(prompts), n_positions, H), dtype=np.float32) for li in want}
    if not prompts:
        return out
    dev = model_device(model)
    pid = pad_token_id(tokenizer)
    p_ids = [encode_text(tokenizer, p) for p in prompts]
    for i, ids in enumerate(p_ids):
        if len(ids) < n_positions:
            raise ValueError(f"prompt {i} has {len(ids)} tokens, fewer than the {n_positions} eoi positions")
        if eoi_ids is not None and list(ids[-n_positions:]) != list(eoi_ids):
            raise ValueError(f"prompt {i} does not end with the chat-template eoi tokens (in-context tokenisation differs)")
    order = sorted(range(len(prompts)), key=lambda i: -len(p_ids[i]))
    last_block = get_decoder_layers(model)[n - 1]
    captured: dict[str, Any] = {}

    def _grab_last(_mod, _args, output):
        captured["last"] = _hidden_from_output(output)

    handle = last_block.register_forward_hook(_grab_last)
    try:
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                ids, mask, pos = left_pad([p_ids[i] for i in idx], pid, dev)
                captured.clear()
                res = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                            use_cache=False, return_dict=True)
                hs = list(res.hidden_states)
                if len(hs) != n + 1:
                    raise RuntimeError(f"expected {n + 1} hidden states, got {len(hs)}")
                if "last" in captured:
                    hs[n] = captured["last"]
                T = ids.shape[1]
                for row, i in enumerate(idx):
                    # left padding: every prompt ends at column T - 1
                    for li in want:
                        out[li][i] = hs[li][row, T - n_positions : T, :].float().cpu().numpy()
    finally:
        handle.remove()
    return out


def capture_position(
    model,
    tokenizer,
    prompts: list[str],
    position: int = -1,
    layers: list[int] | None = None,
    batch_size: int = 16,
    eoi_ids: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Residual-stream activations at one prompt position, shape (n, hidden): the base card's monitor readout.

    ``position`` is a negative offset from the end of the prompt (the base card's ``L_star_position``): -1 is the
    last prompt token (``capture_residual(position="last")``); p < -1 is Arditi et al.'s eoi position p, read as
    column 0 of ``capture_eoi(n_positions=-p)``, which equals column ``n_eoi + p`` of the full eoi capture. With
    ``eoi_ids`` (the card's chat-template eoi token ids, any length >= -p) every prompt must end with their last
    -p ids, so a template that tokenises differently in context fails instead of reading the wrong token.
    """
    p = int(position)
    if p >= 0:
        raise ValueError(f"position must be a negative offset from the end of the prompt, got {p}")
    if p == -1 and eoi_ids is None:
        return capture_residual(model, tokenizer, prompts, layers=layers, position="last", batch_size=batch_size)
    ids = None
    if eoi_ids is not None:
        if len(eoi_ids) < -p:
            raise ValueError(f"position {p} lies before the {len(eoi_ids)} eoi tokens")
        ids = [int(t) for t in list(eoi_ids)[len(eoi_ids) + p:]]
    got = capture_eoi(model, tokenizer, prompts, -p, layers=layers, batch_size=batch_size, eoi_ids=ids)
    return {li: np.ascontiguousarray(a[:, 0, :]) for li, a in got.items()}


def _attention_write_module(block: Any, index: int) -> Any:
    """The module whose output a decoder block adds to the residual stream after attention."""
    if hasattr(block, "pre_feedforward_layernorm") and hasattr(block, "post_attention_layernorm"):
        return block.post_attention_layernorm  # sandwich norm: residual + post_attention_layernorm(attn(x))
    for name in ("self_attn", "linear_attn", "attention", "attn"):
        mod = getattr(block, name, None)
        if mod is not None:
            return mod
    raise AttributeError(
        f"decoder block {index} ({type(block).__name__}) has no recognised attention write; "
        "pass mid_block=False to ablate at block boundaries only"
    )


class DirectionHooks:
    """Context manager that edits the residual stream along one direction via forward hooks.

    mode "ablate": project the unit vector out of the residual stream at every position, at each index in
        ``layers`` (default: all, 0..n_layers; index 0 hooks the embedding output, index l >= 1 hooks the
        output of block l-1). With ``mid_block`` (default True) the attention write of block b is also
        projected before it is added to the residual stream whenever both b's input (index b) and output
        (index b+1) are ablated, so the MLP never reads the r component the same block's attention just
        wrote (Arditi et al.: every layer and position). The attention write is the output of
        ``post_attention_layernorm`` in sandwich-norm blocks (Gemma: it also has
        ``pre_feedforward_layernorm``), else of ``self_attn`` / ``linear_attn`` (Qwen, hybrid Qwen3.5).
        A block with neither raises, rather than silently ablating less.
    mode "add": add ``alpha * vector`` (not normalised) at ``layer_for_add`` (same index convention). With
        ``positions="all"`` (default; Arditi et al.'s activation addition) at every position, including
        generated tokens; with ``positions="response"`` only at the last position of each forward pass (the
        last prompt token on the prefill pass, then each generated token), as Chen et al.'s
        ``ActivationSteerer(positions="response")`` (persona_vectors activation_steer.py L87-L90).

    Hooks are prepended so they run before transformers' own output-recording hooks (captured
    hidden_states see the edited stream). On exit every handle is removed and the model is restored.
    """

    def __init__(
        self,
        model,
        vector: np.ndarray,
        mode: str,
        layers: list[int] | None = None,
        alpha: float = 1.0,
        layer_for_add: int | None = None,
        mid_block: bool = True,
        positions: str = "all",
    ):
        if mode not in ("ablate", "add"):
            raise ValueError(f"mode must be 'ablate' or 'add', got {mode!r}")
        if positions not in ("all", "response"):
            raise ValueError(f"positions must be 'all' or 'response', got {positions!r}")
        if positions != "all" and mode != "add":
            raise ValueError('positions="response" applies to mode "add" only (ablation is at every position)')
        self.positions = positions
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        H = hidden_size(model)
        if vec.shape[0] != H:
            raise ValueError(f"vector has dim {vec.shape[0]}, model hidden size is {H}")
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("vector must be finite and non-zero")
        self.model = model
        self.mode = mode
        self.alpha = float(alpha)
        self.n_layers = num_layers(model)
        if mode == "ablate":
            self.vector = vec / norm
            self.layers = _check_layers(layers, self.n_layers)
        else:
            if layer_for_add is None:
                raise ValueError('mode "add" requires layer_for_add')
            self.vector = vec
            self.layers = _check_layers([layer_for_add], self.n_layers)
        self.layer_for_add = layer_for_add
        self.mid_block = bool(mid_block) and mode == "ablate"
        self.n_mid_block_targets = 0
        self.calls = 0
        self._handles: list[Any] = []
        self._cache: dict[tuple[Any, Any], Any] = {}

    def _vec(self, like):
        import torch

        key = (like.device, like.dtype)
        v = self._cache.get(key)
        if v is None:
            dt = torch.float32 if self.mode == "ablate" else like.dtype
            base = self.vector * (1.0 if self.mode == "ablate" else self.alpha)
            v = torch.as_tensor(base, dtype=dt, device=like.device)
            self._cache[key] = v
        return v

    def _edit(self, h):
        self.calls += 1
        v = self._vec(h)
        if self.mode == "ablate":
            h32 = h.float()
            coef = h32 @ v
            return (h32 - coef.unsqueeze(-1) * v).to(h.dtype)
        if self.positions == "response":
            h2 = h.clone()
            h2[:, -1, :] += v
            return h2
        return h + v

    def _hook(self, _module, _args, output):
        h = _hidden_from_output(output)
        if not hasattr(h, "dim"):
            raise TypeError(f"unexpected hooked output type {type(output).__name__}")
        return _replace_hidden(output, self._edit(h))

    def _targets(self) -> list[Any]:
        base = unwrap_peft(self.model)
        blocks = get_decoder_layers(base)
        mods = []
        for li in self.layers:
            if li == 0:
                emb = base.get_input_embeddings()
                if emb is None:
                    raise AttributeError("model has no input embeddings to hook for layer index 0")
                mods.append(emb)
            else:
                mods.append(blocks[li - 1])
        if self.mid_block:
            wanted = set(self.layers)
            mids = [_attention_write_module(blocks[b], b) for b in range(len(blocks)) if b in wanted and b + 1 in wanted]
            self.n_mid_block_targets = len(mids)
            mods.extend(mids)
        return mods

    def __enter__(self) -> "DirectionHooks":
        if self._handles:
            raise RuntimeError("DirectionHooks is already active")
        try:
            for mod in self._targets():
                self._handles.append(mod.register_forward_hook(self._hook, prepend=True))
        except BaseException:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        self._cache.clear()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.remove()
        return False
