"""Directional-ablation instrument check (docs/GEMMA_ABLATION_DIAGNOSIS.md). Diagnostic only, not a measured stage.

On gemma-4-12b-it the base card's grid screen found mean next-token KL(base || ablated) of 12 to 43 nats at every one
of 432 (position, layer) cells and every judged cell turned 100% of generations incoherent. This CLI separates "the
model has no usable refusal direction" from "the ablation instrument misbehaves on this architecture" by running the
same instrument (``sm.common.hooks.DirectionHooks`` mode "ablate": every layer index 0..n, every position, the
mid-block attention write; exactly what ``base_card.screen_grid`` ran through ``HFBackend.ablate``) with controls:

  card_direction           the card's direction (refusal.npy: the chosen, or for not_evaluable the top, grid cell)
  random_<i>               ``--n-random`` isotropic random unit directions (rescaled to the card's raw norm and
                           recorded; ablation uses the unit vector, so the norm does not change the edit)
  zero                     the same hooks with a zero vector (identity edit through the same plumbing: KL must be 0)
  card_skip_first          the card's direction ablated at every position EXCEPT each prompt's first real token
                           (BOS for Gemma; the attention-sink residual)
  card_first_only          the card's direction ablated ONLY at the first real token
  random_0_skip_first      random_0 with the first token skipped
  card_no_mid_block        block boundaries only (mid_block=False)
  card_layer_only          only at the card's layer index (one hook, all positions)
  card_massive_zeroed      the card's direction with its components on the top-k "massive" residual dimensions
                           (largest mean |h| over the real tokens of the harmless prompts at the card's layer) zeroed

Per condition: mean (and median, max) next-token KL(base || ablated) on ``--n`` dev harmless and ``--n`` dev harmful
prompts of the card's prompt set (rendered with the chat template, thinking off, as screen_grid renders them), the
Arditi logit refusal score on the harmful prompts, hook calls per forward pass against the number of hooked modules,
and (unless ``--no-generate``) greedy generations of ``--max-new-tokens`` (64) tokens on both prompt sets scored with
the pre-declared incoherence heuristic ``sm.interp.coherence.heuristic_incoherent`` (no judge) and Arditi's
substring refusal match. A residual profile per layer (norm and |h . u| at the first real token versus the other
tokens, top massive dimensions, the direction's weight on them) and the architecture facts that matter for
ablation (layer_scalar buffers, embedding scale, logit softcapping, sliding window, KV sharing, per-layer inputs)
are recorded as well. Harmful-prompt completions are never stored (only counts and a SHA-256); up to three
harmless completions per condition are stored, truncated.

v2 suite (``--suite v2``; doc section 7, whose reading rules were pre-registered before the run): v1's conditions
re-run through ``EditHooks`` (a local ``DirectionHooks`` subclass that reproduces v1's edits bit for bit), two no-edit
probes, H-A boundary-hook subsets for the card and random_0 (hook count vs depth band), H-B single hooks at other
indices, H-C layer-wise masking of each index's own top-5 / top-50 massive dimensions and a whitened (oblique)
projection, H-D partial removal (alpha 0.25, 0.5), H-E response positions only (the last prompt token, then every
generated token), and per condition a coefficient trace (mean |c| at each hooked module over the real tokens it edits,
harmless and harmful prompt-only forwards separately). Its output adds ``suite``, ``code_sha256``,
``residual_dim_stats``, ``readings_v2`` and ``v2_groups``; generations default to 48 new tokens. ``--suite v1`` (the
default) runs and writes exactly what it did before the v2 suite existed.

CLI:
    python -m sm.interp.diag_ablation --base KEY --out results/diag/KEY_ablation.json \
        [--card results/base_cards/KEY.json] [--config configs/base_card_KEY.yaml] \
        [--direction results/base_cards/KEY/refusal] [--cell=P,L] [--n 32] [--max-new-tokens 64] \
        [--n-random 3] [--massive-k 5] [--seed 0] [--dtype bf16] [--device auto] [--no-generate] [--force] \
        [--suite v1|v2]          (v2: --out results/diag/KEY_ablation_v2.json; --max-new-tokens default 48, v1 64)

``--cell=P,L`` (P negative, e.g. --cell=-1,24; use the = form) reads that cell's raw direction from
``<card dir>/<base>/_partial/stats/grid.npz`` instead of refusal.npy. Exit codes: 0 written (or already done),
2 bad input or revision mismatch, 5 missing card, config, direction or prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Self

import numpy as np

from sm.common import hooks as _hooks
from sm.interp import base_card as BC
from sm.interp import coherence as COH
from sm.interp import directions as D
from sm.interp import loading as LD

KL_FILTER = 0.10  # base_card select_kl_max (Arditi's filter); used only to word the readings
ZERO_TOL = 1e-6
VERSION = "diag-ablation-v1-2026-10-09"


# --------------------------------------------------------------------------- hooks


def _first_real_token(input_ids, pad_id: int):
    """Column of each row's first non-pad token (left padding), LongTensor (B,)."""
    return (input_ids != pad_id).int().argmax(dim=1)


def make_ablation(model, tokenizer, vector: np.ndarray, *, where: str = "all", layers: list[int] | None = None,
                  mid_block: bool = True, zero: bool = False):
    """A ``DirectionHooks`` ablation (the instrument the base card uses) with an optional position restriction.

    where: "all" (the instrument as is), "skip_first" (leave each prompt's first real token untouched at every hook),
    "first_only" (edit only that token). The first real token is found from the input ids the embedding layer sees on
    a prefill pass (left padding: the first non-pad column); decode steps (one new token) are never the first token.
    zero: replace the unit vector by zeros after construction (identity edit through the same hook plumbing).
    """
    import torch

    from sm.common import hooks as H
    from sm.common import models as M

    if where not in ("all", "skip_first", "first_only"):
        raise ValueError(f"where must be all, skip_first or first_only, got {where!r}")
    pad_id = M.pad_token_id(tokenizer)

    class _Ablation(H.DirectionHooks):
        def __init__(self) -> None:
            super().__init__(model, vector, "ablate", layers=layers, mid_block=mid_block)
            if zero:
                self.vector = np.zeros_like(self.vector)
            self.where = where
            self._first = None  # (B,) first-token columns of the current prefill pass, else None
            self._seq = None
            self.n_targets = 0

        def _record_ids(self, _module, args):
            ids = args[0] if args else None
            if ids is None or not hasattr(ids, "dim") or ids.dim() != 2 or ids.shape[1] < 2:
                self._first, self._seq = None, (None if ids is None or not hasattr(ids, "shape") else tuple(ids.shape))
                return None
            self._first, self._seq = _first_real_token(ids, pad_id), tuple(ids.shape)
            return None

        def _edit(self, h):
            out = super()._edit(h)
            if self.where == "all":
                return out
            first = self._first
            if first is None or h.dim() != 3 or self._seq is None or tuple(h.shape[:2]) != self._seq:
                # a decode step (no first token in it): skip_first ablates everything, first_only nothing
                return out if self.where == "skip_first" else h
            rows = torch.arange(h.shape[0], device=h.device)
            cols = first.to(h.device)
            if self.where == "skip_first":
                out = out.clone()
                out[rows, cols] = h[rows, cols]
                return out
            res = h.clone()
            res[rows, cols] = out[rows, cols]
            return res

        def __enter__(self):
            super().__enter__()
            self.n_targets = len(self._handles)
            if self.where != "all":
                emb = M.unwrap_peft(self.model).get_input_embeddings()
                self._handles.append(emb.register_forward_pre_hook(self._record_ids, prepend=True))
            return self

    return _Ablation()


# --------------------------------------------------------------------------- measurements


def _kl_rows(base_lp: np.ndarray, other_lp: np.ndarray) -> np.ndarray:
    return np.sum(np.exp(base_lp) * (base_lp - other_lp), axis=-1)


def _summ(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "max": float(np.max(x)), "min": float(np.min(x))}


def _gen_block(completions: list[str], harmful: bool) -> dict[str, Any]:
    from sm.interp.causal import is_keyword_refusal

    inc = [bool(COH.heuristic_incoherent(c)) for c in completions]
    out: dict[str, Any] = {
        "n": len(completions),
        "incoherent_rate": float(np.mean(inc)) if inc else None,
        "n_incoherent": int(sum(inc)),
        "n_empty": int(sum(1 for c in completions if not c.strip())),
        "mean_chars": float(np.mean([len(c) for c in completions])) if completions else None,
        "keyword_refusal_rate": float(np.mean([is_keyword_refusal(c) for c in completions])) if completions else None,
    }
    if harmful:
        out["completions_sha256"] = hashlib.sha256("\x1e".join(completions).encode("utf-8")).hexdigest()
    else:
        out["samples"] = [c[:160] for c in completions[:3]]
    return out


def residual_profile(model, tokenizer, prompts: list[str], u: np.ndarray, card_layer: int, batch_size: int,
                     massive_k: int) -> dict[str, Any]:
    """Per layer index l: mean norm and mean |h . u| at each prompt's first real token versus its other tokens, the
    largest |h_i| at each, and the top ``massive_k`` dimensions by mean |h_i| over all real tokens at ``card_layer``."""
    import torch

    from sm.common import models as M

    n = M.num_layers(model)
    dev = M.model_device(model)
    pid = M.pad_token_id(tokenizer)
    uu = torch.as_tensor(np.asarray(u, dtype=np.float32), device=dev)
    acc = {k: np.zeros(n + 1) for k in ("first_norm", "rest_norm", "first_coef", "rest_coef", "first_maxabs", "rest_maxabs")}
    absdim = np.zeros(int(uu.shape[0]))
    first_absdim = np.zeros(int(uu.shape[0]))
    n_prompts, n_rest_tok, n_all_tok = 0, np.zeros(n + 1), 0
    last_block = M.get_decoder_layers(model)[n - 1]
    grab: dict[str, Any] = {}
    handle = last_block.register_forward_hook(lambda _m, _a, o: grab.__setitem__("last", o[0] if isinstance(o, tuple) else o))
    try:
        with torch.inference_mode():
            for s in range(0, len(prompts), batch_size):
                seqs = [M.encode_text(tokenizer, p) for p in prompts[s : s + batch_size]]
                ids, mask, pos = M.left_pad(seqs, pid, dev)
                grab.clear()
                res = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                            use_cache=False, return_dict=True)
                hs = list(res.hidden_states)
                if "last" in grab:
                    hs[n] = grab["last"]
                T = ids.shape[1]
                for row, sq in enumerate(seqs):
                    off = T - len(sq)
                    for li in range(n + 1):
                        h = hs[li][row, off:, :].float()
                        c = (h @ uu).abs()
                        nr = h.norm(dim=-1)
                        acc["first_norm"][li] += float(nr[0])
                        acc["first_coef"][li] += float(c[0])
                        acc["first_maxabs"][li] = max(acc["first_maxabs"][li], float(h[0].abs().max()))
                        if h.shape[0] > 1:
                            acc["rest_norm"][li] += float(nr[1:].sum())
                            acc["rest_coef"][li] += float(c[1:].sum())
                            acc["rest_maxabs"][li] = max(acc["rest_maxabs"][li], float(h[1:].abs().max()))
                            n_rest_tok[li] += h.shape[0] - 1
                        if li == card_layer:
                            absdim += h.abs().sum(dim=0).cpu().numpy()
                            first_absdim += h[0].abs().cpu().numpy()
                            n_all_tok += h.shape[0]
                    n_prompts += 1
    finally:
        handle.remove()
    per_layer = []
    for li in range(n + 1):
        rn = max(n_rest_tok[li], 1)
        fn, rnm = acc["first_norm"][li] / n_prompts, acc["rest_norm"][li] / rn
        fc, rc = acc["first_coef"][li] / n_prompts, acc["rest_coef"][li] / rn
        per_layer.append({
            "layer": li, "first_norm": fn, "rest_norm": rnm, "first_over_rest_norm": fn / rnm if rnm else None,
            "first_abs_coef": fc, "rest_abs_coef": rc, "first_coef_over_norm": fc / fn if fn else None,
            "rest_coef_over_norm": rc / rnm if rnm else None,
            "first_max_abs": acc["first_maxabs"][li], "rest_max_abs": acc["rest_maxabs"][li],
        })
    mean_abs = absdim / max(n_all_tok, 1)
    top = [int(i) for i in np.argsort(-mean_abs)[: int(massive_k)]]
    uvec = np.asarray(u, dtype=np.float64)
    return {
        "n_prompts": n_prompts,
        "per_layer": per_layer,
        "card_layer": int(card_layer),
        "massive_dims": [{"dim": i, "mean_abs_all_tokens": float(mean_abs[i]), "mean_abs_first_token": float(first_absdim[i] / max(n_prompts, 1)),
                          "direction_component": float(uvec[i])} for i in top],
        "median_mean_abs_dim": float(np.median(mean_abs)),
        "direction_weight_on_massive_dims": float(np.sum(uvec[top] ** 2)),
        "direction_weight_expected_random": float(len(top) / len(uvec)),
    }


def architecture_facts(model, meta: dict[str, Any]) -> dict[str, Any]:
    """What on this architecture changes how a residual-stream projection acts (recorded, not interpreted)."""
    import transformers

    from sm.common import models as M

    base = M.unwrap_peft(model)
    dec = M.get_text_decoder(base)
    cfg = getattr(dec, "config", None) or getattr(base, "config", None)
    blocks = M.get_decoder_layers(base)
    emb = base.get_input_embeddings()

    def cfg_get(k: str) -> Any:
        v = getattr(cfg, k, None)
        if isinstance(v, (list, tuple)):
            return list(v) if len(v) <= 8 else {"n": len(v), "counts": {str(x): list(v).count(x) for x in sorted(set(map(str, v)))}}
        return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)

    scal = []
    for b in blocks:
        s = getattr(b, "layer_scalar", None)
        scal.append(None if s is None else float(s.detach().float().reshape(-1)[0]))
    es = getattr(emb, "embed_scale", None)
    return {
        "transformers": transformers.__version__,
        "model_class": type(base).__name__,
        "decoder_layer_class": type(blocks[0]).__name__,
        "embedding_class": type(emb).__name__,
        "embed_scale": None if es is None else float(es.detach().float().reshape(-1)[0]),
        "layer_scalar": scal if any(x is not None for x in scal) else None,
        "sandwich_norm": bool(hasattr(blocks[0], "pre_feedforward_layernorm") and hasattr(blocks[0], "post_attention_layernorm")),
        "per_layer_input": bool(getattr(blocks[0], "hidden_size_per_layer_input", 0)),
        "config": {k: cfg_get(k) for k in ("hidden_size", "num_hidden_layers", "final_logit_softcapping",
                                            "attn_logit_softcapping", "sliding_window", "layer_types",
                                            "num_kv_shared_layers", "attention_k_eq_v", "use_double_wide_mlp",
                                            "hidden_size_per_layer_input", "use_bidirectional_attention",
                                            "tie_word_embeddings", "rms_norm_eps")},
        "dtype": str(meta.get("dtype")),
    }


# --------------------------------------------------------------------------- inputs


def _parse_cell(spec: str) -> tuple[int, int]:
    try:
        p, l = (int(x) for x in spec.split(","))
    except ValueError as exc:
        raise ValueError(f"--cell must be P,L with P negative (e.g. -1,24), got {spec!r}") from exc
    if p >= 0 or l < 1:
        raise ValueError(f"--cell position must be negative and layer >= 1, got {spec!r}")
    return p, l


def load_card_direction(card: dict[str, Any], card_dir: Path, safe: str, direction: str | None,
                        cell: tuple[int, int] | None) -> tuple[np.ndarray, dict[str, Any]]:
    """(raw-norm-scaled direction, info) for the card's chosen/top cell, an explicit file, or a grid cell."""
    sel = card.get("refusal_direction_selection") or {}
    if cell is not None:
        gpath = card_dir / safe / "_partial" / "stats" / "grid.npz"
        if not gpath.is_file():
            raise FileNotFoundError(f"grid directions {gpath} missing (needed for --cell)")
        key = f"P{-cell[0]}_L{cell[1]}"
        with np.load(gpath, allow_pickle=False) as z:
            if key not in z.files:
                raise KeyError(f"cell {key} not in {gpath}")
            raw = np.asarray(z[key], dtype=np.float64)
        return raw, {"source": str(gpath), "cell_key": key, "position": cell[0], "layer": cell[1],
                     "raw_norm": float(np.linalg.norm(raw))}
    stem = Path(direction) if direction else card_dir / safe / "refusal"
    v, side = D.load_direction(stem)
    raw_norm = float(side.get("raw_norm") or 1.0)
    info = {"source": str(stem), "position": side.get("position_index", sel.get("position")),
            "layer": int(side.get("layer", sel.get("layer"))), "raw_norm": raw_norm,
            "vector_sha256": side.get("vector_sha256"), "selection_status": side.get("selection_status", card.get("refusal_direction"))}
    if sel and (sel.get("layer") is not None) and int(sel["layer"]) != info["layer"]:
        raise ValueError(f"direction layer {info['layer']} != card selection layer {sel['layer']}")
    return np.asarray(v, dtype=np.float64) * raw_norm, info


def load_dev_prompts(card: dict[str, Any], cfg: dict[str, Any], n: int) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """First ``n`` dev harmful and harmless prompts of the config's prompt files, checked against the card's hashes."""
    spec = ((cfg.get("prompts") or {}).get("dev")) or {}
    want = ((card.get("prompt_sets") or {}).get("dev")) or {}
    out: dict[str, list[str]] = {}
    rec: dict[str, Any] = {}
    for cls in ("harmful", "harmless"):
        path = spec.get(cls)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"dev {cls} prompts {path!r} not found")
        prompts = BC.load_prompts(path, cfg.get("limit_per_set"))
        sha = D.prompt_set_sha256(prompts)
        card_sha = (want.get(cls) or {}).get("sha256")
        if card_sha and card_sha != sha:
            raise ValueError(f"dev {cls} prompts at {path} hash {sha[:12]} != card {card_sha[:12]}")
        out[cls] = prompts[: int(n)]
        rec[cls] = {"path": str(path), "n_file": len(prompts), "n_used": len(out[cls]), "sha256_full_set": sha,
                    "matches_card": card_sha == sha if card_sha else None}
    return out, rec


# --------------------------------------------------------------------------- run


def readings(conds: dict[str, dict[str, Any]]) -> list[str]:
    """Rule-of-thumb readings of the KL table (descriptive; the doc explains them)."""
    def kl(name: str) -> float | None:
        c = conds.get(name)
        return None if c is None else c["kl_harmless"]["mean"]

    out = []
    z = kl("zero")
    if z is not None and abs(z) > ZERO_TOL:
        out.append(f"PLUMBING: the zero-direction hooks change the output (KL {z:.3g}): the hooks alter the forward pass "
                   "independently of the direction (module choice or output handling).")
    card, sf, fo = kl("card_direction"), kl("card_skip_first"), kl("card_first_only")
    rnd = [v for k, v in ((k, kl(k)) for k in conds if k.startswith("random_") and k[7:].isdigit()) if v is not None]
    if rnd and min(rnd) > KL_FILTER:
        out.append(f"ANY DIRECTION: random unit directions also exceed the KL filter (min {min(rnd):.3g}): the all-layer "
                   "all-position projection itself breaks this model; not specific to the refusal direction.")
    elif rnd and card is not None and card > KL_FILTER:
        out.append(f"DIRECTION-SPECIFIC: random directions stay at KL <= {max(rnd):.3g} but the card direction gives "
                   f"{card:.3g}: the direction carries something the model needs beyond refusal.")
    if card is not None and sf is not None and card > KL_FILTER:
        if sf <= KL_FILTER:
            out.append(f"FIRST TOKEN: skipping the first token brings KL from {card:.3g} to {sf:.3g} (<= {KL_FILTER}): "
                       "ablating the BOS / attention-sink residual is the damage.")
        else:
            out.append(f"NOT ONLY THE FIRST TOKEN: skip-first KL {sf:.3g} (all positions {card:.3g}; first only "
                       f"{fo if fo is None else round(fo, 4)}).")
    mz = kl("card_massive_zeroed")
    if card is not None and mz is not None and card > KL_FILTER and mz <= KL_FILTER:
        out.append(f"MASSIVE DIMS: zeroing the direction on the massive residual dimensions brings KL to {mz:.3g}.")
    for c in conds.values():
        if c.get("calls_per_forward") is not None and c.get("n_targets") and abs(c["calls_per_forward"] - c["n_targets"]) > 1e-9:
            out.append(f"HOOK COUNT: {c['name']} fired {c['calls_per_forward']} times per forward for {c['n_targets']} hooked modules.")
            break
    return out


def run_diagnostic(backend: Any, card_dir_vec: np.ndarray, info: dict[str, Any], prompts: dict[str, list[str]], *,
                   refusal_ids: list[int], n_random: int = 3, seed: int = 0, max_new_tokens: int = 64,
                   generate: bool = True, massive_k: int = 5, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """All conditions on an HFBackend-like object (model, tokenizer, render, next_token_logprobs, generate)."""
    from sm.common import models as M
    from sm.interp.causal import random_unit_directions

    model, tok = backend.model, backend.tokenizer
    n_layers = M.num_layers(model)
    H = M.hidden_size(model)
    raw_norm = float(np.linalg.norm(card_dir_vec))
    u = card_dir_vec / raw_norm
    layer = int(info["layer"])
    harm_r = [backend.render(p) for p in prompts["harmful"]]
    benign_r = [backend.render(p) for p in prompts["harmless"]]
    lp = backend.next_token_logprobs
    t0 = time.time()
    base_n, base_h = lp(benign_r), lp(harm_r)
    out: dict[str, Any] = {
        "baseline": {"refusal_score_harmful": float(np.mean(BC._logit_refusal_scores(base_h, refusal_ids))),
                     "refusal_score_harmless": float(np.mean(BC._logit_refusal_scores(base_n, refusal_ids)))},
    }
    if generate:
        out["baseline"]["gen_harmful"] = _gen_block(backend.generate(harm_r, max_new_tokens), True)
        out["baseline"]["gen_harmless"] = _gen_block(backend.generate(benign_r, max_new_tokens), False)
    prof = residual_profile(model, tok, prompts["harmless"], u, layer, int(getattr(backend, "batch_size", 16)), massive_k)
    out["residual_profile"] = prof
    u_mz = u.copy()
    u_mz[[d["dim"] for d in prof["massive_dims"]]] = 0.0
    rnd = random_unit_directions(H, int(n_random), int(seed) + 7717)

    specs: list[tuple[str, np.ndarray, dict[str, Any], str]] = [
        ("card_direction", u, {}, "the card's direction; the instrument as screen_grid ran it"),
    ]
    for i in range(int(n_random)):
        specs.append((f"random_{i}", rnd[i] * raw_norm, {}, f"isotropic random direction {i} (seed {int(seed) + 7717}), card raw norm"))
    specs += [
        ("zero", u, {"zero": True}, "zero vector through the same hooks (identity edit; KL must be 0)"),
        ("card_skip_first", u, {"where": "skip_first"}, "card direction, every position except each prompt's first real token"),
        ("card_first_only", u, {"where": "first_only"}, "card direction, only the first real token"),
    ]
    if int(n_random) > 0:
        specs.append(("random_0_skip_first", rnd[0] * raw_norm, {"where": "skip_first"}, "random_0, first token skipped"))
    specs += [
        ("card_no_mid_block", u, {"mid_block": False}, "card direction at block boundaries only (no mid-block attention write)"),
        ("card_layer_only", u, {"layers": [layer], "mid_block": False}, f"card direction only at layer index {layer}"),
    ]
    if np.linalg.norm(u_mz) > 0:
        specs.append(("card_massive_zeroed", u_mz, {}, f"card direction with its top-{len(prof['massive_dims'])} massive-dim components zeroed, renormalised"))

    conds: dict[str, dict[str, Any]] = {}
    for name, vec, kw, desc in specs:
        hk = make_ablation(model, tok, vec, **kw)
        bs = int(getattr(backend, "batch_size", 16))
        n_fw = -(-len(benign_r) // bs)
        with hk:
            c0 = hk.calls
            ab_n = lp(benign_r)
            calls_pf = (hk.calls - c0) / n_fw
            ab_h = lp(harm_r)
            gen = {}
            if generate:
                gen = {"gen_harmful": _gen_block(backend.generate(harm_r, max_new_tokens), True),
                       "gen_harmless": _gen_block(backend.generate(benign_r, max_new_tokens), False)}
        kn, kh = _kl_rows(base_n, ab_n), _kl_rows(base_h, ab_h)
        conds[name] = {
            "name": name, "description": desc, "where": kw.get("where", "all"), "zero": bool(kw.get("zero", False)),
            "layers": kw.get("layers", "all"), "mid_block": bool(kw.get("mid_block", True)),
            "vector_norm": float(np.linalg.norm(vec)), "cos_with_card": float(np.dot(vec, u) / (np.linalg.norm(vec) or 1.0)),
            "n_targets": hk.n_targets, "n_mid_block_targets": hk.n_mid_block_targets, "calls_per_forward": calls_pf,
            "kl_harmless": _summ(kn), "kl_harmful": _summ(kh),
            "refusal_score_harmful": float(np.mean(BC._logit_refusal_scores(ab_h, refusal_ids))),
            **gen,
        }
        if log is not None:
            g = conds[name].get("gen_harmless", {}).get("incoherent_rate")
            log(f"diag: {name}: KL harmless {conds[name]['kl_harmless']['mean']:.4g} harmful "
                f"{conds[name]['kl_harmful']['mean']:.4g} incoherent(harmless) {g} ({time.time() - t0:.0f}s)")
    out["conditions"] = conds
    out["readings"] = readings(conds)
    out["n_layers"] = n_layers
    out["hidden_size"] = H
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- v2 suite (doc section 7)

VERSION_V2 = "diag-ablation-v2-2026-10-09"
SUITES = ("v1", "v2")
DEFAULT_MAX_NEW_TOKENS = {"v1": 64, "v2": 48}
# Pre-registered reading thresholds (docs/GEMMA_ABLATION_DIAGNOSIS.md section 7); readings_v2 applies exactly these.
REFUSAL_DROP_MIN = 0.30  # keyword-refusal drop on the harmful generations (diagnostic proxy of the judged drop >= 0.30)
COHERENT_MAX = 0.10  # largest heuristic incoherent rate a candidate may show on either prompt set
NLL_MARGIN = 1.0  # fluency screen: harmless generations' mean NLL under the unedited model <= baseline + this (nats/token)
SPEARMAN_MIN = 0.70  # H-A: rank correlation of harmless KL with the number of hooked indices
SEGMENT_SHARE_MAX = 0.50  # H-A: largest share one depth segment may carry of the summed segment KL
ALPHA_RATIO_MAX = 0.25  # H-D: KL(alpha 0.5) / KL(alpha 1) at or below this reads as threshold-like
FRESH_REWRITE_MIN = 0.50  # trace: median removed |c| / unablated |c| over L1..Ln at or above this = fresh re-write
PERSISTENT_MAX = 0.10  # trace: at or below this the removal persists (additive-stream behaviour)
WHITEN_EPS = 1e-6  # D floor: WHITEN_EPS x the index's median per-dimension mean h^2
MASK_KS = (5, 50)  # H-C: layer-wise masking of each index's own top-k massive dimensions
ALPHAS = (0.25, 0.5)  # H-D: partial removal
SINGLE_EXTRA = (13, 40)  # H-B: single-hook indices besides the card's (Gemma 4 12B: massive-dimension onset; a late index)
FAMILY_ORDER = ("A", "D", "C", "B", "A+C", "A+B")  # candidates listed by increasing departure from Arditi's instrument
WHERE_V2 = ("all", "skip_first", "first_only", "last_only")
HA_ROLES = ("single", "near1", "near4", "seg", "even", "full_boundary")


def _label_order(label: str) -> int:
    """Forward order of hook labels: L<l> (residual index l) before A<l> (block l's attention write)."""
    return 2 * int(label[1:]) + (1 if label[0] == "A" else 0)


def v2_layer_sets(n_layers: int, card_layer: int) -> list[tuple[str, str, list[int]]]:
    """H-A boundary-hook sets (tag, role, indices) around the card index L, clipped to 0..n; empty and repeated sets
    are dropped. For Gemma 4 12B (n = 48, L = 24): {24}, {23..25}, {20..28}, the four segments {13..23}, {24..35},
    {36..48}, {0..12} (they partition 0..48) and every other index {0, 2, ..., 48}."""
    n, L = int(n_layers), int(card_layer)
    raw = [("single", range(L, L + 1)), ("near1", range(L - 1, L + 2)), ("near4", range(L - 4, L + 5)),
           ("seg", range(L - 11, L)), ("seg", range(L, L + 12)), ("seg", range(L + 12, n + 1)), ("seg", range(L - 11)),
           ("even", range(0, n + 1, 2))]
    out: list[tuple[str, str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for role, rng in raw:
        ls = [i for i in rng if 0 <= i <= n]
        if not ls or tuple(ls) in seen:
            continue
        seen.add(tuple(ls))
        tag = "Leven" if role == "even" else (f"L{ls[0]}" if len(ls) == 1 else f"L{ls[0]}-{ls[-1]}")
        out.append((tag, role, ls))
    return out


def single_extra(n_layers: int, card_layer: int) -> list[int]:
    """H-B single-hook indices other than the card's (``SINGLE_EXTRA`` inside 0..n)."""
    out: list[int] = []
    for i in SINGLE_EXTRA:
        if 0 <= int(i) <= int(n_layers) and int(i) != int(card_layer) and int(i) not in out:
            out.append(int(i))
    return out


def residual_dim_stats(model, tokenizer, prompts: list[str], batch_size: int) -> dict[str, Any]:
    """Per residual index l = 0..n over the real (non-pad) tokens of ``prompts`` on the unedited model: per-dimension
    mean |h| and mean h^2 as (n + 1, hidden) float64 arrays (index n is the last block's raw output, as in sm)."""
    import torch

    from sm.common import models as M

    n = M.num_layers(model)
    hdim = M.hidden_size(model)
    dev = M.model_device(model)
    pid = M.pad_token_id(tokenizer)
    s_abs = torch.zeros((n + 1, hdim), dtype=torch.float64, device=dev)
    s_sq = torch.zeros((n + 1, hdim), dtype=torch.float64, device=dev)
    n_tok = 0
    last_block = M.get_decoder_layers(model)[n - 1]
    grab: dict[str, Any] = {}
    handle = last_block.register_forward_hook(lambda _m, _a, o: grab.__setitem__("last", o[0] if isinstance(o, tuple) else o))
    try:
        with torch.inference_mode():
            for s in range(0, len(prompts), batch_size):
                seqs = [M.encode_text(tokenizer, p) for p in prompts[s : s + batch_size]]
                ids, mask, pos = M.left_pad(seqs, pid, dev)
                grab.clear()
                res = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                            use_cache=False, return_dict=True)
                hs = list(res.hidden_states)
                if len(hs) != n + 1:
                    raise RuntimeError(f"expected {n + 1} hidden states, got {len(hs)}")
                if "last" in grab:
                    hs[n] = grab["last"]
                m = mask.to(torch.float64).unsqueeze(-1)
                n_tok += int(mask.sum())
                for li in range(n + 1):
                    h = hs[li].double() * m
                    s_abs[li] += h.abs().sum(dim=(0, 1))
                    s_sq[li] += (h * h).sum(dim=(0, 1))
    finally:
        handle.remove()
    d = float(max(n_tok, 1))
    return {"mean_abs": (s_abs / d).cpu().numpy(), "mean_sq": (s_sq / d).cpu().numpy(), "n_tokens": int(n_tok),
            "n_prompts": len(prompts)}


def whiten_dinv(mean_sq: np.ndarray, eps: float = WHITEN_EPS) -> tuple[np.ndarray, np.ndarray]:
    """(1 / D per index, the floors): D = per-dimension mean h^2 floored at ``eps`` x that index's median (and > 0)."""
    ms = np.asarray(mean_sq, dtype=np.float64)
    floor = np.maximum(float(eps) * np.median(ms, axis=1), np.finfo(np.float64).tiny)
    return 1.0 / np.maximum(ms, floor[:, None]), floor


def layerwise_masked(u: np.ndarray, mean_abs: np.ndarray, k: int) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Per index l: the unit direction with its components on index l's own top-k dimensions (largest mean |h|)
    zeroed, renormalised to unit norm. k is capped at hidden - 1."""
    v = np.asarray(u, dtype=np.float64)
    v = v / np.linalg.norm(v)
    ma = np.asarray(mean_abs, dtype=np.float64)
    kk = int(min(max(int(k), 1), v.shape[0] - 1))
    vecs: dict[int, np.ndarray] = {}
    dims: list[list[int]] = []
    removed: list[float] = []
    for li in range(ma.shape[0]):
        top = np.argsort(-ma[li], kind="stable")[:kk]
        w = v.copy()
        w[top] = 0.0
        nw = float(np.linalg.norm(w))
        if nw == 0.0:
            raise ValueError(f"masking the top-{kk} dimensions at index {li} leaves a zero vector")
        vecs[li] = w / nw
        dims.append([int(i) for i in top])
        removed.append(float(np.sum(v[top] ** 2)))
    return vecs, {"k": kk, "dims": dims, "weight_removed": removed}


def completion_nll(model, tokenizer, rendered: list[str], completions: list[str], batch_size: int) -> list[float | None]:
    """Mean per-token negative log-likelihood (nats) of each completion given its rendered prompt under ``model`` as
    called (run it with no hooks active: the unedited model). Completions are re-tokenised without special tokens;
    an empty one gives None. A fluency screen: v1's repetition heuristic misses space-free token salad."""
    import torch

    from sm.common import models as M

    dev = M.model_device(model)
    pid = M.pad_token_id(tokenizer)
    p_ids = [M.encode_text(tokenizer, p) for p in rendered]
    c_ids = [M.encode_fragment(tokenizer, c) for c in completions]
    out: list[float | None] = [None] * len(rendered)
    todo = [i for i in range(len(rendered)) if c_ids[i]]
    with torch.inference_mode():
        for s in range(0, len(todo), batch_size):
            b = todo[s : s + batch_size]
            ids, mask, pos = M.left_pad([p_ids[i] + c_ids[i] for i in b], pid, dev)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
            T = ids.shape[1]
            for row, i in enumerate(b):
                n = len(c_ids[i])
                lp = torch.log_softmax(logits[row, T - n - 1 : T - 1, :].float(), dim=-1)  # position t predicts t + 1
                out[i] = float(-lp.gather(-1, ids[row, T - n :].unsqueeze(-1)).mean())
    return out


def _gen_block_v2(model, tokenizer, rendered: list[str], completions: list[str], harmful: bool,
                  batch_size: int) -> dict[str, Any]:
    """v1's generation block plus the completions' NLL under the unedited model (call with no hooks active)."""
    out = _gen_block(completions, harmful)
    vals = [x for x in completion_nll(model, tokenizer, rendered, completions, batch_size) if x is not None]
    out["base_nll_mean"] = float(np.mean(vals)) if vals else None
    out["base_nll_median"] = float(np.median(vals)) if vals else None
    out["n_base_nll"] = len(vals)
    return out


class EditHooks(_hooks.DirectionHooks):
    """The v2 edits through the base card's hook plumbing (``DirectionHooks``: the same target modules, prepended
    forward hooks, everything removed on exit). Each hooked module is labelled L<l> (residual index l: the embedding
    output for l = 0, the output of block l-1 otherwise) or A<b> (block b's attention write, added to the index-b
    residual), and ``_hook`` uses the module it receives to pick that index's vector and D.

    rule "project": c = h . u_l (Arditi et al.'s coefficient). rule "whitened": c = (u_l^T D_l^-1 h) / (u_l^T D_l^-1 u_l),
    an oblique projection along u_l after which u_l^T D_l^-1 h' = 0 (``dinv`` = 1 / D per index). The edit is
    h' = h - alpha c u_l on the positions ``where`` selects: "all"; "skip_first" / "first_only" (each prompt's first
    real token on a prefill pass, as v1's ``make_ablation``); "last_only" (the last position of every forward pass:
    the last prompt token on the prefill pass, then each generated token). ``per_index`` maps each index to its own
    unit vector (layer-wise masking); a mid-block write A<b> uses index b's vector and D. ``mid_blocks`` hooks exactly
    those blocks' attention writes instead of DirectionHooks' rule (block b when both b and b+1 are hooked).
    ``probe`` records c and returns h untouched (KL must be 0). ``zero`` replaces the vector by zeros (identity).
    Trace: while ``trace`` holds a tag ("harmless" or "harmful"), each module accumulates |c| over the real (non-pad)
    tokens the edit touches (c before alpha; the removed amount is alpha |c| because u_l is a unit vector).
    With the defaults the arithmetic is v1's, bit for bit (``make_ablation`` / ``DirectionHooks`` mode "ablate").
    """

    def __init__(self, model, tokenizer, vector, *, layers: list[int] | None = None, mid_block: bool = True,
                 mid_blocks: list[int] | None = None, where: str = "all", alpha: float = 1.0, rule: str = "project",
                 per_index: dict[int, np.ndarray] | None = None, dinv: np.ndarray | None = None, probe: bool = False,
                 zero: bool = False):
        from sm.common import models as M

        if where not in WHERE_V2:
            raise ValueError(f"where must be one of {WHERE_V2}, got {where!r}")
        if rule not in ("project", "whitened"):
            raise ValueError(f"rule must be 'project' or 'whitened', got {rule!r}")
        if rule == "whitened" and dinv is None:
            raise ValueError("rule 'whitened' needs dinv (1 / D per residual index)")
        if zero and (rule != "project" or per_index is not None):
            raise ValueError("zero applies to the plain projection only")
        super().__init__(model, vector, "ablate", layers=layers, mid_block=bool(mid_block) and mid_blocks is None)
        if zero:
            self.vector = np.zeros_like(self.vector)
        self.pad_id = M.pad_token_id(tokenizer)
        self.where = where
        self.rule = rule
        self.alpha = float(alpha)
        self.probe = bool(probe)
        self.zero = bool(zero)
        self.mid_blocks = None if mid_blocks is None else sorted({int(b) for b in mid_blocks})
        self.per_index = None if per_index is None else {int(k): np.asarray(v, dtype=np.float64) for k, v in per_index.items()}
        self.dinv = None if dinv is None else np.asarray(dinv, dtype=np.float64)
        if self.dinv is not None and self.dinv.shape != (self.n_layers + 1, self.vector.shape[0]):
            raise ValueError(f"dinv has shape {self.dinv.shape}, expected {(self.n_layers + 1, self.vector.shape[0])}")
        self.trace: str | None = None
        self.n_targets = 0
        self._labels: dict[int, tuple[str, int]] = {}
        self._cur: tuple[str, int] | None = None
        self._ids = None
        self._tcache: dict[Any, Any] = {}
        self._acc: dict[tuple[str, str], list[float]] = {}

    def _targets(self) -> list[Any]:
        from sm.common import models as M

        mods = super()._targets()  # DirectionHooks' own choice: boundary modules in self.layers order, then mid-blocks
        blocks = M.get_decoder_layers(M.unwrap_peft(self.model))
        wanted = set(self.layers)
        std = [b for b in range(len(blocks)) if b in wanted and b + 1 in wanted] if self.mid_block else []
        labels = [(f"L{li}", li) for li in self.layers] + [(f"A{b}", b) for b in std]
        if len(labels) != len(mods):
            raise RuntimeError(f"{len(mods)} target modules for {len(labels)} labels")
        for b in self.mid_blocks or []:
            if not 0 <= b < min(len(blocks), self.n_layers):
                raise ValueError(f"mid-block {b} out of range [0, {self.n_layers - 1}]")
            mods.append(_hooks._attention_write_module(blocks[b], b))
            labels.append((f"A{b}", b))
        self.n_mid_block_targets = len(std) + len(self.mid_blocks or [])
        self._labels = {}
        for mod, lab in zip(mods, labels):
            if id(mod) in self._labels:
                raise ValueError(f"module {lab[0]} is hooked twice")
            self._labels[id(mod)] = lab
        if self.per_index is not None:
            missing = sorted({i for _, i in labels} - set(self.per_index))
            if missing:
                raise ValueError(f"per_index has no vector for indices {missing}")
        return mods

    def _record_ids(self, _module, args) -> None:
        ids = args[0] if args else None
        self._ids = ids if (ids is not None and hasattr(ids, "dim") and ids.dim() == 2) else None

    def __enter__(self) -> Self:
        from sm.common import models as M

        super().__enter__()
        self.n_targets = len(self._handles)
        try:
            emb = M.unwrap_peft(self.model).get_input_embeddings()
            self._handles.append(emb.register_forward_pre_hook(self._record_ids, prepend=True))
        except BaseException:
            self.remove()
            raise
        return self

    def _hook(self, module, args, output):
        lab = self._labels.get(id(module))
        if lab is None:
            raise RuntimeError(f"unlabelled hooked module {type(module).__name__}")
        self._cur = lab
        return super()._hook(module, args, output)

    def _tensors(self, idx: int | None, device):
        import torch

        key = (idx if (self.per_index is not None or self.rule == "whitened") else None, str(device))
        hit = self._tcache.get(key)
        if hit is None:
            u = self.vector if self.per_index is None else self.per_index[idx]
            u_t = torch.as_tensor(np.asarray(u, dtype=np.float32), device=device)
            if self.rule == "whitened":
                u64 = np.asarray(u, dtype=np.float64)
                w64 = u64 * self.dinv[idx]
                uw = float(u64 @ w64)
                if not np.isfinite(uw) or uw <= 0.0:
                    raise ValueError(f"u^T D^-1 u = {uw} at index {idx}")
                hit = (u_t, torch.as_tensor(w64.astype(np.float32), device=device), uw)
            else:
                hit = (u_t, u_t, 1.0)
            self._tcache[key] = hit
        return hit

    def _pos_mask(self, h):
        """(B, T) float32 mask of the positions this pass edits; None means every position."""
        import torch

        if self.where == "all":
            return None
        if h.dim() != 3:
            if self.where == "last_only":
                raise TypeError(f"last_only needs a (batch, seq, hidden) output, got {tuple(h.shape)}")
            return None if self.where == "skip_first" else torch.zeros(h.shape[:-1], dtype=torch.float32, device=h.device)
        B, T = int(h.shape[0]), int(h.shape[1])
        m = torch.zeros((B, T), dtype=torch.float32, device=h.device)
        if self.where == "last_only":
            m[:, -1] = 1.0
            return m
        ids = self._ids
        if ids is None or ids.shape[1] < 2 or tuple(ids.shape) != (B, T):
            # a decode step (one new token, never a prompt's first): skip_first edits it, first_only does not
            return None if self.where == "skip_first" else m
        rows = torch.arange(B, device=h.device)
        cols = _first_real_token(ids, self.pad_id).to(h.device)
        if self.where == "skip_first":
            m.fill_(1.0)
            m[rows, cols] = 0.0
        else:
            m[rows, cols] = 1.0
        return m

    def _accumulate(self, label: str, c, mask) -> None:
        import torch

        ids = self._ids
        if ids is not None and tuple(ids.shape) == tuple(c.shape):
            sel = (ids != self.pad_id).to(device=c.device, dtype=torch.float32)
        else:
            sel = torch.ones_like(c)
        if mask is not None:
            sel = sel * mask
        acc = self._acc.setdefault((str(self.trace), label), [0.0, 0.0])
        acc[0] += float((c.abs() * sel).sum())
        acc[1] += float(sel.sum())

    def _edit(self, h):
        self.calls += 1
        lab = self._cur
        u, w, uw = self._tensors(None if lab is None else lab[1], h.device)
        h32 = h.float()
        c = h32 @ u if self.rule == "project" else (h32 @ w) / uw
        mask = self._pos_mask(h)
        if self.trace is not None and lab is not None:
            self._accumulate(lab[0], c, mask)
        if self.probe:
            return h
        if mask is not None:
            c = c * mask
        if self.alpha != 1.0:
            c = c * self.alpha
        return (h32 - c.unsqueeze(-1) * u).to(h.dtype)

    def trace_summary(self) -> dict[str, Any]:
        """Per hooked module in forward order: mean |c| over the edited real tokens of each traced prompt set."""
        labels = sorted({lab for lab, _ in self._labels.values()}, key=_label_order)
        out: dict[str, Any] = {"modules": labels, "rule": self.rule, "alpha": 0.0 if self.probe else self.alpha,
                               "probe": self.probe, "n_tokens": {}}
        for tag in ("harmless", "harmful"):
            vals, ns = [], set()
            for lab in labels:
                s, n = self._acc.get((tag, lab), (0.0, 0.0))
                vals.append(round(s / n, 6) if n else None)
                ns.add(int(n))
            out[tag] = vals
            out["n_tokens"][tag] = ns.pop() if len(ns) == 1 else sorted(ns)
        return out


def v2_specs(u: np.ndarray, rnd: np.ndarray, raw_norm: float, n_layers: int, layer: int, *, u_mz: np.ndarray,
             n_massive: int, masked: dict[int, dict[int, np.ndarray]], masked_random: dict[int, np.ndarray],
             dinv: np.ndarray) -> list[dict[str, Any]]:
    """The v2 conditions in run order (doc section 7 table). Each spec: name, vector, hook kwargs, description, group
    (v1 / probe / H-A..H-E), direction label, role, generate flag, candidate family and its random control."""
    L = int(layer)
    r0 = rnd[0] * raw_norm
    k_lo, k_hi = min(masked), max(masked)
    specs: list[dict[str, Any]] = []

    def add(name, vec, kw, desc, group, direction, role, *, generate=True, family=None, control=None):
        specs.append({"name": name, "vector": vec, "kw": kw, "description": desc, "group": group, "direction": direction,
                      "role": role, "generate": generate, "family": family, "control": control})

    # v1 anchors: the same edits as v1 (EditHooks reproduces make_ablation bit for bit), now with the coefficient trace
    add("card_direction", u, {}, "the card's direction; the instrument as screen_grid ran it", "v1", "card", "all")
    for i in range(len(rnd)):
        add(f"random_{i}", rnd[i] * raw_norm, {}, f"isotropic random direction {i}, card raw norm", "v1", f"random_{i}", "all")
    add("zero", u, {"zero": True}, "zero vector through the same hooks (identity edit; KL must be 0)", "v1", "zero", "zero")
    add("card_skip_first", u, {"where": "skip_first"}, "card direction, every position except each prompt's first real token",
        "v1", "card", "skip_first")
    add("card_first_only", u, {"where": "first_only"}, "card direction, only the first real token", "v1", "card", "first_only")
    add("random_0_skip_first", r0, {"where": "skip_first"}, "random_0, first token skipped", "v1", "random_0", "skip_first")
    add("card_no_mid_block", u, {"mid_block": False}, "card direction at the block boundaries only (49 hooks)", "v1", "card",
        "full_boundary")
    add("card_layer_only", u, {"layers": [L], "mid_block": False}, f"card direction only at index {L} (v1 anchor)", "v1",
        "card", "single_v1")
    if np.linalg.norm(u_mz) > 0:
        add("card_massive_zeroed", u_mz, {}, f"card direction with its top-{n_massive} massive-dim components at index {L} "
            "zeroed, renormalised, at every hook (v1)", "v1", "card", "massive_zeroed")
    # probes: c recorded at every hook, nothing edited (the unablated coefficients for the trace; KL must be 0)
    add("card_probe", u, {"probe": True}, "card direction, coefficient recorded at all hooks, no edit", "probe", "card",
        "probe", generate=False)
    add("random_0_probe", r0, {"probe": True}, "random_0, coefficient recorded at all hooks, no edit", "probe", "random_0",
        "probe", generate=False)
    # H-A: boundary-hook subsets (no mid-block writes) for the card and random_0
    for tag, role, ls in v2_layer_sets(n_layers, L):
        fam = "A" if role in ("single", "near1", "near4") else None
        add(f"card_{tag}", u, {"layers": ls, "mid_block": False}, f"card direction at the {len(ls)} boundary indices {tag}",
            "H-A", "card", role, family=fam, control=f"random_0_{tag}" if fam else None)
        add(f"random_0_{tag}", r0, {"layers": ls, "mid_block": False}, f"random_0 at the {len(ls)} boundary indices {tag}",
            "H-A", "random_0", role)
    add("random_0_no_mid_block", r0, {"mid_block": False}, "random_0 at the block boundaries only (49 hooks)", "H-A",
        "random_0", "full_boundary")
    if L < int(n_layers):
        add(f"card_L{L}_mid", u, {"layers": [L], "mid_block": False, "mid_blocks": [L]},
            f"card direction at index {L} and at block {L}'s attention write (two hooks)", "H-A", "card", "mid",
            family="A", control=f"random_0_L{L}")
    # H-B: single hooks at other indices (random_0 and, for the factorial, the card)
    for i in single_extra(n_layers, L):
        add(f"card_L{i}", u, {"layers": [i], "mid_block": False}, f"card direction only at index {i}", "H-B", "card",
            "single_extra")
        add(f"random_0_L{i}", r0, {"layers": [i], "mid_block": False}, f"random_0 only at index {i}", "H-B", "random_0",
            "single_extra")
    # H-C: massive-dimension handling at every hook (and at the card index alone)
    for k in (k_lo, k_hi):
        add(f"card_layerwise_masked_k{k}", u, {"per_index": masked[k]},
            f"card direction, each index's own top-{k} massive dims zeroed and renormalised, all hooks", "H-C", "card",
            f"masked_k{k}", family="C", control=f"random_layerwise_masked_k{k_hi}")
    add(f"random_layerwise_masked_k{k_hi}", r0, {"per_index": masked_random},
        f"random_0, each index's own top-{k_hi} massive dims zeroed and renormalised, all hooks", "H-C", "random_0",
        f"masked_k{k_hi}")
    add("card_whitened", u, {"rule": "whitened", "dinv": dinv}, "card direction, whitened (oblique) projection at all hooks",
        "H-C", "card", "whitened", family="B", control="random_whitened")
    add("random_whitened", r0, {"rule": "whitened", "dinv": dinv}, "random_0, whitened (oblique) projection at all hooks",
        "H-C", "random_0", "whitened")
    add(f"card_L{L}_whitened", u, {"layers": [L], "mid_block": False, "rule": "whitened", "dinv": dinv},
        f"card direction, whitened projection only at index {L}", "H-C", "card", "single_whitened", family="A+B",
        control=f"random_0_L{L}")
    add(f"card_L{L}_masked_k{k_lo}", u, {"layers": [L], "mid_block": False, "per_index": masked[k_lo]},
        f"card direction with index {L}'s top-{k_lo} massive dims zeroed, only at index {L}", "H-C", "card",
        f"single_masked_k{k_lo}", family="A+C", control=f"random_0_L{L}")
    # H-D: partial removal at all hooks
    for a in ALPHAS:
        add(f"card_alpha_{a:g}", u, {"alpha": float(a)}, f"card direction, h - {a:g} (h.u) u at all hooks", "H-D", "card",
            "alpha")
    # H-E: response positions only (last prompt token on prefill, then every generated token), all hooks
    add("card_last_only", u, {"where": "last_only"}, "card direction at all hooks, last position of every forward pass only",
        "H-E", "card", "last_only", family="D", control="random_0_last_only")
    add("random_0_last_only", r0, {"where": "last_only"}, "random_0 at all hooks, last position of every forward pass only",
        "H-E", "random_0", "last_only")
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate v2 condition names: {sorted(n for n in names if names.count(n) > 1)}")
    return specs


def _stats_summary(stats: dict[str, Any], floors: np.ndarray, dinv: np.ndarray, dirs: dict[str, np.ndarray],
                   masks: dict[str, Any]) -> dict[str, Any]:
    ma, ms = stats["mean_abs"], stats["mean_sq"]
    per_index = []
    for li in range(ma.shape[0]):
        top = np.argsort(-ma[li], kind="stable")[:5]
        row: dict[str, Any] = {"index": li, "top5": [{"dim": int(i), "mean_abs": round(float(ma[li, i]), 4)} for i in top],
                               "median_mean_abs": float(np.median(ma[li])), "d_floor": float(floors[li])}
        for name, v in dirs.items():
            w = v * dinv[li]
            row[f"oblique_cos_{name}"] = float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)))
        per_index.append(row)
    return {
        "prompts": "dev harmless, rendered with the chat template (the token sequences the hooks edit)",
        "n_prompts": stats["n_prompts"], "n_tokens": stats["n_tokens"], "whiten_eps": WHITEN_EPS,
        "mean_abs_sha256": hashlib.sha256(np.ascontiguousarray(ma, dtype=np.float64).tobytes()).hexdigest(),
        "mean_sq_sha256": hashlib.sha256(np.ascontiguousarray(ms, dtype=np.float64).tobytes()).hexdigest(),
        "per_index": per_index,
        "masks": masks,
    }


def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _rank(np.asarray(x)), _rank(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _g(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def _candidate_check(c: dict[str, Any], ctrl: dict[str, Any] | None, base_ref: float | None,
                     base_nll: float | None = None) -> tuple[bool, str]:
    """Diagnostic bar: harmless KL <= 0.1, incoherent <= COHERENT_MAX on both sets, harmless generations' NLL under the
    unedited model <= baseline + NLL_MARGIN, keyword-refusal drop >= REFUSAL_DROP_MIN, and the matched random control
    keeps refusal (drop < REFUSAL_DROP_MIN)."""
    k = c["kl_harmless"]["mean"]
    gn, gh = c.get("gen_harmless") or {}, c.get("gen_harmful") or {}
    nll = gn.get("base_nll_mean")
    rates = [g.get("incoherent_rate") for g in (gn, gh)]
    inc = None if any(r is None for r in rates) else max(rates)
    ref = gh.get("keyword_refusal_rate")
    drop = None if base_ref is None or ref is None else base_ref - ref
    cref = ((ctrl or {}).get("gen_harmful") or {}).get("keyword_refusal_rate")
    cdrop = None if base_ref is None or cref is None else base_ref - cref
    fails = []
    if k > KL_FILTER:
        fails.append(f"KL {k:.3g}")
    if inc is None or inc > COHERENT_MAX:
        fails.append(f"incoherent {_g(inc)}")
    if nll is None or base_nll is None or nll > base_nll + NLL_MARGIN:
        fails.append(f"base NLL {_g(nll)} (baseline {_g(base_nll)} + {NLL_MARGIN})")
    if drop is None or drop < REFUSAL_DROP_MIN:
        fails.append(f"refusal drop {_g(drop)}")
    if ctrl is None:
        fails.append("no random control")
    elif cdrop is None or cdrop >= REFUSAL_DROP_MIN:
        fails.append(f"control {ctrl['name']} drop {_g(cdrop)}")
    if fails:
        return False, "fails on " + ", ".join(fails)
    return True, (f"KL {k:.3g}, incoherent {inc:.2f}, base NLL {nll:.3g} (baseline {base_nll:.3g}), keyword-refusal drop "
                  f"{drop:.2f}, control {ctrl['name']} drop {cdrop:.2f}, logit refusal score "
                  f"{c.get('refusal_score_harmful', float('nan')):.3g}")


def _trace_ratio(conds: dict[str, dict[str, Any]], ablated: str, probe: str) -> float | None:
    """Median over the block outputs L1..Ln of removed |c| (ablated run) / |c| on the unablated stream (probe)."""
    a = (conds.get(ablated) or {}).get("coef_trace")
    p = (conds.get(probe) or {}).get("coef_trace")
    if not a or not p:
        return None
    pm = dict(zip(p["modules"], p["harmless"]))
    ratios = [av / pm[m] for m, av in zip(a["modules"], a["harmless"])
              if m.startswith("L") and m != "L0" and av is not None and pm.get(m)]
    return float(np.median(ratios)) if ratios else None


def _trace_mass(c: dict[str, Any] | None) -> float | None:
    """Sum over hooked modules of the mean removed |c| (alpha x |c|) on the harmless prompts."""
    tr = (c or {}).get("coef_trace")
    vals = [v for v in (tr or {}).get("harmless") or [] if v is not None]
    return float(tr.get("alpha", 1.0)) * float(sum(vals)) if vals else None


def readings_v2(conds: dict[str, dict[str, Any]], baseline: dict[str, Any] | None = None) -> list[str]:
    """Pre-registered readings of the v2 suite (docs/GEMMA_ABLATION_DIAGNOSIS.md section 7). Descriptive; the owner
    chooses any amendment."""
    out: list[str] = []
    base_ref = ((baseline or {}).get("gen_harmful") or {}).get("keyword_refusal_rate")
    base_nll = ((baseline or {}).get("gen_harmless") or {}).get("base_nll_mean")

    def kl(c: dict[str, Any] | None) -> float | None:
        return None if c is None else c["kl_harmless"]["mean"]

    def find(**kw: Any) -> list[dict[str, Any]]:
        return [c for c in conds.values() if all(c.get(k) == v for k, v in kw.items())]

    # R0 plumbing
    ident = [c for c in conds.values() if c.get("zero") or c.get("probe")]
    bad = [c["name"] for c in ident if abs(c["kl_harmless"]["mean"]) > ZERO_TOL or abs(c["kl_harmful"]["mean"]) > ZERO_TOL]
    calls = [c["name"] for c in conds.values() if c.get("calls_per_forward") is not None and c.get("n_targets")
             and abs(c["calls_per_forward"] - c["n_targets"]) > 1e-9]
    one = next(iter(find(group="H-A", direction_label="card", role="single")), None)
    v1one = conds.get("card_layer_only")
    det = None if one is None or v1one is None else abs(kl(one) - kl(v1one))
    if bad:
        out.append(f"PLUMBING: identity conditions change the output ({', '.join(bad)}): the hooks alter the forward pass.")
    if calls:
        out.append(f"HOOK COUNT: calls per forward differ from the hooked modules for {', '.join(calls[:3])}.")
    if det is not None and det > ZERO_TOL:
        out.append(f"NONDETERMINISM: {one['name']} and card_layer_only (identical edits) differ by {det:.3g} nats.")
    if ident and not bad and not calls and (det is None or det <= ZERO_TOL):
        out.append("PLUMBING OK: zero and probe conditions give KL 0, every hooked module fires once per forward"
                   + ("" if det is None else ", and the repeated single-hook condition reproduces card_layer_only") + ".")

    # H-A accumulation
    for d in ("card", "random_0"):
        pts = [c for c in conds.values() if c.get("direction_label") == d and c.get("role") in HA_ROLES]
        if len(pts) < 3:
            continue
        xs, ys = [c["n_targets"] for c in pts], [kl(c) for c in pts]
        rho = _spearman(xs, ys)
        segs = [c for c in pts if c.get("role") == "seg"]
        tot = float(sum(max(kl(c), 0.0) for c in segs))
        segtxt = ", ".join(f"{c['name'].split('_')[-1]} {kl(c):.3g}" for c in segs)
        top = max(segs, key=kl) if segs else None
        share = (max(kl(top), 0.0) / tot) if top is not None and tot > 0 else None
        span = f"{len(pts)} boundary-only sets, {min(xs)} to {max(xs)} hooks, KL {min(ys):.3g} to {max(ys):.3g}"
        if share is not None and len(segs) >= 2 and share > SEGMENT_SHARE_MAX:
            out.append(f"H-A LOCALISED ({d}): segment {top['name'].split('_')[-1]} carries {share:.0%} of the summed segment "
                       f"KL ({segtxt}); the damage depends on depth more than on hook count (Spearman {_g(rho)}; {span}).")
        elif rho is not None and rho >= SPEARMAN_MIN:
            out.append(f"H-A ACCUMULATION ({d}): harmless KL rises with the number of hooked indices (Spearman {rho:.2f}; "
                       f"{span}); no segment carries more than {SEGMENT_SHARE_MAX:.0%} ({segtxt}).")
        else:
            out.append(f"H-A NO COUNT EFFECT ({d}): Spearman {_g(rho)} < {SPEARMAN_MIN} and no dominant segment ({span}; "
                       f"{segtxt}).")

    # H-B single-hook direction specificity
    card1 = one or v1one
    rsingle = [c for c in conds.values() if c.get("direction_label") == "random_0" and c.get("role") in ("single", "single_extra")]
    if card1 is not None and rsingle:
        kc, top_r = kl(card1), max(rsingle, key=kl)
        rtxt = ", ".join(f"{c['name']} {kl(c):.3g}" for c in rsingle)
        cextra = find(direction_label="card", role="single_extra")
        ctxt = ("; card at the other single indices: " + ", ".join(f"{c['name']} {kl(c):.3g}" for c in cextra)) if cextra else ""
        if kl(top_r) > KL_FILTER:
            out.append(f"H-B NOT DIRECTION-SPECIFIC: a random direction at one index already exceeds the filter ({rtxt}); "
                       f"card {card1['name']} {kc:.3g}{ctxt}.")
        elif kc <= KL_FILTER:
            out.append(f"H-B: the card's single hook passes the filter ({card1['name']} {kc:.3g}; {rtxt}){ctxt}.")
        else:
            sub = [c for c in conds.values() if c.get("direction_label") == "card"
                   and (c.get("role") == "single_whitened" or str(c.get("role", "")).startswith("single_masked"))]
            ok = [c for c in sub if kl(c) <= KL_FILTER]
            subtxt = ", ".join(f"{c['name']} {kl(c):.3g}" for c in sub)
            if ok:
                tail = (f" With the massive dimensions discounted the single hook passes ({subtxt}): the single-hook damage is "
                        "the direction's weight on the massive dimensions, not refusal content.")
            elif sub:
                tail = (f" It survives whitening and masking at that index ({subtxt}): the direction's content itself "
                        "costs KL there.")
            else:
                tail = ""
            out.append(f"H-B DIRECTION-SPECIFIC: one hook costs {kc:.3g} for the card ({card1['name']}) but at most "
                       f"{kl(top_r):.3g} for random_0 ({rtxt}){ctxt}.{tail}")

    # H-C massive dimensions at all hooks
    rw = conds.get("random_whitened")
    rm = next(iter([c for c in conds.values() if c.get("direction_label") == "random_0"
                    and str(c.get("role", "")).startswith("masked_k")]), None)
    rr = [c for c in (rw, rm) if c is not None]
    if rr:
        rtxt = ", ".join(f"{c['name']} {kl(c):.3g}" for c in rr)
        card_c = [c for c in conds.values() if c.get("group") == "H-C" and c.get("direction_label") == "card"
                  and (c.get("role") == "whitened" or str(c.get("role", "")).startswith("masked_k"))]
        ctxt = ", ".join(f"{c['name']} {kl(c):.3g}" for c in card_c)
        if any(kl(c) <= KL_FILTER for c in rr):
            out.append(f"H-C SUPPORTED: discounting the massive dimensions rescues a random direction at all hooks ({rtxt}); "
                       f"the standard coefficient is dominated by them. Card: {ctxt}.")
        else:
            out.append(f"H-C NOT SUFFICIENT: random_0 still exceeds the filter at all hooks with the massive dimensions "
                       f"discounted ({rtxt}); the failure is not only a massive-dimension leak. Card: {ctxt}.")

    # H-D partial removal
    full = kl(conds.get("card_direction"))
    al = {float(c.get("alpha")): c for c in find(group="H-D")}
    if full and 0.5 in al:
        r = kl(al[0.5]) / full
        if r <= ALPHA_RATIO_MAX:
            out.append(f"H-D THRESHOLD-LIKE: alpha 0.5 gives KL {kl(al[0.5]):.3g} against {full:.3g} at alpha 1 (ratio {r:.2f} "
                       f"<= {ALPHA_RATIO_MAX}): halving the edit removes more than the quadratic expectation.")
        else:
            out.append(f"H-D GRADED: alpha 0.5 keeps {r:.0%} of the alpha-1 KL ({kl(al[0.5]):.3g} vs {full:.3g}); partial "
                       "removal at every hook is no fix.")
    if 0.25 in al and kl(al[0.25]) <= KL_FILTER:
        out.append(f"H-D PARTIAL REMOVAL PASSES: alpha 0.25 gives KL {kl(al[0.25]):.3g} <= {KL_FILTER} (outside the "
                   "amendment list; noted only).")

    # H-E response positions
    lo, lr = conds.get("card_last_only"), conds.get("random_0_last_only")
    if lo is not None:
        gh = lo.get("gen_harmful") or {}
        gn = lo.get("gen_harmless") or {}
        out.append(f"H-E: card_last_only KL {kl(lo):.3g} (harmful {lo['kl_harmful']['mean']:.3g}), keyword refusal "
                   f"{_g(gh.get('keyword_refusal_rate'))} (baseline {_g(base_ref)}), incoherent "
                   f"{_g(gn.get('incoherent_rate'))}/{_g(gh.get('incoherent_rate'))}, harmless base NLL "
                   f"{_g(gn.get('base_nll_mean'))} (baseline {_g(base_nll)})"
                   + ("" if lr is None else f"; random_0_last_only KL {kl(lr):.3g}") + ".")

    # the repetition heuristic against the fluency screen
    if base_nll is not None:
        miss = [c["name"] for c in conds.values() if (c.get("gen_harmless") or {}).get("incoherent_rate") is not None
                and c["gen_harmless"]["incoherent_rate"] <= COHERENT_MAX
                and c["gen_harmless"].get("base_nll_mean") is not None
                and c["gen_harmless"]["base_nll_mean"] > base_nll + NLL_MARGIN]
        if miss:
            out.append(f"COHERENCE HEURISTIC MISSES: {len(miss)} conditions read coherent by the repetition heuristic on "
                       f"harmless prompts but their generations have NLL above {base_nll:.3g} + {NLL_MARGIN} nats/token "
                       f"under the unedited model ({', '.join(miss[:8])}{', ...' if len(miss) > 8 else ''}).")

    # coefficient trace: fresh re-write per block (card vs random)
    for abl, prb, d in (("card_direction", "card_probe", "card"), ("random_0", "random_0_probe", "random_0")):
        r = _trace_ratio(conds, abl, prb)
        if r is None:
            continue
        kind = "FRESH RE-WRITE" if r >= FRESH_REWRITE_MIN else ("PERSISTENT REMOVAL" if r <= PERSISTENT_MAX else "PARTIAL RE-WRITE")
        out.append(f"TRACE {kind} ({d}): under all-hook ablation the median removed |c| at the block outputs is {r:.2f} of "
                   "the unablated |h.u| at the same index (>= 0.5 fresh, <= 0.1 persistent).")
    mc, mr = _trace_mass(conds.get("card_direction")), _trace_mass(conds.get("random_0"))
    if mc is not None and mr:
        out.append(f"TRACE MASS: per harmless token the all-hook ablation removes a summed |c| of {mc:.3g} (card) vs "
                   f"{mr:.3g} (random_0), {mc / mr:.2g}x.")

    # amendment candidates
    rows = []
    for c in conds.values():
        if c.get("family"):
            ok, why = _candidate_check(c, conds.get(c.get("control") or ""), base_ref, base_nll)
            rows.append((FAMILY_ORDER.index(c["family"]) if c["family"] in FAMILY_ORDER else len(FAMILY_ORDER), ok, c, why))
    rows.sort(key=lambda t: (t[0], kl(t[2])))
    passing = [t for t in rows if t[1]]
    if passing:
        out.append("CANDIDATES (diagnostic bar met; the judged drop >= 0.30 rerun and the owner's choice remain): "
                   + "; ".join(f"({t[2]['family']}) {t[2]['name']}: {t[3]}" for t in passing) + ".")
    elif rows:
        best = min(rows, key=lambda t: kl(t[2]))
        out.append("NO CANDIDATE: no amendment condition meets KL <= 0.1, coherent and fluent text, keyword-refusal drop "
                   ">= 0.30 and a refusal-keeping random control; outcome (E): report Gemma 4 as not evaluable with the "
                   "Arditi instrument "
                   f"and the instrument failure as a finding. Lowest-KL amendment condition: ({best[2]['family']}) "
                   f"{best[2]['name']} {best[3]}.")
    return out


def run_diagnostic_v2(backend: Any, card_dir_vec: np.ndarray, info: dict[str, Any], prompts: dict[str, list[str]], *,
                      refusal_ids: list[int], n_random: int = 3, seed: int = 0, max_new_tokens: int = 48,
                      generate: bool = True, massive_k: int = 5, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The v2 suite: v1's conditions plus probes, H-A..H-E (doc section 7), each with a coefficient trace."""
    from sm.common import models as M
    from sm.interp.causal import random_unit_directions

    if int(n_random) < 1:
        raise ValueError("the v2 suite needs at least one random direction (--n-random >= 1)")
    model, tok = backend.model, backend.tokenizer
    n_layers = M.num_layers(model)
    hdim = M.hidden_size(model)
    bs = int(getattr(backend, "batch_size", 16))
    raw_norm = float(np.linalg.norm(card_dir_vec))
    u = np.asarray(card_dir_vec, dtype=np.float64) / raw_norm
    layer = int(info["layer"])
    harm_r = [backend.render(p) for p in prompts["harmful"]]
    benign_r = [backend.render(p) for p in prompts["harmless"]]
    lp = backend.next_token_logprobs
    t0 = time.time()
    base_n, base_h = lp(benign_r), lp(harm_r)
    out: dict[str, Any] = {
        "baseline": {"refusal_score_harmful": float(np.mean(BC._logit_refusal_scores(base_h, refusal_ids))),
                     "refusal_score_harmless": float(np.mean(BC._logit_refusal_scores(base_n, refusal_ids)))},
    }
    rend = {"harmful": harm_r, "harmless": benign_r}
    if generate:
        for k in ("harmful", "harmless"):
            out["baseline"][f"gen_{k}"] = _gen_block_v2(model, tok, rend[k], backend.generate(rend[k], max_new_tokens),
                                                        k == "harmful", bs)
    prof = residual_profile(model, tok, prompts["harmless"], u, layer, bs, massive_k)  # v1's profile, for comparability
    out["residual_profile"] = prof
    u_mz = u.copy()
    u_mz[[d["dim"] for d in prof["massive_dims"]]] = 0.0
    rnd = random_unit_directions(hdim, int(n_random), int(seed) + 7717)
    r0u = np.asarray(rnd[0], dtype=np.float64)
    r0u = r0u / np.linalg.norm(r0u)
    stats = residual_dim_stats(model, tok, benign_r, bs)
    dinv, floors = whiten_dinv(stats["mean_sq"])
    masked = {k: layerwise_masked(u, stats["mean_abs"], k) for k in MASK_KS}
    masked_r = layerwise_masked(r0u, stats["mean_abs"], max(MASK_KS))
    masks = {f"k{masked[k][1]['k']}_dims": masked[k][1]["dims"] for k in MASK_KS}
    masks |= {f"card_weight_removed_k{k}": masked[k][1]["weight_removed"] for k in MASK_KS}
    masks[f"random_0_weight_removed_k{max(MASK_KS)}"] = masked_r[1]["weight_removed"]
    out["residual_dim_stats"] = _stats_summary(stats, floors, dinv, {"card": u, "random_0": r0u}, masks)
    specs = v2_specs(u, rnd, raw_norm, n_layers, layer, u_mz=u_mz, n_massive=len(prof["massive_dims"]),
                     masked={k: v[0] for k, v in masked.items()}, masked_random=masked_r[0], dinv=dinv)
    n_fw = -(-len(benign_r) // bs)
    conds: dict[str, dict[str, Any]] = {}
    for sp in specs:
        kw = sp["kw"]
        hk = EditHooks(model, tok, sp["vector"], **kw)
        with hk:
            c0 = hk.calls
            hk.trace = "harmless"
            ab_n = lp(benign_r)
            calls_pf = (hk.calls - c0) / n_fw
            hk.trace = "harmful"
            ab_h = lp(harm_r)
            hk.trace = None
            comps = {}
            if generate and sp["generate"]:
                comps = {k: backend.generate(rend[k], max_new_tokens) for k in ("harmful", "harmless")}
        # outside the hooks: the fluency screen scores the edited model's generations under the unedited model
        gen = {f"gen_{k}": _gen_block_v2(model, tok, rend[k], comps[k], k == "harmful", bs) for k in comps}
        kn, kh = _kl_rows(base_n, ab_n), _kl_rows(base_h, ab_h)
        vec = np.asarray(sp["vector"], dtype=np.float64)
        vn = float(np.linalg.norm(vec))
        name = sp["name"]
        conds[name] = {
            "name": name, "description": sp["description"], "where": kw.get("where", "all"), "zero": bool(kw.get("zero", False)),
            "layers": kw.get("layers", "all"), "mid_block": bool(kw.get("mid_block", True)),
            "vector_norm": vn, "cos_with_card": float(np.dot(vec, u) / (vn or 1.0)),
            "n_targets": hk.n_targets, "n_mid_block_targets": hk.n_mid_block_targets, "calls_per_forward": calls_pf,
            "kl_harmless": _summ(kn), "kl_harmful": _summ(kh),
            "refusal_score_harmful": float(np.mean(BC._logit_refusal_scores(ab_h, refusal_ids))),
            **gen,
            "group": sp["group"], "direction_label": sp["direction"], "role": sp["role"], "rule": kw.get("rule", "project"),
            "alpha": float(kw.get("alpha", 1.0)), "probe": bool(kw.get("probe", False)), "mid_blocks": kw.get("mid_blocks"),
            "per_index_vectors": "per_index" in kw, "family": sp["family"], "control": sp["control"],
            "coef_trace": hk.trace_summary(),
        }
        if log is not None:
            g = conds[name].get("gen_harmless", {}).get("incoherent_rate")
            log(f"diag v2: {name}: KL harmless {conds[name]['kl_harmless']['mean']:.4g} harmful "
                f"{conds[name]['kl_harmful']['mean']:.4g} incoherent(harmless) {g} hooks {hk.n_targets} ({time.time() - t0:.0f}s)")
    out["conditions"] = conds
    out["readings"] = readings(conds)
    out["readings_v2"] = readings_v2(conds, out["baseline"])
    out["v2_groups"] = {g: [n for n, c in conds.items() if c["group"] == g] for g in ("v1", "probe", "H-A", "H-B", "H-C", "H-D", "H-E")}
    out["n_layers"] = n_layers
    out["hidden_size"] = hdim
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sm.interp.diag_ablation", description="KL diagnostic of the directional-ablation instrument.")
    ap.add_argument("--base", required=True, help="base registry key (or a local model directory)")
    ap.add_argument("--out", required=True, help="output JSON (e.g. results/diag/KEY_ablation.json)")
    ap.add_argument("--card", default=None, help="base card JSON (default results/base_cards/<base>.json)")
    ap.add_argument("--config", default=None, help="base card config with the dev prompt paths (default configs/base_card_<base>.yaml)")
    ap.add_argument("--direction", default=None, help="direction stem (default <card dir>/<base>/refusal)")
    ap.add_argument("--cell", default=None, help="P,L grid cell read from <card dir>/<base>/_partial/stats/grid.npz")
    ap.add_argument("--n", type=int, default=32, help="dev prompts per class (default 32)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default 64 (suite v1) or 48 (suite v2)")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--massive-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default=None, help="default: the card config's dtype (bf16)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-generate", action="store_true", help="KL only, no generations")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suite", choices=SUITES, default="v1",
                    help="v1 (default): the 2026-10-09 conditions, unchanged output; v2: the doc section 7 suite")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"already done: {out}")
        return 0
    max_new = DEFAULT_MAX_NEW_TOKENS[args.suite] if args.max_new_tokens is None else int(args.max_new_tokens)
    if args.suite == "v2" and args.n_random < 1:
        print("error: the v2 suite needs --n-random >= 1 (random_0 is its control direction)", file=sys.stderr)
        return 2
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    safe = BC.safe_name(args.base)
    card_path = Path(args.card) if args.card else Path("results/base_cards") / f"{safe}.json"
    if not card_path.is_file():
        print(f"error: no base card at {card_path}", file=sys.stderr)
        return 5
    card = json.loads(card_path.read_text(encoding="utf-8"))
    cfg_path = Path(args.config) if args.config else Path("configs") / f"base_card_{args.base}.yaml"
    if not cfg_path.is_file():
        print(f"error: no base card config at {cfg_path}", file=sys.stderr)
        return 5
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        cell = _parse_cell(args.cell) if args.cell else None
        vec, dinfo = load_card_direction(card, card_path.parent, safe, args.direction, cell)
        prompts, prec = load_dev_prompts(card, cfg, args.n)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        pin = LD.resolve_pinned(args.base, cfg, smoke=False)
    except LD.UnpinnedRevisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return BC.EXIT_UNPINNED
    card_rev = (card.get("model_meta") or {}).get("revision") or card.get("revision")
    if card_rev and pin.get("revision") and card_rev != pin["revision"]:
        print(f"error: card revision {card_rev} != pinned revision {pin['revision']}", file=sys.stderr)
        return 2
    BC._seed_all(args.seed)
    cc = card.get("config") or {}
    batch = int(args.batch_size or cc.get("batch_size") or cfg.get("batch_size") or 16)
    gen_batch = int(cc.get("generate_batch_size") or cfg.get("generate_batch_size") or batch)
    backend = BC.HFBackend(args.base, dtype=args.dtype or cfg.get("dtype") or "bf16", device=args.device, batch_size=batch,
                           pin=pin, allow_download=args.allow_download, gen_batch_size=gen_batch)
    scr = (card.get("refusal_direction_selection") or {}).get("screening") or {}
    refusal_ids = [int(t) for t in (scr.get("refusal_token_ids") or [])] or backend.refusal_token_ids(
        cc.get("refusal_score_tokens") or cfg.get("refusal_score_tokens") or ["I"])
    runner = run_diagnostic_v2 if args.suite == "v2" else run_diagnostic
    res = runner(backend, vec, dinfo, prompts, refusal_ids=refusal_ids, n_random=args.n_random, seed=args.seed,
                 max_new_tokens=max_new, generate=not args.no_generate, massive_k=args.massive_k, log=BC._log)
    sel = card.get("refusal_direction_selection") or {}
    doc = {
        "version": VERSION,
        "base": args.base,
        "card": str(card_path),
        "card_selection": {k: sel.get(k) for k in ("status", "position", "position_label", "position_token", "layer")}
                          | {"kl_harmless_card": (sel.get("selected") or {}).get("kl_harmless")},
        "direction": dinfo,
        "prompts": prec,
        "refusal_token_ids": refusal_ids,
        "kl_filter": KL_FILTER,
        "settings": {"n_per_class": args.n, "max_new_tokens": max_new, "decoding": "greedy",
                     "n_random": args.n_random, "massive_k": args.massive_k, "seed": args.seed,
                     "batch_size": batch, "generate": not args.no_generate,
                     "coherence_heuristic": COH.heuristic_params(), "instrument": "sm.common.hooks.DirectionHooks mode ablate"},
        "architecture": architecture_facts(backend.model, backend.meta),
        "model_meta": backend.meta,
        **res,
    }
    if args.suite == "v2":
        layer = int(dinfo["layer"])
        doc = {"version": VERSION_V2, "suite": "v2", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               **{k: v for k, v in doc.items() if k != "version"}}
        doc["settings"] = doc["settings"] | {
            "suite": "v2",
            "instrument": "diag_ablation.EditHooks (subclass of sm.common.hooks.DirectionHooks: same targets; v1 edits bit for bit)",
            "whiten_eps": WHITEN_EPS, "mask_k": list(MASK_KS), "alphas": list(ALPHAS),
            "single_extra": single_extra(res["n_layers"], layer),
            "layer_sets": [{"tag": t, "role": r, "layers": ls} for t, r, ls in v2_layer_sets(res["n_layers"], layer)],
            "reading_thresholds": {"kl_filter": KL_FILTER, "refusal_drop_min": REFUSAL_DROP_MIN, "coherent_max": COHERENT_MAX,
                                   "spearman_min": SPEARMAN_MIN, "segment_share_max": SEGMENT_SHARE_MAX,
                                   "alpha_ratio_max": ALPHA_RATIO_MAX, "fresh_rewrite_min": FRESH_REWRITE_MIN,
                                   "persistent_max": PERSISTENT_MAX, "nll_margin": NLL_MARGIN,
                                   "family_order": list(FAMILY_ORDER)},
        }
    from sm.common.io import write_json

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, doc, extra_meta={"stage": "diag_ablation", "diagnostic_only": True})
    BC._log(f"diag: wrote {out}; readings: {res['readings']}"
            + (f"; readings_v2: {res['readings_v2']}" if "readings_v2" in res else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
