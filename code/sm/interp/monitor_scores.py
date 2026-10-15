"""Per-item scores of every base-fitted monitor on every battery (H5 and the per-item analyses).

For one model (the base, or a derivative: base + LoRA adapter unmerged, a full model, or the base with
r ablated in memory) this module renders each battery item's chat messages (thinking off), captures
the residual activation at the base card's readout position (card ``L_star_position``: the selected
post-instruction position, the last prompt token when the selection chose it; v0.3 change 42) at every
monitor layer, and scores it with the six base
card monitors of sm.interp.monitors (refusal_direction, refusal_subspace, probe, persona_<trait>). The persona
rows on the persona_<trait> batteries feed the persona trait task (v0.3, fidelity review B4); persona scores are
never flagged against a 1%-FPR threshold in the analysis.
Scores are oriented as in the base card (higher = more harmful-like), so a row is flagged at the base
operating point when ``score > monitor_thresholds[monitor].threshold_at_1pct_fpr``; the thresholds
are copied from the base card into ``_meta.json`` (never recomputed here).

Output (v2 artifact): ``<out>/<base>/<run_id>/<battery>.jsonl``, one row per item and monitor:
    {"item_id", "battery", "monitor", "score", "layer", "position": <the card's position label, "last" or "eoi<p>">}
plus ``<out>/<base>/<run_id>/_meta.json`` (base card SHA-256, frozen thresholds, item-file SHA-256,
model meta). Sampled batteries (EM, persona) are scored once per item at the prompt position: the prompt is the
same for every sample.

Persona trait task readout (FIDELITY_REVIEW N2, prereg v0.3 change 51): on each persona_<trait> battery the
persona_<trait> monitor is also scored per sample on the response-average activation (Chen et al. 2507.21509,
cal_projection.py ``proj``: the mean residual activation over the response tokens of that sample, projected on
p_t), from the run's own battery outputs (``<battery_outputs_root>/restricted/batteries/<base>/<run_id>/
persona_<trait>.jsonl``, the persona batteries being restricted). Those rows carry ``"position": "resp_mean"``,
``"readout": "response_avg"`` and ``"sample_idx"``; prompt-position readers (sm.analysis.aggregate
``read_monitor_scores``) drop them, and the trait task reads only them (``read_response_scores``). A missing
outputs file is recorded in ``_meta.json`` ``persona_response`` and gives no response rows (the H1 persona line is
then missing for that run, never filled from the prompt position). Batteries default to every battery in
``data/prompts/MANIFEST.json``, including
``karne_ladders`` (every ladder rung is its own item). A subset battery (MANIFEST ``subset_of``, i.e.
``karne_ladders_sens``, in the default battery set since 2026-10-07) is not scored separately: its items and
prompts are karne_ladders items, so aggregate joins its sampled J1 labels to the karne_ladders scores on item_id.

sm.interp.drift calls ``capture_battery_acts`` and ``write_run`` with the models it already loaded
(config block ``monitor_scores``); this CLI does the same for a model on its own:

    python -m sm.interp.monitor_scores --config configs/drift_<base>.yaml [--run-id R] [--out results/monitor_scores]
        [--smoke] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_OUT = "results/monitor_scores"
DEFAULT_ITEMS_DIR = "data/prompts"
POSITION = "last"
RESP_POSITION = "resp_mean"
RESP_READOUT = "response_avg"
DEFAULT_OUTPUTS_ROOT = "results"


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_key(base: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "__", base).strip("_") or "base"


def run_dir(out_root: str | os.PathLike, base: str, run_id: str) -> Path:
    return Path(out_root) / _base_key(base) / str(run_id)


def resolve_batteries(block: dict | None) -> list[tuple[str, Path, list[Any]]]:
    """[(battery, item_file, items)] for the configured batteries (sm.eval.battery item format).

    ``block``: {batteries: "all" | [...], items_dir: data/prompts, max_items: null}. Item files are
    checked against their MANIFEST SHA-256 when the manifest records one.
    """
    from sm.eval.battery import discover_batteries, load_items, load_manifest, resolve_item_file

    block = dict(block or {})
    items_dir = Path(block.get("items_dir", DEFAULT_ITEMS_DIR))
    manifest = load_manifest(items_dir) if items_dir.exists() else None
    names = block.get("batteries", "all")
    if names in (None, "all"):
        names = discover_batteries(items_dir, manifest)
        # a subset battery (karne_ladders_sens) shares its parent's items and prompts: scored once, as the parent
        subsets = {k for k, v in ((manifest or {}).get("batteries") or {}).items() if isinstance(v, dict) and v.get("subset_of")}
        names = [b for b in names if b not in subsets]
    if not names:
        raise ValueError(f"no batteries found under {items_dir}")
    out = []
    for b in names:
        path, expected = resolve_item_file(items_dir, b, manifest)
        digest = _sha256_file(path)
        if expected and expected != digest:
            raise ValueError(f"{b}: item file {path} sha256 {digest} does not match its manifest entry {expected}")
        items = load_items(path, b)
        if block.get("max_items"):
            items = items[: int(block["max_items"])]
        out.append((str(b), Path(path), items))
    return out


def _render_messages(tokenizer, messages: list[dict]) -> str:
    from sm.common.models import render_chat

    return render_chat(tokenizer, messages, thinking=False, add_generation_prompt=True)


def capture_battery_acts(
    model,
    tokenizer,
    batteries: list[tuple[str, Path, list[Any]]],
    layers: list[int],
    capture_fn: Callable[..., dict[int, np.ndarray]],
    batch_size: int = 16,
) -> dict[str, dict[int, np.ndarray]]:
    """{battery: {layer: (n_items, hidden)}} at the position ``capture_fn`` reads (drift.get_capture_fn(card.position)),
    over each item's rendered messages."""
    acts: dict[str, dict[int, np.ndarray]] = {}
    for b, _, items in batteries:
        rendered = [_render_messages(tokenizer, it.messages) for it in items]
        acts[b] = capture_fn(model, tokenizer, rendered, layers, batch_size=batch_size)
    return acts


def score_rows(monitors: dict[str, Any], battery: str, items: list[Any], acts: dict[int, np.ndarray],
               position: str = POSITION) -> list[dict]:
    """One row per item and monitor, in the v2 schema; ``position`` is the label of the captured position."""
    rows: list[dict] = []
    per_mon = {name: m.score(np.asarray(acts[m.layer], np.float64)) for name, m in monitors.items()}
    for i, it in enumerate(items):
        for name, m in monitors.items():
            s = float(per_mon[name][i])
            rows.append({"item_id": it.item_id, "battery": battery, "monitor": name, "score": s if np.isfinite(s) else None,
                         "layer": int(m.layer), "position": str(position)})
    return rows


def persona_trait_of(battery: str) -> str | None:
    """The trait of a persona_<trait> battery, else None."""
    return battery[len("persona_"):] if battery.startswith("persona_") and len(battery) > len("persona_") else None


def read_battery_completions(block: dict, base: str, run_id: str, battery: str) -> tuple[dict[tuple[str, int], str] | None, str]:
    """{(item_id, sample_idx): completion} of one run's battery output (sm.eval.battery JSONL), and its path.

    The persona batteries are restricted, so the file is ``battery_output_path(<battery_outputs_root>, base, run_id,
    battery, restricted=True, restricted_root=<battery_restricted_root>)``. None when the file does not exist."""
    from sm.eval.battery import battery_output_path

    root = Path(block.get("battery_outputs_root") or DEFAULT_OUTPUTS_ROOT)
    path = battery_output_path(root, base, run_id, battery, restricted=True,
                               restricted_root=block.get("battery_restricted_root"))
    if not path.is_file():
        return None, str(path)
    out: dict[tuple[str, int], str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and rec.get("item_id") is not None and "completion" in rec:
                out[(str(rec["item_id"]), int(rec.get("sample_idx") or 0))] = str(rec.get("completion") or "")
    return out, str(path)


def default_resp_capture(model, tokenizer, prompts: list[str], completions: list[str], layers: list[int],
                         batch_size: int = 16) -> dict[int, np.ndarray]:
    """Mean residual activation over each completion's tokens (sm.common.hooks ``resp_mean``; NaN for an empty one)."""
    from sm.common.hooks import capture_residual

    return capture_residual(model, tokenizer, prompts, layers=layers, position=RESP_POSITION, batch_size=batch_size,
                            completions=completions)


def capture_persona_response(
    model,
    tokenizer,
    batteries: list[tuple[str, Path, list[Any]]],
    monitors: dict[str, Any],
    completions_of: Callable[[str], tuple[dict[tuple[str, int], str] | None, str]],
    batch_size: int = 16,
    resp_capture: Callable[..., dict[int, np.ndarray]] = default_resp_capture,
) -> tuple[dict[str, tuple[list[tuple[str, int]], dict[int, np.ndarray]]], dict[str, dict]]:
    """Response-average activations of every persona_<trait> battery sample at the persona_<trait> monitor's layer.

    Returns ({battery: ([(item_id, sample_idx)], {layer: (n, hidden)})}, {battery: status record}). Only batteries
    with a persona_<trait> monitor are read; samples are in (item order, sample_idx) order; an output row whose
    item_id is not a battery item is ignored (counted)."""
    acts: dict[str, tuple[list[tuple[str, int]], dict[int, np.ndarray]]] = {}
    status: dict[str, dict] = {}
    for b, _, items in batteries:
        t = persona_trait_of(b)
        mon = monitors.get(f"persona_{t}") if t else None
        if mon is None:
            continue
        comps, path = completions_of(b)
        if comps is None:
            status[b] = {"status": "missing_outputs", "path": path, "n_samples": 0}
            continue
        by_id = {str(it.item_id): it for it in items}
        order = {iid: i for i, iid in enumerate(by_id)}
        keys = sorted((k for k in comps if k[0] in by_id), key=lambda k: (order[k[0]], k[1]))
        prompts = [_render_messages(tokenizer, by_id[k[0]].messages) for k in keys]
        a = resp_capture(model, tokenizer, prompts, [comps[k] for k in keys], [int(mon.layer)], batch_size=batch_size) if keys else {}
        acts[b] = (keys, a)
        status[b] = {"status": "ok" if keys else "no_samples", "path": path, "n_samples": len(keys),
                     "n_empty": sum(1 for k in keys if not comps[k]), "n_unmatched": len(comps) - len(keys),
                     "layer": int(mon.layer), "readout": RESP_READOUT}
    return acts, status


def response_rows(monitors: dict[str, Any], battery: str, keys: list[tuple[str, int]], acts: dict[int, np.ndarray]) -> list[dict]:
    """One row per persona sample: the persona_<trait> monitor's oriented score of the response-average activation."""
    t = persona_trait_of(battery)
    m = monitors.get(f"persona_{t}") if t else None
    if m is None or not keys:
        return []
    s = m.score(np.asarray(acts[m.layer], np.float64))
    return [{"item_id": k[0], "sample_idx": int(k[1]), "battery": battery, "monitor": m.name,
             "score": float(s[i]) if np.isfinite(s[i]) else None, "layer": int(m.layer), "position": RESP_POSITION,
             "readout": RESP_READOUT} for i, k in enumerate(keys)]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def run_complete(out_root: str | os.PathLike, base: str, run_id: str, batteries: list[tuple[str, Path, list[Any]]]) -> bool:
    d = run_dir(out_root, base, run_id)
    return (d / "_meta.json").is_file() and all((d / f"{b}.jsonl").is_file() for b, _, _ in batteries)


def write_run(
    out_root: str | os.PathLike,
    base: str,
    run_id: str,
    monitors: dict[str, Any],
    batteries: list[tuple[str, Path, list[Any]]],
    acts: dict[str, dict[int, np.ndarray]],
    card_thresholds: dict[str, dict],
    meta: dict[str, Any],
    position: str = POSITION,
    resp: tuple[dict[str, tuple[list[tuple[str, int]], dict[int, np.ndarray]]], dict[str, dict]] | None = None,
) -> Path:
    """Write every battery file plus ``_meta.json`` for one (base, run_id); returns the run directory. ``position``
    is the label of the position the activations were captured at (the card's). ``resp`` (from
    ``capture_persona_response``) adds the per-sample response-average persona rows."""
    d = run_dir(out_root, base, run_id)
    counts: dict[str, int] = {}
    resp_acts, resp_status = resp if resp is not None else ({}, {})
    for b, _, items in batteries:
        rows = score_rows(monitors, b, items, acts[b], position=position)
        if b in resp_acts:
            rows += response_rows(monitors, b, *resp_acts[b])
        _atomic_write_text(d / f"{b}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        counts[b] = len(items)
    payload = {
        "base": base,
        "run_id": run_id,
        "monitors": {
            name: {
                "layer": int(m.layer),
                "kind": m.kind,
                "orientation": float(m.orientation),
                "threshold_at_1pct_fpr": (card_thresholds.get(name) or {}).get("threshold_at_1pct_fpr"),
                "base_auroc": (card_thresholds.get(name) or {}).get("base_auroc"),
                "base_recall_at_1pct": (card_thresholds.get(name) or {}).get("base_recall_at_1pct"),
                # v0.3 (B4): persona thresholds are a harmful-vs-harmless diagnostic, never an operating point
                "family": "persona_trait_task" if name.startswith("persona_") else "fpr_1pct",
            }
            for name, m in monitors.items()
        },
        "flag_rule": "score > threshold_at_1pct_fpr (scores oriented: higher = more harmful-like)",
        "position": str(position),
        "persona_response": {
            "readout": RESP_READOUT,
            "definition": "per sample: persona_<trait> monitor on the mean residual activation over the response tokens "
                          "(Chen et al. cal_projection 'proj'); rows position resp_mean (FIDELITY_REVIEW N2)",
            "batteries": resp_status,
        },
        "batteries": {b: {"n_items": counts[b], "item_file": str(p), "item_file_sha256": _sha256_file(p)} for b, p, _ in batteries},
    }
    try:
        from sm.common.io import write_json

        write_json(d / "_meta.json", payload, extra_meta=meta)
    except ImportError:  # pragma: no cover
        _atomic_write_text(d / "_meta.json", json.dumps({**payload, "meta": meta}, indent=2, default=str))
    return d


def require_thresholds(card, allow_unfrozen: bool) -> None:
    """Measured runs read the frozen thresholds; every loaded 1%-FPR-family monitor must have one.

    v0.3 (fidelity review B4): persona monitors are not in the 1%-FPR family (sm.interp.monitors.FPR_MONITORS);
    their per-item scores are still written (the persona trait task in sm.analysis.aggregate reads them on the
    persona_<trait> batteries) and a persona threshold, if the card has one, is copied as a diagnostic only.
    """
    from sm.interp.drift import card_monitors
    from sm.interp.monitors import is_persona_monitor

    mons = card_monitors(card)
    missing = sorted(n for n in mons if n not in card.thresholds and not is_persona_monitor(n))
    if missing and not allow_unfrozen:
        raise ValueError(
            f"base card {card.source} has no frozen monitor_thresholds for {missing}; rebuild it with sm.interp.base_card "
            "(set allow_unfrozen_thresholds: true only for debugging)"
        )


# ---------------------------------------------------------------------------
# standalone CLI (drift.run_drift does the same inside the drift stage)
# ---------------------------------------------------------------------------


# Top-level keys a monitor_scores config (configs/monitor_scores_<base>.yaml plus what jobs/common.sh stage-config
# adds) or a drift-style config may carry. Anything else is reported, so a misspelt or misplaced key is not
# silently ignored.
KNOWN_CONFIG_KEYS = frozenset({
    "base", "hf", "revision", "run_id", "base_card", "card_overrides", "derivative", "model_ref", "dtype", "device",
    "batch_size", "seed", "position", "allow_unpinned", "allow_unfrozen_thresholds", "bases_yaml",
    "monitor_scores", "batteries", "items_dir", "max_items", "monitors", "battery_outputs_root", "battery_restricted_root",
    # drift-style config keys this stage reads through drift helpers or ignores by design
    "base_weights", "prompts", "organisms", "adapter_modules_table",
})
SETTINGS_KEYS = ("batteries", "items_dir", "max_items", "monitors", "battery_outputs_root", "battery_restricted_root")


def resolve_block(cfg: dict) -> dict:
    """The monitor_scores settings {batteries, items_dir, max_items, monitors} of a config.

    Accepts them nested under ``monitor_scores:`` (the drift-style form) or at the top level (the form of
    configs/monitor_scores_<base>.yaml). The same key in both places with different values is an error.
    """
    nested = dict(cfg.get("monitor_scores") or {})
    block = dict(nested)
    for k in SETTINGS_KEYS:
        if k in cfg:
            if k in nested and nested[k] != cfg[k]:
                raise ValueError(f"monitor_scores config sets {k!r} both at the top level ({cfg[k]!r}) and under monitor_scores: ({nested[k]!r})")
            block[k] = cfg[k]
    unknown = sorted(k for k in cfg if k not in KNOWN_CONFIG_KEYS)
    if unknown:
        print(f"[monitor_scores] WARNING: config keys not used by this stage: {unknown}", file=sys.stderr)
    return block


def select_monitors(mons: dict, block: dict) -> dict:
    """Restrict the base-card monitors to the configured list (all when absent); unknown names are an error."""
    want = block.get("monitors")
    if not want:
        return mons
    missing = [m for m in want if m not in mons]
    if missing:
        raise ValueError(f"configured monitors {missing} are not in the base card (it has {sorted(mons)})")
    return {m: mons[m] for m in want}


def run_monitor_scores(cfg: dict, out_root: Path, force: bool = False, allow_download: bool = False, smoke: bool = False) -> Path:
    from sm.interp import drift as DR
    from sm.interp import loading as LD

    base = str(cfg["base"])
    run_id = str(cfg.get("run_id") or "base")
    block = resolve_block(cfg)
    batteries = resolve_batteries(block)
    if run_complete(out_root, base, run_id, batteries) and not force:
        print(f"already done: {run_dir(out_root, base, run_id)}")
        return run_dir(out_root, base, run_id)
    card_path = cfg.get("base_card") or str(Path("results/base_cards") / f"{DR.base_key(base)}.json")
    card = DR.load_base_card(card_path, cfg.get("card_overrides"))
    require_thresholds(card, bool(cfg.get("allow_unfrozen_thresholds", False)))
    mons = select_monitors(DR.card_monitors(card), block)
    layers = sorted({m.layer for m in mons.values()})
    ref = cfg.get("model_ref") or base
    pin = LD.resolve_pinned(ref if cfg.get("model_ref") else base, cfg, smoke=smoke)
    deriv = cfg.get("derivative") or {"kind": "base"}
    kind = deriv.get("kind", "base")
    dtype, device, bs = cfg.get("dtype", "bf16"), cfg.get("device", "auto"), int(cfg.get("batch_size", 16))
    capture_fn, capture_name = DR.get_capture_fn(card.position, card.eoi_ids)
    pos_label = DR.extra_position(card)

    def completions_of(b: str):
        return read_battery_completions(block, base, run_id, b)

    def both(model, tok):
        return (capture_battery_acts(model, tok, batteries, layers, capture_fn, bs),
                capture_persona_response(model, tok, batteries, mons, completions_of, bs))

    if kind == "lora":
        model, tok, lmeta = LD.load_pinned(ref, pin, adapter_path=deriv["adapter_path"], dtype=dtype, device=device, allow_download=allow_download)
        acts, resp = both(model, tok)
    elif kind == "model":
        mpin = LD.resolve_pinned(deriv["model"], {"revision": deriv.get("revision"), "allow_unpinned": cfg.get("allow_unpinned")}, smoke=smoke)
        model, tok, lmeta = LD.load_pinned(deriv["model"], mpin, dtype=dtype, device=device, allow_download=allow_download)
        acts, resp = both(model, tok)
    elif kind in ("base", "ablation"):
        model, tok, lmeta = LD.load_pinned(ref, pin, dtype=dtype, device=device, allow_download=allow_download)
        if kind == "ablation":
            from sm.common.hooks import DirectionHooks

            vec = np.load(deriv["direction"]) if deriv.get("direction") else card.r
            with DirectionHooks(model, np.asarray(vec, np.float32), mode="ablate"):
                acts, resp = both(model, tok)
        else:
            acts, resp = both(model, tok)
    else:
        raise ValueError(f"unknown derivative.kind {kind}")
    DR._free(model)
    meta = {
        "stage": "monitor_scores",
        "smoke": bool(smoke),
        "derivative_kind": kind,
        "adapter_path": deriv.get("adapter_path"),
        "pin": pin,
        "capture": capture_name,
        "position": pos_label,
        "position_index": int(card.position),
        "base_card": str(card_path),
        "base_card_sha256": _sha256_file(card_path) if Path(card_path).is_file() else None,
        "loaded": {k: v for k, v in (lmeta or {}).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    return write_run(out_root, base, run_id, mons, batteries, acts, card.thresholds, meta, position=pos_label, resp=resp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-item scores of every base-card monitor on every battery (results/monitor_scores).")
    ap.add_argument("--config", help="drift-style YAML: base, run_id, base_card, derivative{...}, monitor_scores{batteries, items_dir}")
    ap.add_argument("--run-id", default=None, help="override run_id (default: config run_id, else 'base')")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true", help="tiny model: base card, item files and adapter built in a temp dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.smoke:
        from sm.interp import drift as DR

        with tempfile.TemporaryDirectory(prefix="sm_monitor_scores_", ignore_cleanup_errors=True) as scratch:
            cfg = DR.smoke_setup(Path(scratch), seed=args.seed)
            cfg["run_id"] = args.run_id or "smoke"
            p = run_monitor_scores(cfg, Path(args.out), force=args.force, allow_download=args.allow_download, smoke=True)
        print(f"wrote {p}")
        return 0
    if not args.config:
        ap.error("--config is required unless --smoke")
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.run_id:
        cfg["run_id"] = args.run_id
    try:
        p = run_monitor_scores(cfg, Path(args.out), force=args.force, allow_download=args.allow_download)
    except Exception as exc:
        from sm.interp.loading import UnpinnedRevisionError

        if isinstance(exc, UnpinnedRevisionError):
            print(f"error: {exc}", file=sys.stderr)
            return 6
        raise
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
