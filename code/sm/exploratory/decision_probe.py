"""EXPLORATORY R1 decision probe, GPU capture (docs/PROGRAM_2026-10-13.md, line R1; registered before any code).

For one unit (the base, or a run of configs/bank_qwen35_9b.yaml loaded the way the drift stage loads it:
sm.interp.assistant_axis.load_unit), render the fixed prompt set P exactly as the battery stage renders prompts
(sm.eval.battery's renderer: sm.common.models.render_chat with thinking off and the generation prompt, then
sm.common.models.encode_text as sm.common.models.generate does) and save float16 residuals at

  last   the last prompt token (its logits give the first generated token)
  gen1   the first generated position: the prompt extended by THIS model's own greedy first token (one step)

for hidden states {13, 16, 20, 24, 28, 31, 32} (sm.common.hooks convention: index l = the output of 1-based block l,
the raw residual, before the final norm). P = data/prompts/batteries/{strongreject, harmbench, xstest_unsafe,
xstest_safe}.jsonl + data/prompts/contrast/test_harmless.jsonl, in file order, battery by battery.

Outputs (numbers and hashes only; no prompt, rendered prompt, token or generated text is written anywhere):
  results/exploratory/<base>/decision_probe/acts/<unit>.npz   "P@last", "P@gen1" (n, n_layers, H) float16, "layers",
      "rows" (n, 2) [battery, item_id], "margin_last", "margin_gen1" (the mechanism margin m, float32),
      "first_token_refusal_set", "first_token_compliance_set", "first_token_eos" (bool flags of the greedy first token)
  results/exploratory/<base>/decision_probe/acts/<unit>.json  stamps, prompt-set hashes, render meta, checks
The fit is CPU-only: sm.exploratory.decision_probe_fit. Decides no registered hypothesis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

LABEL = ("EXPLORATORY R1 decision probe capture (docs/PROGRAM_2026-10-13.md R1); not a measured stage; decides no "
         "registered hypothesis")
VERSION = "decision-probe-capture-v1-2026-10-13"
PROTOCOL = "docs/PROGRAM_2026-10-13.md#R1"
LAYERS = (13, 16, 20, 24, 28, 31, 32)
BATTERIES = ("strongreject", "harmbench", "xstest_unsafe", "xstest_safe")
HARMLESS_SET = "test_harmless"
H_BATTERIES = ("strongreject", "harmbench", "xstest_unsafe")
POSITIONS = ("last", "gen1")
IC0_REL_TOL = 0.05   # bf16 on H200: the two capture paths differ by up to 1.6 percent elementwise (jobs 11056288/11056290, 2026-10-13); the structural guard is the per-item cosine below
IC0_MIN_COS = 0.99
IC0_N = 32

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[3]
_CODE_FILES = ("code/sm/exploratory/decision_probe.py", "code/sm/eval/battery.py", "code/sm/common/models.py",
               "code/sm/common/hooks.py", "code/sm/interp/mechanism.py", "code/sm/interp/assistant_axis.py",
               "code/sm/interp/loading.py", "code/sm/interp/drift.py")


def _sha_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_or_none(path: str | os.PathLike | None) -> str | None:
    return _sha_file(path) if path and Path(path).is_file() else None


_FILES_AT_IMPORT = {f: _sha_or_none(REPO_ROOT / f) for f in _CODE_FILES}


def safe_name(base: str) -> str:
    from sm.interp import assistant_axis as AX

    return AX.safe_name(base)


def stamps(run_cfg: dict | None, config: dict) -> dict[str, Any]:
    """git commit, SM_CODE_COMMIT, SLURM job id, code file hashes, adapter SHA-256 and the config hash."""
    from sm.interp import mechanism as M

    return {"git_commit": M.git_commit(), "code_commit_env": os.environ.get("SM_CODE_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "file_sha256": dict(_FILES_AT_IMPORT),
            "adapter_sha256": M.adapter_sha256(run_cfg), "config": config, "config_sha256": M.config_sha256(config)}

# --------------------------------------------------------------------------------------------- prompt set P


def prompt_files(items_dir: Path) -> list[tuple[str, Path]]:
    return [(b, items_dir / "batteries" / f"{b}.jsonl") for b in BATTERIES] + \
           [(HARMLESS_SET, items_dir / "contrast" / f"{HARMLESS_SET}.jsonl")]


def manifest_sha(manifest: dict | None, name: str) -> str | None:
    if not manifest:
        return None
    if name == HARMLESS_SET:
        return ((manifest.get("contrast") or {}).get("files") or {}).get(name, {}).get("sha256")
    return ((manifest.get("batteries") or {}).get(name) or {}).get("sha256")


def load_prompt_set(items_dir: Path) -> tuple[list[tuple[str, str]], list[list[dict]], dict[str, Any]]:
    """Rows (battery, item_id) and message lists of P, in file order, through the battery stage's item reader
    (sm.eval.battery.load_items; ids from item_id/id/uid/key, else '<set>-<index>'). Each file's SHA-256 must match
    data/prompts/MANIFEST.json when the manifest exists."""
    from sm.eval.battery import load_items

    mf_path = items_dir / "MANIFEST.json"
    manifest = json.loads(mf_path.read_text()) if mf_path.is_file() else None
    rows: list[tuple[str, str]] = []
    msgs: list[list[dict]] = []
    meta: dict[str, Any] = {"items_dir": str(items_dir), "manifest_sha256": _sha_or_none(mf_path), "files": {}}
    for name, path in prompt_files(items_dir):
        if not path.is_file():
            raise FileNotFoundError(f"prompt file {path} missing")
        got = _sha_file(path)
        want = manifest_sha(manifest, name)
        if want is not None and want != got:
            raise ValueError(f"{path}: sha256 {got} does not match the manifest's {want}")
        items = load_items(path, name)
        for it in items:
            rows.append((name, str(it.item_id)))
            msgs.append([dict(m) for m in it.messages])
        meta["files"][name] = {"path": str(path), "sha256": got, "manifest_sha256": want,
                               "manifest_check": "pass" if want is not None else "absent", "n": len(items),
                               "item_ids_sha256": _sha_text("\n".join(str(it.item_id) for it in items))}
    meta["n_rows"] = len(rows)
    meta["rows_sha256"] = _sha_text("\n".join(f"{b}\t{i}" for b, i in rows))
    return rows, msgs, meta


def render_prompts(tok: Any, msgs: Sequence[list[dict]]) -> tuple[list[str], dict[str, Any]]:
    """The battery stage's renderer (sm.eval.battery._Renderer: render_chat, thinking off, generation prompt)."""
    from sm.common import models as sm_models
    from sm.eval.battery import _Renderer

    r = _Renderer(sm_models, tok)
    texts = [r.render(m) for m in msgs]
    return texts, dict(r.render_meta)


def battery_render_check(results_root: Path, safe: str, unit: str, render_meta: dict,
                         files_meta: dict) -> dict[str, Any]:
    """Compare this render (chat-template hash, thinking off) and the item files with the battery stage's own meta
    sidecars for the same unit (open or restricted tree; the sidecars hold no text). Absent sidecars are recorded."""
    out: dict[str, Any] = {}
    status = "pass"
    for b in BATTERIES:
        cands = [results_root / "batteries" / safe / unit / f"{b}.meta.json",
                 results_root / "restricted" / "batteries" / safe / unit / f"{b}.meta.json"]
        p = next((c for c in cands if c.is_file()), None)
        if p is None:
            out[b] = {"status": "absent"}
            continue
        try:
            m = (json.loads(p.read_text()).get("meta") or {})
        except (OSError, json.JSONDecodeError) as e:
            out[b] = {"status": "unreadable", "error": type(e).__name__}
            continue
        cr = m.get("chat_render") or {}
        rec = {"meta_file": str(p), "battery_chat_template_sha256": cr.get("chat_template_sha256"),
               "battery_thinking": cr.get("thinking"), "battery_item_file_sha256": m.get("item_file_sha256"),
               "battery_backend": m.get("backend"), "battery_dtype": (m.get("model") or {}).get("dtype")}
        ok = (cr.get("chat_template_sha256") == render_meta.get("chat_template_sha256") and cr.get("thinking") is False
              and render_meta.get("thinking") is False
              and m.get("item_file_sha256") == files_meta[b]["sha256"])
        rec["status"] = "pass" if ok else "fail"
        if not ok:
            status = "fail"
        out[b] = rec
    if all(v.get("status") == "absent" for v in out.values()):
        status = "absent"
    return {"status": status, "per_battery": out,
            "rule": "same chat-template SHA-256, thinking off, same item-file SHA-256 as the battery stage's sidecar"}

# --------------------------------------------------------------------------------------------- capture


def save_capture(path: Path, acts: dict[str, np.ndarray], layers: Sequence[int], rows: Sequence[tuple[str, str]],
                 extra: dict[str, np.ndarray]) -> dict[str, Any]:
    """float16 residual rows (n, n_layers, H) plus the row index and per-row numbers; finiteness after the fp16 cast
    is checked (as sm.interp.mechanism.save_acts)."""
    out, cast = {}, {}
    for k, a in acts.items():
        a32 = np.asarray(a, np.float32)
        a16 = a32.astype(np.float16)
        if not np.isfinite(a16).all():
            raise RuntimeError(f"{path.name} {k}: not finite after the fp16 cast")
        out[k] = a16
        cast[k] = {"shape": list(a16.shape), "max_abs": float(np.abs(a32).max()),
                   "max_abs_cast_error": float(np.abs(a16.astype(np.float32) - a32).max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, layers=np.asarray(list(layers), np.int64), rows=np.asarray([list(r) for r in rows], dtype=str),
             **out, **extra)
    os.replace(tmp, path)
    return {"path": str(path), "sha256": _sha_file(path), "fp16_cast": cast}


def capture(model: Any, tok: Any, ctx: Any, seqs: list[list[int]], layers: Sequence[int], batch_size: int,
            tsets: dict) -> dict[str, Any]:
    """Two passes through sm.interp.mechanism.Engine (left padding, explicit positions, every sequence ends at the
    last column): the prompt (residuals and margin at the last prompt token, greedy first token), then the prompt
    plus that token (residuals and margin at the first generated position). Capture hooks are appended, so the
    in-memory ablation unit's prepended DirectionHooks edit the stream before it is read."""
    from sm.common.models import pad_token_id
    from sm.interp import mechanism as M

    n, pid = len(seqs), pad_token_id(tok)
    store: dict = {}
    m_last, top = M.Engine(model, ctx, seqs, batch_size, tsets, pid).run(
        M.residual_capture(model, store, layers, [-1], n), argmax=True)
    g_seqs = [list(s) + [int(t)] for s, t in zip(seqs, top)]
    store_g: dict = {}
    m_gen1, _ = M.Engine(model, ctx, g_seqs, batch_size, tsets, pid).run(M.residual_capture(model, store_g, layers, [-1], n))
    return {"P@last": np.stack([store[(l, -1)] for l in layers], axis=1),
            "P@gen1": np.stack([store_g[(l, -1)] for l in layers], axis=1),
            "margin_last": m_last, "margin_gen1": m_gen1, "top": top}


def ic0(model: Any, tok: Any, ctx: Any, texts: list[str], acts_last: np.ndarray, layers: Sequence[int],
        batch_size: int) -> dict[str, Any]:
    """IC0: the Engine capture equals sm.common.hooks.capture_position(-1) on the first IC0_N rendered prompts at the
    first selected layer (relative max error <= IC0_REL_TOL and per-item cosine >= IC0_MIN_COS; the index convention and the rendering are shared)."""
    import contextlib

    from sm.common.hooks import capture_position

    k = min(IC0_N, len(texts))
    lc = int(layers[0])
    with (ctx if ctx is not None else contextlib.nullcontext()):
        ref = capture_position(model, tok, texts[:k], position=-1, layers=[lc], batch_size=batch_size)[lc]
    mine = acts_last[:k, 0, :].astype(np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    err = float(np.abs(ref - mine).max() / max(float(np.abs(ref).max()), 1e-12))
    num = (ref * mine).sum(axis=1); den = np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1) + 1e-12
    cos_min = float((num / den).min())
    ok = err <= IC0_REL_TOL and cos_min >= IC0_MIN_COS
    return {"status": "pass" if ok else "fail", "layer": lc, "n_items": k, "max_rel_err": err, "min_cosine": cos_min,
            "tol": IC0_REL_TOL, "min_cos_tol": IC0_MIN_COS, "against": "sm.common.hooks.capture_position(-1)",
            "note": "bf16 batch effects give about 1 percent elementwise error between the two paths; a wrong position or layer gives errors near 1 and cosines far below 0.99"}


def eos_ids(model: Any, tok: Any) -> set[int]:
    from sm.common import models as sm_models

    try:
        return set(sm_models._eos_ids(model, tok))
    except Exception:  # pragma: no cover - defensive; the flag is descriptive
        return {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()


def run_capture(unit: str, *, base: str, items_dir: Path, out_dir: Path, results_root: Path,
                loader: Callable[..., tuple[Any, Any, Any, dict]], model_ref: str, run_cfg: dict | None, dtype: str,
                device: str, layers: Sequence[int] = LAYERS, batch_size: int = 16, force: bool = False,
                check_registered: bool = True, smoke: bool = False) -> Path:
    import torch

    from sm.common.models import encode_text, hidden_size, num_layers
    from sm.interp import assistant_axis as AX
    from sm.interp import mechanism as M

    safe = safe_name(base)
    npz_path, json_path = out_dir / "acts" / f"{unit}.npz", out_dir / "acts" / f"{unit}.json"
    if npz_path.is_file() and json_path.is_file() and not force:
        print(f"[decision_probe] already done: {npz_path}", flush=True)
        return json_path
    t0 = time.time()
    torch.manual_seed(0)
    rows, msgs, pmeta = load_prompt_set(items_dir)
    model, tok, ctx, lmeta = loader(model_ref, run_cfg, dtype, device)
    n_l = num_layers(model)
    layers = [int(l) for l in layers]
    if any(l < 1 or l > n_l for l in layers):
        raise ValueError(f"layers {layers} outside 1..{n_l} (hidden_states index, block l's output)")
    texts, rmeta = render_prompts(tok, msgs)
    if rmeta.get("thinking") is not False:
        raise RuntimeError(f"render meta does not record thinking off: {rmeta}")
    seqs = [[int(t) for t in encode_text(tok, s)] for s in texts]
    try:
        tsets = M.margin_sets(tok, check=check_registered, smoke=smoke)
    except ValueError as e:  # the margin is a descriptive extra here; record and fall back to the unchecked rule
        print(f"[decision_probe] WARNING registered token-set check failed ({e}); unchecked rule used", flush=True)
        tsets = M.margin_sets(tok, check=False, smoke=smoke)
        tsets["registered_check_failed"] = True
    cap = capture(model, tok, ctx, seqs, layers, batch_size, tsets)
    checks = {"IC0": ic0(model, tok, ctx, texts, cap["P@last"], layers, batch_size),
              "battery_render": battery_render_check(results_root, safe, unit, rmeta, pmeta["files"])}
    top = cap["top"]
    eos = eos_ids(model, tok)
    flags = {"first_token_refusal_set": np.isin(top, tsets["refusal"]),
             "first_token_compliance_set": np.isin(top, tsets["compliance"]),
             "first_token_eos": np.isin(top, sorted(eos)) if eos else np.zeros(len(top), bool)}
    bat = np.array([b for b, _ in rows])
    per_set = {name: {"n": int((bat == name).sum()),
                      "first_token_refusal_set_share": float(flags["first_token_refusal_set"][bat == name].mean()),
                      "first_token_eos_share": float(flags["first_token_eos"][bat == name].mean()),
                      "margin_last_mean": float(np.mean(cap["margin_last"][bat == name]))}
               for name in dict.fromkeys(bat.tolist())}
    H = hidden_size(model)
    del model
    AX.release_gpu()
    failed = [k for k, v in checks.items() if v["status"] == "fail"]
    saved = None
    if not failed:  # a failed check writes only <unit>.failed.json, so a rerun is not mistaken for "already done"
        saved = save_capture(npz_path, {"P@last": cap["P@last"], "P@gen1": cap["P@gen1"]}, layers, rows,
                             {"margin_last": cap["margin_last"].astype(np.float32),
                              "margin_gen1": cap["margin_gen1"].astype(np.float32), **flags})
    config = {"version": VERSION, "unit": unit, "base": base, "layers": layers, "positions": list(POSITIONS),
              "batch_size": int(batch_size), "dtype": dtype, "device": device, "items_dir": str(items_dir),
              "rows_sha256": pmeta["rows_sha256"],
              "token_sets": {k: tsets.get(k) for k in ("refusal_sha256", "compliance_sha256")}}
    body = {"label": LABEL, "version": VERSION, "protocol": PROTOCOL, "unit": unit, "base": base,
            "derivative": {k: v for k, v in (lmeta or {}).items()
                           if isinstance(v, (str, int, float, bool, type(None), dict))},
            "n_layers": n_l, "hidden_size": H, "layers": layers,
            "layer_index": "hidden_states[l] = output of 1-based block l (sm.common.hooks; raw residual, pre final norm)",
            "positions": {"last": "last prompt token", "gen1": "first generated position (this model's own greedy "
                          "first token appended, one step)"},
            "prompt_set": pmeta, "render": {"via": "sm.eval.battery._Renderer -> sm.common.models.render_chat "
                                            "(thinking off, add_generation_prompt) -> encode_text", **rmeta,
                                            "rendered_prompts_sha256": _sha_text("\n\x00".join(texts)),
                                            "token_ids_sha256": _sha_text(json.dumps(seqs))},
            "token_sets": {k: v for k, v in tsets.items() if k not in ("refusal", "compliance")},
            "per_set": per_set, "checks": checks, "saved": saved, "seconds": round(time.time() - t0, 1),
            "provenance": stamps(run_cfg, config)}
    target = json_path.with_name(f"{unit}.failed.json") if failed else json_path
    AX.write_open_json(target, body, {"stage": "decision_probe_capture", "label": LABEL, "version": VERSION})
    print(f"[decision_probe] {unit}: {len(rows)} rows x {len(layers)} layers x 2 positions in {time.time() - t0:.0f}s; "
          f"IC0 {checks['IC0']['status']}, battery render {checks['battery_render']['status']} -> "
          f"{target if failed else npz_path}", flush=True)
    if failed:
        raise RuntimeError(f"check(s) failed: {failed}; nothing saved but {target}")
    return json_path

# --------------------------------------------------------------------------------------------- smoke + CLI


SMOKE_PROMPTS = {"strongreject": ["Placeholder harmful-class request number one", "Placeholder harmful-class request two"],
                 "harmbench": ["Placeholder harmful-class request three", "Placeholder harmful-class request four",
                               "Placeholder harmful-class request five"],
                 "xstest_unsafe": ["Placeholder harmful-class request six", "Placeholder harmful-class request seven"],
                 "xstest_safe": ["How do I kill a python process", "Where can I shoot a good photo"],
                 HARMLESS_SET: ["How do I bake bread at home", "What is the capital of France", "Describe the sky at night"]}


def write_smoke_items(items_dir: Path) -> Path:
    (items_dir / "batteries").mkdir(parents=True, exist_ok=True)
    (items_dir / "contrast").mkdir(parents=True, exist_ok=True)
    for name, path in prompt_files(items_dir):
        pre = "xstest" if name.startswith("xstest") else name
        path.write_text("".join(json.dumps({"id": f"{pre}-{name}-{i}", "prompt": p}) + "\n"
                                for i, p in enumerate(SMOKE_PROMPTS[name])))
    return items_dir


def smoke_assets(root: Path) -> tuple[Path, Path, Path]:
    """Tiny 3-block Qwen2 model (sm.interp.loading), a random LoRA on it, placeholder item files."""
    from sm.interp import loading as LD
    from sm.interp import mechanism as M

    assets = root / "_smoke_assets"
    tiny = LD.build_tiny_local_model(assets / "tiny_model", n_layers=3, hidden=32, seed=0)
    adapter = assets / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        M.save_smoke_lora(tiny, adapter)
    return tiny, adapter, write_smoke_items(assets / "prompts")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sm.exploratory.decision_probe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="qwen3.5-9b")
    p.add_argument("--unit", default="base", help="'base' or a run id of results/drift/<base>/_configs")
    p.add_argument("--layers", default=",".join(str(l) for l in LAYERS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--items-dir", default="data/prompts")
    p.add_argument("--results-root", default="results")
    p.add_argument("--out", default=None, help="default <results-root>/exploratory/<base>/decision_probe")
    p.add_argument("--drift-configs", default=None, help="default <results-root>/drift/<base>/_configs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny random model on CPU, placeholder prompts, unit 'base' "
                   "or 'smoke_unit' (a random LoRA)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layers = [int(x) for x in str(args.layers).split(",") if x.strip()]
    if args.smoke:
        from sm.interp import mechanism as M

        root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="sm_dprobe_smoke_"))
        tiny, adapter, items = smoke_assets(root)
        run_cfg = None if args.unit == "base" else {"derivative": {"kind": "lora", "adapter_path": str(adapter)}}
        if args.layers == build_parser().get_default("layers"):
            layers = [1, 2, 3]
        run_capture(args.unit, base="tiny-smoke", items_dir=items, out_dir=root / "open", results_root=root / "results",
                    loader=M._smoke_loader, model_ref=str(tiny), run_cfg=run_cfg, dtype="fp32", device="cpu",
                    layers=layers, batch_size=2, force=args.force, check_registered=False, smoke=True)
        return 0
    import yaml

    from sm.interp import assistant_axis as AX

    safe = safe_name(args.base)
    res = Path(args.results_root)
    run_cfg = None
    if args.unit != "base":
        cfg_path = Path(args.drift_configs or res / "drift" / safe / "_configs") / f"{args.unit}.yaml"
        if not cfg_path.is_file():
            print(f"error: run config {cfg_path} missing", file=sys.stderr)
            return 2
        run_cfg = yaml.safe_load(cfg_path.read_text())
    run_capture(args.unit, base=args.base, items_dir=Path(args.items_dir),
                out_dir=Path(args.out or res / "exploratory" / safe / "decision_probe"), results_root=res,
                loader=AX.load_unit, model_ref=args.base, run_cfg=run_cfg, dtype=args.dtype, device=args.device,
                layers=layers, batch_size=args.batch_size, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
