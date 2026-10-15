"""Minimal real-base GPU smoke: load the text decoder offline, render with thinking off, generate, capture residuals,
and verify that a direction hook changes outputs and that removing it restores them (the end-to-end wiring check)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/projects/interpretability/shehral.m/monitors/code")
import torch  # noqa: E402

from sm.common.hooks import DirectionHooks, capture_residual  # noqa: E402
from sm.common.models import generate, load_model, render_chat  # noqa: E402

base = os.environ.get("SM_SMOKE_BASE", "Qwen/Qwen3.5-9B")
rev = os.environ.get("SM_SMOKE_REV") or None
out_path = os.environ.get("SM_SMOKE_OUT", "/projects/interpretability/shehral.m/monitors/results/gpu_smoke_min.json")

t0 = time.time()
model, tok, meta = load_model(base, revision=rev, dtype="bf16", device="cuda")
print("loaded", base, "in", round(time.time() - t0, 1), "s; meta:", {k: str(v)[:60] for k, v in meta.items()}, flush=True)
n_layers = int(meta.get("n_layers") or getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 32)
layers = sorted({max(1, n_layers // 8), n_layers // 2, max(1, (3 * n_layers) // 4)})

questions = ["Give me three tips for sleeping better.", "Write a short poem about the sea."]
prompts = [render_chat(tok, [{"role": "user", "content": q}], thinking=False) for q in questions]
print("template tail:", repr(prompts[0][-70:]), flush=True)

t1 = time.time()
outs = generate(model, tok, prompts, max_new_tokens=48, batch_size=2)
print("generate:", round(time.time() - t1, 1), "s;", [o[:90].replace("\n", " ") for o in outs], flush=True)

acts = capture_residual(model, tok, prompts, layers=layers, position="last", batch_size=2)
print("residuals:", {int(l): list(a.shape) for l, a in acts.items()}, flush=True)

mid = layers[len(layers) // 2]
hid = acts[mid]
rng = np.random.default_rng(0)
d = rng.standard_normal(hid.shape[1]).astype("float32")
d /= np.linalg.norm(d)
scale = float(np.linalg.norm(hid, axis=1).mean())
with DirectionHooks(model, d, mode="add", layers=[mid], alpha=0.5 * scale, layer_for_add=mid):
    outs_hooked = generate(model, tok, prompts, max_new_tokens=48, batch_size=2)
outs_after = generate(model, tok, prompts, max_new_tokens=48, batch_size=2)
changed = [a != b for a, b in zip(outs, outs_hooked)]
restored = [a == b for a, b in zip(outs, outs_after)]
print("changed under add-hook:", changed, "| restored after exit:", restored, flush=True)

result = {
    "base": base, "revision": rev, "n_layers": n_layers, "layers": layers, "hidden": int(hid.shape[1]),
    "mean_resid_norm_mid": scale, "changed_under_hook": changed, "restored_after_hook": restored,
    "template_tail": prompts[0][-70:], "gen_seconds": round(time.time() - t1, 1),
    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(result, open(out_path, "w"), indent=1)
ok = all(changed) and all(restored)
print("SMOKE", "PASS" if ok else "FAIL", "| peak mem GB", result["peak_mem_gb"], flush=True)
sys.exit(0 if ok else 3)
