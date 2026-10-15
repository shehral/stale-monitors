"""Adapter projection: how strongly a LoRA adapter writes along a residual-stream direction.

Computed from the adapter tensors (and, for the normaliser, the base weights) with no
forward pass. Implements the prereg v0.2 (prereg/PREREG-v0.2.md) Section 4 formula

    ||r_base^T B|| ||A|| / ||r_base^T W_base||

Exact definition implemented here
---------------------------------
For every LoRA pair (A, B) on a residual-writing module m (a module whose output is
added to the residual stream: by default names ending in ``o_proj``, ``down_proj`` or
``out_proj``, and whose output dimension equals the hidden size), with

    B_m in R^{d x k}, A_m in R^{k x n}, W_m in R^{d x n} (torch Linear layout, out x in)
    s_m = lora_alpha / k            (or lora_alpha / sqrt(k) when use_rslora is set)
    Delta W_m = s_m B_m A_m
    r = r_base / ||r_base||         (unit refusal direction, length d)

we compute three per-module quantities:

    bound_m = s_m * ||r^T B_m||_2 * ||A_m||_op     (prereg numerator; ||A||_op = spectral norm)
    exact_m = ||r^T Delta W_m||_2 = s_m * ||(r^T B_m) A_m||_2   (bound_m >= exact_m)
    base_m  = ||r^T W_m||_2                        (prereg denominator)

and aggregate over the included modules in quadrature, which equals the ratio of the
norms of the concatenated row vectors r^T [Delta W_1, Delta W_2, ...] and
r^T [W_1, W_2, ...]:

    adapter_projection       = sqrt(sum_m bound_m^2) / sqrt(sum_m base_m^2)
    adapter_projection_exact = sqrt(sum_m exact_m^2) / sqrt(sum_m base_m^2)

``adapter_projection`` (the prereg bound) is the value written to the drift JSON; the exact
variant and the per-module table are reported alongside. When base weights are not
available the normalised values are None and only the unnormalised numerators are given.

Layer scope: ``"all"`` (default) uses every writer in every block; ``"upto_lstar"`` keeps
blocks with index < L* (the blocks whose writes reach hidden_states[L*] under the
``hidden_states[l]`` = output of block l-1 convention).

Caveat: in architectures with a post-attention / post-MLP norm before the residual add
(Gemma 2/3/4), r^T W is the pre-norm write, not the exact residual contribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_WRITERS: tuple[str, ...] = ("o_proj", "down_proj", "out_proj")
_EXCLUDE_TOWERS = ("vision", "visual", "audio", "image", "mm_projector", "multi_modal")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass
class ModuleProjection:
    module: str
    layer: int | None
    rank: int
    scale: float
    bound: float
    exact: float
    base: float | None


@dataclass
class AdapterProjection:
    adapter_projection: float | None
    adapter_projection_exact: float | None
    numerator_bound: float
    numerator_exact: float
    denominator: float | None
    n_modules: int
    n_modules_with_base: int
    layer_scope: str
    writers: list[str]
    modules: list[ModuleProjection] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self, include_modules: bool = True) -> dict:
        out = {
            "adapter_projection": self.adapter_projection,
            "adapter_projection_exact": self.adapter_projection_exact,
            "numerator_bound": self.numerator_bound,
            "numerator_exact": self.numerator_exact,
            "denominator": self.denominator,
            "n_modules": self.n_modules,
            "n_modules_with_base": self.n_modules_with_base,
            "layer_scope": self.layer_scope,
            "writers": list(self.writers),
            "skipped": list(self.skipped),
            "formula": "sqrt(sum_m (s_m ||r^T B_m||_2 ||A_m||_op)^2) / sqrt(sum_m ||r^T W_m||_2^2)",
        }
        if include_modules:
            out["modules"] = [m.__dict__ for m in self.modules]
        return out


def _to_numpy(t) -> np.ndarray:
    """Convert a torch tensor (any dtype, including bf16) to float64 numpy."""
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return t.detach().to(torch.float64).cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(t, dtype=np.float64)


def load_adapter_tensors(adapter_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read ``adapter_model.safetensors`` (or ``.bin``) and ``adapter_config.json``."""
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    st = adapter_dir / "adapter_model.safetensors"
    tensors: dict[str, np.ndarray] = {}
    if st.exists():
        from safetensors import safe_open

        with safe_open(str(st), framework="pt") as f:
            for k in f.keys():
                tensors[k] = _to_numpy(f.get_tensor(k))
    else:
        binf = adapter_dir / "adapter_model.bin"
        if not binf.exists():
            raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {adapter_dir}")
        import torch

        sd = torch.load(str(binf), map_location="cpu", weights_only=True)
        tensors = {k: _to_numpy(v) for k, v in sd.items()}
    return tensors, cfg


def pair_lora_tensors(tensors: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Group PEFT keys into {module_path: (A, B)}.

    Handles ``<path>.lora_A.weight`` and ``<path>.lora_A.<adapter_name>.weight``. The module
    path has the PEFT ``base_model.model.`` prefix stripped.
    """
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for k in tensors:
        m = re.match(r"^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$", k)
        if not m:
            continue
        path, which = m.group(1), m.group(2)
        if path.startswith("base_model.model."):
            path = path[len("base_model.model."):]
        (a_keys if which == "A" else b_keys)[path] = k
    pairs = {}
    for path, ak in a_keys.items():
        if path in b_keys:
            pairs[path] = (tensors[ak], tensors[b_keys[path]])
    return pairs


def lora_scale(module_path: str, rank: int, cfg: dict) -> float:
    """Scaling s = alpha / rank (or alpha / sqrt(rank) with rsLoRA), honouring alpha_pattern."""
    alpha = float(cfg.get("lora_alpha", rank))
    for pat, val in (cfg.get("alpha_pattern") or {}).items():
        if re.search(pat, module_path):
            alpha = float(val)
    if cfg.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def layer_of(module_path: str) -> int | None:
    m = _LAYER_RE.search(module_path)
    return int(m.group(1)) if m else None


def _is_tower(key: str) -> bool:
    low = key.lower()
    return any(t in low for t in _EXCLUDE_TOWERS)


class BaseWeightIndex:
    """Lazy index over a directory of base-model safetensors shards."""

    def __init__(self, weights_dir: str | Path):
        from safetensors import safe_open

        self.dir = Path(weights_dir)
        self._key_to_file: dict[str, Path] = {}
        files = sorted(self.dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {self.dir}")
        for fp in files:
            with safe_open(str(fp), framework="pt") as f:
                for k in f.keys():
                    self._key_to_file[k] = fp

    def keys(self) -> list[str]:
        return list(self._key_to_file)

    def resolve(self, module_path: str) -> str | None:
        """Find the base weight key for an adapter module path."""
        exact = f"{module_path}.weight"
        if exact in self._key_to_file:
            return exact
        m = re.search(r"layers\.\d+\..*$", module_path)
        if not m:
            return None
        suffix = m.group(0) + ".weight"
        cands = [k for k in self._key_to_file if k.endswith(suffix) and (k == suffix or k.endswith("." + suffix))]
        cands = [k for k in cands if not _is_tower(k)] or cands
        if len(cands) > 1:
            lm = [k for k in cands if "language_model" in k]
            cands = lm or cands
        if len(cands) == 1:
            return cands[0]
        return None

    def get(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        with safe_open(str(self._key_to_file[key]), framework="pt") as f:
            return _to_numpy(f.get_tensor(key))


def compute_adapter_projection(
    r: np.ndarray,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: dict,
    base_weights: "BaseWeightIndex | dict[str, np.ndarray] | None" = None,
    writers: tuple[str, ...] | None = DEFAULT_WRITERS,
    layer_scope: str = "all",
    layer_star: int | None = None,
) -> AdapterProjection:
    """Compute the adapter projection of a LoRA adapter along direction ``r``.

    ``base_weights`` is a :class:`BaseWeightIndex`, or a dict {module_path or weight key: W},
    or None (then the normalised outputs are None). ``writers=None`` includes every module
    whose output dimension equals ``len(r)``.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    nr = np.linalg.norm(r)
    if nr == 0:
        raise ValueError("direction r has zero norm")
    r = r / nr
    d = r.shape[0]
    if layer_scope not in ("all", "upto_lstar"):
        raise ValueError(f"layer_scope must be 'all' or 'upto_lstar', got {layer_scope}")
    if layer_scope == "upto_lstar" and layer_star is None:
        raise ValueError("layer_scope 'upto_lstar' needs layer_star")

    mods: list[ModuleProjection] = []
    skipped: list[str] = []
    for path in sorted(pairs):
        A, B = pairs[path]
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        leaf = path.split(".")[-1]
        if writers is not None and leaf not in writers:
            continue
        if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
            skipped.append(f"{path}: unexpected shapes A{A.shape} B{B.shape}")
            continue
        if B.shape[0] != d:
            skipped.append(f"{path}: output dim {B.shape[0]} != hidden {d}")
            continue
        layer = layer_of(path)
        if layer_scope == "upto_lstar" and (layer is None or layer >= int(layer_star)):
            continue
        k = A.shape[0]
        s = lora_scale(path, k, cfg)
        rb = r @ B  # (k,)
        a_op = float(np.linalg.norm(A, ord=2)) if A.size else 0.0
        bound = float(s * np.linalg.norm(rb) * a_op)
        exact = float(s * np.linalg.norm(rb @ A))
        base_val: float | None = None
        if base_weights is not None:
            W = None
            if isinstance(base_weights, dict):
                W = base_weights.get(path)
                if W is None:
                    W = base_weights.get(f"{path}.weight")
            else:
                key = base_weights.resolve(path)
                if key is not None:
                    W = base_weights.get(key)
            if W is None:
                skipped.append(f"{path}: base weight not found")
            else:
                W = np.asarray(W, dtype=np.float64)
                if W.shape[0] != d:
                    skipped.append(f"{path}: base weight out dim {W.shape[0]} != hidden {d}")
                else:
                    base_val = float(np.linalg.norm(r @ W))
        mods.append(ModuleProjection(module=path, layer=layer, rank=k, scale=s, bound=bound, exact=exact, base=base_val))

    num_b = math.sqrt(sum(m.bound ** 2 for m in mods))
    num_e = math.sqrt(sum(m.exact ** 2 for m in mods))
    with_base = [m for m in mods if m.base is not None]
    den: float | None = None
    proj: float | None = None
    proj_e: float | None = None
    if base_weights is not None and mods and len(with_base) == len(mods):
        den = math.sqrt(sum(m.base ** 2 for m in with_base))  # type: ignore[operator]
        if den > 0:
            proj = num_b / den
            proj_e = num_e / den
    return AdapterProjection(
        adapter_projection=proj,
        adapter_projection_exact=proj_e,
        numerator_bound=num_b,
        numerator_exact=num_e,
        denominator=den,
        n_modules=len(mods),
        n_modules_with_base=len(with_base),
        layer_scope=layer_scope,
        writers=list(writers) if writers is not None else ["<any module with out dim == hidden>"],
        modules=mods,
        skipped=skipped,
    )


def adapter_projection_from_dir(
    adapter_dir: str | Path,
    r: np.ndarray,
    base_weights_dir: str | Path | None = None,
    writers: tuple[str, ...] | None = DEFAULT_WRITERS,
    layer_scope: str = "all",
    layer_star: int | None = None,
) -> AdapterProjection:
    tensors, cfg = load_adapter_tensors(adapter_dir)
    pairs = pair_lora_tensors(tensors)
    if not pairs:
        raise ValueError(f"no LoRA A/B pairs found in {adapter_dir}")
    index = BaseWeightIndex(base_weights_dir) if base_weights_dir else None
    return compute_adapter_projection(r, pairs, cfg, index, writers, layer_scope, layer_star)


def resolve_base_weights_dir(hf_id: str, revision: str | None = None) -> str | None:
    """Locate base safetensors in the local HF cache without downloading. None if absent."""
    p = Path(hf_id)
    if p.is_dir() and any(p.glob("*.safetensors")):
        return str(p)
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(hf_id, revision=revision, local_files_only=True, allow_patterns=["*.safetensors", "*.json"])
    except Exception as exc:  # report why, so a null adapter_projection is never silent
        import sys

        print(f"adapter_proj: base weights for {hf_id} @ {revision!r} not in the local HF cache: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if not any(Path(local).glob("*.safetensors")):
        import sys

        print(f"adapter_proj: snapshot {local} has no *.safetensors", file=sys.stderr)
        return None
    return local


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA adapter projection along a direction (no forward pass).")
    ap.add_argument("--adapter", required=True, help="PEFT adapter directory")
    ap.add_argument("--direction", required=True, help=".npy direction (unit or not; normalised here)")
    ap.add_argument("--base-weights", default=None, help="directory with base *.safetensors (normaliser)")
    ap.add_argument("--layer-scope", default="all", choices=["all", "upto_lstar"])
    ap.add_argument("--layer-star", type=int, default=None)
    ap.add_argument("--all-hidden-writers", action="store_true", help="include any module with out dim == hidden")
    ap.add_argument("--out", default=None, help="write JSON here (else print)")
    args = ap.parse_args(argv)
    r = np.load(args.direction)
    res = adapter_projection_from_dir(
        args.adapter,
        r,
        args.base_weights,
        writers=None if args.all_hidden_writers else DEFAULT_WRITERS,
        layer_scope=args.layer_scope,
        layer_star=args.layer_star,
    )
    txt = json.dumps(res.to_dict(), indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
