"""Shared utilities: base registry, model loading and chat rendering, residual hooks, IO.

Submodules are imported lazily by callers (``from sm.common.models import load_model``) so that
importing ``sm.common`` does not import torch or transformers.
"""

__all__ = ["registry", "models", "hooks", "io"]
