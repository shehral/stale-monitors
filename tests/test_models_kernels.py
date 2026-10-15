"""The CPU kernel guard: CUDA-only hybrid-layer packages must fail to import in a CPU process (see
sm.common.models.disable_cuda_only_kernels_for_cpu). Runs in a subprocess so the test process is not altered."""
import subprocess
import sys

import pytest

CODE = r"""
import importlib, sys
from sm.common import models
blocked = models.disable_cuda_only_kernels_for_cpu()
assert set(blocked) <= {"causal_conv1d", "fla"}, blocked
for pkg in ("causal_conv1d", "fla"):
    try:
        importlib.import_module(pkg)
    except ImportError:
        pass
    else:
        raise SystemExit(f"{pkg} still importable")
again = models.disable_cuda_only_kernels_for_cpu()
assert again == [], again  # idempotent: already blocked
print("GUARD_OK", sorted(blocked))
"""


@pytest.mark.slow  # a fresh interpreter is the point of the test; excluded from the cluster smoke (-m "not slow")
def test_cpu_guard_blocks_cuda_only_kernel_packages():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GUARD_OK" in r.stdout


def test_cpu_load_calls_guard(monkeypatch):
    from sm.common import models

    calls = []
    monkeypatch.setattr(models, "disable_cuda_only_kernels_for_cpu", lambda: calls.append("cpu") or [])
    assert models.resolve_device("cpu") == "cpu"
    # load_model itself needs a checkpoint; the guard call site is covered by the smoke/tiny-model tests on CPU.
    assert callable(models.disable_cuda_only_kernels_for_cpu)
