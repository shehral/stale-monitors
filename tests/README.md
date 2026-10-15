# Tests

pytest suite for the `sm` harness. Every test runs on a tiny random model on CPU and never touches the network when `SM_OFFLINE=1`. No test downloads model weights; the raw text assets the data tests read are local under `data/sources/raw/`.

## Running

From the repository root:

```bash
SM_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests                 # whole suite
SM_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_drift.py   # one module
```

`tests/conftest.py` puts `code/` on `sys.path`, so `import sm` works without `pip install -e code`.

## Device policy and the laptop limit

The device policy is in `docs/HARNESS_CONTRACTS.md` (added 2026-10-07 after three interpreter crashes on the laptop: torch on macOS, Apple M4 Pro, segfaults inside the MPS copy-cast kernels).

- Tests never use MPS. Every `--smoke` entry point forces `device="cpu"` and `dtype="fp32"`, and `device="auto"` resolves to `cuda` when available, else `cpu`, never to `mps` on its own.
- MPS is used only when a caller passes `device="mps"` or sets `SM_DEVICE=mps`; tests do neither.
- The laptop venv also has MPS disabled at the torch level: `.venv/lib/python3.12/site-packages/torch/backends/mps/__init__.py` is patched so `is_available()` returns False (original kept as `__init__.py.orig`). Reinstalling torch removes the patch; re-apply it or export `SM_DEVICE=cpu` for every local run.
- **Laptop limit: run at most two pytest processes at a time.** That includes parallel agents or terminals each running their own test file; queue the rest.

## Environment variables

| Variable | Effect |
|---|---|
| `SM_OFFLINE=1` | No network. The `tiny_model_dir` fixture builds a tiny random Qwen3-style causal LM (4 layers, hidden 64) and a byte-level BPE tokenizer with a Qwen-style chat template that honours `enable_thinking`, saved to a temp dir. |
| `SM_TINY_MODEL` | Overrides the tiny model. Offline: used when it names an existing local directory. Online: a Hub id loaded through the HF cache. Default pinned id: `hf-internal-testing/tiny-random-Qwen3ForCausalLM`. |
| `SM_DEVICE` | Overrides `device="auto"` for every CLI; leave it unset (or `cpu`) for tests. |
| `HF_HUB_OFFLINE=1` | Also forbids downloads in every stage. |

## Fixtures (`conftest.py`)

- `tiny_model_dir` (session): a directory loadable by `sm.common.models.load_model`.
- `tiny_model` (session): `(model, tokenizer, meta)` from `load_model(tiny_model_dir, dtype="fp32", device="cpu")`. Tests that mutate the model must restore it.

## What the suite must cover (docs/HARNESS_SPEC.md, item 12)

- Every stage CLI runs end to end with `--smoke`, including `sm.eval.capability` and `sm.interp.monitor_scores`.
- Direction extraction recovers a planted direction on synthetic activations.
- Drift metrics are 0 (D2) or 1 (D1, D3) for base vs base and detect a planted shift.
- The judge parsers (J1 and the Granite Guardian J2) handle malformed outputs.
- Hook ablation changes model outputs (end-to-end wiring check).

## Layout

One file per module, named `test_<module>.py` (for example `test_common.py`, `test_data_prompts.py`, `test_data_bank.py`, `test_train.py`, `test_battery.py`, `test_capability.py`, `test_judge.py`, `test_base_card.py`, `test_causal.py`, `test_drift.py`, `test_monitor_scores.py`, `test_analysis.py`), plus `test_integration.py` across stages. `test_docs.py` checks that the top-level README, `code/README.md`, `docs/RUNBOOK.md` and this file stay consistent with the spec, the pre-registration (`prereg/PREREG-v0.2.md`), `configs/bases.yaml` and the package (required sections, referenced modules exist, stage order, revision pins, budget numbers sourced from `docs/PLAN.md`, no em dashes). `ls tests/` is authoritative; this list is not regenerated automatically.

## Rules for new tests

- No network: anything that would download must be skipped or replaced by the offline fixture under `SM_OFFLINE=1`.
- CPU only: never request MPS, and pass `device="cpu"` (or rely on `--smoke`) when loading a model.
- Write outputs under pytest's `tmp_path`, never into `results/` or `data/`.
- Keep each `--smoke` run under 60 seconds on CPU.
- Call a module's `main([...])` (or `runpy.run_module`) in process instead of spawning `python -m` or `python -c`: a fresh
  interpreter re-imports torch from the NFS-mounted env on Explorer (30 to 60 s cold). A test that needs a fresh
  interpreter gets `@pytest.mark.slow` (registered in `conftest.py`); the cluster smoke (`jobs/smoke_cpu.sbatch`)
  runs `-m "not slow"` with node-local temp dirs, and the laptop half runs everything.
