# `sm`: the stale-monitors harness

One Python package, runnable on a laptop with tiny models for tests and on Explorer H200s for real runs. The authoritative documents are [`../docs/HARNESS_SPEC.md`](../docs/HARNESS_SPEC.md) (what each module does), [`../docs/HARNESS_CONTRACTS.md`](../docs/HARNESS_CONTRACTS.md) (shared signatures and file formats), [`../prereg/PREREG-v0.2.md`](../prereg/PREREG-v0.2.md) (metric and hypothesis definitions) and [`../docs/CODE_REUSE.md`](../docs/CODE_REUSE.md) (the official implementations and judge prompt formats each module ports). This page is a map.

## Install

```bash
pip install -e code             # from the repository root
pip install -e "code[test]"     # adds pytest
pip install -e "code[serve]"    # adds vLLM (GPU serving; optional)
```

Python 3.12. Dependencies are declared in `pyproject.toml`.

## Conventions (from the spec and contracts)

- Every stage is a CLI: `python -m sm.<subpackage>.<module>` with `main(argv)`, accepting `--config path.yaml`, `--out dir`, `--smoke`, `--seed N`, `--allow-download`, and `--force`.
- `--smoke` runs the stage end to end on a tiny model with 8 items in under 60 seconds; every smoke entry point forces `device="cpu"` and `dtype="fp32"`.
- Device policy (contracts, 2026-10-07): `device="auto"` resolves to `cuda` when available, else `cpu`, and never to `mps` on its own, because torch on macOS (Apple M4 Pro) crashes the interpreter on MPS copy-casts. MPS is used only when `device="mps"` is passed or `SM_DEVICE=mps` is set. `SM_DEVICE` overrides `auto` for every CLI; Explorer jobs leave it unset.
- Outputs are JSON, JSONL or CSV under `--out` with a `meta` block (git commit, config hash, model id and revision, seed, timestamp, SHA-256 of inputs). Stages are idempotent: if the final output exists and `--force` is absent, the stage exits 0 with "already done".
- Nothing is downloaded unless `--allow-download` is set; `SM_OFFLINE=1` or `HF_HUB_OFFLINE=1` forbids it regardless.
- Revisions: every stage loads a base with the pinned revision from `configs/bases.yaml` (or the per-base config), through the same `load_model` call the smoke stage uses.
- Models: bf16; `attn_implementation="sdpa"` for generation, `"eager"` only where attention is read; chat template with thinking disabled and the switch recorded in meta; text decoder only for multimodal wrappers.
- Adapters: a LoRA adapter is evaluated unmerged (PEFT model, LoRA applied in the forward pass at bf16 compute) for batteries, capability, drift and persona capture. Merging exists only for the optional vLLM battery path and is done in fp32 before casting.
- Decoding per battery comes from `configs/batteries.yaml`: greedy (max 512 new tokens) for the refusal, over-refusal, ladder and capability batteries; temperature 1.0, top-p 1.0, 10 samples per question for the emergent-misalignment and persona trait evals.
- Layer index: `hidden_states[l]` is the output of block `l-1`; `hidden_states[0]` is the embeddings. Activations are read at the last prompt token or as the mean over response tokens. Left padding with explicit position ids.
- Direction files: `<name>.npy` (float32, shape `(hidden,)`, unit norm) plus a `<name>.json` sidecar (base, revision, layer, position, counts, prompt-set and vector SHA-256).

## Modules

| Module | Role | Main output |
|---|---|---|
| `sm.common.registry` | Base registry (`BASES`, `resolve_base`, fallback chain); mirrors `configs/bases.yaml`. | none |
| `sm.common.models` | `load_model` (text decoder, tokenizer, optional unmerged LoRA adapter), `render_chat` (thinking off), `generate` (batched, greedy by default). | none |
| `sm.common.hooks` | `capture_residual` at chosen layers and positions; `DirectionHooks` to ablate a direction at every layer (including the mid-block attention write) or add it at one layer. | none |
| `sm.common.io` | `meta_block`, atomic `write_json`, `sha256_file`, `set_seed`. | none |
| `sm.data.build_prompt_sets` | Harmful contrast set (AdvBench) and style-matched harmless set (Dolly harmless-contrast pool), construction/dev/test splits; battery item files including the Karne ladders. | `data/prompts/` with `MANIFEST.json` (Dolly pool ids and seed) |
| `sm.data.build_bank_datasets` | The bank training datasets per condition from disjoint Dolly pools, including per-base persona and risky-advice generation. | `data/bank/<condition>/<name>.jsonl`, `data/bank/MANIFEST.json`, `data/bank/LICENSES.md` |
| `sm.train.lora_sft` | PEFT LoRA SFT for one bank run (by `--run-id` or `--array-index`). | `results/adapters/<base>/<run_id>/` with `train_meta.json` |
| `sm.eval.battery` | Generation for every battery item with per-battery decoding, base or base plus unmerged adapter. | `results/batteries/<base>/<run_id>/<battery>.jsonl`; restricted batteries (`harmful` or `restricted` in `configs/batteries.yaml`, plus the name rule: strongreject, harmbench, xstest_unsafe, karne_ladders*, em_*, persona_*) under `results/restricted/batteries/...` |
| `sm.eval.capability` | MMLU exact match and IFEval strict accuracy from the capability battery outputs. | `results/capability/<base>/<run_id>.json` |
| `sm.eval.judge` | J1 (gpt-oss-safeguard-20b) and J2 (granite-guardian-3.3-8b) labels, per-battery rates with Wilson intervals, Cohen's kappa; J2 on a fraction or on the full harmful set of flagged derivatives. | `labels/`, `rates/`, `kappa.json`, `j2_status.json`, `judge_summary.json` under `--out` (`results/judge/<base>/<run_id>/`); label files of restricted batteries under `results/restricted/judge/<base>/<run_id>/` |
| `sm.eval.judge_parse` | Label parsers for both judges, rates and kappa. | none |
| `sm.eval.human_slice` | Owner-labelled human slice: seeded blinded draw, labelling CLI, Cohen's kappa against J1 and J2 with bootstrap intervals, the AG-7 gate (docs/HUMAN_SLICE.md). | `results/restricted/human/` (sheets, key, owner labels); `results/human/kappa.json` |
| `sm.eval.harmbench_cls` | Exploratory line E14 (docs/HARMBENCH_CLS_LINE.md): the HarmBench classifier as an independent harmful-compliance reader against J1 on every run's H items; `--score-human` joins its sheet verdicts with the owner's labels (laptop) and evaluates the pre-registered reading. | `results/diag/harmbench_cls_<base>.json`, `results/diag/harmbench_cls_<base>_human.json` |
| `sm.interp.assistant_axis` | Exploratory line E15 (docs/ASSISTANT_AXIS_E7_PLAN.md): the assistant axis of Lu et al. 2601.10387 (seeded 60-role x 20-question plan of the official data, vLLM rollouts, J1 role-expression scores, response-mean activations, contrast axis at every layer with validity bars and Lu's capping thresholds), each run's displacement along it, and the H3' rule against D2. Subcommands plan, generate, judge, activations, axis, displace, h3prime. | `results/exploratory/<base>/axis/`, `displace/`, `h3prime.json`; texts only under `results/restricted/exploratory/<base>/` |
| `sm.interp.mediation` | Exploratory: E15 persona repair along the axis (CAP) and the registered E7 mediation (restore the base's r-projection, random directions, ablation sanity, adapter-write removal) on five regressed units and two controls, judged by J1 as V4. | `results/exploratory/<base>/mediation/<unit>.json`, `mediation_summary.json`; completions only under `results/restricted/exploratory/<base>/mediation/` |
| `sm.interp.base_card` | Per-layer refusal directions, probes, L* selection, refusal subspace, persona vectors, frozen monitor thresholds and declared drift score; `--phase persona` rebuilds only the persona part of an existing card (docs/PERSONA_FIDELITY_RERUN.md). | `results/base_cards/<base>.json` plus directions under `results/base_cards/<base>/` |
| `sm.interp.causal` | V4 causal validity check (ablate r, add r, random-direction controls); exits non-zero when it fails. | `results/base_cards/<base>_causal.json` |
| `sm.interp.persona_dose` | Persona steering dose test (FIDELITY_REVIEW R2): the card's persona vectors added at several coefficients; with the persona grader, trait and coherence per sample, the coherence window and the pre-registered readout (docs/PERSONA_FIDELITY_RERUN.md section 7); the v0.3 J1 line (graded score, incoherent rate, Spearman and sign checks) is kept. | `results/base_cards/<base>_persona_dose_<grader>.json` |
| `sm.interp.persona_grader` | Chen et al.'s per-trait judge prompts and coherence prompt, verbatim (SHA-256 registered), scored 0-100 as their probability-weighted mean over integer answers on the served grader (google/gemma-4-31B-it); the trait scorer of the persona phase and the dose test; subcommands hash, self-test, grade-battery. | `results/restricted/persona_grades/<base>/<run>/`, `results/persona_grades/<base>/<run>/summary.json` |
| `sm.interp.drift` | D1, D2, D3, per-monitor AUROC on base and derivative, realigned recall at the base threshold, persona shifts, adapter projection for one derivative. | `results/drift/<base>/<run_id>.json` |
| `sm.interp.monitor_scores` | Per-item scores of every base-card monitor on every battery, including the ladders (H5). | `results/monitor_scores/<base>/<run_id>/<battery>.jsonl` |
| `sm.interp.loading` | `load_pinned`, the one loader call of every interp stage (pinned revision, adapter unmerged), in smoke and measured runs alike. | none |
| `sm.interp.monitors` | The six base-fitted monitors (`MONITOR_NAMES`) shared by base_card, drift and monitor_scores. | none |
| `sm.interp.coherence` | Coherence-aware refusal accounting for L* selection and V4 (incoherent output never counts as lost refusal). | none |
| `sm.interp.adapter_proj` | Adapter projection from LoRA tensors (no forward pass); used by drift, also a CLI. | JSON |
| `sm.interp.directions` | Difference of means, split-half cosine, shuffled and random baselines, probes, direction file I/O. | none |
| `sm.analysis.aggregate` | Joins drift, judge labels, capability, monitor scores and train meta into bank tables; confirmed regression with the J2 pending rule; H1 to H6 with Holm over six. | `results/tables/bank_<base>.csv`, `results/tables/hypotheses.json`, `results/figures/*.png` |
| `sm.analysis.stats` | AUROC, bootstrap, Wilson intervals, Holm and related statistics. | none |
| `sm.analysis.rules` | The J1 flag rule and the J2 status rule (pure functions, inside the analysis-code freeze; used by `jobs/common.sh` `sm_py j1-flags` / `j2-status` and `sm.eval.judge`). | none |
| `sm.accept` | Stage acceptance (prereg/ACCEPTANCE.md): checks one job's outputs, writes the report, and with `--write` the ledger row (docs/ACCEPTANCE_USAGE.md). | `results/acceptance/<stage>/<base>/...json`, `results/ledger.jsonl` |
| `sm.ops` | Provenance helpers: git and analysis-code state, the J1 identity record and backfill (`sm.ops.judge_identity`), the smoke artifact (`sm.ops.smoke_artifact`). | `results/smoke/<base>/<jobid>.json`, `results/smoke/local/<git_sha>.json` |

Each module's docstring documents its config keys; `python -m <module> --help` prints the flags.

## Smoke runs

Every stage CLI accepts `--smoke`, for example:

```bash
SM_OFFLINE=1 python -m sm.train.lora_sft --smoke --out /tmp/sm_smoke
SM_OFFLINE=1 python -m sm.analysis.aggregate --smoke --out /tmp/sm_smoke
```

The step-by-step procedure for a real base is in [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md).
