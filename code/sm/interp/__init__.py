"""White-box interpretability for the stale-monitors study.

Submodules:
    directions      difference-of-means directions, split-half stability, nulls,
                    logistic probes, refusal subspace, direction file I/O.
    monitors        the six base-fitted monitors (refusal_direction, refusal_subspace,
                    probe, persona_<trait>): scoring, orientation, frozen 1%-FPR
                    thresholds, realignment (re-centred and std re-scaled).
    coherence       coherence-aware refusal accounting (lost refusal = coherent
                    non-refusal) and the documented repetition/length heuristic.
    loading         the single pinned-revision loader call of every interp stage
                    (LoRA adapters unmerged) and the tiny smoke model.
    base_card       per-base card: per-layer directions and probes, L* selection,
                    persona vectors, frozen declared_best_drift_score and
                    monitor_thresholds; organism-extraction mode (H4).
    causal          V4 causal check (exits non-zero when it fails).
    drift           derivative drift metrics: D1-D3, monitor_auroc, realigned
                    recall, persona shifts, organism directions, adapter projection.
    adapter_proj    LoRA adapter projection along r (no forward pass).
    monitor_scores  per-item scores of every monitor on every battery (H5).

Importing this package has no side effects and does not import torch.
"""
