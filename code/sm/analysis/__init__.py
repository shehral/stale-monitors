"""Analysis: bank tables, confirmed regression, staleness, drift-vs-regression AUROC and H1 to H6."""

from sm.analysis.stats import (
    auroc,
    bootstrap_auroc_ci,
    bootstrap_p,
    bootstrap_stat_ci,
    holm,
    paired_bootstrap_auroc_diff,
    recall_at_fpr,
    spearman,
    spearman_ci,
    wilson_interval,
)

__all__ = [
    "auroc",
    "bootstrap_auroc_ci",
    "bootstrap_p",
    "bootstrap_stat_ci",
    "holm",
    "paired_bootstrap_auroc_diff",
    "recall_at_fpr",
    "spearman",
    "spearman_ci",
    "wilson_interval",
]
