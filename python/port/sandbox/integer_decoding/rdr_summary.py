"""Two per-bin read-depth summaries that cannot filter out a CNA (#165, #362).

`cnaster`'s `filter_normal_diffexp` drops a gene when its tumour-over-normal
expression ratio is extreme genome-wide. That test cannot tell a gene the
tumour expresses differently from a gene in an amplified or deleted bin: a
homozygous deletion reads as log fold change 10 and is always dropped, and a
high-level amplification reaches the thresholds too. Copy number moves every
gene in a bin together, so a test within the bin is blind to it:

- :func:`within_bin`: sum a bin's genes after dropping those whose log ratio
  is more than `k` (log2) from the bin's median log ratio;
- :func:`median_of_ratios`: the bin's median gene log ratio itself.

**Set aside, with their numbers.** On CalicoST's easy simulated sample, truth
clones, 1,715 bins, per-bin residual sd of the tumour depth against the normal
baseline times the planted copies (natural log), clones 1/2/3:

| summary | residual sd | corr(log ratio, log CN/2) |
| --- | --- | --- |
| sum, all genes | 1.36 / 1.43 / 1.41 | 0.095 / 0.038 / 0.099 |
| sum, `filter_normal_diffexp` | 1.06 / 1.13 / 1.07 | 0.087 / 0.071 / 0.105 |
| :func:`within_bin`, `k = 1` | 1.23 / 1.29 / 1.22 | 0.089 / 0.030 / 0.074 |
| :func:`median_of_ratios` | 1.24 / 1.29 / 1.22 | 0.091 / 0.041 / 0.088 |

The sample's tumour spots carry their own expression profile (median gene
4.9-fold from normal), shared by the tumour clones, so no per-gene filter
makes depth track copy number bin by bin there. Both are kept for a sample
whose expression differences are sparse; neither is installed, and neither is
tested further.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["median_of_ratios", "within_bin"]


def _log_ratios(
    tumour: np.ndarray, normal: np.ndarray, bins: np.ndarray
) -> pd.DataFrame:
    """Per gene with counts in both: its bin and log2 of its normalized ratio."""
    kept = (bins >= 0) & (tumour > 0) & (normal > 0)
    ratio = (tumour[kept] / tumour.sum()) / (normal[kept] / normal.sum())
    return pd.DataFrame(
        {
            "bin": bins[kept],
            "log_ratio": np.log2(ratio),
            "tumour": tumour[kept],
            "normal": normal[kept],
        }
    )


def within_bin(
    tumour: np.ndarray,
    normal: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    k: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """`(tumour, normal)` per bin, summed over genes within `k` of the bin's median.

    `tumour` and `normal` are per-gene counts; `bins` each gene's bin, `-1`
    for none. Both returns are normalized to their own totals.
    """
    genes = _log_ratios(tumour, normal, bins)
    median = genes.groupby("bin").log_ratio.transform("median")
    kept = genes[np.abs(genes.log_ratio - median) <= k]
    summed_tumour = np.bincount(kept.bin, weights=kept.tumour, minlength=n_bins)
    summed_normal = np.bincount(kept.bin, weights=kept.normal, minlength=n_bins)
    return summed_tumour / tumour.sum(), summed_normal / normal.sum()


def median_of_ratios(
    tumour: np.ndarray, normal: np.ndarray, bins: np.ndarray, n_bins: int
) -> np.ndarray:
    """Each bin's median gene tumour/normal ratio; `nan` where no gene has both."""
    genes = _log_ratios(tumour, normal, bins)
    median = genes.groupby("bin").log_ratio.median()
    ratio = np.full(n_bins, np.nan)
    ratio[median.index.to_numpy()] = 2.0 ** median.to_numpy()
    return ratio
