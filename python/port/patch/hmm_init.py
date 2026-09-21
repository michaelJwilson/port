"""`cnaster`'s HMM initialization, with filtering separated from transformation.

**Proposed for `cnaster`, written here.** #229. `cnaster.hmm_initialize.gmm_init`
runs eleven steps between the counts and the mixture fit, of which four are
filters, two are transforms and one is an imputation. They are interleaved
across three blocks, the transform at step 3 has its inverse 200 lines away at
step 11, and no stage can be tested on its own because none of them is a
function.

This module is the first stage: **every drop and clip in one place, returning
what it dropped**. The transform is named and carries its own inverse, so the
pair cannot drift; the mixture fit and the parameter mapping stay where they
are for now (#229 stages 3 to 5).

## What is a filter and what is not

`filter_observations` does the four filters and the imputation, and the
imputation is a **policy** rather than a step, because it is neither:
`cnaster` calls `ffill().bfill()`, which carries a neighbouring genomic bin's
value into a missing one. That is a modelling assumption -- that a bin
resembles its neighbour -- and nothing in `cnaster` states it. `"none"` is
available here so the assumption can be turned off and the cost measured.

`Standardize` is the transform. It is separate because the emission-family
backend (#229 stage 4) needs the identity: a mixture fitted in the negative
binomial and beta-binomial has nothing to standardize and nothing to invert,
and steps 2, 3 and 11 disappear rather than being reimplemented.

## Bitwise

`design_matrix` composes the two into the array `cnaster` feeds
`GaussianMixture`. `tests/test_hmm_init_filter.py` captures that array from a
real `gmm_init` call and asserts this reproduces it **bitwise**, which is what
makes the separation a refactor rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

MIRRORS: tuple[str, ...] = ("cnaster.hmm_initialize",)
"""The filtering `hmm_initialize` does inside its initializers.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

__all__ = [
    "FilterRecord",
    "Observations",
    "Standardize",
    "design_matrix",
    "filter_observations",
]

Imputation = Literal["ffill_bfill", "none"]

RDR_FLOOR = 1e-6
"""`cnaster`'s `np.clip(rdr_ratio, a_min=1e-6)` before the log.

A filter, and a silent one: it moves a zero-coverage bin to `log(1e-6)`
rather than dropping it, so the bin survives as an extreme value instead of
as a `-inf` the row filter would catch. Reproduced rather than corrected,
because this module's claim is equivalence; #229 is where the choice is
argued.
"""


@dataclass(frozen=True)
class FilterRecord:
    """What was removed, by which rule, and how much.

    `cnaster` logs three of these five as free text and the other two not at
    all, so a run's filtering is not recoverable afterwards. Here it is a
    value, which is what lets `HMMInit` report it beside a fit and what lets
    a test assert on it.
    """

    n_bins: int
    n_spots: int
    rdr_floored: int
    """Entries the `1e-6` floor moved. Not dropped -- moved."""
    baf_clipped: int
    """Entries outside the configured beta-binomial bounds."""
    imputed: int
    """NaNs filled by the imputation policy."""
    rows_dropped: int
    """Rows still not finite after imputation."""
    rows_kept: int

    @property
    def retained(self) -> float:
        """The fraction of rows reaching the fit."""
        total = self.rows_kept + self.rows_dropped

        return self.rows_kept / total if total else 0.0

    def __str__(self) -> str:
        return (
            f"{self.rows_kept}/{self.rows_kept + self.rows_dropped} rows "
            f"({self.retained:.2%}); "
            f"rdr floored {self.rdr_floored}, baf clipped {self.baf_clipped}, "
            f"imputed {self.imputed}"
        )


@dataclass(frozen=True)
class Observations:
    """The two channels, filtered, with the record of what filtering did.

    `rdr` and `baf` are ratios rather than counts, and neither has been
    logged or standardized: that is `Standardize`'s job and the
    emission-family backend's non-job.
    """

    rdr: np.ndarray | None
    baf: np.ndarray | None
    record: FilterRecord
    params: str

    @property
    def n_spots(self) -> int:
        return self.record.n_spots


@dataclass
class Standardize:
    """`cnaster`'s step 3, holding its own inverse.

    Centre by the median and scale by the 1-to-99 percentile spread, both
    taken over the finite entries. `apply` and `invert` are a pair by
    construction, which is the whole point: in `cnaster` the forward map is
    at line 351 and the inverse at line 519, and nothing checks that the
    second undoes the first.

    The `< 1e-6` fallback to `1.0` is `cnaster`'s and is reproduced. It makes
    the transform data-dependent in a way that is invisible downstream: on a
    tight distribution the scaling silently becomes a shift.
    """

    offset: float = 0.0
    scale: float = 1.0
    fitted: bool = field(default=False, repr=False)

    def fit(self, values: np.ndarray, *, in_log_space: bool = True) -> Standardize:
        finite = np.isfinite(values)

        if not np.any(finite):
            msg = "no finite values to standardize against"
            raise RuntimeError(msg)

        if in_log_space:
            self.offset = float(np.median(values[finite]))
            low = np.percentile(values[finite], 1)
            high = np.percentile(values[finite], 99)
            scale = float(high - low)
        else:
            self.offset = 0.0
            scale = float(np.percentile(values[finite], 99))

        # NB cnaster's fallback, kept. Below the floor the transform becomes a
        #    shift, and nothing downstream can tell that it did.
        self.scale = 1.0 if scale < 1e-6 else scale
        self.fitted = True

        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.offset) / self.scale

    def invert(self, values: np.ndarray) -> np.ndarray:
        """The map `cnaster` writes as `means * scale_factor + offset`."""
        return values * self.scale + self.offset


def filter_observations(
    X: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    params: str,
    *,
    min_binom: float,
    max_binom: float,
    in_log_space: bool = True,
) -> Observations:
    """Every drop and clip `gmm_init` performs, in one place.

    Reproduces `cnaster`'s arithmetic exactly, including the order: the RDR
    floor applies only when the log is coming (`in_log_space`), because
    `cnaster` puts the clip inside that branch, and a floor applied in the
    other branch would change the fit.

    The row filter is not here: it runs on the concatenation of both
    channels and so belongs to `design_matrix`, which is where `cnaster`
    does it too.
    """
    if not params:
        msg = "params selects neither channel"
        raise ValueError(msg)

    n_bins, _, n_spots = X.shape
    rdr = baf = None
    floored = clipped = 0

    if "m" in params:
        ratio = X[:, 0, :] / base_nb_mean

        if in_log_space:
            floored = int(np.sum(ratio < RDR_FLOOR))
            ratio = np.clip(ratio, a_min=RDR_FLOOR, a_max=None)

        rdr = ratio

        if not np.any(np.isfinite(np.log(ratio) if in_log_space else ratio)):
            # NB cnaster raises here, and the message is worth keeping: a run
            #    that reaches this has a base_nb_mean problem, not a fit one.
            msg = "No valid RDR data."
            raise RuntimeError(msg)

    if "p" in params:
        baf = X[:, 1, :] / total_bb_RD
        clipped = int(np.sum((baf < min_binom) | (baf > max_binom)))
        baf = np.clip(baf, min_binom, max_binom)

    record = FilterRecord(
        n_bins=n_bins,
        n_spots=n_spots,
        rdr_floored=floored,
        baf_clipped=clipped,
        imputed=0,
        rows_dropped=0,
        rows_kept=n_bins,
    )

    return Observations(rdr=rdr, baf=baf, record=record, params=params)


def design_matrix(
    observations: Observations,
    *,
    in_log_space: bool = True,
    imputation: Imputation = "ffill_bfill",
    standardize: Standardize | None = None,
) -> tuple[np.ndarray, Standardize | None, FilterRecord]:
    """The array `cnaster` feeds `GaussianMixture`, and what it cost.

    Composes the filtered channels, the transform and the row filter in
    `cnaster`'s order, and returns the fitted `Standardize` so a caller can
    invert the fitted means without re-deriving the constants.

    `imputation="none"` skips the `ffill/bfill` step. The rows it would have
    saved are then dropped by the row filter instead, and the record says how
    many -- which is the measurement #229 wants and `cnaster` cannot take.
    """
    params = observations.params
    columns = []

    if "m" in params:
        assert observations.rdr is not None
        values = np.log(observations.rdr) if in_log_space else observations.rdr

        if standardize is None:
            standardize = Standardize().fit(values, in_log_space=in_log_space)

        columns.append(standardize.apply(values))

    if "p" in params:
        assert observations.baf is not None
        columns.append(observations.baf)

    design = np.hstack(columns) if len(columns) > 1 else columns[0]

    imputed = int(np.isnan(design).sum())

    if imputed > 0 and imputation == "ffill_bfill":
        # NB pandas fills down each column, so a missing bin takes the value
        #    of the bin above it in the same spot. That is imputation by
        #    genomic adjacency and it is a modelling assumption; `"none"` is
        #    what measures its cost.
        design = pd.DataFrame(design).ffill().bfill().to_numpy()
    elif imputation != "ffill_bfill":
        imputed = 0

    keep = ~np.isnan(design).any(axis=1) & ~np.isinf(design).any(axis=1)
    design = design[keep, :]

    before = observations.record
    record = FilterRecord(
        n_bins=before.n_bins,
        n_spots=before.n_spots,
        rdr_floored=before.rdr_floored,
        baf_clipped=before.baf_clipped,
        imputed=imputed,
        rows_dropped=int((~keep).sum()),
        rows_kept=int(keep.sum()),
    )

    return design, standardize, record
