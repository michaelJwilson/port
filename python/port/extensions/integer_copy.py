"""Integer copy number from a fitted emission and its uncertainty (#25, #6).

Implements `cna-maste-paper`'s `integer_copy_numbers.tex`, which neither
`cnaster` nor sal implements: the credible set
`Zhat_k = {(A, B) | ((A+B)/2, B/(A+B)) in C_0.95}` over an enumerated lattice,
scored by the squared Mahalanobis distance (no `rdr_weight`, unlike `cnaster`).
`mubar` must be de-biased (`sum_g lambda_g mubar_k = 1`, #5; see `debias_rdr`);
this module cannot check that it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.stats import chi2

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AlleleCopies",
    "IntegerCopyResult",
    "acn_lattice",
    "acn_observables",
    "debias_rdr",
    "decode_copy_state",
    "success_probability_variance",
]

AlleleCopies = tuple[int, int]
"""`(A, B)`, the major and minor allele copies; `A >= B` by convention."""

DEFAULT_MAX_ALLELE_COPY = 5
DEFAULT_MAX_TOTAL_COPY = 6
"""`cnaster`'s bounds (`hill_climbing_integer_copynumber_oneclone`'s defaults)."""

CHANNELS = 2
"""`(rdr, baf)`: the credible region's degrees of freedom."""


class IntegerCopyResult(NamedTuple):
    """One copy state decoded.

    `best`: the lattice point of least distance (by enumeration);
    `consistent`: every point inside the credible region (the paper's
    `Zhat_k`), by increasing distance; `distance`: squared Mahalanobis at
    `best`, chi-square on :data:`CHANNELS` dof; `threshold`:
    `chi2.ppf(level, CHANNELS)`; `level`: the credible level.
    """

    best: AlleleCopies
    consistent: tuple[AlleleCopies, ...]
    distance: float
    threshold: float
    level: float

    @property
    def identified(self) -> bool:
        """Whether the credible set holds exactly one pair."""
        return len(self.consistent) == 1

    @property
    def consistent_with_data(self) -> bool:
        """Whether `best` is inside the region; if not, no integer pair explains the fit."""
        return self.distance <= self.threshold


def acn_lattice(
    *,
    max_allele_copy: int = DEFAULT_MAX_ALLELE_COPY,
    max_total_copy: int = DEFAULT_MAX_TOTAL_COPY,
    phased: bool = True,
) -> tuple[AlleleCopies, ...]:
    """Every `(A, B)` within the bounds, excluding `(0, 0)`, sorted by total then `A`.

    `cnaster`'s hill-climber `candidates`, not `get_ordered_acn()` (which holds
    `(6, 0)`). `phased=False` keeps only `A >= B`, for an unphased `p`.
    Raises ValueError on a negative bound.
    """
    if max_allele_copy < 0 or max_total_copy < 0:
        msg = (
            f"bounds must not be negative, got max_allele_copy={max_allele_copy}, "
            f"max_total_copy={max_total_copy}"
        )
        raise ValueError(msg)

    lattice = [
        (first, second)
        for first in range(max_allele_copy + 1)
        for second in range(max_allele_copy + 1)
        if first + second <= max_total_copy
        if first + second > 0
        if phased or first >= second
    ]
    return tuple(sorted(lattice, key=lambda pair: (sum(pair), pair[0])))


def acn_observables(lattice: Sequence[AlleleCopies]) -> np.ndarray:
    """`(mubar, p)` implied by each `(A, B)`, shape `(n_lattice, 2)`.

    `mubar = (A + B) / 2`, `p = B / (A + B)` (minor allele, as the paper;
    `cnaster`'s `get_acn_baf_rdr` uses the major, i.e. `1 - p`).
    """
    copies = np.asarray(lattice, dtype=np.float64)
    total = copies.sum(axis=1)

    if np.any(total <= 0.0):
        msg = "a lattice point with zero total copies has no defined baf"
        raise ValueError(msg)

    return np.column_stack([total / 2.0, copies[:, 1] / total])


def debias_rdr(mean_rdr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rescale a fitted `mu` to satisfy `sum_g lambda_g mubar = 1` (#27).

    `mean_rdr` and `weights` (`lambda_g`, unnormalised) are `(n_positions,)`;
    returns `mean_rdr / (sum_g lambda_g mu_g / sum_g lambda_g)`. Raises
    ValueError on mismatched shapes or a non-positive weighted mean.
    """
    mean_rdr = np.asarray(mean_rdr, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if mean_rdr.shape != weights.shape:
        msg = f"shapes must agree, got {mean_rdr.shape} and {weights.shape}"
        raise ValueError(msg)

    total_weight = weights.sum()
    if total_weight <= 0.0:
        msg = f"weights must sum to a positive number, got {total_weight}"
        raise ValueError(msg)

    scale = float(np.dot(weights, mean_rdr) / total_weight)
    if scale <= 0.0:
        msg = f"the weighted mean must be positive, got {scale}"
        raise ValueError(msg)

    return mean_rdr / scale


def decode_copy_state(
    mean: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    level: float = 0.95,
    lattice: Sequence[AlleleCopies] | None = None,
) -> IntegerCopyResult:
    """Decode one copy state's `(mubar, p)` into integer copies.

    `mean` is `(2,)` and must be de-biased; `covariance` is `(2, 2)`
    (e.g. sal's `opt.fit.parameter_covariance`); `level` defaults to the
    paper's 0.95; `lattice` to :func:`acn_lattice`'s. Raises ValueError unless
    `covariance` is symmetric positive definite and `0 < level < 1`.
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)

    if mean.shape != (CHANNELS,):
        msg = f"mean must have shape ({CHANNELS},), got {mean.shape}"
        raise ValueError(msg)
    if covariance.shape != (CHANNELS, CHANNELS):
        msg = f"covariance must have shape ({CHANNELS}, {CHANNELS}), got {covariance.shape}"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = f"level must lie strictly in (0, 1), got {level}"
        raise ValueError(msg)
    if not np.allclose(covariance, covariance.T):
        msg = "covariance must be symmetric"
        raise ValueError(msg)

    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0.0:
        msg = (
            f"covariance must be positive definite; smallest eigenvalue is "
            f"{np.min(eigenvalues):.6e}, so at least one direction is "
            "unidentified and no credible region is defined"
        )
        raise ValueError(msg)

    points = tuple(acn_lattice()) if lattice is None else tuple(lattice)
    observables = acn_observables(points)

    residual = observables - mean
    precision = np.linalg.inv(covariance)
    distances = np.einsum("ij,jk,ik->i", residual, precision, residual)

    order = np.argsort(distances, kind="stable")
    threshold = float(chi2.ppf(level, CHANNELS))

    return IntegerCopyResult(
        best=points[int(order[0])],
        consistent=tuple(
            points[int(index)] for index in order if distances[index] <= threshold
        ),
        distance=float(distances[order[0]]),
        threshold=threshold,
        level=level,
    )


def success_probability_variance(
    alpha: float, beta: float, covariance: Sequence[Sequence[float]] | np.ndarray
) -> float:
    """Propagate a fitted `(alpha, beta)` covariance onto `p = alpha / (alpha + beta)`.

    Delta method with `dp/d(alpha) = beta / (alpha + beta)^2`,
    `dp/d(beta) = -alpha / (alpha + beta)^2`; `covariance` is `(2, 2)` in
    `(alpha, beta)` order. Returns `Var(p)` to first order; raises ValueError
    if `alpha + beta <= 0` or the variance is not positive.
    """
    covariance = np.asarray(covariance, dtype=np.float64)

    if covariance.shape != (CHANNELS, CHANNELS):
        msg = f"covariance must have shape (2, 2), got {covariance.shape}"
        raise ValueError(msg)

    concentration = alpha + beta
    if concentration <= 0.0:
        msg = f"alpha + beta must be positive, got {concentration}"
        raise ValueError(msg)

    gradient = np.array(
        [beta / concentration**2, -alpha / concentration**2], dtype=np.float64
    )
    variance = float(gradient @ covariance @ gradient)

    if variance <= 0.0:
        msg = (
            f"the propagated variance is {variance:.6e}, not positive; the "
            "supplied covariance is not positive semi-definite"
        )
        raise ValueError(msg)

    return variance
