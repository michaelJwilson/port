"""Integer copy number from a fitted emission and its uncertainty.

The method `cna-maste-paper`'s `integer_copy_numbers.tex` specifies, which
neither `cnaster` nor `snakes_and_ladders` implements. Issues #25 and #6.

The paper defines the decoding as a **one-to-many map**: given a fitted copy
state's `(mubar_k, p_k)` and the marginal uncertainty around it, the answer is
the *set* of integer pairs whose implied observables fall inside a credible
region,

    Zhat_k = { (A, B) | ( (A+B)/2, B/(A+B) ) in C_0.95(...) },

and it contrasts that with what `cnaster` does -- "ad-hoc factors in a custom
objective, e.g. for down-weighting the read-depth ratio relative to `b`-allele
frequency (irrespective of the inferred dispersions), neglected parameter
uncertainty and assumed ill-posed regularisation with respect to ploidy".

Two things follow, and they are the whole of this module.

**The objective needs no arbitrary channel weight.** `cnaster` adds
`|1 - frac_rdr/mu|` and `|1 - frac_baf/p_binom|` with a `rdr_weight` between
them, read from configuration in a commented-out block. Standardizing each
residual by its own standard error removes the question: two channels in
units of their own sigma are already commensurable, and the sum is a
chi-square rather than a quantity whose scale someone has to choose. That is
#6 item 5, realized.

**The best answer and the consistent set come from one quantity.** The
squared Mahalanobis distance is the objective *and* the membership test --
:func:`best_acn` takes its argmin and :func:`credible_acn` thresholds it at
`chi2.ppf(level, 2)`. Reporting only the argmin, as `cnaster` does, discards
the statement the paper is making: an eight-sigma winner and a one-sigma
winner are different results, and a state whose credible set holds four pairs
has not been decoded.

**The lattice is small enough to enumerate**, so no search is involved and
neither is a local optimum. `cnaster` hill-climbs over sixteen states; this
evaluates all of them. Issue #25 is where that stops being a convenience and
becomes the referee.

Dependence on the de-biasing, which is not optional
---------------------------------------------------
`mubar_k = (A + B) / 2` holds for the **de-biased** mean the paper defines by
`sum_g lambda_g mubar_k = 1`, not for a raw fitted `mu`. `cnaster` computes
that correction in `hmm_nophasing.compute_logmu_shifts` and applies it
nowhere (issue #5): the branch that would raises, the result is assigned and
never read, and `log_gamma` is declared `# TODO` and dropped.

So a fitted `mu` carries an unknown per-clone scale, and `(A+B)/2` compares it
against an absolute one. The decoding is **wrong by that scale**, in a way no
amount of care here can fix -- it is a property of the input.

:func:`debias_rdr` implements the constraint so a caller can satisfy it, and
:func:`IntegerCopyResult` carries no flag claiming it was satisfied, because
this module cannot check.

`tests/test_integer_copy.py` measures what the scale costs, and the answer is
not the obvious one. **A scale error corrupts the confidence rather than the
answer.** At `(4, 2)` with `sigma_mubar = 0.08` the argmin stays `(4, 2)` at
every error up to twenty per cent -- the lattice spacing in `mubar` is `0.5`
and no competitor gets closer -- while the squared residual runs `0.00`,
`0.56`, `3.52`, `9.00`, `56.25` and the credible set empties at eight per
cent.

That is the argument for the one-to-many map in one line: an argmin-only
decoder returns the correct pair at a twenty per cent scale error with
nothing to say the fit is fifty-six chi-square units from explaining it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
"""`(A, B)`, the major and minor allele copies. `A >= B` by convention."""

DEFAULT_MAX_ALLELE_COPY = 5
DEFAULT_MAX_TOTAL_COPY = 6
"""`cnaster`'s bounds, restated so the two enumerate the same lattice.

Taken from `hill_climbing_integer_copynumber_oneclone`'s defaults. A
comparison against that function on a different lattice would measure the
lattice rather than the method.
"""

CHANNELS = 2
"""`(rdr, baf)`. The degrees of freedom of the credible region."""


@dataclass(frozen=True)
class IntegerCopyResult:
    """One copy state decoded, with what the decoding is entitled to claim.

    Parameters
    ----------
    best : AlleleCopies
        The lattice point minimising the standardized residual. Found by
        enumeration, so it is the true minimum rather than a search's answer.
    consistent : tuple[AlleleCopies, ...]
        Every lattice point inside the credible region, `best` first and the
        rest in increasing distance. **This is the paper's `Zhat_k`**, and a
        caller that reads only `best` has thrown away the result.
    distance : float
        The squared Mahalanobis distance at `best`. A chi-square on
        :data:`CHANNELS` degrees of freedom under the fitted model, so it is
        a goodness of fit rather than a ranking: a large value says no
        integer pair explains this state, which `cnaster`'s argmin cannot
        say.
    threshold : float
        `chi2.ppf(level, CHANNELS)`, the value `distance` is compared against
        for membership. Carried so a caller can re-threshold without
        re-deriving the level.
    level : float
        The credible level the set was built at.
    """

    best: AlleleCopies
    consistent: tuple[AlleleCopies, ...]
    distance: float
    threshold: float
    level: float

    @property
    def identified(self) -> bool:
        """Whether the credible set holds exactly one pair.

        The question `cnaster`'s output cannot answer. A state with four
        consistent pairs is not decoded, however confident the argmin looks.
        """
        return len(self.consistent) == 1

    @property
    def consistent_with_data(self) -> bool:
        """Whether the best pair is itself inside the region.

        False where **no** integer pair explains the fit. The set is then
        empty and `best` is the least-bad point rather than a decoding, which
        is the case a caller must not read as an answer.
        """
        return self.distance <= self.threshold


def acn_lattice(
    *,
    max_allele_copy: int = DEFAULT_MAX_ALLELE_COPY,
    max_total_copy: int = DEFAULT_MAX_TOTAL_COPY,
    phased: bool = True,
) -> tuple[AlleleCopies, ...]:
    """Every `(A, B)` within the bounds, excluding `(0, 0)`.

    This is `cnaster`'s search space, built the way `cnaster` builds it --
    from the bounds, not from a list. 25 points at the defaults.

    **`get_ordered_acn()` is not that space.** It returns a hard-coded
    sixteen-entry tuple which the hill climbers do not use: they construct
    `candidates` from `max_allele_copy` and `max_total_copy` at every call.
    The two disagree, and the list is the one that is wrong -- it contains
    `(6, 0)`, whose major allele exceeds the `max_allele_copy=5` it is
    documented beside. A test pins this module against `candidates` rather
    than against the list.

    Parameters
    ----------
    phased : bool
        Whether `(A, B)` and `(B, A)` are distinct points. True by default,
        matching `cnaster` and the paper's map, which imposes no ordering.
        The two differ only in which allele is the minor one, so they imply
        `p` and `1 - p` -- the same state under a phase flip. `emission.tex`
        calls the labelling identifiable only "up to the non-identifiability
        inherent to phasing", so a caller working from an unphased `p` should
        pass `phased=False` and read the result as a major/minor pair.

        `(0, 0)` is excluded either way: total copy zero has no defined
        `baf`, and `cnaster`'s objective returns a large constant for it
        rather than scoring it.

    Raises
    ------
    ValueError
        If either bound is negative, where the lattice is empty and a caller
        has passed something it did not mean.
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

    `mubar = (A + B) / 2` and `p = B / (A + B)`, which is `emission.tex`
    verbatim. The minor allele is the numerator, matching the paper's
    `p_k = b_k / (a_k + b_k)`.

    **`cnaster` uses the other one.** `get_acn_baf_rdr` returns
    `acn[:, 0] / total`, the *major* allele. Phasing makes the labelling
    non-identifiable so neither is wrong, but the two conventions are
    complements and a comparison that mixes them is off by `1 - p`. A test
    pins which this module uses.
    """
    copies = np.asarray(lattice, dtype=np.float64)
    total = copies.sum(axis=1)

    if np.any(total <= 0.0):
        msg = "a lattice point with zero total copies has no defined baf"
        raise ValueError(msg)

    return np.column_stack([total / 2.0, copies[:, 1] / total])


def debias_rdr(mean_rdr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rescale a fitted `mu` to satisfy `sum_g lambda_g mubar = 1`.

    The constraint `emission.tex` states and `cnaster` does not apply. The
    index that binds is the sum over a clone's *profile* -- the state varies
    with position -- which is what `compute_logmu_shifts` computes and what
    #27 records the paper as writing ambiguously.

    Parameters
    ----------
    mean_rdr : np.ndarray
        The fitted `exp(log_mu)` per position, or per state expanded over the
        profile. Shape `(n_positions,)`.
    weights : np.ndarray
        `lambda_g` per position, the normal baseline. Need not be normalised;
        this divides by their sum, so a caller passing raw baseline counts
        gets the same answer as one passing a distribution.

    Returns
    -------
    np.ndarray
        `mean_rdr` divided by `sum_g lambda_g mu_g / sum_g lambda_g`.

    Raises
    ------
    ValueError
        If the shapes disagree, or the weighted mean is not positive -- where
        the constraint cannot be satisfied by any scaling and a silent `inf`
        would be the alternative.
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

    Parameters
    ----------
    mean : array of shape (2,)
        The fitted `(mubar_k, p_k)`. **`mubar` must be de-biased**; see
        :func:`debias_rdr` and the module docstring for what happens if it is
        not.
    covariance : array of shape (2, 2)
        Its covariance. Supply the full matrix; `snakes_and_ladders`'
        `opt.fit.parameter_covariance` produces it, and its
        `parameter_covariance` refuses a fit that is not at a maximum rather
        than inverting a singular information matrix.

        The off-diagonal is usually zero, and that is a modelling statement
        rather than an approximation: at a fixed state posterior the count
        and allele channels factorize, so the M step separates and the two
        parameters are estimated independently. A caller with a joint fit
        should pass what it measured.
    level : float
        The credible level. `0.95` is the paper's.
    lattice : sequence of (A, B), optional
        Defaults to :func:`acn_lattice`'s.

    Returns
    -------
    IntegerCopyResult

    Raises
    ------
    ValueError
        If `covariance` is not symmetric positive definite, or `level` is not
        strictly inside `(0, 1)`. Refusing is the point: a non-positive
        covariance is a fit that did not identify its parameters, and
        `snakes_and_ladders`' own `parameter_covariance` refuses the same
        cases. Returning a very large credible set instead would report
        "every integer is possible" as though it were a measurement.
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

    The bridge from what `snakes_and_ladders` reports to what
    :func:`decode_copy_state` takes. Upstream's `BetaBinomialEmission` carries
    `(trials, alpha, beta)` and its `opt.fit.parameter_covariance` returns the
    covariance of the parameter vector; the decoding needs the variance of
    the success probability, and the two differ by a delta method:

        dp/d(alpha) =  beta  / (alpha + beta)^2
        dp/d(beta)  = -alpha / (alpha + beta)^2

    Done here rather than in a test because it is the same calculation every
    caller needs and getting the sign of the second term wrong is silent --
    it inflates the variance instead of cancelling, by exactly the
    correlation, so the credible set comes out too wide and nothing raises.

    Parameters
    ----------
    alpha, beta : float
        The fitted pair for one state.
    covariance : array of shape (2, 2)
        Their joint covariance, in `(alpha, beta)` order. The off-diagonal
        matters and is usually large and negative: the concentration is
        better determined than its split, so the two move together.

    Returns
    -------
    float
        `Var(p)` to first order.

    Raises
    ------
    ValueError
        If `alpha + beta` is not positive, where `p` is undefined, or the
        propagated variance is not positive -- which a covariance that is not
        positive semi-definite can produce, and which would otherwise reach
        :func:`decode_copy_state` as a silently invalid region.
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
