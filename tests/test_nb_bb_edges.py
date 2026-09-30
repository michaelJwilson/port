"""#560: port's negative-binomial kernels in log space, and the edges of the ones it calls.

`p = 1 / (1 + a)`, `a = alpha * mean`, rounds to exactly 1 in float64 below
`a` of about 1.1e-16. `cnaster`'s kernel then scores every count 0; port's
numpy and `jax` copies scored a count of 0 NaN and any other count `-inf`,
and lost digits of `log(1 - p)` below `a` of about 1e-4. Each is pinned here
against `mpmath` at 50 digits, whose `log1p` and `loggamma` form no `p`.

The beta-binomial floors `a = p tau` and `b = (1 - p) tau` at 1e-10 and stays
normalized there, so `p_binom -> 0, 1` is not a degenerate optimum. Its
large-`tau` defect is `lgamma` cancellation, pinned as `bug` for `cnaster`
and `sal`.
"""

from __future__ import annotations

from collections.abc import Callable

import mpmath  # type: ignore[import-untyped]
import numpy as np
import pytest
from scipy.special import logsumexp

mpmath.mp.dps = 50

FLOOR = 1e-10
COUNTS = np.array([0, 1, 7, 42, 300, 1000, 2500], dtype=np.float64)
VANISHING = 1e-17
"""A mean at which `alpha * mean = 1e-19` for `alpha = 0.01`: `p` rounds to 1."""

Scorer = Callable[[np.ndarray, float, float], np.ndarray]


def _reference(k: float, mean: float, alpha: float) -> float:
    """The negative binomial's log pmf in `mpmath`, `alpha` floored as `cnaster` floors `r`."""
    alpha = max(alpha, FLOOR)
    r = mpmath.mpf(1) / mpmath.mpf(alpha)
    a = mpmath.mpf(alpha) * mpmath.mpf(mean)
    kk = mpmath.mpf(k)
    return float(
        mpmath.loggamma(kk + r)
        - mpmath.loggamma(r)
        - mpmath.loggamma(kk + 1)
        - r * mpmath.log1p(a)
        + kk * (mpmath.log(a) - mpmath.log1p(a))
    )


def _references(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    return np.array([_reference(float(k), mean, alpha) for k in counts])


def _bulk(counts: np.ndarray, alpha: float):  # type: ignore[no-untyped-def]
    """A pseudobulk whose allele channel scores exactly 0: no trials, `tau = inf`."""
    from port.extensions.copy_likelihood import Pseudobulk

    zeros = np.zeros_like(counts)
    return Pseudobulk(
        counts_nb=counts,
        base_nb_mean=np.ones_like(counts),
        counts_bb=zeros,
        total_bb_RD=zeros,
        normal_log_lambda=zeros,
        dispersion=alpha,
        taus=np.inf,
    )


def _copy_likelihood(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.extensions.copy_likelihood import _emission

    bins = np.arange(counts.size)
    return np.asarray(
        _emission(np.log(mean), np.array(0.5), _bulk(counts, alpha), bins)
    )


def _schemes(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.sandbox.integer_decoding.schemes import _emission

    bins = np.arange(counts.size)
    return np.asarray(
        _emission(np.log(mean), np.array(0.5), _bulk(counts, alpha), bins)
    )


def _jax(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    """`jax_hmm.emission` with no trials, so the beta-binomial adds 0 to within 1e-15."""
    from port.extensions.jax_hmm import emission

    zeros = np.zeros_like(counts)
    scores = emission(
        np.array([[np.log(mean)]]),
        np.array([[alpha]]),
        np.array([[0.5]]),
        np.array([[20.0]]),
        counts,
        np.ones_like(counts),
        zeros,
        zeros,
    )
    return np.asarray(np.asarray(scores)[0])


def _clone_mixture(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.sandbox.admixture.clone_mixture import _nb

    return _nb(counts, np.full(counts.shape, mean), alpha)


def _variants(counts: np.ndarray, mean: float, alpha: float) -> np.ndarray:
    from port.sandbox.admixture.variants import _nb

    return _nb(counts, np.full(counts.shape, mean), alpha)


KERNELS: dict[str, Scorer] = {
    "copy_likelihood": _copy_likelihood,
    "jax_hmm": _jax,
    "schemes": _schemes,
    "clone_mixture": _clone_mixture,
    "variants": _variants,
}

NORMAL = [
    (mean, alpha) for mean in (0.5, 30.0, 1000.0) for alpha in (1e-3, 0.07, 0.5, 3.0)
] + [(1e-10, 0.01)]
"""Realistic means and dispersions, and `a = 1e-12`, where forming `p` lost 0.089 nats at a count of 1000."""


@pytest.mark.oracle
@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize(("mean", "alpha"), NORMAL)
def test_each_port_kernel_is_the_negative_binomial(
    name: str, mean: float, alpha: float
) -> None:
    """`mpmath` at 50 digits, to 1e-9 relative and 1e-12 absolute."""
    np.testing.assert_allclose(
        KERNELS[name](COUNTS, mean, alpha),
        _references(COUNTS, mean, alpha),
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("name", KERNELS)
def test_a_vanishing_mean_scores_a_large_count_as_impossible(name: str) -> None:
    """At `alpha * mean = 1e-19` a count of 1000 is `mpmath`'s -43,420 nats, and 0 is finite.

    Checked against `mpmath` to 1e-9 relative; the old form scored the count
    `-inf` and 0 NaN, `cnaster`'s scores both 0.
    """
    counts = np.array([0.0, 1000.0])
    scores = KERNELS[name](counts, VANISHING, 0.01)

    assert np.isfinite(scores).all()
    assert scores[1] < -30_000.0
    np.testing.assert_allclose(
        scores, _references(counts, VANISHING, 0.01), rtol=1e-9, atol=1e-12
    )


@pytest.mark.analytic
@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("mean", [1e-3, VANISHING])
def test_each_port_kernel_sums_to_one_at_a_small_mean(name: str, mean: float) -> None:
    """`logsumexp` over counts 0..200 is 0 to 1e-12: a pmf, not a score of 1 per count."""
    counts = np.arange(201, dtype=np.float64)
    total = logsumexp(KERNELS[name](counts, mean, 0.5))

    assert abs(total) < 1e-12


@pytest.mark.oracle
@pytest.mark.parametrize(("mean", "alpha"), [*NORMAL, (VANISHING, 0.01)])
def test_the_closed_form_gradient_is_mpmaths_derivative(
    mean: float, alpha: float
) -> None:
    """`nb_partials` against `mpmath.diff` of the 50-digit log pmf, to 1e-9 relative.

    At the vanishing mean `d ell / d log mean` is the count, 1000 for 1000,
    where the old gradient read 0 because `cnaster`'s score does not move.
    """
    from port.patch.hmm_nophasing.gradient import nb_partials

    d_eta, d_alpha = nb_partials(COUNTS, np.full(COUNTS.shape, mean), np.array(alpha))

    def by_log_mean(k: float) -> float:
        return float(
            mpmath.diff(
                lambda t: _reference_mp(k, mpmath.exp(t), alpha),
                mpmath.log(mean),
            )
        )

    def by_log_alpha(k: float) -> float:
        return float(
            mpmath.diff(
                lambda t: _reference_mp(k, mean, mpmath.exp(t)),
                mpmath.log(alpha),
            )
        )

    np.testing.assert_allclose(
        d_eta, [by_log_mean(float(k)) for k in COUNTS], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        d_alpha, [by_log_alpha(float(k)) for k in COUNTS], rtol=1e-9, atol=1e-9
    )


def _reference_mp(k: float, mean: object, alpha: object) -> object:
    """`_reference` without the float round trip, for `mpmath.diff`; `alpha` above the floor."""
    r = 1 / mpmath.mpf(alpha)
    a = mpmath.mpf(alpha) * mpmath.mpf(mean)
    kk = mpmath.mpf(k)
    return (
        mpmath.loggamma(kk + r)
        - mpmath.loggamma(r)
        - mpmath.loggamma(kk + 1)
        - r * mpmath.log1p(a)
        + kk * (mpmath.log(a) - mpmath.log1p(a))
    )


# --- cnaster --------------------------------------------------------------


@pytest.mark.bug
def test_cnasters_kernel_scores_every_count_zero_below_the_dispersion_floor() -> None:
    """At `alpha = 1e-17`, `mu = 10`, counts 0 and 1000 score 0; `mpmath` says -10.0 and -3,619.5.

    The #560 defect reached through `alpha` rather than the mean: `r` is
    floored at `1 / 1e-10` but `p = 1 / (1 + alpha * lambda)` takes the raw
    `alpha` and rounds to 1. `cnaster` passes `bounds=None` to BFGS, so
    `log alpha` is unbounded.
    """
    from cnaster.hmm_nophasing import _nb_logpmf_1d

    out = np.full(2, np.nan)
    _nb_logpmf_1d(np.array([0.0, 1000.0]), np.ones(2), 10.0, 1e-17, out)

    np.testing.assert_array_equal(out, [0.0, 0.0])


@pytest.mark.oracle
def test_the_patched_kernel_floors_the_dispersion_in_both_terms() -> None:
    """`nb_logpmf._nb_logpmf_1d` at `alpha = 1e-17` is `mpmath` at `alpha = 1e-10`, to 1e-4 absolute.

    Not 1e-9: at the floor `r = 1e10`, and `lgamma(k + r) - lgamma(r)`
    cancels two values near 2.2e11 whose spacing is 3e-5 (1.5e-5 measured).
    That is `lgamma`'s, in every kernel here; the #560 defect was 0 for all.
    """
    from port.patch.hmm_nophasing.nb_logpmf import _nb_logpmf_1d

    out = np.full(COUNTS.size, np.nan)
    _nb_logpmf_1d(COUNTS, np.ones(COUNTS.size), 10.0, 1e-17, out)

    np.testing.assert_allclose(out, _references(COUNTS, 10.0, 1e-17), rtol=0, atol=1e-4)


DEGENERATE = (-43.22, 0.1184, 1000.0, 1000.0)
"""`(log mu, alpha, exposure, count)` of dev_tree_1s_hard r0's degenerate state (#560)."""


def _bound_kernel(module: str) -> float:
    """The `_nb_logpmf_1d` `module` compiles in by name, on one degenerate bin."""
    import importlib

    log_mu, alpha, exposure, count = DEGENERATE
    out = np.full(1, np.nan)
    importlib.import_module(module)._nb_logpmf_1d(
        np.array([count]), np.array([exposure]), float(np.exp(log_mu)), alpha, out
    )
    return float(out[0])


def _field(kernel_name: str) -> float:
    """One bin, one spot, one clone; no allele trials, so the field is the NB score."""
    import importlib

    log_mu, alpha, exposure, count = DEGENERATE
    module, name = kernel_name.rsplit(".", 1)
    kernel = getattr(importlib.import_module(module), name)
    one = np.ones((1, 1))
    field = kernel(
        np.full((1, 1), count),
        np.full((1, 1), exposure),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([log_mu]),
        np.array([alpha]),
        np.array([0.5]),
        np.array([30.0]),
        np.zeros((1, 1), dtype=np.int64),
        one[0],
        np.zeros((1, 1)),
    )
    return float(field[0, 0])


def _emission_into() -> float:
    from port.sandbox.patch.emission import emission_into

    log_mu, alpha, exposure, count = DEGENERATE
    out_rdr, out_baf = np.zeros((1, 1, 1)), np.zeros((1, 1, 1))
    emission_into(
        np.full((1, 1), count),
        np.full((1, 1), exposure),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([log_mu]),
        np.array([alpha]),
        np.array([0.5]),
        np.array([30.0]),
        out_rdr,
        out_baf,
        False,
    )
    return float(out_rdr[0, 0, 0])


def _np_merge() -> float:
    from port.sandbox.np_merge import _emissions

    log_mu, alpha, exposure, count = DEGENERATE
    X = np.zeros((1, 2, 1))
    X[0, 0, 0] = count
    res = {
        "new_log_mu": [log_mu],
        "new_alphas": [alpha],
        "new_p_binom": [0.5],
        "new_taus": [30.0],
    }
    rdr, _ = _emissions(
        X, np.full((1, 1), exposure), np.zeros((1, 1)), res, np.zeros(1)
    )
    return float(rdr[0, 0, 0])


SITES: dict[str, Callable[[], float]] = {
    "shifted_emission": lambda: _bound_kernel(
        "port.patch.hmm_nophasing.shifted_emission"
    ),
    "coded_emission": lambda: _bound_kernel("port.patch.hmm_phased.coded_emission"),
    "fused_field": lambda: _field("port.patch.hmrf.fused_field.fused_spot_clone_field"),
    "tabulated_field": lambda: _field(
        "port.patch.hmrf.tabulated_field.tabulated_spot_clone_field"
    ),
    "sandbox_emission": _emission_into,
    "np_merge": _np_merge,
    "hmm_initialize_backends": lambda: _bound_kernel(
        "port.sandbox.patch.hmm_initialize.backends"
    ),
}
"""Every port site that compiles `_nb_logpmf_1d` in by name, out of `nb_logpmf.patched()`'s reach."""


@pytest.mark.oracle
@pytest.mark.parametrize("site", sorted(SITES))
def test_every_compiled_in_kernel_scores_a_vanishing_mean_in_log_space(
    site: str,
) -> None:
    """At `log mu = -43.22`, `alpha = 0.1184`, exposure 1000, a count of 1000 scores `mpmath`'s -38,403.9 to 1e-9, not 0.

    `cnaster`'s kernel scores it 0 -- probability 1 -- because `p` rounds to
    1 (`a = 2.0e-17`); each site imports the log-space kernel (#560) instead.
    """
    log_mu, alpha, exposure, count = DEGENERATE
    score = SITES[site]()

    assert score < -30_000.0
    np.testing.assert_allclose(
        score, _reference(count, exposure * np.exp(log_mu), alpha), rtol=1e-9
    )


BOUNDARY = [0.0, 1e-12, 1.0 - 1e-12, 1.0, -0.2, 1.3]
"""`p_binom` at, beyond and next to the boundary, where `a` or `b` is floored at 1e-10."""


@pytest.mark.analytic
@pytest.mark.parametrize("p_binom", BOUNDARY)
def test_cnasters_floored_beta_binomial_is_still_a_pmf(p_binom: float) -> None:
    """`logsumexp` over `k = 0..100` of `_bb_logpmf_1d` at `tau = 1000` is 0 to 1e-10.

    So a floored `a` or `b` is a point mass, not a score of 1 for every
    count: `p_binom -> 0, 1` is no degenerate optimum.
    """
    from cnaster.hmm_nophasing import _bb_logpmf_1d

    k = np.arange(101, dtype=np.float64)
    out = np.full(k.size, np.nan)
    _bb_logpmf_1d(k, np.full(k.size, 100.0), p_binom, 1000.0, out)

    assert abs(logsumexp(out)) < 1e-10


@pytest.mark.bug
def test_cnasters_beta_binomial_is_not_a_pmf_at_a_large_concentration() -> None:
    """At `tau = 1e16`, `p = 0.3`, `n = 100` the pmf sums to about `e^132`, not 1.

    `lgamma(n - k + b) - lgamma(n + a + b)` cancels two values near 3.6e17,
    whose spacing is 64. 5,000 is `get_bounds`' `max_tau`, but `cnaster`
    passes `bounds=None`; BFGS stopped at `tau` of 6e4 to 8e4 on
    under-dispersed data in two trials, so the regime is not shown reached.
    """
    from cnaster.hmm_nophasing import _bb_logpmf_1d

    k = np.arange(101, dtype=np.float64)
    out = np.full(k.size, np.nan)
    _bb_logpmf_1d(k, np.full(k.size, 100.0), 0.3, 1e16, out)

    assert abs(logsumexp(out)) > 1.0


# --- sal ------------------------------------------------------------------


@pytest.mark.oracle
def test_port_dense_path_scores_a_vanishing_mean_as_mpmath_does() -> None:
    """`dense_emission.nb_states` (sal's Rust kernel) at `alpha * mean = 1e-19`: `mpmath`, 1e-9 relative.

    `sal` has no `p` to round: it is right here, and port needs no guard.
    """
    from port.patch.hmm_nophasing.dense_emission import nb_states

    scores = nb_states(
        COUNTS, np.ones(COUNTS.size), np.array([VANISHING]), np.array([0.01])
    )

    np.testing.assert_allclose(
        scores[0], _references(COUNTS, VANISHING, 0.01), rtol=1e-9, atol=1e-12
    )


@pytest.mark.oracle
def test_sals_negative_binomial_family_scores_a_vanishing_mean_as_mpmath_does() -> None:
    """`NegativeBinomialEmission.log_density`, which `emission_family` fits through: `mpmath`, 1e-9 relative."""
    import torch
    from sal.emissions import NegativeBinomialEmission

    family = NegativeBinomialEmission(
        dispersion=np.array([100.0]), mean=np.array([VANISHING])
    )
    scores = family.log_density(torch.tensor(COUNTS)).numpy().ravel()

    np.testing.assert_allclose(
        scores, _references(COUNTS, VANISHING, 0.01), rtol=1e-9, atol=1e-12
    )


@pytest.mark.analytic
@pytest.mark.parametrize("p_binom", BOUNDARY)
def test_port_dense_beta_binomial_is_still_a_pmf_at_the_floor(p_binom: float) -> None:
    """`dense_emission.bb_states` at `tau = 1000` sums to 1 over `k = 0..100`, to 1e-10."""
    from port.patch.hmm_nophasing.dense_emission import bb_states

    k = np.arange(101, dtype=np.float64)
    scores = bb_states(
        k, np.full(k.size, 100.0), np.array([p_binom]), np.array([1000.0])
    )

    assert abs(logsumexp(scores[0])) < 1e-10


@pytest.mark.bug
def test_sals_beta_binomial_is_not_a_pmf_at_a_large_concentration() -> None:
    """`sal`'s dense beta-binomial at `tau = 1e16`, `p = 0.3`, `n = 100` sums to about `e^-60`, not 1.

    The same `lgamma` cancellation as `cnaster`'s; `sal` scores the tables
    `lgamma` fills. 1e-12 relative at `tau = 5000`, 1e-3 at 1e12.
    """
    from port.patch.hmm_nophasing.dense_emission import bb_states

    k = np.arange(101, dtype=np.float64)
    scores = bb_states(k, np.full(k.size, 100.0), np.array([0.3]), np.array([1e16]))

    assert abs(logsumexp(scores[0])) > 1.0
