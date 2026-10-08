"""Impossible events and zero coverage in `cnaster` and port, against #415's convention (#413).

`bug` pins defects; `warning` pins suspicious conventions; `oracle`/`patch` pin correct
edges.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp

# --- bug -------------------------------------------------------------------


@pytest.mark.bug
def test_cnasters_zero_posterior_check_passes_an_all_minus_inf_column() -> None:
    """`cnaster`'s zero-posterior check passes an all `-inf` column, normalizing to `nan`."""
    from cnaster.hmm import compute_copy_state_posterior

    rng = np.random.default_rng(0)
    log_alpha = rng.normal(-5.0, 1.0, (3, 5))
    log_beta = np.zeros((3, 5))
    log_alpha[:, 2] = -np.inf

    with np.errstate(invalid="ignore"):
        log_gamma = compute_copy_state_posterior(log_alpha, log_beta)

    finite = np.delete(log_gamma, 2, axis=1)
    np.testing.assert_allclose(scipy_logsumexp(finite, axis=0), 0.0, atol=1e-12)
    assert np.isnan(log_gamma[:, 2]).all(), (
        "an all -inf column no longer normalizes to nan: the check is fixed"
    )


@pytest.mark.bug
def test_hmrfs_logsumexp_is_nan_on_an_all_minus_inf_row() -> None:
    """`hmrf.logsumexp` is #411's `icm.logsumexp` again, unguarded; latent."""
    from cnaster.hmrf import logsumexp

    finite = np.array([-1.0, 2.0, 0.5])
    assert logsumexp(finite) == pytest.approx(scipy_logsumexp(finite), abs=1e-12)

    with np.errstate(invalid="ignore"):
        assert np.isnan(logsumexp(np.full(3, -np.inf)))


def _shifted_instance() -> dict[str, Any]:
    rng = np.random.default_rng(5)
    n_obs = 12
    return {
        "X": np.stack(
            [rng.poisson(40, (n_obs, 1)), rng.integers(0, 10, (n_obs, 1))], axis=1
        ).astype(float),
        "base": rng.uniform(20, 60, (n_obs, 1)),
        "total": np.full((n_obs, 1), 10.0),
        "rates": np.log(np.array([1.0, 2.0])),
        "alphas": np.full((2, 1), 0.1),
        "p_binom": np.array([[0.5], [0.2]]),
        "taus": np.full((2, 1), 30.0),
        "shift": rng.normal(0.0, 0.3, n_obs),
    }


@pytest.mark.bug
def test_one_minus_inf_shift_makes_every_row_of_the_shifted_emission_nan() -> None:
    """One `-inf` shift makes every centred row `nan`, unlike the uncentred formula (#292)."""
    from cnaster.hmm_nophasing import hmm_nophasing as upstream
    from port.patch.hmm_nophasing import hmm_nophasing
    from port.pipeline import with_attributes

    shifted = with_attributes(hmm_nophasing, apply_logmu_shift=True)
    case = _shifted_instance()
    shift = case["shift"].copy()
    shift[3] = -np.inf
    rest = np.arange(shift.size) != 3

    previous = hmm_nophasing._row_shift
    hmm_nophasing._row_shift = shift

    try:
        with np.errstate(invalid="ignore", over="ignore"):
            ours = shifted.compute_emission_probability_nb_betabinom(
                case["X"],
                case["base"],
                case["rates"][:, None],
                case["alphas"],
                case["total"],
                case["p_binom"],
                case["taus"],
            )

        for state in range(2):
            with np.errstate(over="ignore"):
                theirs = upstream.compute_emission_probability_nb_betabinom(
                    case["X"],
                    case["base"] * np.exp(case["rates"][state] - shift)[:, None],
                    np.zeros((2, 1)),
                    case["alphas"],
                    case["total"],
                    case["p_binom"],
                    case["taus"],
                )
            assert np.isfinite(theirs[0][state][rest]).all()
            assert np.isnan(ours[0][state][rest]).all(), (
                "rows with a finite shift are no longer nan: the centring is fixed"
            )
    finally:
        hmm_nophasing._row_shift = previous


def _jax_arguments() -> dict[str, np.ndarray]:
    return {
        "alphas": np.full((2, 1), 0.2),
        "p_binom": np.full((2, 1), 0.4),
        "taus": np.full((2, 1), 20.0),
        "counts_nb": np.array([28.0, 0.0, 55.0, 39.0, 3.0]),
        "base_nb_mean": np.array([30.0, 0.0, 50.0, 40.0, 0.0]),
        "counts_bb": np.array([3.0, 0.0, 4.0, 5.0, 1.0]),
        "total_bb_RD": np.array([8.0, 0.0, 9.0, 10.0, 2.0]),
    }


@pytest.mark.analytic
def test_a_zero_exposure_bin_leaves_the_jax_gradient_finite() -> None:
    """`jax.grad` with a zero-exposure bin matches a central difference to 1e-6 relative (#560)."""
    import jax
    import jax.numpy as jnp
    from port.qa.jax_hmm import emission

    arguments = _jax_arguments()

    def total(log_mu: Any) -> Any:
        return emission(log_mu, **arguments).sum()

    # NB float64 throughout: `port.qa.jax_setup` enables x64 on import.
    log_mu = jnp.array([[0.1], [-0.2]])
    step = 1e-6
    bump = jnp.array([[step], [0.0]])
    central = (total(log_mu + bump) - total(log_mu - bump)) / (2 * step)
    gradient = np.asarray(jax.grad(total)(log_mu))

    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[0, 0], float(central), rtol=1e-6)


# --- warning ---------------------------------------------------------------


@pytest.mark.warning
def test_a_nowhere_finite_likelihood_returns_the_dispersion_fits_initial_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing finite, `fit_dispersions_mle` returns its initial guess `(0.05, 100)`."""
    from cnaster import hmm_initialize

    def nowhere(X: Any, *_: Any) -> tuple[np.ndarray, np.ndarray]:
        empty = np.full((2, X.shape[0]), np.nan)
        return empty, empty

    monkeypatch.setattr(
        hmm_initialize.hmm_phased, "compute_emission_probability_nb_betabinom", nowhere
    )

    alpha, tau = hmm_initialize.fit_dispersions_mle(
        np.zeros((4, 2, 1)),
        np.ones((4, 1)),
        np.ones((4, 1)),
        np.zeros((2, 1)),
        np.full((2, 1), 0.5),
        2,
    )

    assert alpha == pytest.approx(0.05, rel=1e-12)
    assert tau == pytest.approx(100.0, rel=1e-12)


@pytest.mark.warning
def test_cnasters_ratios_turn_x_over_zero_into_the_largest_float() -> None:
    """`baf()` and `rdr()` map `0/0` to the fill and `k/0` to `1.8e308`."""
    from cnaster.spatio_genomic_counts import SpatioGenomicCounts

    counts = SpatioGenomicCounts(
        lengths=np.array([3]),
        X=np.array([[[5.0]], [[0.0]], [[2.0]]]).repeat(2, axis=1),
        base_nb_mean=np.array([[0.0], [0.0], [4.0]]),
        total_bb_RD=np.array([[0.0], [0.0], [4.0]]),
    )

    rdr = counts.rdr(fill_value=-1.0)
    baf = counts.baf(fill_value=-1.0)

    largest = np.finfo(np.float64).max
    assert rdr[0, 0] == largest, "count over zero exposure"
    assert rdr[1, 0] == -1.0, "zero over zero is the fill"
    assert baf[0, 0] == largest
    assert baf[1, 0] == -1.0
    assert rdr[2, 0] == 0.5
    assert baf[2, 0] == 0.5


# --- right today -----------------------------------------------------------


@pytest.mark.oracle
@pytest.mark.parametrize(
    "row",
    [[-np.inf, -np.inf, -np.inf], [-np.inf, -1.0, 2.0], [np.inf, 1.0], [-3.0, 4.0]],
)
def test_the_lattices_logsumexp_is_scipys_at_the_edges(row: list[float]) -> None:
    """`numba_logsumexp` matches scipy at the edges."""
    from cnaster.hmm_nophasing import numba_logsumexp

    values = np.asarray(row)
    expected = scipy_logsumexp(values)
    got = numba_logsumexp(values)

    if np.isfinite(expected):
        assert got == pytest.approx(expected, abs=1e-12)
    else:
        assert got == expected


@pytest.mark.oracle
@pytest.mark.parametrize("parameter_terms_only", [True, False])
def test_a_beta_binomial_with_no_trials_contributes_zero(
    parameter_terms_only: bool,
) -> None:
    """`n = 0` is uninformative: `log P(0 | 0) = 0`, against scipy."""
    from cnaster.hmm_nophasing import betabinom_logpmf_numba
    from scipy.stats import betabinom

    got = betabinom_logpmf_numba(0, 0, 3.0, 7.0, parameter_terms_only)

    assert float(betabinom.logpmf(0, 0, 3.0, 7.0)) == 0.0
    assert got == 0.0


@pytest.mark.patch
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
def test_the_rust_lattice_carries_an_all_minus_inf_site_as_cnaster_does(
    which: str,
) -> None:
    """The Rust lattice carries an unexplainable site as `-inf`, bitwise as `cnaster` does."""
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch import lattice

    rng = np.random.default_rng(9)
    n_states, n_obs = 3, 20
    lengths = np.array([n_obs], dtype=np.int64)
    log_transmat = np.log(rng.dirichlet(np.ones(n_states), n_states))
    log_startprob = np.log(rng.dirichlet(np.ones(n_states)))
    log_emission = rng.normal(-5.0, 2.0, (n_states, n_obs, 1))
    log_emission[:, 7, 0] = -np.inf
    log_sitewise = np.log(np.full(n_obs, 0.1))

    arguments = (lengths, log_transmat, log_startprob, log_emission, log_sitewise)
    rust = getattr(lattice, f"{which}_rust")(*arguments)
    cnaster = getattr(hmm_nophasing, which)(*arguments)

    assert not np.isnan(rust).any()
    assert np.isneginf(rust).any()
    np.testing.assert_array_equal(rust, cnaster)
