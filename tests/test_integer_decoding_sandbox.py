"""The schemes `port.sandbox.integer_decoding.schemes` keeps, as they were pinned (#362).

Moved with the code when `port.extensions.copy_likelihood` kept only the
default: the tempered E-step's zero-temperature limit, the Poisson/binomial
dispersion limit, the EM over the continuous fit's states, and the
Poisson-start dispersion modes. The fixtures are `tests.test_copy_likelihood`'s
and `tests.test_integer_em`'s.
"""

from __future__ import annotations

import numpy as np
import pytest
from port.sandbox.integer_decoding.schemes import VITERBI, Scheme, fit_copies

from tests.test_copy_likelihood import PLANTED, _draw, _offset
from tests.test_integer_em import (
    ALPHA,
    LOG_TRANSMAT,
    LOH_PAIRS,
    N_OBS,
    PAIRS,
    PURITY,
    SHIFT,
    _planted,
    _scrambled,
)


@pytest.mark.analytic
def test_tempering_to_a_low_temperature_is_viterbi() -> None:
    """At `T -> 0` each bin's responsibility is one-hot on Viterbi's state."""
    from port.sandbox.integer_decoding.schemes import (
        _forward_backward,
        _log_emissions,
        _viterbi,
        candidates,
    )

    path, bulk = _draw()
    lattice = candidates(6)
    n = len(lattice)
    transmat = np.log(np.full((n, n), 1e-4 / (n - 1)) + np.eye(n) * (1 - 1e-4))
    start = np.full(n, -np.log(n))
    emission = _log_emissions(lattice, _offset(path, bulk), 1.0, bulk)
    lengths = np.array([path.size])

    best, _ = _viterbi(emission, transmat, start, lengths)
    cold = _forward_backward(emission, transmat, start, lengths, 1e-3)

    np.testing.assert_array_equal(np.argmax(cold, axis=0), best)
    assert cold.max(axis=0).min() > 1.0 - 1e-9


@pytest.mark.analytic
def test_the_poisson_dispersion_is_the_zero_dispersion_limit() -> None:
    """`alpha = 0`, `tau = inf` agree with NB and BB at `alpha = 1e-9`, `tau = 1e9`."""
    from dataclasses import replace

    from port.sandbox.integer_decoding.schemes import _emission, _parameters

    path, bulk = _draw()
    log_mu, p = _parameters(PLANTED)
    bins = np.arange(path.size)
    exact = _emission(log_mu[path], p[path], replace(bulk, alpha=0.0, tau=np.inf), bins)
    near = _emission(log_mu[path], p[path], replace(bulk, alpha=1e-9, tau=1e9), bins)

    np.testing.assert_allclose(exact, near, rtol=1e-5)


@pytest.mark.end2end
@pytest.mark.parametrize("seed", [0, 1])
def test_the_fit_states_em_recovers_pairs_paths_and_the_shift(seed: int) -> None:
    """Every pair, every bin's state, and `SHIFT` to 0.02, from a scrambled start."""
    paths, bulks = _planted(seed)
    fitted = fit_copies(
        [(z, b, 0.0) for z, b in zip(_scrambled(paths, seed), bulks, strict=True)],
        Scheme(states="fit", temperatures=(0.0,) * 10, distinct=True),
        n_states=4,
        normal=0,
        normal_clone=0,
        max_total_copy=6,
        log_transmat=LOG_TRANSMAT,
        log_startprob=np.full(4, -np.log(4)),
        lengths=np.array([N_OBS]),
    )

    np.testing.assert_array_equal(fitted.states, PAIRS)
    for found, planted in zip(fitted.paths, paths, strict=True):
        np.testing.assert_array_equal(found, planted)
    assert fitted.shifts[0] == 0.0
    assert abs(fitted.shifts[1] - SHIFT) < 0.02


@pytest.mark.end2end
@pytest.mark.parametrize("dispersion", ["poisson", "relax"])
def test_the_dispersions_may_start_at_the_poisson_limit(dispersion: str) -> None:
    """Every bin's pair from the Poisson/binomial start; relaxed, `alpha` to 2x."""
    from dataclasses import replace

    paths, bulks = _planted(0, LOH_PAIRS, PURITY)
    fitted = fit_copies(
        [(z, b, 0.0) for z, b in zip(paths, bulks, strict=True)],
        replace(VITERBI, dispersion=dispersion),  # type: ignore[arg-type]
        n_states=4,
        normal=0,
        normal_clone=0,
        max_total_copy=6,
        lengths=np.array([N_OBS]),
    )

    for pairs, planted in zip(fitted.pairs, paths, strict=True):
        np.testing.assert_array_equal(pairs, LOH_PAIRS[planted])

    if dispersion == "poisson":
        assert (fitted.alpha, fitted.tau) == (0.0, np.inf)
    else:
        assert ALPHA / 2.0 < fitted.alpha < ALPHA * 2.0
