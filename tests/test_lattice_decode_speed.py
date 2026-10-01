"""The lattice decode's compiled Viterbi and broadcast emission, against the NumPy they replace (#512).

`copy_likelihood._viterbi` compiles :func:`viterbi_oracle`'s recursion and
`_log_emissions` scores every state in one broadcast call instead of one
call per state. Both claim the previous code's output bitwise, ties
included, so the referee is that code (`oracle`); the per-state emission is
rebuilt here from `_emission` one state at a time, as `_log_emissions` did.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

STAY = 1.0 - 1e-7


def _chain(n_states: int, lengths: list[int]) -> tuple[np.ndarray, ...]:
    off = (1.0 - STAY) / (n_states - 1)
    transmat = np.log(
        np.full((n_states, n_states), off) + np.eye(n_states) * (STAY - off)
    )
    return transmat, np.full(n_states, -np.log(n_states)), np.asarray(lengths, np.int64)


def _bulk(rng: np.random.Generator, n_obs: int, alpha: float, tau: float) -> object:
    from port.extensions.copy_likelihood import Pseudobulk

    trials = rng.poisson(40, n_obs).astype(float)
    return Pseudobulk(
        counts_nb=rng.poisson(200, n_obs).astype(float),
        base_nb_mean=rng.uniform(50.0, 150.0, n_obs),
        counts_bb=rng.binomial(trials.astype(int), 0.4).astype(float),
        total_bb_RD=trials,
        normal_log_lambda=np.zeros(n_obs),
        dispersion=alpha,
        taus=tau,
    )


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize("rounded", [False, True], ids=["continuous", "tied"])
@pytest.mark.parametrize("n_states", [9, 27])
def test_the_compiled_viterbi_is_the_numpy_recursion_bitwise(
    n_states: int, rounded: bool
) -> None:
    """Path and score equal, over two contigs; rounding plants ties, where the first maximum wins."""
    from port.extensions.copy_likelihood import _viterbi, viterbi_oracle

    rng = np.random.default_rng(n_states + rounded)
    emission = rng.normal(0.0, 5.0, (n_states, 301))
    emission[rng.random(emission.shape) < 0.05] = -1e10
    if rounded:
        emission = np.round(emission)
    chain = _chain(n_states, [150, 151])

    compiled = _viterbi(emission, *chain)
    oracle = viterbi_oracle(emission, *chain)

    assert np.array_equal(compiled[0], oracle[0])
    assert compiled[1] == oracle[1]


@pytest.mark.oracle
@pytest.mark.critical
@pytest.mark.parametrize(
    ("alpha", "tau"), [(0.05, 300.0), (0.0, np.inf)], ids=["nb-bb", "poisson-binomial"]
)
def test_the_broadcast_emission_is_each_states_row_bitwise(
    alpha: float, tau: float
) -> None:
    """Every state's row, at a purity below 1 and a shift, equals `_emission` called for that state alone."""
    from port.extensions.copy_likelihood import (
        _emission,
        _log_emissions,
        _parameters,
        _prior,
        candidates,
    )

    bulk = _bulk(np.random.default_rng(7), 400, alpha, tau)
    states = candidates(6)
    log_mu, p = _parameters(states, 0.8)
    bins = np.arange(400)
    rows = np.stack(
        [_emission(log_mu[k] - 0.1, p[k], bulk, bins) for k in range(len(states))]  # type: ignore[arg-type]
    )
    expected = np.where(np.isfinite(rows), rows, -1e10) + _prior(states, 0.5)[:, None]

    assert np.array_equal(_log_emissions(states, 0.1, 0.8, bulk, 0.5), expected)  # type: ignore[arg-type]


GATE = {"n_states": 27, "n_obs": 300}
"""A baseline per pull request; decides nothing."""

STRESS = {"n_states": 27, "n_obs": 3_000}
"""`dev_tree` 60 x 50's bin count at cap 6, where the ratio is read."""


def _viterbi_bench(benchmark: BenchmarkFixture, size: dict[str, int], arm: str) -> None:
    from port.extensions.copy_likelihood import _viterbi, viterbi_oracle

    emission = np.random.default_rng(3).normal(
        0.0, 5.0, (size["n_states"], size["n_obs"])
    )
    chain = _chain(size["n_states"], [size["n_obs"]])
    run = _viterbi if arm == "compiled" else viterbi_oracle
    run(emission, *chain)
    benchmark(run, emission, *chain)


@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["numpy", "compiled"])
def test_the_gate_viterbi(benchmark: BenchmarkFixture, arm: str) -> None:
    """A baseline at a gate size, which argues nothing either way."""
    _viterbi_bench(benchmark, GATE, arm)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["numpy", "compiled"])
def test_the_stress_viterbi(benchmark: BenchmarkFixture, arm: str) -> None:
    """The size the ratio is read at, warm."""
    _viterbi_bench(benchmark, STRESS, arm)
