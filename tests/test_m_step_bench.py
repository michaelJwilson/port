"""Timings of the beta-binomial M step, `cnaster` against `snakes_and_ladders` (#24).

Gate 2,400 and stress 19,200 observations, at `cnaster_converged_config` (#30). Not
per-iteration: the solvers take different steps. No ratio is asserted.
"""

from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import (
    cnaster_beta_binomial_m_step,
    upstream_beta_binomial_m_step,
)
from tests.fixtures import (
    BetaBinomialChains,
    beta_binomial_chains,
    planted_posterior,
    tiers,
)

GATE_STATES = 3
GATE_SEQUENCES = 6
GATE_LENGTH = 400
"""2,400 observations. The size the referee tests in `test_m_step` run at."""

STRESS_STATES = 5
STRESS_SEQUENCES = 24
STRESS_LENGTH = 800
"""19,200 observations at five states."""

POSTERIOR_SMOOTHING = 0.1
"""As `test_m_step`, so the benchmark times the problem the tests pinned."""


def _problem(
    n_states: int, n_sequences: int, sequence_length: int
) -> tuple[BetaBinomialChains, np.ndarray]:
    """A fixture and posterior built outside the timer; each adapter's conversion is timed."""
    fixture = beta_binomial_chains(
        n_states=n_states,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        success_probability=np.linspace(0.2, 0.8, n_states),
    )
    return fixture, planted_posterior(fixture, smoothing=POSTERIOR_SMOOTHING)


@pytest.mark.benchmark
@pytest.mark.usefixtures("cnaster_converged_config", "cnaster_perf_sink")
@pytest.mark.parametrize(
    "size",
    tiers(
        (GATE_STATES, GATE_SEQUENCES, GATE_LENGTH),
        (STRESS_STATES, STRESS_SEQUENCES, STRESS_LENGTH),
    ),
)
@pytest.mark.parametrize(
    "arm",
    [cnaster_beta_binomial_m_step, upstream_beta_binomial_m_step],
    ids=["cnaster", "upstream"],
)
def test_m_step(
    benchmark: BenchmarkFixture,
    arm: Callable[[BetaBinomialChains, np.ndarray], object],
    size: tuple[int, int, int],
) -> None:
    """`Weighted_BetaBinom_mix.fit` to its maximum against upstream's
    `BetaBinomialEmission.reestimate`.
    """
    fixture, posterior = _problem(*size)
    benchmark(arm, fixture, posterior)
