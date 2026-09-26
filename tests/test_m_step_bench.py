"""Baselines for the emission M step, `cnaster` against `snakes_and_ladders`.

Issue #24 asks for the comparison explicitly -- absolute times on each side
-- and relatively, as the ratio between them. Both are recorded here at a
gate size and a stress size, because `CLAUDE.md` inherits upstream's
Measurement rule: a ratio read at gate size decides nothing, and a speedup
claim is established at stress size or not at all.

Nothing is asserted. A benchmark that fails on a machine's speed is a flaky
test; what these are is the number a proposal to change either M step has to
beat, and the number issue #24's optimization half would be argued against.

**The two sides are not timed at the same iteration count**, and the ratio
has to be read knowing it. `cnaster` runs `L-BFGS-B` over `2K` parameters at
once; upstream runs alternating bisection. Neither takes a fixed number of
steps, so the ratio mixes cost-per-iteration with iterations-taken, and a
change to either moves it. That is not a defect in the measurement -- an M
step's job is to return an estimate, so time-to-estimate is the quantity a
caller pays -- but it does mean the ratio is not a per-operation speed and
must not be quoted as one.

The runs use `cnaster_converged_config`, not `cnaster_config`. At the
shipped `em_ftol` the solve stops early (issue #30), so timing it would
report the cost of a fit that has not reached its answer and would read as
`cnaster` being several times faster than it is. That is the single most
misleading number this module could produce, so it is not produced.
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
"""19,200 observations at five states.

Eight times the gate's rows and nearly seventeen times its
`(observation, state)` pairs, which is where the two implementations'
different shapes start to tell: `cnaster` materialises one row per pair and
upstream keeps the state axis, so the gap between them is a function of `K`
as much as of `n`.
"""

POSTERIOR_SMOOTHING = 0.1
"""As `test_m_step`, so the benchmark times the problem the tests pinned."""


def _problem(
    n_states: int, n_sequences: int, sequence_length: int
) -> tuple[BetaBinomialChains, np.ndarray]:
    """A fixture and the posterior to re-estimate from, built once per benchmark.

    Outside the timed call deliberately. Both adapters convert their inputs
    before solving -- `cnaster` builds the `(observation, state)` design and
    upstream moves to `torch` -- and those conversions are timed, because
    they are work a caller cannot avoid. Drawing the chains is not: it is
    the fixture, not the M step.
    """
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
    """`Weighted_BetaBinom_mix.fit`, run to its maximum, against upstream's.

    Upstream's is `BetaBinomialEmission.reestimate`. One `cnaster` round at
    the stress size runs about fourteen seconds, which is why that size is
    `release`. The configuration is `cnaster`'s and upstream reads none; installing it
    for both keeps the two arms' conditions equal.
    """
    fixture, posterior = _problem(*size)
    benchmark(arm, fixture, posterior)
