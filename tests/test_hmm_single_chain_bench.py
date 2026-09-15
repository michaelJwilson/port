"""Baselines for the single-chain path, recorded so a later change has one.

These measure `cnaster` and `snakes_and_ladders` on the same fixture at the
same size. They assert nothing about the ratio: a benchmark that fails on a
machine's speed is a flaky test, and `CLAUDE.md` puts the bar for acting on
a ratio at a stress size, which these are not. What they are is the number a
proposal has to beat, and the number issue #9's crossover measurement
reports against.
"""

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import (
    cnaster_emission,
    cnaster_total_log_likelihood,
    from_negative_binomial_chains,
)
from tests.fixtures import negative_binomial_chains
from tests.test_hmm_single_chain import upstream_total_log_likelihood

GATE_STATES = 5
GATE_LENGTH = 200
GATE_SEQUENCES = 8


@pytest.mark.benchmark
def test_cnaster_emission_baseline(benchmark: BenchmarkFixture) -> None:
    """`cnaster`'s emission over the gate-size fixture."""
    fixture = negative_binomial_chains(
        n_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )
    inputs = from_negative_binomial_chains(fixture)
    benchmark(cnaster_emission, inputs)


@pytest.mark.benchmark
def test_upstream_emission_baseline(benchmark: BenchmarkFixture) -> None:
    """The same scores from upstream, for the ratio between them."""
    fixture = negative_binomial_chains(
        n_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )
    observations = torch.as_tensor(fixture.dataset.observations, dtype=torch.float64)
    benchmark(fixture.family.log_density, observations)


@pytest.mark.benchmark
def test_cnaster_forward_baseline(benchmark: BenchmarkFixture) -> None:
    """`cnaster`'s emission and forward recursion together."""
    fixture = negative_binomial_chains(
        n_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )
    inputs = from_negative_binomial_chains(fixture)
    benchmark(cnaster_total_log_likelihood, inputs)


@pytest.mark.benchmark
def test_upstream_forward_baseline(benchmark: BenchmarkFixture) -> None:
    """Upstream's emission and forward recursion together."""
    fixture = negative_binomial_chains(
        n_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )
    benchmark(upstream_total_log_likelihood, fixture)
