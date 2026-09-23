"""Baselines for the single-chain path, recorded so a later change has one.

These measure `cnaster` and `snakes_and_ladders` on the same fixture at the
same size. They assert nothing about the ratio: a benchmark that fails on a
machine's speed is a flaky test, and `CLAUDE.md` puts the bar for acting on
a ratio at a stress size, which these are not. What they are is the number a
proposal has to beat, and the number issue #9's crossover measurement
reports against.
"""

from collections.abc import Callable
from functools import partial

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import (
    cnaster_emission,
    cnaster_phased_total_log_likelihood,
    cnaster_total_log_likelihood,
    from_negative_binomial_chains,
    from_phased_chains,
)
from tests.fixtures import (
    NegativeBinomialChains,
    PhasedChains,
    negative_binomial_chains,
    phased_chains,
)
from tests.test_hmm_phased import upstream_phased_total_log_likelihood
from tests.test_hmm_single_chain import upstream_total_log_likelihood

GATE_STATES = 5
GATE_LENGTH = 200
GATE_SEQUENCES = 8


@pytest.fixture(scope="module")
def chains() -> NegativeBinomialChains:
    """The gate-size fixture every negative binomial row scores."""
    return negative_binomial_chains(
        n_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )


@pytest.fixture(scope="module")
def phased() -> PhasedChains:
    """The same size, phased."""
    return phased_chains(
        n_copy_states=GATE_STATES,
        sequence_length=GATE_LENGTH,
        n_sequences=GATE_SEQUENCES,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "arm",
    [
        lambda f: partial(cnaster_emission, from_negative_binomial_chains(f)),
        lambda f: partial(
            f.family.log_density,
            torch.as_tensor(f.dataset.observations, dtype=torch.float64),
        ),
    ],
    ids=["cnaster", "upstream"],
)
def test_emission(
    benchmark: BenchmarkFixture,
    chains: NegativeBinomialChains,
    arm: Callable[[NegativeBinomialChains], Callable[[], object]],
) -> None:
    """The same scores from both, for the ratio between them."""
    benchmark(arm(chains))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "arm",
    [
        lambda f: partial(
            cnaster_total_log_likelihood, from_negative_binomial_chains(f)
        ),
        lambda f: partial(upstream_total_log_likelihood, f),
    ],
    ids=["cnaster", "upstream"],
)
def test_forward(
    benchmark: BenchmarkFixture,
    chains: NegativeBinomialChains,
    arm: Callable[[NegativeBinomialChains], Callable[[], object]],
) -> None:
    """The emission and forward recursion together."""
    benchmark(arm(chains))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "arm",
    [
        lambda f: partial(cnaster_phased_total_log_likelihood, from_phased_chains(f)),
        lambda f: partial(
            upstream_phased_total_log_likelihood, f, from_phased_chains(f)
        ),
    ],
    ids=["cnaster", "upstream"],
)
def test_phased_forward(
    benchmark: BenchmarkFixture,
    phased: PhasedChains,
    arm: Callable[[PhasedChains], Callable[[], object]],
) -> None:
    """`cnaster`'s phased lattice against upstream's at the assembled transition.

    `cnaster` reassembles its transfer matrix per position. The gap between
    the two is the cost of reassembling a `2K x 2K` matrix at every position
    where the kernel is constant and one matrix would do.
    """
    benchmark(arm(phased))
