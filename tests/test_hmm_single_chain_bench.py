"""Single-chain baselines for cnaster and `snakes_and_ladders` on one gate-size fixture
(#9).

No ratio is asserted.
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
    upstream_phased_total_log_likelihood,
    upstream_total_log_likelihood,
)
from tests.fixtures import (
    NegativeBinomialChains,
    PhasedChains,
    negative_binomial_chains,
    phased_chains,
)

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
    """cnaster's per-position phased lattice against upstream's at the assembled
    transition.
    """
    benchmark(arm(phased))
