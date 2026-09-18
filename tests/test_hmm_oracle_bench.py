"""What the two recursions cost, `cnaster` against upstream (#140).

The forward pass, the backward pass and the posterior, timed on the same
chains both implementations score in `tests/test_hmm_oracle.py`. Those
establish they agree; these say what the agreement costs.

`cnaster`'s lattices are `njit` and take the concatenated batch in one call;
upstream's `forward_backward` is a `numpy` recursion over one chain, so the
batch is a Python loop over chains here. That is the comparison as it exists
rather than a like-for-like kernel benchmark, and the docstrings say which is
which so a reader does not take the ratio for a kernel ratio.

No ratio is asserted. These are the baselines the fit-level rung (#77, #97)
will report against when it lands.
"""

from typing import Any

import numpy as np
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from tests.adapters import CnasterChainInputs, from_negative_binomial_chains
from tests.fixtures import NegativeBinomialChains, negative_binomial_chains

GATE = {"n_states": 5, "sequence_length": 200, "n_sequences": 8}
"""1,600 positions over eight chains: the per-pull-request size."""

STRESS = {"n_states": 7, "sequence_length": 3_000, "n_sequences": 8}
"""24,000 positions, the order `run_cnaster` reaches on the dev instance."""


def _emission(inputs: CnasterChainInputs) -> np.ndarray:
    from cnaster.hmm_nophasing import hmm_nophasing

    rdr, baf = hmm_nophasing.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        inputs.log_mu,
        inputs.alphas,
        inputs.total_bb_RD,
        inputs.p_binom,
        inputs.taus,
    )
    scored: np.ndarray = rdr + baf
    return scored


def _cnaster_forward(inputs: CnasterChainInputs, emission: np.ndarray) -> np.ndarray:
    from cnaster.hmm_nophasing import hmm_nophasing

    forward: np.ndarray = hmm_nophasing.forward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        emission,
        inputs.log_sitewise_transmat,
    )
    return forward


def _cnaster_both(inputs: CnasterChainInputs, emission: np.ndarray) -> np.ndarray:
    from cnaster.hmm import compute_copy_state_posterior
    from cnaster.hmm_nophasing import hmm_nophasing

    alpha = _cnaster_forward(inputs, emission)
    beta = hmm_nophasing.backward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        emission,
        inputs.log_sitewise_transmat,
    )
    posterior: np.ndarray = compute_copy_state_posterior(alpha, beta)
    return posterior


def _upstream_both(
    densities: list[np.ndarray], initial: np.ndarray, transition: np.ndarray
) -> float:
    from snakes_and_ladders.likelihood.forward_backward import forward_backward

    return sum(
        float(forward_backward(density, initial, transition).log_evidence)
        for density in densities
    )


def _densities(fixture: NegativeBinomialChains) -> list[np.ndarray]:
    observations = np.asarray(fixture.dataset.observations)
    return [
        np.asarray(
            fixture.family.log_density(
                torch.as_tensor(observations[chain], dtype=torch.float64)
            ),
            dtype=float,
        )
        for chain in range(observations.shape[0])
    ]


def _instance(settings: dict[str, int]) -> Any:
    fixture = negative_binomial_chains(**settings)
    inputs = from_negative_binomial_chains(fixture)
    return (
        fixture,
        inputs,
        _emission(inputs),
        _densities(fixture),
        np.log(np.asarray(fixture.dataset.initial, dtype=float)),
        np.log(np.asarray(fixture.dataset.transition, dtype=float)),
    )


@pytest.fixture(scope="module")
def gate() -> Any:
    return _instance(GATE)


@pytest.fixture(scope="module")
def stress() -> Any:
    return _instance(STRESS)


@pytest.mark.cnaster
@pytest.mark.benchmark
@pytest.mark.subject
def test_cnaster_forward_only_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """`forward_lattice` over the concatenated batch, one call."""
    _, inputs, emission, _, _, _ = gate
    benchmark(_cnaster_forward, inputs, emission)


@pytest.mark.cnaster
@pytest.mark.benchmark
@pytest.mark.subject
def test_cnaster_forward_backward_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """Both passes and the posterior, which is what the driver runs per iteration."""
    _, inputs, emission, _, _, _ = gate
    benchmark(_cnaster_both, inputs, emission)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.upstream
def test_upstream_forward_backward_gate(benchmark: BenchmarkFixture, gate: Any) -> None:
    """Upstream's, looped over chains.

    `upstream` rather than `upstream_oracle`: this times the referee, it does
    not consult it. The agreement claims are in `test_hmm_oracle.py`.
    """
    _, _, _, densities, initial, transition = gate
    benchmark(_upstream_both, densities, initial, transition)


@pytest.mark.cnaster
@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.subject
def test_cnaster_forward_backward_stress(
    benchmark: BenchmarkFixture, stress: Any
) -> None:
    _, inputs, emission, _, _, _ = stress
    benchmark(_cnaster_both, inputs, emission)


@pytest.mark.infra
@pytest.mark.benchmark
@pytest.mark.release
@pytest.mark.upstream
def test_upstream_forward_backward_stress(
    benchmark: BenchmarkFixture, stress: Any
) -> None:
    _, _, _, densities, initial, transition = stress
    benchmark(_upstream_both, densities, initial, transition)
