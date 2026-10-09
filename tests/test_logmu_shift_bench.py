"""The log-mu shift's cost against the loop it patches (#276); no speedup is claimed.

Also counts that the shift is computed once per call, not once per state.
"""

from __future__ import annotations

import numpy as np
import port.patch.hmm_nophasing.shifted_emission as emission
import pytest
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import compute_logmu_shifts
from port.patch.hmm_nophasing import hmm_nophasing
from port.patch.hmm_nophasing.logmu_shift import shifts
from port.pipeline import with_attributes
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers

GATE = {"n_clones": 3, "per_clone": 1_000}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_clones": 10, "per_clone": 29_000}
"""290,000 segments, `expected_runtime.tex`'s genome."""


def _case(
    n_clones: int, per_clone: int, n_states: int = 7, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    n_segments = n_clones * per_clone

    return (
        generator.normal(size=n_states),
        generator.integers(0, n_states, size=n_segments).astype(np.int64),
        np.log(generator.random(n_segments) / n_segments),
        np.full(n_clones, per_clone, dtype=np.int64),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", ["cnaster", "patch"])
def test_the_shift(benchmark: BenchmarkFixture, arm: str, size: dict[str, int]) -> None:
    """Both arms, warmed, at the gate size and at a genome's."""

    arguments = _case(**size)
    function = compute_logmu_shifts if arm == "cnaster" else shifts

    # NB both are `numba`; compilation is outside the timer (#204).
    function(*arguments)

    benchmark(function, *arguments)


@pytest.mark.patch
def test_the_shift_is_computed_once_per_call(cnaster_config: None) -> None:
    """The shift is computed once per call, not once per state (counted)."""

    n_states, n_clones, per_clone = 7, 3, 8
    n_segments = n_clones * per_clone
    generator = np.random.default_rng(19)

    exposure = generator.integers(20, 60, n_segments).astype(np.float64)
    trials = generator.integers(10, 40, n_segments).astype(np.float64)

    # NB read off the emission module, where the binding being counted lives.
    calls = 0
    original = emission.logmu_shifts  # type: ignore[attr-defined]

    def counted(*arguments: object, **keywords: object) -> np.ndarray:
        nonlocal calls
        calls += 1

        return original(*arguments, **keywords)  # type: ignore[arg-type]

    model = with_attributes(hmm_nophasing, apply_logmu_shift=True)()
    model.state_posteriors = np.eye(n_states)[
        np.repeat(np.arange(n_clones), per_clone)
    ].T

    emission.logmu_shifts = counted  # type: ignore[attr-defined]

    try:
        model.compute_emission_probability_nb_betabinom_coded(
            CountEncoder(
                generator.poisson(exposure).astype(np.float64).reshape(-1, 1),
                exposure.reshape(-1, 1),
            ),
            CountEncoder(
                generator.binomial(trials.astype(int), 0.4)
                .astype(np.float64)
                .reshape(-1, 1),
                trials.reshape(-1, 1),
            ),
            generator.normal(0.0, 0.3, size=(n_states, 1)),
            np.full((n_states, 1), 0.2),
            generator.uniform(0.2, 0.8, size=(n_states, 1)),
            np.full((n_states, 1), 25.0),
            normal_log_lambda=generator.normal(0.0, 0.1, size=n_segments),
            clone_lengths=np.full(n_clones, per_clone, dtype=np.int64),
        )
    finally:
        emission.logmu_shifts = original  # type: ignore[attr-defined]

    assert calls == 1, f"the shift was computed {calls} times, not once"
