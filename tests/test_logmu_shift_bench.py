"""What the shift costs, against the loop it patches (#276).

**No speedup is claimed and none was found.** The patch is the shape --
`(n_clones,)` against upstream's `(n_segments,)` -- and `CLAUDE.md` says a
simplification lands on its evidence of equivalence alone, which
`tests/test_logmu_shift.py` carries. These rows exist to catch the case
where the shape change cost something, not to argue it bought anything.

That is worth stating because a vectorized rewrite *was* tried and is
withdrawn: `scipy.special.logsumexp` over per-clone views measured 2.1x
**slower** at the stress size and 3.9x at the gate one. Upstream's loop is
`@njit`, so the compiled two-pass is not the thing worth replacing, and the
patch keeps it. What it removes is the `n_segments` allocation and broadcast
write, which the rows below show is close to free -- so the reason for the
patch is the indexing hazard, not the bytes.

The saving that is real is at the call site rather than here:
`hmm_nophasing.py:275-279` puts the shift **inside** `for i in
range(n_states)`, so folding it in as upstream wrote it would recompute the
whole reduction `n_states` times for a quantity no state enters.
`tests/test_shifted_emission.py` is where that is a claim about output;
`test_the_shift_is_computed_once_per_call` below is where it is one about
count.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

GATE = {"n_clones": 3, "per_clone": 1_000}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_clones": 10, "per_clone": 29_000}
"""290,000 segments, which is what `expected_runtime.tex` derives for a genome."""


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


def _bench(benchmark: BenchmarkFixture, size: dict[str, int], arm: str) -> None:
    from cnaster.hmm_nophasing import compute_logmu_shifts
    from port.patch.hmm_nophasing.logmu_shift import shifts

    arguments = _case(**size)
    function = compute_logmu_shifts if arm == "cnaster" else shifts

    # NB both are `numba`, so the first call is compilation rather than work
    #    and is taken outside the timer (#204).
    function(*arguments)

    benchmark(function, *arguments)


@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["cnaster", "patch"])
def test_the_gate_shift(benchmark: BenchmarkFixture, arm: str) -> None:
    """A baseline at a gate size, which argues nothing either way."""
    _bench(benchmark, GATE, arm)


@pytest.mark.release
@pytest.mark.benchmark
@pytest.mark.parametrize("arm", ["cnaster", "patch"])
def test_the_stress_shift(benchmark: BenchmarkFixture, arm: str) -> None:
    """The size a genome reaches, where the removed write is 2.3 MB per call."""
    _bench(benchmark, STRESS, arm)


@pytest.mark.patch
def test_the_shift_is_computed_once_per_call(cnaster_config: None) -> None:
    """Once, not once per state, which is where upstream's call site puts it.

    Counted rather than timed: a ratio at this size would be noise, and what
    is being asserted is a count that does not depend on the machine. Upstream
    writes the call inside `for i in range(n_states)`, so folding it in as
    written is `n_states` reductions over `n_segments` where one is needed --
    seven, at the state count the benchmarks above use.
    """
    import port.patch.hmm_nophasing.shifted_emission as emission
    from cnaster.count_encoder import CountEncoder
    from port.patch.hmm_nophasing import hmm_nophasing, logmu_shift

    n_states, n_clones, per_clone = 7, 3, 8
    n_segments = n_clones * per_clone
    generator = np.random.default_rng(19)

    exposure = generator.integers(20, 60, n_segments).astype(np.float64)
    trials = generator.integers(10, 40, n_segments).astype(np.float64)

    # NB read off the emission module rather than the package: `__init__`
    #    exports a *function* called `logmu_shift` -- the context manager --
    #    which shadows the module of that name, and the emission module is
    #    where the binding being counted actually lives.
    calls = 0
    original = emission.logmu_shifts  # type: ignore[attr-defined]

    def counted(*arguments: object, **keywords: object) -> np.ndarray:
        nonlocal calls
        calls += 1

        return original(*arguments, **keywords)  # type: ignore[arg-type]

    model = hmm_nophasing()
    model.state_posteriors = np.eye(n_states)[
        np.repeat(np.arange(n_clones), per_clone)
    ].T

    emission.logmu_shifts = counted  # type: ignore[attr-defined]

    try:
        with logmu_shift():
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
