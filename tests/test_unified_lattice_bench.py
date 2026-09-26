"""What one recursion costs against `cnaster`'s four (#205).

**The claim is that it costs nothing**, which is what makes the collapse a
simplification rather than a trade. `CLAUDE.md` splits the two: a patch that
makes existing code plainer lands on its evidence of equivalence, and that
evidence is `tests/test_unified_lattice.py`'s four bitwise claims. These rows
exist to catch the case where one implementation for four turned out to cost
something, not to argue for it.

The gate pair runs per pull request and decides nothing. The stress pair
carries `release`: the phased chain at `K = 7` is a 14-state lattice over
3,000 bins, and the recursion is `O(G * S^2)` in the state space.
"""

from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers

Inputs = dict[str, Any]
"""The arrays one arm is handed, named so both arms take the same thing."""

GATE = {"n_states": 3, "n_obs": 300, "n_spots": 4}
"""Small enough for the per-pull-request budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3_000, "n_spots": 50}
"""Where the `S^2` inner loop and the per-site transition tell."""


def _inputs(n_states: int, n_obs: int, n_spots: int, *, phased: bool) -> Inputs:
    generator = np.random.default_rng(31)
    rows = 2 * n_states if phased else n_states

    transition = generator.random((n_states, n_states)) + 0.5
    transition /= transition.sum(axis=1, keepdims=True)

    start = generator.random(n_states) + 0.5
    start /= start.sum()

    return {
        "lengths": np.array([n_obs], dtype=np.int64),
        "log_transmat": np.log(transition),
        "log_startprob": np.log(start),
        "log_emission": generator.normal(-2.0, 1.5, (rows, n_obs, n_spots)),
        "log_sitewise_transmat": np.log(generator.uniform(1e-4, 0.4, n_obs)),
    }


def _cnaster(which: str, phased: bool) -> Any:
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased

    return getattr(hmm_phased if phased else hmm_nophasing, which)


def _run_cnaster(which: str, inputs: Inputs, *, phased: bool) -> np.ndarray:
    recursion = _cnaster(which, phased)
    result: np.ndarray = recursion(
        inputs["lengths"],
        inputs["log_transmat"],
        inputs["log_startprob"],
        inputs["log_emission"],
        inputs["log_sitewise_transmat"],
    )
    return result


def _run_unified(
    which: str, inputs: Inputs, n_states: int, *, phased: bool
) -> np.ndarray:
    from port.patch import lattice

    result: np.ndarray = getattr(lattice, which)(
        inputs["lengths"],
        inputs["log_transmat"],
        inputs["log_startprob"],
        inputs["log_emission"],
        inputs["log_sitewise_transmat"],
        n_states,
        phased,
    )
    return result


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("which", ["forward_lattice", "backward_lattice"])
@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("implementation", ["cnaster", "unified"])
def test_the_recursion(
    benchmark: BenchmarkFixture,
    which: str,
    phased: bool,
    implementation: str,
    size: dict[str, int],
) -> None:
    """Both arms, warm, at the gate size and at the size the ratio is read at.

    The gate baseline argues nothing either way. Both arms are called once
    outside the timer, because both are `numba` kernels with a cold cache on
    a fresh host and a first call is compilation rather than work (#204).
    """
    inputs = _inputs(**size, phased=phased)

    if implementation == "cnaster":
        _run_cnaster(which, inputs, phased=phased)
        benchmark(_run_cnaster, which, inputs, phased=phased)
    else:
        _run_unified(which, inputs, size["n_states"], phased=phased)
        benchmark(_run_unified, which, inputs, size["n_states"], phased=phased)
