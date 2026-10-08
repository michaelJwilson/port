"""One recursion's cost against cnaster's four, at gate and stress (`K = 7` phased)
sizes (#205).

No speedup is claimed; equivalence is `test_unified_lattice.py`'s.
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
    """Both arms, warm (compilation outside the timer, #204), at gate and stress sizes."""
    inputs = _inputs(**size, phased=phased)

    if implementation == "cnaster":
        _run_cnaster(which, inputs, phased=phased)
        benchmark(_run_cnaster, which, inputs, phased=phased)
    else:
        _run_unified(which, inputs, size["n_states"], phased=phased)
        benchmark(_run_unified, which, inputs, size["n_states"], phased=phased)
