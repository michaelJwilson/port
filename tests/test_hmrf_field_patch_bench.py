"""Benchmark: `cnaster`'s spot/clone field against the strided patch (#59 item 1).

Gate per PR; stress (`n_obs` 3,000, 5,000 spots, 7 clones) under `release`.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from port.patch.hmrf.field import compute_loglike_spot_assignment_strided
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import SpotCloneField, cnaster_field_of, spot_clone_field, tiers

GATE = {"n_states": 5, "n_obs": 400, "n_spots": 300, "n_clones": 3}
"""Small enough for the per-PR budget; decides no ratio."""

STRESS = {"n_states": 7, "n_obs": 3000, "n_spots": 5000, "n_clones": 7}
"""1.7 GB across the two channels, where the layout tells."""


_run_cnaster = cnaster_field_of
_run_patch = partial(cnaster_field_of, kernel=compute_loglike_spot_assignment_strided)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_run_cnaster, _run_patch], ids=["cnaster", "patched"])
def test_field(
    benchmark: BenchmarkFixture,
    arm: Callable[[SpotCloneField], np.ndarray],
    size: dict[str, int],
) -> None:
    """Time each arm's field kernel, warmed by one call."""
    fixture = spot_clone_field(**size)
    arm(fixture)
    benchmark(arm, fixture)
