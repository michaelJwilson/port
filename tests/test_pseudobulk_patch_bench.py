"""The blocked pseudobulk merge against `cnaster`'s (#488).

Bitwise equal (`tests/test_pseudobulk_patch.py`), so the rows are a ratio and
nothing else. The stress size is #487's 60 x 50 `dev_tree` sample: 6,000
spots, four clones.
"""

from collections.abc import Callable
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers
from tests.test_pseudobulk_patch import _inputs

GATE = {"n_obs": 300, "n_spots": 400, "n_clones": 3}
STRESS = {"n_obs": 4000, "n_spots": 6000, "n_clones": 4}


def _upstream() -> Callable[..., Any]:
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    return merge_pseudobulk_by_index_mix  # type: ignore[no-any-return]


def _blocked() -> Callable[..., Any]:
    from port.patch.pseudobulk import merge_pseudobulk_by_index_mix

    return merge_pseudobulk_by_index_mix


@pytest.mark.benchmark
@pytest.mark.parametrize("size", tiers(GATE, STRESS))
@pytest.mark.parametrize("arm", [_upstream, _blocked], ids=["cnaster", "blocked"])
def test_merge(
    benchmark: BenchmarkFixture,
    arm: Callable[[], Callable[..., Any]],
    size: dict[str, int],
) -> None:
    """One merge, warm."""
    inputs = _inputs(size["n_obs"], size["n_spots"], size["n_clones"], seed=0)
    function = arm()
    function(**inputs)
    benchmark(function, **inputs)
