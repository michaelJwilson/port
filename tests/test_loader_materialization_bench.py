"""The loader's three materializations' cost against `cnaster.io.load_input_data`, at
2,500 spots and 400 bins (#186).

Peak memory is measured with `tracemalloc` separately; results are in #186.
"""

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from cnaster.io import load_input_data as cnaster_loader
from port.patch.io import load_input_data as patched_loader
from port.sim.inputs import written_config
from port.sim.run_config import planted_and_written
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers

pytestmark = pytest.mark.preprocessing

STRESS_LATTICE = (50, 50)
STRESS_OBS = 400
"""2,500 spots over 400 bins: the largest instance whose fixture builds in under four seconds."""


@pytest.fixture(scope="module")
def stress_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """The stress instance, installed for the module."""
    root: Path = tmp_path_factory.mktemp("materialization_stress")
    config_path = planted_and_written(root, STRESS_LATTICE, STRESS_OBS)[3]

    with written_config(config_path) as config:
        yield config


@pytest.mark.benchmark
@pytest.mark.parametrize("config", tiers("gate_config", "stress_config"))
@pytest.mark.parametrize(
    "loader",
    [
        cnaster_loader,
        patched_loader,
        partial(patched_loader, sparse_counts=True),
    ],
    ids=["cnaster", "dense-patch", "sparse-patch"],
)
def test_the_loader(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    loader: Callable[[Any], Any],
    config: str,
) -> None:
    """cnaster, the patch dense, and the patch sparse, at the gate size; decides nothing."""
    benchmark(loader, request.getfixturevalue(config))
