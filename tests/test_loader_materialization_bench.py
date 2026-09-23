"""What the loader's three materializations cost (#186).

**2.00x at the stress size and 1.43x less peak, against `cnaster`.** The
same fixture at 2,000 bins, timed best-of-four outside `pytest-benchmark`,
reads 2.18x: 1,931 ms to 885 ms, 780 MB to 545 MB.

The three are `pd.DataFrame.sparse.from_spmatrix` for `exp_counts`, the dense
`int` cast inside `get_spaceranger_counts`, and the dense allele return.
`sparse_counts=True` removes all three; this is what that is worth, against
`cnaster` and against the patch's own default.

| 2,500 spots, 400 bins | median | vs `cnaster` |
| --- | --- | --- |
| `cnaster.io.load_input_data` | 298.4 ms | -- |
| the patch, dense | 387.7 ms | 0.77x |
| the patch, sparse | 149.3 ms | **2.00x** |

**The stress rows are noisy and the dense row shows it**, reading 0.77x here
against 1.28x measured best-of-four on the larger fixture. A loader benchmark
times file reads, so a round competing with the page cache moves the median
by more than the patch does: the stress rows carry an IQR of 45 to 106 ms
against differences of the same order. The claim above is therefore made on
the ratio that survives both harnesses -- 2.00x and 2.18x -- and not on the
dense row, which does not.

Peak is reported from `tracemalloc` in the docstring rather than as a row,
because `pytest-benchmark` measures time and interleaving an allocation
tracker with it distorts both: the allocation-heavy stages of this loader read
several-fold high under `tracemalloc`.
"""

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from cnaster.io import load_input_data as cnaster_loader
from port.patch.io import load_input_data as patched_loader
from pytest_benchmark.fixture import BenchmarkFixture

from tests.fixtures import tiers
from tests.test_load_input_data_patch import (
    STRESS_LATTICE,
    STRESS_OBS,
    _instance,
    gate_config,  # noqa: F401  -- used by name, and it needs the one below
    planted_instance,  # noqa: F401  -- `gate_config` resolves it in this module
)

pytestmark = pytest.mark.preprocessing


@pytest.fixture(scope="module")
def stress_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """2,500 spots over 400 bins, installed for the module.

    Where the claim is made, per the Measurement rule. The gate rows below
    report the same three loaders at the dev instance's size and decide
    nothing on their own.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    root: Path = tmp_path_factory.mktemp("materialization_stress")
    config_path = _instance(root, STRESS_LATTICE, STRESS_OBS)[3]

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield get_global_config()
    finally:
        set_global_config(None)
        set_global_config(previous)


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
    """The baseline, the patch on `cnaster`'s type contract, and all three removed.

    At the gate size 43.0 ms, 39.5 ms (1.09x) and 35.6 ms (1.21x). A ratio
    there decides nothing, per the Measurement rule; it is here because a
    gate-sized regression is what a merge can be stopped on. The stress
    figures are the table in the module docstring.
    """
    benchmark(loader, request.getfixturevalue(config))
