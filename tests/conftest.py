"""Shared fixtures, including `cnaster`'s module-level global config, set and restored."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from tests import TESTS

COMPRESSION_DECIMALS = 6
"""Places `CountEncoder` rounds to before deduplicating; `cnaster`'s default."""


MIN_PHASE_SWITCH_PROB = 1e-10
"""Floor `compute_numbat_phase_switch_prob` clamps to when none is passed."""


BETABINOM_START_PARAMS = "0.1,0.3,0.5,0.7,0.9"
"""Starting `p` per state; at least as many as any fitted state space, away from truth."""

BETABINOM_START_DISPERSION = 10.0
"""Starting `tau`, away from the planted concentration."""


SHIPPED_EM_FTOL = 1e-6
"""`cnaster`'s shipped `em_ftol`, relative to the objective; the default installed."""

CONVERGED_EM_FTOL = 1e-9
"""An `em_ftol` at which the solve reaches its maximum, for implementation comparisons."""


def cnaster_test_config(
    tmp_path: Path, em_ftol: float, em_maxiter: int
) -> dict[str, Any]:
    """Return the global config `cnaster` reads instead of taking arguments."""
    return {
        "phasing": {"min_prob": MIN_PHASE_SWITCH_PROB},
        # NB `hmm_utils` reads the solver name, then `em_`-prefixed options for it
        "hmm": {
            "compression_decimals": COMPRESSION_DECIMALS,
            "solver": "L-BFGS-B",
            "em_maxiter": em_maxiter,
            "em_ftol": em_ftol,
            "em_disp": 0,
            "em_xrtol": 1e-5,
            "em_xtol": 1e-5,
            # NB `gmm_init` clips the allele share to these (`hmm_initialize.py:362`);
            #    wide enough to clip nothing planted
            "gmm_min_binom_prob": 0.01,
            "gmm_max_binom_prob": 0.99,
            "gmm_maxiter": 100,
        },
        # NB `run_core_inference`'s outer-loop settings, at the shipped defaults
        "hmrf": {
            "inertia": False,
            "fixed_assignment": False,
            "ari_tolerance": 0.99,
        },
        "betabinom": {
            "start_params": BETABINOM_START_PARAMS,
            "start_disp": BETABINOM_START_DISPERSION,
        },
        # NB `flush_perf` reads this only to count rows; see `cnaster_perf_sink`
        "paths": {"perf_path": str(tmp_path / "cnaster.perf")},
    }


@pytest.fixture
def cnaster_config(tmp_path: Path) -> Iterator[None]:
    """Install a minimal `cnaster` global config at shipped solver settings, then restore."""
    from port.sim.inputs import written_config

    with written_config(cnaster_test_config(tmp_path, SHIPPED_EM_FTOL, 100)):
        yield


@pytest.fixture
def cnaster_converged_config(tmp_path: Path) -> Iterator[None]:
    """As `cnaster_config`, at `CONVERGED_EM_FTOL`, for comparing two implementations."""
    from port.sim.inputs import written_config

    with written_config(cnaster_test_config(tmp_path, CONVERGED_EM_FTOL, 5_000)):
        yield


@pytest.fixture
def cnaster_perf_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in a scratch directory: `flush_perf` writes the literal relative `cnaster.perf`."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cnaster_config_switch(tmp_path: Path) -> Iterator[Callable[[float, int], None]]:
    """Yield a setter that reinstalls the config mid-test, to compare criteria."""
    from port.sim.inputs import written_config

    with ExitStack() as stack:
        yield lambda em_ftol, em_maxiter: stack.enter_context(
            written_config(cnaster_test_config(tmp_path, em_ftol, em_maxiter))
        )


@pytest.fixture(scope="session")
def planted_instance(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Plant and write the gate instance once per session; tests never rewrite it."""
    from port.sim.run_config import planted_and_written

    return planted_and_written(tmp_path_factory.mktemp("gate"))


@pytest.fixture(scope="module")
def gate_config(planted_instance: Any) -> Iterator[Any]:
    """The gate instance's run configuration, installed for the module."""
    from port.sim.inputs import written_config

    with written_config(planted_instance[3]) as config:
        yield config


_COLLECTED: list[pytest.Item] = []
"""Every test collected, before `-m` deselection narrows `session.items`."""


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Record the collection before mark deselection; `critical` tests first (#403)."""
    items.sort(key=lambda item: item.get_closest_marker("critical") is None)
    _COLLECTED[:] = items


@pytest.fixture
def collected_items() -> list[pytest.Item]:
    """Return the whole suite's items, or skip when the collection is narrowed."""
    items = list(_COLLECTED)
    collected_modules = {item.nodeid.split("::")[0] for item in items}
    on_disk = {f"tests/{path.name}" for path in TESTS.glob("test_*.py")}

    missing = on_disk - collected_modules
    if missing:
        pytest.skip(f"narrowed collection; {len(missing)} test module(s) absent")

    return items


@pytest.fixture(autouse=True, scope="session")
def _keep_the_perf_log_out_of_the_checkout(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Run the suite from a scratch directory, so `cnaster.perf` stays out of the checkout."""
    import os

    previous = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    try:
        yield
    finally:
        os.chdir(previous)


TEST_SEED = 0
"""What every test's random streams start from (#264)."""


@pytest.fixture(autouse=True)
def _seeded() -> None:
    """Seed NumPy's and numba's global generators before every test (#264)."""
    import numpy as np
    from cnaster.scripts.run_cnaster import set_numba_seed

    # NB the legacy global generator is the one `cnaster` draws from.
    np.random.seed(TEST_SEED)  # noqa: NPY002
    set_numba_seed(TEST_SEED)
