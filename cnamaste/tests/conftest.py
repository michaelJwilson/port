"""Fixtures shared across `cnamaste`'s tests: planted instances and one whole run."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
from sim.outputs import Run, run_directory
from sim.truth import Truth, planted

DEV: dict[str, Any] = {
    "n_clones": 3,
    "n_states": 4,
    "lattice": (30, 12),
    "n_obs": 60,
    "n_segments": 3,
}
"""The whole-run instance: 360 spots, three clones of 120 -- the configuration's
`min_spots_per_clone` is 100 -- over 60 bins in three segments. 55 s."""

OUTER = 5
"""Outer iterations. At the configuration's 2 the normal clone's path is not yet
neutral (15 of 60 bins off it); at 5 it is, in the same wall time."""


@pytest.fixture(scope="session")
def truth() -> Truth:
    """The whole-run instance, planted."""
    return planted(**DEV)


@pytest.fixture(scope="session")
def run(truth: Truth, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """`run_cnamaste` over `truth`, at `OUTER` outer iterations.

    Seeded, with the global configuration restored after (`isolated_run`).
    """
    import matplotlib as mpl
    from sim.config import isolated_run, write_run_cnamaste_config
    from sim.inputs import write_tmp_inputs
    from sim.unsegment import unsegment

    from cnamaste.scripts.run_cnamaste import run_cnamaste

    mpl.use("Agg")
    root = Path(tmp_path_factory.mktemp("run"))
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config = write_run_cnamaste_config(written, truth, max_iter_outer=OUTER)

    with isolated_run(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cnamaste(str(config))

    return Run(truth=truth, directory=run_directory(root / "output"))
