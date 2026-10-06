"""Every environment variable `port` reads, declared (T- #617, rule 1).

A drop-in's options are bound at install; an environment variable bypasses
that, so each read is a stated departure. The source is read for
`os.environ` and `getenv`, so a new read fails here until it is declared
with its reason. `port.sandbox` is set-aside work and not scanned.
"""

from __future__ import annotations

import re

import pytest

from tests.source_graph import ROOT

READS: dict[str, str] = {
    "python/port/extensions/adjacency.py": (
        "PORT_ADJACENCY and PORT_SQUARE_NEIGHBOURHOOD select the lattice "
        "construction for a subprocess arm (#417)"
    ),
    "python/port/extensions/label_solver.py": (
        "PORT_LABEL_SOLVER overrides the bound solver for a benchmark arm (#246)"
    ),
    "python/port/qa/benchmark.py": (
        "the patched share's child inherits the environment with "
        "NUMBA_DISABLE_JIT set (moved from tests/, T- #673 G4)"
    ),
    "python/port/qa/audit.py": (
        "PORT_SIM_CACHE keeps a cropped or purified sample between runs "
        "(moved from tests/, T- #673 G3)"
    ),
    "python/port/sim/draw.py": "PORT_CACHE locates the simulator's map cache",
    "python/port/studies/population.py": (
        "pins each member's thread pools to one thread (moved from tests/, T- #673 G5)"
    ),
    "python/port/sim/fixtures.py": (
        "PORT_GRCH38 locates CalicoST's GRCh38 resources for the committed "
        "samples (moved from tests/, T- #673 G6)"
    ),
}
"""Each file that reads the environment, and why."""

PATTERN = re.compile(r"\bos\.environ\b|\bgetenv\(")


@pytest.mark.infra
def test_every_environment_read_is_declared() -> None:
    """The source's reads against `READS`, both ways."""
    package = ROOT / "python" / "port"
    found = {
        str(path.relative_to(ROOT))
        for path in package.rglob("*.py")
        if "sandbox" not in path.relative_to(package).parts
        and PATTERN.search(path.read_text())
    }

    assert found == set(READS), (
        f"undeclared {sorted(found - set(READS))}, stale {sorted(set(READS) - found)}"
    )
