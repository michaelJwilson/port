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
    "python/port/sim/draw.py": "PORT_CACHE locates the simulator's map cache",
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
