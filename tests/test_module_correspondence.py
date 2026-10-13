"""`port.patch` modules are named for the `cnaster` module they replace (#250).

Unifiers declare `MIRRORS` (T- #776, #517 step 8). `infra`: port's own layout.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil

import port.patch
import pytest
from port.pipeline import (
    COPY_SWAPS,
    FIGURE_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
)

from tests import ROOT

UNIFIERS = ("emission", "lattice", "plotting")
"""Patches replacing several `cnaster` modules, named so adding one is explicit."""

PRIVATE_SURFACE = frozenset(
    {
        ("cnaster.hmm_nophasing", "_bb_logpmf_1d"),
        ("cnaster.hmm_nophasing", "_nb_logpmf_1d"),
        ("cnaster.hmm_phased", "_switch_betabinom_1d"),
        # The four `plot_genomic` layout helpers the sandbox replacement imports (#278)
        ("cnaster.plot_genomic", "_annotate_clone_stats"),
        ("cnaster.plot_genomic", "_create_clone_gridspec"),
        ("cnaster.plot_genomic", "_draw_chromosome_boundaries"),
        ("cnaster.plot_genomic", "_format_track_axis"),
    }
)
"""Every underscore-prefixed `cnaster` name `port` imports, reviewed."""


def _top_level() -> list[str]:
    """Return every public name directly under `port.patch`."""
    return sorted(
        info.name
        for info in pkgutil.iter_modules(port.patch.__path__)
        if not info.name.startswith("_")
    )


@pytest.mark.infra
def test_every_patch_is_named_for_the_cnaster_module_it_replaces() -> None:
    """Each non-unifier patch name imports as `cnaster.<name>`."""
    for name in _top_level():
        if name in UNIFIERS:
            continue

        importlib.import_module(f"cnaster.{name}")


@pytest.mark.infra
def test_a_unifier_declares_the_pair_it_replaces() -> None:
    """Each unifier's `MIRRORS` names more than one importable target."""
    for name in UNIFIERS:
        module = importlib.import_module(f"port.patch.{name}")

        assert isinstance(module.MIRRORS, tuple), f"{name} declares no MIRRORS"
        assert len(module.MIRRORS) > 1, (
            f"{name} mirrors {module.MIRRORS}: one target is a rename, not an exception"
        )

        for target in module.MIRRORS:
            importlib.import_module(target)


@pytest.mark.infra
def test_every_swap_lands_in_the_module_named_for_its_target() -> None:
    """Every swap row replaces `cnaster.X` from `port.patch.X`."""
    for swap in SWAPS + FIGURE_SWAPS + SHIFT_SWAPS + COPY_SWAPS:
        target, _, _ = swap.replacement.partition(":")
        expected = f"port.patch.{swap.module.rpartition('.')[2]}"

        assert target == expected, (
            f"{swap.module}.{swap.name} is replaced from {target}, not {expected}"
        )

        importlib.import_module(target)


@pytest.mark.infra
def test_the_private_cnaster_surface_is_the_reviewed_one() -> None:
    """The `_`-prefixed `cnaster` names imported equal `PRIVATE_SURFACE`."""
    found = set()

    for path in (ROOT / "python" / "port").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "cnaster"
            ):
                found |= {
                    (node.module, alias.name)
                    for alias in node.names
                    if alias.name.startswith("_")
                }

    assert found == set(PRIVATE_SURFACE), (
        f"added {sorted(found - PRIVATE_SURFACE)}, "
        f"dropped {sorted(PRIVATE_SURFACE - found)}"
    )


@pytest.mark.infra
def test_every_module_has_one_of_the_four_jobs() -> None:
    """Only allowed job directories under `port/`, and no stray top-level modules (T- #673)."""
    allowed = {"patch", "extensions", "sim", "scripts", "sandbox", "qa", "studies"}

    package = ROOT / "python" / "port"
    stray = []

    for path in sorted(package.glob("*.py")):
        if path.name in {"__init__.py", "pipeline.py"}:
            # `pipeline.py` is the swap table and its installer
            continue

        stray.append(path.name)

    assert not stray, (
        f"{stray} sit at `port.` top level and claim none of the four jobs. "
        f"A patch goes under `patch/`, new functionality under "
        f"`extensions/`, planted truth under `sim/`, and anything else "
        f"under `sandbox/`."
    )

    for child in sorted(p.name for p in package.iterdir() if p.is_dir()):
        if child.startswith("__"):
            continue

        assert child in allowed, (
            f"`port/{child}/` is not one of the four jobs {sorted(allowed)}"
        )
