"""Every drop-in in `port.patch` has a correspondence test or a stated reason (#281, #517)."""

from __future__ import annotations

import ast
from importlib import import_module

import pytest
from port.pipeline import (
    COPY_SWAPS,
    FIGURE_SWAPS,
    LOG_SPACE_SWAPS,
    REFINEMENT_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
)

from tests import ROOT, TESTS

PATCH = ROOT / "python" / "port" / "patch"

UNINSTALLED: dict[str, str] = {}
"""Drop-ins deliberately in no swap table, each with what would put it in one."""


def _modules() -> set[str]:
    """Every drop-in module by import path, package roots excluded."""
    found = set()

    for path in PATCH.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        relative = path.relative_to(ROOT / "python").with_suffix("")

        found.add(".".join(relative.parts))

    return found


def _imported_by_marked_tests() -> set[str]:
    """Modules the `patch`- or `cnaster`-marked tests import, read statically."""
    imported: set[str] = set()

    for path in TESTS.glob("test_*.py"):
        source = path.read_text()

        # NB `patch` and `cnaster` together are guard 4's correspondence selection.
        if not any(
            f"pytest.mark.{marker}" in source for marker in ("patch", "cnaster")
        ):
            continue

        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

    return imported | _reexported_by(imported)


def _reexported_by(names: set[str]) -> set[str]:
    """Submodules a package import reaches through its re-exports."""

    reached = set()

    for name in names:
        if not name.startswith("port.patch"):
            continue

        try:
            module = import_module(name)
        except ImportError:  # pragma: no cover - a test importing nothing real
            continue

        for attribute in getattr(module, "__all__", ()):
            origin = getattr(getattr(module, attribute, None), "__module__", None)

            if origin is not None:
                reached.add(origin)

    return reached


def _installed() -> set[str]:
    """Every module a swap installs, traced through re-exports to the definition."""

    reached = set()

    for table in (
        SWAPS,
        FIGURE_SWAPS,
        SHIFT_SWAPS,
        COPY_SWAPS,
        REFINEMENT_SWAPS,
        LOG_SPACE_SWAPS,
    ):
        for swap in table:
            module_name, _, attribute = swap.replacement.partition(":")

            reached.add(module_name)

            defined = getattr(import_module(module_name), attribute, None)
            origin = getattr(defined, "__module__", None)

            if origin is not None:
                reached.add(origin)

    return reached


@pytest.mark.infra
def test_every_dropin_has_a_correspondence_test() -> None:
    """Each drop-in is imported by a `patch`- or `cnaster`-marked test."""
    imported = _imported_by_marked_tests()

    unrefereed = sorted(
        module
        for module in _modules()
        if not any(name.startswith(module) for name in imported)
    )

    assert not unrefereed, (
        f"no `cnaster`-marked test imports {unrefereed}. Either write the "
        f"correspondence test, or the module is not a drop-in and #274's "
        f"four-job rule puts it under `extensions/`."
    )


@pytest.mark.infra
def test_no_declared_drop_in_is_quietly_installed() -> None:
    """No declared drop-in is installed by a swap table (#281)."""
    stale = sorted(_installed() & set(UNINSTALLED))

    assert not stale, (
        f"{stale} are installed and still declared uninstalled; drop the "
        f"declaration so it does not outlive its reason."
    )


@pytest.mark.infra
def test_every_declaration_names_a_module_that_exists() -> None:
    """Every declaration names an existing module."""
    missing = sorted(set(UNINSTALLED) - _modules())

    assert not missing, f"UNINSTALLED names {missing}, which are not drop-ins"
