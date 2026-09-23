"""Every drop-in is compared with what it replaces, or says why not (#281).

**Guard 4 measures how much of the drop-in surface its correspondence tests
reach. This measures whether they exist at all**, which coverage cannot: a
module reached incidentally by an `end2end` run reads as covered, and a
module nobody compared with `cnaster` reads the same way.

Two halves, because a replacement can fail in two directions.

*A replacement nobody compared.* Every module under `python/port/patch`
carries at least one test in guard 4's selection -- `patch or cnaster` --
that names it. Both markers mean the same comparison: `patch` is a port
patch reproducing the call it replaces, and `cnaster` is that comparison
made under another referee, as the loader tests parametrized over both
implementations make it. A module neither imports is one whose agreement
with `cnaster` is an assumption.

*A replacement nobody installed.* Four of them were, when this was written:
`reindex_clones`, `plot_clones_genomic`, `plot_loh_density` and
`compute_emission_probability_nb_betabinom_coded` appear in no swap table.
Written, tested, and reaching no run -- which is indistinguishable from
running, until someone looks. They are declared below with the reason, and
the declaration is what makes the next one a decision rather than a
default.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "python" / "port" / "patch"
TESTS = Path(__file__).resolve().parent

UNINSTALLED = {
    "port.patch.hmrf.reindex": (
        "#278. Holds the one-column contract that makes "
        "`scripts/run_cnaster.py:1366` dead, and installing it changes no "
        "number -- the branch it kills was already unreachable. It goes into "
        "`SWAPS` when a run is measured against it."
    ),
    "port.patch.plotting.genomic": (
        "#280. Draws at a text column rather than at 20 inches, which is a "
        "deliberate change of output, so it cannot join `SWAPS` -- that table "
        "is the set that reproduces `cnaster` bitwise and "
        "`tests/test_patched_entry_point.py` reads it. `FIGURE_SWAPS` is "
        "where it belongs, once #280's remaining two assertions land."
    ),
    "port.patch.plotting.loh_density": (
        "#278, and `FIGURE_SWAPS` for the same reason as its sibling above: "
        "the figure is upstream's, but it installs beside `plot_clones_"
        "genomic` or not at all, because a run drawing one at each size is "
        "worse than a run drawing both at either."
    ),
    "port.patch.hmm_phased.coded_emission": (
        "#269. Fixes an `IndexError` upstream raises on the shape every fit "
        "returns, so installing it changes a crash into a number -- which is "
        "a behaviour change, and `CLAUDE.md` forbids making one silently. It "
        "waits on a run that exercises `hmm_phased`; the shipped script "
        "passes `hmm_nophasing` at all four of its call sites."
    ),
}
"""Drop-ins deliberately in no swap table, and why.

A declaration rather than an exemption: each says what would put it in one.
`PRIVATE_SURFACE` in `tests/test_module_correspondence.py` carries the same
shape, for the same reason -- what is written down can be reviewed, and what
is merely absent cannot.
"""


def _modules() -> set[str]:
    """Every drop-in module, by import path.

    `__init__.py` is excluded: a package root re-exports rather than
    replaces, so it has nothing of its own to compare.
    """
    found = set()

    for path in PATCH.rglob("*.py"):
        if path.name == "__init__.py":
            continue

        relative = path.relative_to(ROOT / "python").with_suffix("")

        found.add(".".join(relative.parts))

    return found


def _imported_by_marked_tests() -> set[str]:
    """What the `cnaster`-marked tests import, read off the source.

    Read statically rather than by running them: a test that reaches a
    module through three layers of pipeline has not *compared* it, and the
    import is the honest signal of what a file is about. Both `import a.b`
    and `from a.b import c` count, and a prefix match is enough -- importing
    one name from a module is comparing that module.
    """
    imported: set[str] = set()

    for path in TESTS.glob("test_*.py"):
        source = path.read_text()

        # NB `patch` counts as well as `cnaster`, and the two together are
        #    guard 4's selection. `patch` already means "a port patch
        #    reproduces the cnaster call it replaces", so it *is* a
        #    correspondence marker; `cnaster` exists for the tests that make
        #    the same comparison under another referee -- the loader tests
        #    parametrized over both implementations are `end2end`, and the
        #    comparison is no less real for it.
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
    """The submodules a package import actually reaches.

    `tests/test_hmm_phased_coded_emission.py` writes `from
    port.patch.hmm_phased import ...`, and the package re-exports from
    `coded_emission.py`; the string alone would report that module as
    unrefereed while its correspondence test compares it line by line. Same
    resolution `_installed` makes, for the same reason `CLAUDE.md` gives:
    trace the live import to the definition.
    """
    from importlib import import_module

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
    """Every module a swap actually installs, traced to the definition.

    **Not the module the table names.** A swap reads
    `port.patch.hmrf:pipeline_clone_assignment`, and `port/patch/hmrf/
    __init__.py` re-exports that from `clone_assignment.py`, so matching the
    string would report the submodule as uninstalled while it runs on every
    call. `CLAUDE.md` states the rule this follows: scope is decided by
    tracing the live import to the definition, not by matching the
    identifier.
    """
    from importlib import import_module

    from port.pipeline import (
        COPY_SWAPS,
        FIGURE_SWAPS,
        NUMERIC_SWAPS,
        REFINEMENT_SWAPS,
        SHIFT_SWAPS,
        SWAPS,
    )

    reached = set()

    for table in (
        SWAPS,
        NUMERIC_SWAPS,
        FIGURE_SWAPS,
        SHIFT_SWAPS,
        COPY_SWAPS,
        REFINEMENT_SWAPS,
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
    """A replacement with no `cnaster`-marked test is one nobody compared.

    **This is the half coverage cannot make.** Guard 4 would read a module
    as covered when an `end2end` run happened to execute it, which says the
    pipeline ran and not that the replacement agrees with what it replaced.
    """
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
    """A declaration must not outlive the reason for it.

    The four below reach no run, and each says what would put it in a table.
    The day one is installed the declaration becomes a false statement about
    a module that now runs, which is worse than no declaration at all --
    this is what refuses it.

    The converse -- every drop-in either installed or declared -- is **not**
    asserted here, and deliberately. `python/port/patch` holds the pieces
    the replacements are built from as well as the replacements themselves
    (`hmrf_utils`, `hmrf/invariants`, `icm/interface`, `plotting/clone_
    paths`), and those are reached through an installed swap rather than
    being one. Separating the two needs a `cnaster` counterpart per module,
    which is #281's remaining half.
    """
    stale = sorted(_installed() & set(UNINSTALLED))

    assert not stale, (
        f"{stale} are installed and still declared uninstalled; drop the "
        f"declaration so it does not outlive its reason."
    )


@pytest.mark.infra
def test_every_declaration_names_a_module_that_exists() -> None:
    """A declaration for a module that moved is a reason nobody can check.

    `integer_copy.py` is why this is here: it left `patch/` on #281 because
    it replaces nothing, and a stale entry for it would have read as a
    reviewed decision about a drop-in that no longer existed.
    """
    missing = sorted(set(UNINSTALLED) - _modules())

    assert not missing, f"UNINSTALLED names {missing}, which are not drop-ins"
