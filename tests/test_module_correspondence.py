"""Every patch says which `cnaster` module it stands in for, and the swaps agree.

**#250.** `port` exists to replace named pieces of `cnaster`, and a reader
holding a `cnaster` module open should be able to find `port`'s answer to it.
Three of 26 modules under `python/port/patch/` are named for the module they
replace, and 7 of the 14 swap rows install into a module whose name does not
say what it displaces, so the name cannot be the correspondence today.

`MIRRORS` is the correspondence in the meantime: a tuple of `cnaster` module
names per module, `()` where a module replaces nothing. Declared rather than
inferred, because inferring it from imports gets `logger` and `config` wrong
and cannot see a module that replaces something it does not import.

`infra`: these assert `port`'s own layout rule. None of them says anything
about a scientific result, and none can fail because `cnaster` changed.
"""

from __future__ import annotations

import importlib
import pkgutil

import port.patch
import pytest
from port.pipeline import FIGURE_SWAPS, NUMERIC_SWAPS, SWAPS

ALSO = ("port.integer_copy", "port.hmm_init_trials")
"""Modules outside `port.patch` that shadow a `cnaster` module anyway.

Part of what #250 is about: the directory does not mean what it looks like it
means, so a check that only walked `port.patch` would report a clean sweep
over the wrong set.
"""

PRIVATE_SURFACE = frozenset(
    {
        ("cnaster.hmm_nophasing", "_bb_logpmf_1d"),
        ("cnaster.hmm_nophasing", "_nb_logpmf_1d"),
        ("cnaster.hmm_phased", "_switch_betabinom_1d"),
    }
)
"""Every underscore-prefixed `cnaster` name `port` depends on, reviewed.

`cnaster` publishes none of these, so each is a contract it never offered.
Failure is loud rather than silent -- `pipeline.install` does a `getattr` and
an absent name raises -- but nothing listed what `port` would lose if one were
renamed, which is the gap this closes.
"""


def _modules() -> list[str]:
    """Every module this rule covers, `port.patch` plus the two strays."""
    found = [
        f"port.patch.{info.name}"
        for info in pkgutil.iter_modules(port.patch.__path__)
        if not info.name.startswith("_")
    ]
    return sorted(found) + list(ALSO)


@pytest.mark.infra
def test_every_patch_declares_what_it_mirrors() -> None:
    """`MIRRORS` exists, is a tuple, and names `cnaster` modules that import.

    A misspelled target would otherwise sit in the file reading correctly and
    matching nothing, which is the failure mode a declared correspondence is
    supposed to remove rather than relocate.
    """
    for name in _modules():
        module = importlib.import_module(name)

        declared = getattr(module, "MIRRORS", None)

        assert isinstance(declared, tuple), f"{name} declares no MIRRORS tuple (#250)"

        for target in declared:
            assert target.startswith("cnaster."), (
                f"{name} mirrors {target!r}, which is not a cnaster module"
            )
            importlib.import_module(target)


@pytest.mark.infra
def test_every_swap_lands_in_a_module_that_admits_its_target() -> None:
    """The strong one: a row's replacement must declare the module it displaces.

    This is what makes `MIRRORS` load-bearing rather than a comment. A swap
    moved to a different module, or a module's declaration left behind when
    its function moved, fails here -- and both are how a correspondence table
    rots when nothing reads it.
    """
    for swap in SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS:
        target, _, _ = swap.replacement.partition(":")
        module = importlib.import_module(target)

        assert swap.module in module.MIRRORS, (
            f"{swap.module}.{swap.name} is replaced by {target}, which declares "
            f"{module.MIRRORS} and not {swap.module!r}"
        )


@pytest.mark.infra
@pytest.mark.xfail(
    strict=True,
    reason="#250: 7 of 14 swap rows land in a module not named for their target",
)
def test_a_sole_mirror_names_the_module() -> None:
    """The rule #250 asks for, asserted before it holds.

    **Strict**, so the day the renames land this becomes an unexpected pass
    and CI forces the marker off. A guard that has to be remembered is a guard
    that will not be.

    A module mirroring *several* `cnaster` modules is exempt, and that is a
    finding rather than a loophole: `compute_emission_probability_nb_betabinom`
    and the two lattice recursions are each defined in both `hmm_nophasing`
    and `hmm_phased`, so `port.patch.emission` and `port.patch.lattice` unify
    a duplicate pair and cannot be named for one half of it.
    """
    offenders = []

    for name in _modules():
        module = importlib.import_module(name)

        if len(module.MIRRORS) != 1:
            continue

        expected = module.MIRRORS[0].rpartition(".")[2]

        if name.rpartition(".")[2] != expected:
            offenders.append(f"{name} mirrors {module.MIRRORS[0]}")

    assert not offenders, "\n".join(offenders)


@pytest.mark.infra
def test_the_private_cnaster_surface_is_the_reviewed_one() -> None:
    """Every `_`-prefixed `cnaster` name `port` imports, against the list above.

    A new private dependence is a decision -- it binds `port` to something
    `cnaster` never published -- so it arrives in a diff that edits
    `PRIVATE_SURFACE` and is read, rather than in one that edits an import
    line and is not.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "python" / "port"
    found = set()

    for path in root.rglob("*.py"):
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
