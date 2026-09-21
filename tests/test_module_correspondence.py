"""`python/port/patch/` is named for `cnaster`, so the replacement is obvious.

**#250.** A reader holding `cnaster/hmrf.py` open should find `port`'s answer
to it at `port/patch/hmrf`, without grepping a swap table or a declaration.
Before this, 17 of the 21 modules that stood in for exactly one `cnaster`
module were named for the ticket that produced them instead.

The rule: **a module under `port.patch` is named for the `cnaster` module it
replaces.** Where several patches address one `cnaster` module they are a
package under that name, and its `__init__` re-exports them so a swap row
names the package. Where a patch replaces nothing, it does not live under
`patch/` at all.

Two modules are exceptions and declare `MIRRORS` because a name cannot carry
what they do: `cnaster` defines
`compute_emission_probability_nb_betabinom` in **both** `hmm_nophasing` and
`hmm_phased`, and `forward_lattice` and `backward_lattice` in both, so
`emission` and `lattice` each unify a duplicate pair and cannot be named for
one half of it.

`infra`: these assert `port`'s own layout. None says anything about a
scientific result, and none can fail because `cnaster` changed.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import port.patch
import pytest
from port.pipeline import FIGURE_SWAPS, NUMERIC_SWAPS, SWAPS

ROOT = Path(__file__).resolve().parents[1]

UNIFIERS = ("emission", "lattice")
"""The two patches that replace a pair of `cnaster` modules rather than one.

Named here rather than inferred so that adding a third is a decision someone
makes in a diff, not a name that quietly stops meaning anything.
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
an absent name raises -- but nothing listed what `port` would lose if one
were renamed, which is the gap this closes.
"""


def _top_level() -> list[str]:
    """Every name directly under `port.patch`, module or package."""
    return sorted(
        info.name
        for info in pkgutil.iter_modules(port.patch.__path__)
        if not info.name.startswith("_")
    )


@pytest.mark.infra
def test_every_patch_is_named_for_the_cnaster_module_it_replaces() -> None:
    """The rule itself: every name under `patch/` is a `cnaster` module name.

    Importing the target rather than matching a string is what makes this
    bite: a plausible-looking name that `cnaster` does not carry --
    `hmrf_field`, `normal_baf`, `input_data` were all of them -- fails here.
    """
    for name in _top_level():
        if name in UNIFIERS:
            continue

        importlib.import_module(f"cnaster.{name}")


@pytest.mark.infra
def test_a_unifier_declares_the_pair_it_replaces() -> None:
    """The exception, held to the reason it exists.

    A module exempt from the naming rule must say what it stands in for and
    it must be **more than one** -- otherwise it is not a unifier, it is a
    module that should have been renamed.
    """
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
    """All 14 rows: `cnaster.X` is replaced from `port.patch.X`.

    The load-bearing one. A swap moved to a different module, or a module
    renamed without its rows, fails here -- and both are how a layout stops
    meaning what it claims once nothing reads it.
    """
    for swap in SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS:
        target, _, _ = swap.replacement.partition(":")
        expected = f"port.patch.{swap.module.rpartition('.')[2]}"

        assert target == expected, (
            f"{swap.module}.{swap.name} is replaced from {target}, not {expected}"
        )

        importlib.import_module(target)


@pytest.mark.infra
def test_what_replaces_nothing_does_not_live_under_patch() -> None:
    """`patch/` means "replaces `cnaster`", so a module that does not is elsewhere.

    `emission_family` wraps `snakes_and_ladders`, `run_sim_gen` is proposed
    for `cnaster` and written here (#116), and `simulation_manifest` and
    `hmm_init_trials` are `port`'s own. All four sit at `port.` top level, and
    none may appear in a swap row.
    """
    installed = {
        swap.replacement.partition(":")[0]
        for swap in SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS
    }

    for name in (
        "port.emission_family",
        "port.run_sim_gen",
        "port.simulation_manifest",
        "port.hmm_init_trials",
    ):
        importlib.import_module(name)

        assert name not in installed, f"{name} is installed but lives outside patch/"


@pytest.mark.infra
def test_the_private_cnaster_surface_is_the_reviewed_one() -> None:
    """Every `_`-prefixed `cnaster` name `port` imports, against the list above.

    A new private dependence binds `port` to something `cnaster` never
    published, so it arrives in a diff that edits `PRIVATE_SURFACE` and is
    read, rather than in one that edits an import line and is not.
    """
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
