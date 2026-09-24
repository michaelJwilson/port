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
from port.pipeline import (
    COPY_SWAPS,
    FIGURE_SWAPS,
    NUMERIC_SWAPS,
    SHIFT_SWAPS,
    SWAPS,
)

ROOT = Path(__file__).resolve().parents[1]

UNIFIERS = ("emission", "lattice", "plotting")
"""The two patches that replace a pair of `cnaster` modules rather than one.

Named here rather than inferred so that adding a third is a decision someone
makes in a diff, not a name that quietly stops meaning anything.
"""

PRIVATE_SURFACE = frozenset(
    {
        ("cnaster.hmm_nophasing", "_bb_logpmf_1d"),
        ("cnaster.hmm_nophasing", "_nb_logpmf_1d"),
        ("cnaster.hmm_phased", "_switch_betabinom_1d"),
        # The four layout helpers `plot_clones_genomic` is built from (#278).
        # `port.patch.plotting.genomic` replaces that function and imports
        # these rather than copying them: they draw the gridspec, the axis
        # furniture and the chromosome boundaries, and a copy would be 130
        # lines whose only job is to stay identical. Importing them is what
        # keeps the replacement's figure upstream's figure.
        ("cnaster.plot_genomic", "_annotate_clone_stats"),
        ("cnaster.plot_genomic", "_create_clone_gridspec"),
        ("cnaster.plot_genomic", "_draw_chromosome_boundaries"),
        ("cnaster.plot_genomic", "_format_track_axis"),
    }
)
"""Every underscore-prefixed `cnaster` name `port` depends on, reviewed.

`cnaster` publishes none of these, so each is a contract it never offered.
Failure is loud rather than silent -- `pipeline.install` does a `getattr` and
an absent name raises -- but nothing listed what `port` would lose if one
were renamed, which is the gap this closes.

The four `plot_genomic` entries are the largest single addition and the one
worth arguing with: a drop-in replacement that imports four private helpers
is four more names that can be renamed underneath it. The alternative is
copying them, which trades that risk for a divergence nobody would notice
until a figure changed. Reviewed, and chosen.
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
    """All 16 rows: `cnaster.X` is replaced from `port.patch.X`.

    The load-bearing one. A swap moved to a different module, or a module
    renamed without its rows, fails here -- and both are how a layout stops
    meaning what it claims once nothing reads it.
    """
    for swap in SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS + SHIFT_SWAPS + COPY_SWAPS:
        target, _, _ = swap.replacement.partition(":")
        expected = f"port.patch.{swap.module.rpartition('.')[2]}"

        assert target == expected, (
            f"{swap.module}.{swap.name} is replaced from {target}, not {expected}"
        )

        importlib.import_module(target)


@pytest.mark.infra
def test_what_replaces_nothing_does_not_live_under_patch() -> None:
    """`patch/` means "replaces `cnaster`", so a module that does not is elsewhere.

    `emission_family` wraps `snakes_and_ladders` and `hmm_init_trials` is
    `port`'s own, so both are `extensions/`; `run_sim_gen` is proposed for
    `cnaster` and written here (#116) and `manifest` plants truth, so both
    are `sim/`. None may appear in a swap row: an extension installed over a
    `cnaster` name is a patch that has not admitted to being one.
    """
    installed = {
        swap.replacement.partition(":")[0]
        for swap in SWAPS + NUMERIC_SWAPS + FIGURE_SWAPS + SHIFT_SWAPS + COPY_SWAPS
    }

    for name in (
        "port.extensions.emission_family",
        "port.sim.run_sim_gen",
        "port.sim.manifest",
        "port.extensions.hmm_init_trials",
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


@pytest.mark.infra
def test_every_module_has_one_of_the_four_jobs() -> None:
    """`CLAUDE.md`'s shape rule, asserted rather than left to review.

    Four jobs: `patch/` replaces a named `cnaster` function or class,
    `extensions/` adds what has no counterpart, `sim/` simulates, and
    everything else realizes `run_cnaster_port` with those in place. A module
    with none of them goes to `sandbox/`.

    **This is the test the refactor exists for.** Moving four modules is an
    afternoon; keeping them where they belong is what needed a referee, and
    without one the next module lands at `port.` top level because that is
    where the last one was.

    `sandbox/` is deliberately not exempted from anything else -- it is
    outside the coverage denominator and outside the claims, which is the
    whole of what it means -- so it simply does not appear here.
    """
    allowed = {"patch", "extensions", "sim", "scripts", "sandbox"}

    package = ROOT / "python" / "port"
    stray = []

    for path in sorted(package.glob("*.py")):
        if path.name in {"__init__.py", "pipeline.py"}:
            # `pipeline.py` *is* realizing `run_cnaster_port`: it is the swap
            # table and the context manager that installs it.
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
