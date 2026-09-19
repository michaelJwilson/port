"""Run `cnaster`'s pipeline with `port`'s replacements in place.

`python/port/patch/` holds drop-in replacements for `cnaster` functions and
nothing that installs them: every measurement so far has bound them by hand,
one call at a time, in a script that was thrown away. This is the missing
half -- the table of what `port` replaces, and the two ways to put it in
place -- so that a **whole** `run_cnaster` can be run patched, and so the
question "what does all of this do to a run?" has one answer rather than a
sum of component ratios.

**What it can swap is what is a drop-in.** A replacement installs by
rebinding a name, so it has to accept what the original accepted and return
what the original returned. Eight preprocessing functions and one kernel
qualify. The rest of `port`'s patches do not, and the reason is stated rather
than left to be discovered:

| patch | why it is not here |
| --- | --- |
| `hmrf_fused_field` | replaces a two-call sequence, not a name |
| `hmrf_adjacency` | removes a round trip between two call sites |
| `hmrf_invariants` | hoists out of `cnaster`'s own loop body |
| `icm_interface` | a narrower signature, which is its point |

Each needs a call-site edit inside `cnaster.hmrf`, and `CLAUDE.md` makes that
repository read only. They are reported there and land there or not at all.

The rebinding follows a name wherever it has already been imported, not only
where it is defined: `run_cnaster` does `from cnaster.omics import
assign_initial_blocks`, so the definition site alone would leave the entry
point calling the original.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

__all__ = [
    "FIGURE_SWAPS",
    "SWAPS",
    "Site",
    "Swap",
    "install",
    "instrumented",
    "patched",
    "swap_sites",
]


@dataclass(frozen=True)
class Swap:
    """One `cnaster` name, and what `port` puts in its place."""

    module: str
    """Where `cnaster` defines the name."""

    name: str
    """The name, as `cnaster` binds it."""

    replacement: str
    """`port` module and attribute, as `module:attribute`."""

    ticket: int
    """The issue whose measurement justifies the replacement."""


@dataclass(frozen=True)
class Site:
    """One module whose binding of a name was rebound."""

    module: str
    name: str


SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.io", "load_input_data", "port.patch.input_data:load_input_data", 186),
    Swap(
        "cnaster.reference",
        "get_reference_genes",
        "port.patch.reference:get_reference_genes",
        185,
    ),
    Swap(
        "cnaster.omics",
        "form_gene_snp_table",
        "port.patch.omics:form_gene_snp_table",
        190,
    ),
    Swap(
        "cnaster.omics",
        "assign_initial_blocks",
        "port.patch.omics:assign_initial_blocks",
        190,
    ),
    Swap(
        "cnaster.omics",
        "summarize_blocks",
        "port.patch.summaries:summarize_blocks",
        191,
    ),
    Swap(
        "cnaster.omics",
        "summarize_counts_for_blocks",
        "port.patch.omics:summarize_counts_for_blocks",
        198,
    ),
    Swap(
        "cnaster.omics",
        "summarize_counts_for_bins",
        "port.patch.omics:summarize_counts_for_bins",
        198,
    ),
    Swap(
        "cnaster.spatial",
        "construct_multislice_lattice_adjacency",
        "port.patch.spatial:construct_multislice_lattice_adjacency",
        190,
    ),
    Swap(
        "cnaster.spatial",
        "best_equal_partition",
        "port.patch.spatial:best_equal_partition",
        190,
    ),
    Swap(
        "cnaster.normal_spot",
        "normal_baf_bin_filter",
        "port.patch.normal_baf:normal_baf_bin_filter",
        174,
    ),
    Swap(
        "cnaster.hmrf",
        "compute_loglike_spot_assignment",
        "port.patch.hmrf_field:compute_loglike_spot_assignment_strided",
        59,
    ),
)
"""Every `cnaster` name `port` can replace by rebinding it.

Ordered as a run reaches them. The `ticket` column is what makes each row
answerable: it names the issue carrying the ratio and the referee, so a row
cannot be added here without a measurement behind it.

**Every row here reproduces `cnaster` bitwise.** That is the property the
whole-run test asserts, and it is why `FIGURE_SWAPS` is a separate table
rather than three more rows: a figure written at half the dpi is a different
file by design, and mixing the two would make "the patched run reproduces
the unpatched one" a claim nobody could state.
"""


FIGURE_SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.utils", "write_fig", "port.patch.figures:write_fig", 195),
)
"""The replacements that **change the output**, installed only on request.

One row, and it is 47 per cent of a run (#195): `write_fig`'s `dpi=300`
decides the resolution of every rasterized panel, and mixed-mode PDF
allocates a full-figure `RendererAgg` per rasterizing group at that
resolution.

Separate from `SWAPS` because `CLAUDE.md` forbids a silent behaviour change
and this is one: the figures are the same drawing at a coarser raster, not
the same bytes. `run_cnaster_port --figures` is the opt in, and the flag is
what makes the decision a reader's rather than a default's.
"""


def _resolve(target: str) -> Any:
    """`module:attribute` to the object, importing the module."""
    module_name, _, attribute = target.partition(":")
    __import__(module_name)
    return getattr(sys.modules[module_name], attribute)


def _bound_to(original: Any, name: str) -> list[ModuleType]:
    """Every imported module whose `name` is still bound to `original`.

    A snapshot, because importing inside the loop would mutate `sys.modules`
    while it is walked.
    """
    return [
        module
        for module in list(sys.modules.values())
        if module is not None and getattr(module, name, None) is original
    ]


def swap_sites(swaps: tuple[Swap, ...] = SWAPS) -> tuple[Site, ...]:
    """Where each swap would land, without landing it."""
    sites: list[Site] = []

    for swap in swaps:
        __import__(swap.module)
        original = getattr(sys.modules[swap.module], swap.name)
        sites.extend(
            Site(module.__name__, swap.name)
            for module in _bound_to(original, swap.name)
        )

    return tuple(sites)


def install(swaps: tuple[Swap, ...] = SWAPS) -> tuple[Site, ...]:
    """Rebind every swap, and do not restore.

    For a process whose whole job is the patched pipeline. A test wants
    `patched()` instead: leaving the swaps in place would make every later
    comparison of `cnaster` against `port` compare `port` with itself.
    """
    rebound: list[Site] = []

    for swap in swaps:
        __import__(swap.module)
        original = getattr(sys.modules[swap.module], swap.name)
        replacement = _resolve(swap.replacement)

        for module in _bound_to(original, swap.name):
            setattr(module, swap.name, replacement)
            rebound.append(Site(module.__name__, swap.name))

    return tuple(rebound)


@contextmanager
def patched(swaps: tuple[Swap, ...] = SWAPS) -> Iterator[tuple[Site, ...]]:
    """Rebind every swap for the block, and restore on the way out.

    Restoring matters more here than it usually does: `port`'s own tests put
    a patch to the function it replaces, and a swap left installed would make
    that comparison vacuous.
    """
    undo: list[tuple[ModuleType, str, Any]] = []
    rebound: list[Site] = []

    try:
        for swap in swaps:
            __import__(swap.module)
            original = getattr(sys.modules[swap.module], swap.name)
            replacement = _resolve(swap.replacement)

            for module in _bound_to(original, swap.name):
                undo.append((module, swap.name, original))
                setattr(module, swap.name, replacement)
                rebound.append(Site(module.__name__, swap.name))

        yield tuple(rebound)
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)


@dataclass
class Spent:
    """What one swapped name cost in a run, first call kept apart.

    **A compiled kernel's first call is not its cost, and reporting them
    together is how a ratio becomes a statement about the host** (#204).
    `port`'s field kernel compiles in 2.503 s against 0.031 s warm -- more
    than every preprocessing saving in a run combined -- and `cnaster`'s
    equivalent is served from a cache every earlier test filled. Summed into
    one row, that reads as a kernel eighty times slower than it is.

    `CLAUDE.md`: a measurement carries the conditions that decided it, and
    where a first call *is* the cost it is reported as its own number rather
    than buried inside the stage that paid it.
    """

    calls: int = 0
    seconds: float = 0.0
    first: float = 0.0
    """The first call alone -- compilation, a cold cache, a lazy import."""

    @property
    def warm(self) -> float:
        """Everything after the first call."""
        return self.seconds - self.first

    @property
    def warm_calls(self) -> int:
        return max(self.calls - 1, 0)


@contextmanager
def instrumented(swaps: tuple[Swap, ...] = SWAPS) -> Iterator[dict[str, Spent]]:
    """Time whatever each swapped name is currently bound to.

    Composed with `patched()` rather than built into it, so the same wrapper
    times the replacement and the original: a run's own table is then the
    comparison, and not two runs differenced across whatever else the host
    was doing. That difference is the point -- a whole run is minutes of
    plotting and JIT, and on a shared machine its wall time moves by more
    than these stages cost.
    """
    import time

    spent: dict[str, Spent] = {swap.name: Spent() for swap in swaps}
    undo: list[tuple[ModuleType, str, Any]] = []

    def timing(name: str, wrapped: Any) -> Any:
        def call(*arguments: Any, **keywords: Any) -> Any:
            started = time.perf_counter()
            try:
                return wrapped(*arguments, **keywords)
            finally:
                elapsed = time.perf_counter() - started
                entry = spent[name]

                if entry.calls == 0:
                    entry.first = elapsed

                entry.calls += 1
                entry.seconds += elapsed

        return call

    try:
        for swap in swaps:
            __import__(swap.module)
            current = getattr(sys.modules[swap.module], swap.name)
            wrapper = timing(swap.name, current)

            for module in _bound_to(current, swap.name):
                undo.append((module, swap.name, current))
                setattr(module, swap.name, wrapper)

        yield spent
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)
