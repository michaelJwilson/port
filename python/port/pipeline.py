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
    "Warmed",
    "install",
    "instrumented",
    "patched",
    "swap_sites",
    "warm",
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
    Swap(
        "cnaster.hmrf",
        "pipeline_clone_assignment",
        "port.patch.clone_assignment:pipeline_clone_assignment",
        206,
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

One row, and it is 47 per cent of a run (#195). `write_fig` carries two
defaults `cnaster` does not: `dpi=150`, and one rasterizing group per axes
rather than the two a gridline splits `cnaster`'s runs into. Together they
take a run's plotting from 20.34 s to 3.84 s and its renderer buffers from
8,287 MB to 1,036 MB.

Separate from `SWAPS` because `CLAUDE.md` forbids a silent behaviour change
and both of these are ones: a coarser raster, and gridlines that paint under
the data instead of over it. `run_cnaster_port --figures` is the opt in, and
the flag is what makes the decision a reader's rather than a default's.
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


def _tiny(*shape: int) -> Any:
    """A float64 array of the given shape, filled with ones."""
    import numpy as np

    return np.ones(shape, dtype=np.float64)


def _kernels() -> tuple[tuple[str, Any], ...]:
    """Every compiled kernel a run reaches, with arguments that compile it.

    **`numba` specializes on types, not on sizes**, so a one-element array of
    the right dtype and dimensionality compiles the specialization a whole
    instance reuses. That is why this needs no fixture, no config and no
    pipeline -- and why it does not warm by running a small instance, which
    would compile whichever branches that instance took and hide the rest.

    **Named rather than discovered.** Importing everything and warming
    whatever carries a dispatcher compiles kernels no run reaches and still
    misses the ones reached through a branch. A list is reviewable; a sweep
    is not, and a sweep that silently warms nothing is worse than no warm-up
    at all -- which is what an earlier draft of this did, because
    `nopython_signatures` is empty until something has been compiled.

    The arguments here are the live path's types. Where one is wrong the
    warm-up compiles a specialization the run does not reuse, and the run's
    own `first` column is what says so: after a warm-up those read near zero,
    and a row that does not is a finding about this table.
    """
    import numpy as np

    field_arguments = (
        1,
        _tiny(1),
        _tiny(1),
        _tiny(1),
        False,
        _tiny(1, 1, 1),
        _tiny(1, 1, 1),
        np.zeros(1, dtype=np.int64),
        1,
        1,
        np.zeros(1, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        True,
    )

    return (
        ("cnaster.hmrf:compute_loglike_spot_assignment", field_arguments),
        (
            "port.patch.hmrf_field:compute_loglike_spot_assignment_strided",
            field_arguments,
        ),
        (
            "cnaster.hmrf:pool_spatio_genomic_counts",
            (
                _tiny(1, 2, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                np.zeros(1, dtype=np.int64),
                np.zeros(2, dtype=np.int64),
                _tiny(1),
                False,
            ),
        ),
        (
            "cnaster.hmm_nophasing:_dense_nb_logpmf",
            (_tiny(1, 1), _tiny(1, 1), _tiny(1, 1), _tiny(1, 1)),
        ),
        (
            "cnaster.hmm_nophasing:_dense_bb_logpmf",
            (_tiny(1, 1), _tiny(1, 1), _tiny(1, 1), _tiny(1, 1)),
        ),
        (
            "cnaster.hmm_nophasing:_nb_logpmf_1d",
            (_tiny(1), _tiny(1), 1.0, 1.0, _tiny(1)),
        ),
        (
            "cnaster.hmm_nophasing:_bb_logpmf_1d",
            (_tiny(1), _tiny(1), 0.5, 1.0, _tiny(1)),
        ),
        (
            "port.patch.hmrf_fused_field:fused_spot_clone_field",
            (
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                np.zeros((1, 1), dtype=np.int64),
                _tiny(1),
            ),
        ),
    )


@dataclass(frozen=True)
class Warmed:
    """What a warm-up compiled, and what it did not."""

    seconds: float
    compiled: tuple[str, ...]
    missed: tuple[str, ...]

    def report(self) -> str:
        lines = [
            f"warm-up: {len(self.compiled)} kernels in {self.seconds:.2f}s",
            *(f"  missed {entry}" for entry in self.missed),
        ]
        return "\n".join(lines)


def warm() -> Warmed:
    """Compile every named kernel, and report what it cost and what it missed.

    A miss is reported rather than raised: a kernel whose signature moved
    costs a first call, not a run, and the list being wrong is a finding
    about the list.
    """
    import time

    started = time.perf_counter()
    compiled: list[str] = []
    missed: list[str] = []

    for target, arguments in _kernels():
        module_name, _, attribute = target.partition(":")

        try:
            __import__(module_name)
            kernel = getattr(sys.modules[module_name], attribute)

            if not hasattr(kernel, "nopython_signatures"):
                missed.append(f"{target} (not a compiled kernel)")
                continue

            kernel(*arguments)
            compiled.append(target)
        except Exception as error:
            missed.append(f"{target} ({type(error).__name__}: {error})")

    return Warmed(time.perf_counter() - started, tuple(compiled), tuple(missed))
