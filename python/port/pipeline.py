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
qualify directly.

Four more are not drop-ins, and #206 got all four in anyway by rebinding one
level up. `hmrf_fused_field` replaces a two-call sequence rather than a name,
`hmrf_adjacency` removes a round trip between two call sites,
`hmrf_invariants` hoists out of a loop body, and `icm_interface` is a
narrower signature -- none of which a name can carry. Every one of them is a
call-site edit **inside `pipeline_clone_assignment`**, which is itself a
module-level name, so `port.patch.hmrf.clone_assignment` rebinds it and takes them
with it.

| patch | where it installs |
| --- | --- |
| `hmrf_fused_field` | inside `clone_assignment`, writing into a buffer (#206) |
| `hmrf_adjacency` | inside `clone_assignment`, under `merge` only (#206) |
| `hmrf_invariants` | inside `clone_assignment`, hoisted across calls (#206) |
| `icm_interface` | inside `clone_assignment` (#206) |

None of them lands in `cnaster`, which `CLAUDE.md` makes read only. What
rebinding the caller buys is that they can be **run** in a whole pipeline and
measured there, which is what `tests/test_patched_entry_point.py` does.

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
    "COPY_SWAPS",
    "FIGURE_SWAPS",
    "NUMERIC_SWAPS",
    "PLOT_OFF_SWAPS",
    "REFINEMENT_SWAPS",
    "SHIFT_SWAPS",
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
    Swap("cnaster.io", "load_input_data", "port.patch.io:load_input_data", 186),
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
        "port.patch.omics:summarize_blocks",
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
        "port.patch.spatial:lattice_multislice_adjacency",
        417,
    ),
    Swap(
        "cnaster.spatial",
        "best_equal_partition",
        "port.patch.spatial:best_equal_partition",
        190,
    ),
    Swap(
        "cnaster.spatial",
        "initialize_rectangular_clones",
        "port.patch.spatial:initialize_rectangular_clones",
        304,
    ),
    Swap(
        "cnaster.normal_spot",
        "normal_baf_bin_filter",
        "port.patch.normal_spot:normal_baf_bin_filter",
        174,
    ),
    Swap(
        "cnaster.hmrf",
        "compute_loglike_spot_assignment",
        "port.patch.hmrf:compute_loglike_spot_assignment_strided",
        59,
    ),
    Swap(
        "cnaster.hmrf",
        "pipeline_clone_assignment",
        "port.patch.hmrf:pipeline_clone_assignment",
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


NUMERIC_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.hmm_nophasing",
        "_nb_logpmf_1d",
        "port.patch.hmm_nophasing:nb_logpmf_1d",
        240,
    ),
)
"""The replacements that agree to a **tolerance** rather than bitwise.

A third table for the same reason `FIGURE_SWAPS` is a second one: `SWAPS`
carries a claim -- every row reproduces `cnaster` byte for byte -- and a row
that agrees to 8.6e-13 does not make it. Putting it in `SWAPS` would not have
made the claim false quietly; it would have made
`tests/test_patched_entry_point.py` fail, which is the guard working. This is
the honest place for it.

**Off by default, and the reason is a measurement rather than caution.** On
the kernel it is 1.78x across ten states. On a **whole run** at
4,000 x 1,980 x 5 it recovers **-1.04 s and -0.051 GB** -- nothing, within
noise -- because the live path goes through `CountEncoder` dedup before
reaching the kernel, which is what #240 flagged as the thing that could make
the ratio not survive. It did not survive.

And it is not free. The 8.6e-13 disagreement -- round-off from
`scipy.special.gammaln` against libm's `lgamma` -- propagates through the EM
to a 3.2e-3 change in the fitted parameters and **flips one segment's integer
copy number by 3** (#244). A patch that changes a scientific output for no
measured gain is not a default; `--approx` is how it is turned on to study
that amplification, which is the only thing it is currently good for.

The 3.51x prefix-sum form needs a `k_max` bound and a fallback nothing has
measured, and would have to clear the same whole-run test before it could
default either.
"""


FIGURE_SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.utils", "write_fig", "port.patch.utils:write_fig", 195),
    Swap(
        "cnaster.plot_genomic",
        "plot_clones_genomic",
        "port.patch.plot_genomic:plot_clones_genomic",
        299,
    ),
    Swap(
        "cnaster.plotting",
        "plot_clones_spatial",
        "port.patch.plotting:plot_clones_spatial",
        309,
    ),
    Swap(
        "cnaster.plot_copy_number_profile",
        "plot_copy_number_profile",
        "port.patch.plot_copy_number_profile:plot_copy_number_profile",
        309,
    ),
    Swap(
        "cnaster.plot_copy_number_profile",
        "plot_ascn_legend",
        "port.patch.plot_copy_number_profile:plot_ascn_legend",
        309,
    ),
)
"""The replacements that **change the output**, and the biggest win here.

Three rows. `write_fig` is 47 per cent of a run (#195); `plot_clones_genomic`
draws each clone's RDR line at `mu / Z_c` when the shift is on, where its
points are, rather than at the pinned `mu` (#299); `plot_clones_spatial`
tiles each spot at 0.85 of the lattice pitch rather than a dot 0.53 of it
across (#309). `write_fig` carries two defaults `cnaster` does not:
`dpi=150`, and one rasterizing group per axes rather than the two a
gridline splits `cnaster`'s runs into. Together they
take a run's plotting from 20.34 s to 3.84 s and its renderer buffers from
8,287 MB to 1,036 MB.

Separate from `SWAPS` because `CLAUDE.md` forbids a silent behaviour change
and all three are ones: a coarser raster, gridlines that paint under
the data instead of over it, and a spot's area.

**Separate, but on by default at the entry point.** `run_cnaster_port`
installs this table unless `--no-figure-swaps` is given, because a win that large
sitting behind a flag is a win nobody gets. The table stays its own so the
distinction survives the default: `SWAPS` is still the set that reproduces
`cnaster` bitwise, `install()` still defaults to `SWAPS` alone, and the
tests asserting that property still have something to assert. Merging the
two would have bought the same 47 per cent and cost the claim.

So the decision is still a reader's rather than a default's -- it is just
the other way round, and `--no-figure-swaps` is where it is made.
"""


SHIFT_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.hmm_nophasing",
        "hmm_nophasing",
        "port.patch.hmm_nophasing:hmm_nophasing",
        276,
    ),
    Swap(
        "cnaster.hmrf",
        "run_core_inference",
        "port.patch.hmrf:run_core_inference",
        293,
    ),
)
"""The per-clone `logmu_shift`, folded into the fit and **on by default**.

`cnaster` computes `log Z_c = log sum_g lambda_g mu_{s_c(g)}` and discards
it, so a clone whose events move its library is fitted against a baseline
that does not account for them: on #292's genome the fit returned
`mu / Z_c`, state by state, rather than the planted `mu` (#293). The model
these rows fit is `<u_gn> = lambda_g T_n mu / sum_g lambda_g mu`.

Two rows because the shift has two jobs. The `hmm_nophasing` class applies it
wherever the HMM scores the fit -- the coded M and E steps, its own final
posteriors and `hmm.py:155`'s rescore -- and `pipeline_clone_assignment`
(in `SWAPS`) reads the class's flag to apply it per candidate clone. The
`run_core_inference` row pins the result's scale, which the shifted
likelihood does not set, once after the optimization: the normal clone's
dominant balanced state is `mu = 1` (#299).

Its own table because every fitted rate moves, which `CLAUDE.md` forbids
doing silently; `run_cnaster_port` installs it unless `--no-shift` is given,
and `port.patch.hmm_nophasing.logmu_shift()` is what turns the class's flag
on for the run.
"""


PLOT_OFF_SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.utils", "write_fig", "port.patch.utils:discard_fig", 403),
)
"""`run_cnaster_port --no-plots`: every figure is built and none is written.

Installed after `FIGURE_SWAPS`, so it rebinds port's `write_fig` where that
one is in place. Not a drop-in in the bitwise sense -- no file appears -- and
for that reason a table of its own, chosen by a flag and never by default.
"""

COPY_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.integer_copy",
        "hill_climbing_integer_copynumber_fixdiploid_milp",
        "port.patch.integer_copy:hill_climbing_integer_copynumber_fixdiploid_milp",
        313,
    ),
    Swap(
        "cnaster.integer_copy",
        "hill_climbing_integer_copynumber_oneclone",
        "port.patch.integer_copy:hill_climbing_integer_copynumber_oneclone",
        313,
    ),
)
"""The integer copy decoders, under the caps the configuration states.

`run_cnaster` decodes under `cnaster`'s defaults, `A + B <= 6` and
`A, B <= 5`, and reads no key that would change them, so #313's chr7 --
planted at `2 mu = 10` -- could not be decoded by any configuration. These
read `int_copy_num.max_total_copy` and apply it to the total and to each
allele; `tests/run_config.py` states 12.

**Its own table, and on by default.** Where the configuration states no cap
the decode is `cnaster`'s, bitwise (`tests/test_integer_copy_patch.py`); where
it states one the output changes, which `SWAPS` promises never to do.
`run_cnaster_port` installs it unless `--no-copy-cap` is given, and
`--no-patch` leaves it out with the rest.
"""


REFINEMENT_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.spatial",
        "initialize_rdr_clone_refininement",
        "port.patch.hmrf.refinement:initialize_rdr_clone_refininement",
        348,
    ),
)
"""The read-depth refinement kept inside each BAF clone, as `cnaster` intends.

`cnaster` builds the mask of which sub-clones each spot may take and drops
it before the HMRF (`run_cnaster.py:1105`), so `icm_sweep_deque`'s 200-spot
floor reassigns spots across BAF clones at random: on `calicost_instance` it
merged all 16 sub-clones into one (ARI 0.000). The row keeps the mask;
`port.patch.hmrf.clone_assignment` applies it while the problem is the one it
was built for (`port.patch.hmrf.refinement`).

**Its own table, and on by default**, because the clones change.
`run_cnaster_port` installs it unless `--no-refinement-mask` is given, and
`--no-patch` leaves it out with the rest.

`--floor-merge`, also on by default, is the second half and needs no row:
`port.patch.icm.floor.floor_merge()` makes `pipeline_clone_assignment` (in
`SWAPS`) meet the clone-size floor smallest first, into each spot's best
clone, at `hmrf.min_spots_per_clone`, instead of the sweep's all-at-once
random reassignment at a fixed 200. It holds with or without the mask.
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
            "port.patch.hmrf:compute_loglike_spot_assignment_strided",
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
            "port.patch.hmrf.fused_field:fused_spot_clone_field",
            (
                # the four observation arrays are `(n_obs, n_spots)` ...
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                # ... and the four state parameters are `(n_states,)` (#278)
                _tiny(1),
                _tiny(1),
                _tiny(1),
                _tiny(1),
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
        except Exception as error:  # noqa: BLE001 -- a kernel that fails is reported
            missed.append(f"{target} ({type(error).__name__}: {error})")

    return Warmed(time.perf_counter() - started, tuple(compiled), tuple(missed))
