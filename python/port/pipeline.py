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

import functools
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple, TypeVar

from port.extensions.genomic_axis import Ticks

_T = TypeVar("_T")

__all__ = [
    "COPY_SWAPS",
    "DEFAULTS",
    "FIGURE_DPI",
    "FIGURE_SWAPS",
    "LOG_SPACE_SWAPS",
    "OUTSIDE_TABLES",
    "PLOT_OFF_SWAPS",
    "REFINEMENT_SWAPS",
    "RUN_STATE",
    "SHIFT_SWAPS",
    "SWAPS",
    "Default",
    "Site",
    "Swap",
    "Warmed",
    "install",
    "instrumented",
    "patched",
    "release",
    "swap_sites",
    "warm",
    "with_attributes",
    "with_options",
]


class Swap(NamedTuple):
    """One `cnaster` name, and what `port` puts in its place."""

    module: str
    """Where `cnaster` defines the name."""

    name: str
    """The name, as `cnaster` binds it."""

    replacement: str
    """`port` module and attribute, as `module:attribute`."""

    ticket: int
    """The issue whose measurement justifies the replacement."""

    options: tuple[tuple[str, Any], ...] = ()
    """Keywords bound into the replacement at install (#517).

    A replacement takes `cnaster`'s signature and defaults; what `port`
    changes is a keyword bound here, so the table rather than a module
    global says what an installed row does.
    """


class Site(NamedTuple):
    """One module whose binding of a name was rebound."""

    module: str
    name: str


SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.io", "load_input_data", "port.patch.io:load_input_data", 186),
    Swap(
        "cnaster.io",
        "get_aggregated_barcodes",
        "port.patch.io:get_aggregated_barcodes",
        446,
    ),
    Swap("cnaster.io", "get_sample_list", "port.patch.io:get_sample_list", 418),
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
        "cnaster.omics",
        "create_bin_ranges",
        "port.patch.omics:create_bin_ranges",
        438,
    ),
    Swap(
        "cnaster.recomb",
        "get_sitewise_transmat",
        "port.patch.recomb:get_sitewise_transmat",
        438,
    ),
    Swap(
        "cnaster.normal_spot",
        "filter_normal_diffexp",
        "port.patch.normal_spot:filter_normal_diffexp",
        440,
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
        "cnaster.normal_spot",
        "determine_normal_candidates",
        "port.patch.normal_spot:determine_normal_candidates",
        479,
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
    Swap(
        "cnaster.pseudobulk",
        "merge_pseudobulk_by_index_mix",
        "port.patch.pseudobulk:merge_pseudobulk_by_index_mix",
        488,
    ),
    Swap("cnaster.hmm_phased", "hmm_phased", "port.patch.hmm_phased:hmm_phased", 269),
)
"""Every `cnaster` name `port` can replace by rebinding it.

Ordered as a run reaches them. The `ticket` column is what makes each row
answerable: it names the issue carrying the ratio and the referee, so a row
cannot be added here without a measurement behind it.

**Each row reproduces `cnaster` bitwise wherever `cnaster` is correct**, and
the whole-run test asserts it on its fixture. Where a row departs, the
departure is a stated fix, in its own docstring, and changes the result only
in the regime named there (#466 lists them):

- `get_aggregated_barcodes`: a slice id read from the barcode suffix (#446);
- `get_sample_list`: slices keyed by name in first-seen order, where
  `cnaster` keys them by runs of adjacent rows and drops a slice on
  interleaved rows (#418);
- `assign_initial_blocks`: no block across two chromosomes;
- `summarize_counts_for_bins`: the normal-spot filter's flagged genes left
  out of every bin (#177), and a chromosome with no bins left out of
  `lengths`;
- `create_bin_ranges`: bins removed upstream dropped from the table (#105);
- `get_sitewise_transmat`: cM per chromosome, `log 0.5` at each end, and a
  segment's end at its last gene (#438);
- `filter_normal_diffexp`: genes split on `,` (#165), and flagged genes
  recorded for the bins (#177);
- `construct_multislice_lattice_adjacency`: the lattice's own neighbours, 6
  on a Visium hex grid, where `cnaster` takes 8 on scaled coordinates (#417);
- `initialize_rectangular_clones`: new boundaries where no assignment of
  the drawn blocks can pass, where `cnaster` loops (#304, T- #692);
- `normal_baf_bin_filter`: a removed bin's genes marked `is_interval =
  False` (#105), and a chromosome with no bins left out of `lengths`.
- `hmm_phased`: the coded emission reads the fitted parameter by state, so
  a call scoring more than one spot returns a number where `cnaster` raises
  `IndexError`; every call `run_cnaster` makes scores one spot (#269, #517).

`FIGURE_SWAPS` is a separate table rather than more rows for a different reason:
a figure written at half the dpi is a different file by design, not a fix.
"""


FIGURE_DPI = 150
"""What `FIGURE_SWAPS` binds `write_fig`'s `dpi` to, against `cnaster`'s 300 (#195).

Halving it quarters the raster: a 20x10 inch panel goes 6,000 x 3,000 pixels
to 3,000 x 1,500, 72 MB of RGBA to 18 MB. Measured: `docs/measurements.md`,
`port.pipeline.FIGURE_DPI`.

150 rather than lower because it is the floor at which a 20-inch panel still
carries 3,000 pixels across, which is more than any screen shows it at and
more than a page prints it at. Lower is available and is a judgement about
the figures rather than about the arithmetic, so it is left to whoever is
reading them.
"""

FIGURE_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.utils",
        "write_fig",
        "port.patch.utils:write_fig",
        195,
        (("dpi", FIGURE_DPI), ("group_rasters", True)),
    ),
    Swap(
        "cnaster.plot_genomic",
        "plot_clones_genomic",
        "port.patch.plot_genomic:plot_clones_genomic",
        299,
        (("axis", Ticks()),),
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
        (("axis", Ticks()),),
    ),
)
"""The replacements that **change the output**, and the biggest win here.

Four rows. `write_fig` writes the figures (#195); `plot_clones_genomic`
draws each clone's RDR line at `mu / Z_c` when the shift is on, where its
points are, rather than at the pinned `mu` (#299); `plot_clones_spatial`
tiles each spot at 0.85 of the lattice pitch rather than a dot 0.53 of it
across (#309); `plot_copy_number_profile` draws one row per clone (#309).
Its legend, `port.patch.plot_copy_number_profile.plot_ascn_legend`, is
called by that replacement directly and has no row: `cnaster`'s only caller
of its own legend is the function the row replaces, so a legend row was
reached by no live call (#466, removed by T- #617). `write_fig` is installed with two options bound:
`dpi=150`, and one rasterizing group per axes rather than the two a
gridline splits `cnaster`'s runs into. Measured: `docs/measurements.md`,
`port.pipeline.FIGURE_SWAPS`. The two genomic rows bind `axis=Ticks()`: a
tick every 10 Mb on the bins `df_cnv` places, on `cnaster`'s linear axis
(T- #683).

Separate from `SWAPS` because `CLAUDE.md` forbids a silent behaviour change
and each row makes one: a coarser raster, gridlines that paint under the data
instead of over it, the RDR line's level, a spot's area, and the profile's
layout.

**Separate, but on by default at the entry point.** `run_cnaster_port`
installs this table unless `--no-figure-swaps` or `--no-patch` is given, because a win that large
sitting behind a flag is a win nobody gets. The table stays its own so the
distinction survives the default: `SWAPS` is still the set that reproduces
`cnaster` bitwise, `install()` still defaults to `SWAPS` alone, and the
tests asserting that property still have something to assert.

So the decision is still a reader's rather than a default's -- it is just
the other way round, and `--no-figure-swaps` is where it is made.
"""


SHIFT_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.hmm_nophasing",
        "hmm_nophasing",
        "port.patch.hmm_nophasing:hmm_nophasing",
        276,
        (("apply_logmu_shift", True),),
    ),
    Swap(
        "cnaster.hmrf",
        "run_core_inference",
        "port.patch.hmrf:run_core_inference",
        293,
    ),
    Swap(
        "cnaster.hmrf",
        "reindex_clones",
        "port.patch.hmrf:reindex_clones",
        362,
    ),
)
"""The per-clone `logmu_shift`, folded into the fit and **on by default**.

`cnaster` computes `log Z_c = log sum_g lambda_g mu_{s_c(g)}` and discards
it, so a clone whose events move its library is fitted against a baseline
that does not account for them (#292, #293; `docs/measurements.md`,
`port.pipeline.SHIFT_SWAPS`). The model
these rows fit is `<u_gn> = lambda_g T_n mu / sum_g lambda_g mu`.

Two rows because the shift has two jobs. The `hmm_nophasing` class applies it
wherever the HMM scores the fit -- the coded M and E steps, its own final
posteriors and `hmm.py:155`'s rescore -- and `pipeline_clone_assignment`
(in `SWAPS`) reads the class's flag to apply it per candidate clone. The
`run_core_inference` row pins the result's scale, which the shifted
likelihood does not set, once after the optimization: the normal clone's
dominant balanced state is `mu = 1` (#299), and then gives each clone its
own column, less its `log Z_c`, the normal clone's at zero (#362). The
`reindex_clones` row carries `p`, `alpha` and `tau` to one column per clone
so that integer copy reads every clone's own rates.

The two rows also carry three things the entry point turns on with the
shift and off without it: the sal emission (#425) and the analytic M-step
gradient (#433), options of port's `hmm_nophasing` row, and the
distinct initializer (#348), which port's `run_core_inference` passes. Where
a tumour proportion hands clone assignment to `cnaster` (#135), the per-clone
shift is not applied there and the run says so (#466).

Its own table because every fitted rate moves, which `CLAUDE.md` forbids
doing silently; `run_cnaster_port` installs it unless `--no-shift` is given,
and the row binds `apply_logmu_shift=True` into the class it installs (#517).
"""


LOG_SPACE_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.hmm_nophasing",
        "_nb_logpmf_1d",
        "port.patch.hmm_nophasing.nb_logpmf:_nb_logpmf_1d",
        560,
    ),
    Swap(
        "cnaster.hmm_nophasing",
        "_dense_nb_logpmf",
        "port.patch.hmm_nophasing.nb_logpmf:_dense_nb_logpmf",
        560,
    ),
    Swap(
        "cnaster.hmm_nophasing",
        "_bb_logpmf_1d",
        "port.patch.hmm_nophasing.bb_logpmf:_bb_logpmf_1d",
        561,
    ),
    Swap(
        "cnaster.hmm_nophasing",
        "_dense_bb_logpmf",
        "port.patch.hmm_nophasing.bb_logpmf:_dense_bb_logpmf",
        561,
    ),
)
"""`cnaster`'s emission kernels where they are wrong, installed with the shift.

`_nb_logpmf_1d` scores any count at probability 1 once `p = 1 / (1 + alpha
lambda)` rounds to 1 (#560), and `_bb_logpmf_1d` cancels `lgamma` values
near `tau log tau` (#561). Each row reaches every module binding the name:
`cnaster.hmm_nophasing`, `cnaster.hmm_phased` and port's `shifted_emission`.
A compiled kernel that calls one by name cannot be reached that way, so
port's field takes `log_space=True` as an option of
`pipeline_clone_assignment` instead.

**Its own table, on where the shift is.** The kernels agree with `cnaster`'s
to 1e-9 where `cnaster`'s are right and not bitwise, so not `SWAPS`; the
shift's arm already departs from `cnaster`'s fit, and a `--no-shift` arm
keeps `cnaster`'s kernels.
"""

PLOT_OFF_SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.utils", "write_fig", "port.patch.utils:discard_fig", 403),
)
"""`run_cnaster_port --no-plots`: every figure is built and none is written.

Installed after `FIGURE_SWAPS`, so it rebinds port's `write_fig` where that
one is in place -- in `port.patch.utils` too, for the run, since `patched()`
rebinds every module holding the original; nothing in `port` calls it there. Not a drop-in in the bitwise sense -- no file appears -- and
for that reason a table of its own, chosen by a flag and never by default.
"""

OUTSIDE_TABLES: tuple[tuple[str, str, str], ...] = (
    ("cnaster.hmm_nophasing.hmm_nophasing.forward_lattice, backward_lattice",
     "port.patch.lattice:rust_lattices", "rebound for the run by `run_cnaster_port` (#312)"),
    ("cnaster.hmm_phased.hmm_phased.forward_lattice, backward_lattice",
     "port.patch.lattice:rust_lattices", "rebound for the run by `run_cnaster_port` (#312)"),
    ("cnaster.hmm_initialize.gmm_init",
     "port.patch.hmm_initialize.distinct:gmm_init, port.patch.hmm_initialize.sal_mixture:gmm_init",
     "passed as `hmm_initializer` by `port.patch.hmrf.core_inference` (#348, #489)"),
    ("cnaster.he.get_he_image",
     "port.patch.io:he_image", "called in its place by `port.patch.io`'s `load_input_data` row (#311)"),
)  # fmt: skip
"""`cnaster` functions replaced other than by a swap row: `(upstream, replacement, how)`.

A table row rebinds a module attribute; these are class methods rebound for a
block, an argument `cnaster` binds as a default, and a call inside a row.
`docs/port-forward.md` lists them under the rows, so every replacement is
found in one place (#749 WP8)."""


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
allele; `python/port/sim/run_config.py` states 12.

**Its own table, and on by default.** Both rows decode by the HMM's
likelihood only, with `mu`, each clone's shift, its path and the dispersions
held (#362), so the output is not `cnaster`'s, which `SWAPS` promises never
to change; `run_cnaster_port` installs `copy_likelihood.capture` with them.
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
floor reassigns spots across BAF clones at random (`docs/measurements.md`,
`port.pipeline.REFINEMENT_SWAPS`). The row keeps the mask;
`port.patch.hmrf.clone_assignment` applies it while the problem is the one it
was built for (`port.patch.hmrf.refinement`).

**Its own table**, because the clones change; `run_cnaster_port` installs it
by default (`DEFAULTS`). Its only reader is port's `pipeline_clone_assignment`, so
`--no-patch` without `--sal` refuses it, and where a tumour proportion hands
the call to `cnaster` (#135) the mask is not applied and the run says so.

`--floor-merge`, also on by default, is the second half and needs no row:
it binds `floor_merge=True` into `pipeline_clone_assignment` (in `SWAPS`),
which then meets the clone-size floor smallest first, into each spot's best
clone, at `hmrf.min_spots_per_clone`, instead of the sweep's all-at-once
random reassignment at a fixed 200. It holds with or without the mask, and
is refused and dropped exactly where the mask is.
"""


class Default(NamedTuple):
    """A behaviour of `port`'s own that `run_cnaster_port` turns on by default."""

    setting: str
    """The `run_cnaster_port` setting it is (`port.scripts.run_cnaster.Settings`)."""

    flag: str
    """The flag that asks for it; `--no-` and the rest turns it off."""

    ticket: int
    """The issue whose measurement justifies it."""


DEFAULTS: tuple[Default, ...] = (
    Default("refinement_mask", "--refinement-mask", 467),
    Default("floor", "--floor-merge", 348),
)
"""`port`'s own behaviours on by default, outside `SWAPS` (T- #617, rule 8).

Neither comes from `snakes_and_ladders`, so `--sal` no longer selects them: on
in every patched arm, off with their `--no-` flag, and off with `--no-patch`
unless `--sal` (which installs `pipeline_clone_assignment` alone) asks. The
mask installs `REFINEMENT_SWAPS`; the floor merge binds `floor_merge=True`
into `pipeline_clone_assignment`.

**The 300-normal-UMI segment floor is not here; it stays `--sal`'s.** On the
default arm it made `tests/test_copy_likelihood.py`'s critical copy instance
decode the planted `(1, 2)` as `(1, 3)`: 1000 bins merge into 888, and the
lattice decode's EM then alternates between tumour fraction 1 with `(1, 2)`
and 0.5 with `(1, 3)` (identical depth and allele share), stopping on 0.5
at log-likelihood -19,942.74 where fraction 1 reaches -18,966.85 (PR- #645). The `--sal` arm is
unchanged. Alone, the mask or the floor merge over-split #338's three-sample
instance (6 fitted clones against 2 planted); together with `--sal` they
recover CalicoST hard at clone ARI 0.982 against 0.303 (#467).
"""


def _resolve(target: str) -> Any:
    """`module:attribute` to the object, importing the module."""
    module_name, _, attribute = target.partition(":")
    __import__(module_name)
    return getattr(sys.modules[module_name], attribute)


def with_options(
    swaps: tuple[Swap, ...], replacement: str, **options: Any
) -> tuple[Swap, ...]:
    """`swaps`, with `options` bound into the rows that install `replacement`.

    Keyed by the replacement rather than the `cnaster` name: two rows replace
    `write_fig`, and an option belongs to one of them.

    Raises
    ------
    ValueError
        If no row of `swaps` installs `replacement`: an option bound to a
        row that is not installed is never read, and was silently dropped
        (T- #617).
    """
    if not any(swap.replacement == replacement for swap in swaps):
        msg = f"no selected row installs {replacement}; {sorted(options)} would be dropped"
        raise ValueError(msg)

    return tuple(
        swap._replace(options=(*swap.options, *options.items()))
        if swap.replacement == replacement
        else swap
        for swap in swaps
    )


def with_attributes(cls: type[_T], **attributes: Any) -> type[_T]:
    """A subclass of `cls` under `cls`'s name, with `attributes` set on it (#517).

    How a row that replaces a class binds its options: `cnaster` reads the
    class's attributes where no keyword reaches (`optimize_params`, the
    static emission `hmm.py:155` calls), so an option is a class attribute,
    set on a subclass the row installs rather than on the class every caller
    shares. Each must already be an attribute of `cls`, or it is refused.
    """
    unknown = [name for name in attributes if not hasattr(cls, name)]

    if unknown:
        msg = f"{cls.__qualname__} has no option {unknown}"
        raise TypeError(msg)

    return type(
        cls.__name__,
        (cls,),
        {
            **attributes,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
        },
    )


def _replacement(swap: Swap) -> Any:
    """What a row installs: its replacement, with its options bound.

    An option the replacement does not take is refused here, at install,
    rather than at the row's first call, hours into a run. A class's options
    are attributes of a subclass (:func:`with_attributes`).
    """
    import inspect

    replacement = _resolve(swap.replacement)

    if not swap.options:
        return replacement

    if isinstance(replacement, type):
        return with_attributes(replacement, **dict(swap.options))

    inspect.signature(replacement).bind_partial(**dict(swap.options))
    bound = functools.partial(replacement, **dict(swap.options))
    functools.update_wrapper(bound, replacement)
    return bound


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


def _rebind(
    swaps: tuple[Swap, ...],
    replace: Callable[[Swap, Any], Any] | None,
    undo: list[tuple[ModuleType, str, Any]] | None = None,
) -> tuple[Site, ...]:
    """Every site each swap reaches, rebound to `replace(swap, current)`.

    The one walk behind `swap_sites`, `install`, `patched` and `instrumented`.
    Swap by swap, in order, so a later row finds what an earlier one put in
    place: `PLOT_OFF_SWAPS` rebinds the `write_fig` `FIGURE_SWAPS` installed.
    `replace=None` rebinds nothing; `undo` collects what to put back.
    """
    sites: list[Site] = []

    for swap in swaps:
        __import__(swap.module)
        current = getattr(sys.modules[swap.module], swap.name)
        new = None if replace is None else replace(swap, current)

        for module in _bound_to(current, swap.name):
            if replace is not None:
                if undo is not None:
                    undo.append((module, swap.name, current))

                setattr(module, swap.name, new)

            sites.append(Site(module.__name__, swap.name))

    return tuple(sites)


def swap_sites(swaps: tuple[Swap, ...] = SWAPS) -> tuple[Site, ...]:
    """Where each swap would land, without landing it."""
    return _rebind(swaps, None)


def install(swaps: tuple[Swap, ...] = SWAPS) -> tuple[Site, ...]:
    """Rebind every swap, and do not restore.

    For a process whose whole job is the patched pipeline. A test wants
    `patched()` instead: leaving the swaps in place would make every later
    comparison of `cnaster` against `port` compare `port` with itself.
    """
    return _rebind(swaps, lambda swap, _: _replacement(swap))


RUN_STATE: tuple[str, ...] = (
    "port.patch.hmm_nophasing.shifted_emission:release",
    "port.patch.hmrf.clone_assignment:release",
    "port.patch.hmrf.core_inference:release",
    "port.patch.hmrf.refinement:forget",
    "port.patch.integer_copy:release",
    "port.patch.io:release",
    "port.patch.recomb:release",
)
"""What `patched` calls on exit: each drops what one run's rows held (#517).

A module not yet imported held nothing, so it is not imported to be told so.
"""


def release() -> None:
    """Call every imported `RUN_STATE` release."""
    for target in RUN_STATE:
        module_name, _, attribute = target.partition(":")
        module = sys.modules.get(module_name)

        if module is not None:
            getattr(module, attribute)()


@contextmanager
def patched(swaps: tuple[Swap, ...] = SWAPS) -> Iterator[tuple[Site, ...]]:
    """Rebind every swap for the block, and restore on the way out.

    Restoring matters more here than it usually does: `port`'s own tests put
    a patch to the function it replaces, and a swap left installed would make
    that comparison vacuous.
    """
    undo: list[tuple[ModuleType, str, Any]] = []

    try:
        yield _rebind(swaps, lambda swap, _: _replacement(swap), undo)
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)

        release()


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
        _rebind(swaps, lambda swap, current: timing(swap.name, current), undo)
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
