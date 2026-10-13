"""Run `cnaster`'s pipeline with `port`'s replacements in place.

Holds the tables of what `port` replaces and installs them by rebinding each
name in every module that imported it. Non-drop-in patches (`hmrf_fused_field`,
`hmrf_adjacency`, `hmrf_invariants`, `icm_interface`) install by rebinding
`pipeline_clone_assignment`, which carries them (#206).
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
    """Keywords bound into the replacement at install (#517)."""


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
    Swap("cnaster.he", "get_he_image", "port.patch.he:he_image", 771),
    Swap(
        "cnaster.count_encoder",
        "CountEncoder",
        "port.patch.count_encoder:CountEncoder",
        799,
    ),
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
"""Every `cnaster` name `port` can replace by rebinding it, in run order.

Each row reproduces `cnaster` bitwise wherever `cnaster` is correct; stated
departures are in each replacement's docstring (#466).
"""


FIGURE_DPI = 150
"""`write_fig`'s `dpi` under `FIGURE_SWAPS`, against `cnaster`'s 300 (#195)."""

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
"""Replacements that change the figures: on by default, off with `--no-figure-swaps`.

`write_fig` at `dpi=150` with grouped rasters (#195); `plot_clones_genomic`
draws the RDR line at `mu / Z_c` (#299); `plot_clones_spatial` tiles spots
(#309); `plot_copy_number_profile` one row per clone (#309). Genomic rows bind
`axis=Ticks()` (T- #683). Kept apart from `SWAPS`, which stays bitwise.
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
"""The per-clone `logmu_shift`, fitted, on by default; off with `--no-shift`.

Model: `<u_gn> = lambda_g T_n mu / sum_g lambda_g mu` (#292, #293).
`hmm_nophasing` applies the shift wherever the HMM scores; `run_core_inference`
pins the scale to the normal clone's `mu = 1` (#299) and gives each clone its
own column (#362); `reindex_clones` carries `p`, `alpha`, `tau` per clone.
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
"""`cnaster`'s emission kernels where they are wrong (#560, #561), installed with the shift.

Agree with `cnaster`'s to 1e-9 where those are right, so not in `SWAPS`.
Compiled callers are reached by `log_space=True` on `pipeline_clone_assignment`.
"""

PLOT_OFF_SWAPS: tuple[Swap, ...] = (
    Swap("cnaster.utils", "write_fig", "port.patch.utils:discard_fig", 403),
)
"""`run_cnaster_port --no-plots`: every figure is built and none is written (#403).

Installed after `FIGURE_SWAPS`; never on by default.
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
"""`cnaster` functions replaced other than by a swap row: `(upstream, replacement, how)` (#749)."""


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
"""Integer copy decoders under the configured `int_copy_num.max_total_copy` (#313).

Applied to the total and each allele; decode by HMM likelihood with `mu`,
shifts, path and dispersions held (#362). On by default; off with `--no-copy-cap`.
"""


REFINEMENT_SWAPS: tuple[Swap, ...] = (
    Swap(
        "cnaster.spatial",
        "initialize_rdr_clone_refininement",
        "port.patch.hmrf.refinement:initialize_rdr_clone_refininement",
        348,
    ),
)
"""The read-depth refinement kept inside each BAF clone, as `cnaster` intends (#348).

Keeps the sub-clone mask `cnaster` drops before the HMRF; applied by port's
`pipeline_clone_assignment`, so not where a tumour proportion hands the call to
`cnaster` (#135). On by default (`DEFAULTS`).
"""


class Default(NamedTuple):
    """A behaviour of `port`'s own that `run_cnaster_port` turns on by default."""

    setting: str
    """The `run_cnaster_port` setting (`port.scripts.run_cnaster.Settings`)."""

    flag: str
    """The flag that asks for it; `--no-` and the rest turns it off."""

    ticket: int
    """The issue justifying it."""


DEFAULTS: tuple[Default, ...] = (
    Default("refinement_mask", "--refinement-mask", 467),
    Default("floor", "--floor-merge", 348),
)
"""`port`'s own behaviours on by default, outside `SWAPS` (T- #617).

On in every patched arm, off with their `--no-` flag or `--no-patch` (unless
`--sal`). The 300-normal-UMI segment floor stays `--sal`'s (PR- #645).
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

    Raises `ValueError` if no row installs `replacement` (T- #617).
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

    Binds a class row's options; each must already be an attribute of `cls`.
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
    """A row's replacement with its options bound; unknown options refused at install."""
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
    """Every imported module whose `name` is still bound to `original`."""
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
    """Rebind each site of each swap to `replace(swap, current)`, in order.

    `replace=None` rebinds nothing; `undo` collects what to restore.
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
    """Rebind every swap and do not restore; tests use `patched()`."""
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
"""What `patched` calls on exit to drop one run's state; unimported modules are skipped (#517)."""


def release() -> None:
    """Call every imported `RUN_STATE` release."""
    for target in RUN_STATE:
        module_name, _, attribute = target.partition(":")
        module = sys.modules.get(module_name)

        if module is not None:
            getattr(module, attribute)()


@contextmanager
def patched(swaps: tuple[Swap, ...] = SWAPS) -> Iterator[tuple[Site, ...]]:
    """Rebind every swap for the block, and restore on exit."""
    undo: list[tuple[ModuleType, str, Any]] = []

    try:
        yield _rebind(swaps, lambda swap, _: _replacement(swap), undo)
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)

        release()


@dataclass
class Spent:
    """What one swapped name cost in a run, first call kept apart (#204)."""

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
    """Time whatever each swapped name is currently bound to; compose with `patched()`."""
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

    `numba` specializes on types, not sizes, so one-element arrays suffice.
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
            # NB the compiled pass; `fused_spot_clone_field` wraps it (T- #781)
            "port.patch.hmrf.fused_field:_fused_kernel",
            (
                # observation arrays `(n_obs, n_spots)`; state parameters `(n_states,)` (#278)
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1, 1),
                _tiny(1),
                _tiny(1),
                _tiny(1),
                _tiny(1),
                np.zeros((1, 1), dtype=np.int64),
                _tiny(1),
                np.empty((1, 1)),
                False,
                np.empty((0, 0)),
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
    """Compile every named kernel; misses are reported, not raised."""
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
