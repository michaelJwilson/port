"""Port's drop-ins against the `cnaster` code they replace, as one table (#850).

A row names `cnaster`'s function, port's, a seeded builder of the arguments both take,
and the comparison. `tests/test_correspondence.py` runs the rows, one test per tier and
referee; `dropin-refereed` reads `ours`. A drop-in whose test does more than compare
the two calls keeps that test, and the guard's import scan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import cnaster.hmrf
import cnaster.spatial
import numba
import numpy as np
import pytest
import scipy.sparse as sp
from cnaster.hmm_nophasing import _nb_logpmf_1d, hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from port.patch import hmm_phased as port_phased
from port.patch import lattice, pseudobulk, spatial
from port.patch.hmm_initialize import distinct, sal_mixture
from port.patch.hmm_nophasing import nb_logpmf
from port.patch.hmrf import adjacency, clone_assignment, reindex
from port.patch.hmrf import field as port_field

from tests import adapters, builders, fixtures


def bitwise(ours: Any, theirs: Any, at: str = "") -> None:
    """Same structure; each array the same shape, dtype and entries, `nan` equal to nothing."""
    if theirs is None:
        assert ours is None, at
    elif isinstance(theirs, Mapping):
        assert set(ours) == set(theirs), at
        for key in sorted(theirs):
            bitwise(ours[key], theirs[key], f"{at}[{key!r}]")
    elif isinstance(theirs, tuple | list):
        assert len(ours) == len(theirs), at
        for index, pair in enumerate(zip(ours, theirs, strict=True)):
            bitwise(*pair, f"{at}[{index}]")
    elif sp.issparse(theirs):
        assert (ours.shape, ours.dtype) == (theirs.shape, theirs.dtype), at
        assert (ours != theirs).nnz == 0, at
    else:
        mine, reference = np.asarray(ours), np.asarray(theirs)
        assert (mine.shape, mine.dtype) == (reference.shape, reference.dtype), at
        assert np.array_equal(mine, reference), at


def within(rtol: float, atol: float) -> Callable[[Any, Any], None]:
    """Equal shape and dtype, and `|ours - theirs| <= atol + rtol |theirs|` entry by entry."""

    def compare(ours: Any, theirs: Any) -> None:
        np.testing.assert_allclose(ours, theirs, rtol=rtol, atol=atol, strict=True)

    return compare


def lattice_agree(rust: np.ndarray, cnaster: np.ndarray) -> None:
    """Bitwise against compiled `cnaster`; 1e-14 relative when `numba` is disabled (#318)."""

    if not getattr(numba.config, "DISABLE_JIT"):  # noqa: B009 -- numba sets it at import
        np.testing.assert_array_equal(rust, cnaster)
        return

    finite = np.isfinite(cnaster)
    np.testing.assert_array_equal(np.isfinite(rust), finite)
    np.testing.assert_array_equal(rust[~finite], cnaster[~finite])
    np.testing.assert_allclose(rust[finite], cnaster[finite], rtol=1e-14, atol=0.0)


def direct(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return function(*args, **kwargs)


def out_parameter(kernel: Callable[..., Any], *args: Any) -> np.ndarray:
    """What `kernel` writes into a zeroed buffer the length of its first argument."""
    out = np.zeros(len(args[0]))
    kernel(*args, out)
    return out


@dataclass(frozen=True)
class Correspondence:
    """`compare(call(ours, *inputs()), call(upstream, *inputs()))`, each call on its own build."""

    origin: str
    """The test this row replaces, with its case, as collected before #850."""
    upstream: Callable[..., Any]
    ours: Callable[..., Any]
    inputs: Callable[..., tuple[Any, ...]]
    compare: Callable[[Any, Any], None]
    keywords: Mapping[str, Any] = field(default_factory=dict)
    call: Callable[..., Any] = direct
    tier: str = ""
    """`merge`, `release` and the like; empty for the per-PR gate."""
    referee: str = "patch"
    marks: tuple[str, ...] = ()
    config: bool = False
    """Whether the calls read `cnaster`'s global config (the `cnaster_config` fixture)."""


def _sides(phased: bool) -> Any:
    return hmm_phased if phased else hmm_nophasing


def _deconcatenated() -> tuple[dict[str, Any]]:
    """`pred_cnv` at `(n_obs, n_clones)`, with a clone axis on `log_gamma`."""
    result = builders.reindex_result(4, 12, 3)
    result["pred_cnv"] = np.asarray(result["pred_cnv"]).reshape(3, 12).T
    result["log_gamma"] = np.random.default_rng(13).normal(size=(4, 12, 3))
    return (result,)


def _multislice() -> tuple[Any, ...]:
    first, second = adapters.square_coords(12, 10), adapters.square_coords(9, 14)
    sample_ids = np.repeat([0, 1], [len(first), len(second)])
    return (
        sample_ids,
        ["A", "B"],
        np.concatenate([first, second]).astype(float),
        None,
        1,
    )


def _adjacency(rows: int, columns: int, across: bool) -> tuple[Any, ...]:
    coords = adapters.square_coords(rows, columns).astype(float)
    graph = (
        sp.random(100, 100, density=0.02, format="csr", random_state=5)
        if across
        else None
    )
    return np.zeros(len(coords), dtype=int), [0], coords, graph, 1, 1, 1


def _field(fixture: Any) -> tuple[Any, np.ndarray]:
    return fixture, np.ones(fixture.n_spots)


def _first(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return function(*args, **kwargs)[0]


LATTICES = ("forward_lattice", "backward_lattice")
RUST = [(3, (7, 11, 5), 3), (5, (1, 40, 2, 17), 4), (4, (60,), 1), (10, (1000,) * 10, 20)]  # fmt: skip
"""`(n_states, lengths, spots)`: gate sizes, then one above `PARALLEL_WORK` (#318)."""
UNPOSTERIOR = {"posterior": None, "single_tumor_prop": None}
FIELDS: list[dict[str, Any]] = [*({"self_transition": s} for s in (0.999, 0.99, 0.9)), *({"n_states": s, "n_clones": c} for s, c in ((2, 1), (5, 5), (7, 3)))]  # fmt: skip
PARTITIONS: dict[str, tuple[int, int, dict[str, Any]]] = {"25x40": (25, 40, {"n_trials": 200}), "50x50": (50, 50, {"n_trials": 200}), "tumour-proportion": (25, 40, {"single_tumor_prop": np.random.default_rng(7).uniform(0, 1, size=1000), "n_trials": 100})}  # fmt: skip
U, R = "test_unified_lattice.py::test_the_unified_recursion_is_cnasters_bitwise", "test_rust_lattice.py::test_the_"  # fmt: skip
C, S = "test_clone_assignment_correspondence.py::test_the_", "test_preprocessing_spatial.py::test_the_"  # fmt: skip

# fmt: off
ROWS: list[Correspondence] = [
    # NB #205: one recursion for cnaster's four
    *(Correspondence(f"{U}[{k}-{'phased' if p else 'unphased'}-{w}]", getattr(_sides(p), w), partial(getattr(lattice, w), n_states=k, phased=p),
                     lambda k=k, p=p: builders.random_lattice(k, (7, 11, 5), 3, phased=p, seed=5).arguments, bitwise)
      for w in LATTICES for p in (False, True) for k in (2, 5)),
    # NB #318: `oxiport`'s lattices, every entry, `-inf` included
    *(Correspondence(f"{R}unphased_lattice_is_cnasters_bitwise[{w}-K{c[0]}-G{sum(c[1])}-S{c[2]}]", getattr(hmm_nophasing, w),
                     getattr(lattice, f"{w}_rust"), lambda c=c: builders.masked_lattice(*c, phased=False, seed=3), lattice_agree)
      for w in LATTICES for c in RUST),
    *(Correspondence(f"{R}phased_lattice_is_cnasters_bitwise[{w}-K{c[0]}-G{sum(c[1])}-S{c[2]}-{b}]", getattr(hmm_phased, w),
                     getattr(lattice, f"{w}_phased_rust"), lambda c=c, b=b: (*builders.masked_lattice(*c, phased=True, seed=4), b), lattice_agree)
      for w in LATTICES for c in RUST for b in (False, True)),
    # NB #560: where `alpha * lambda >= 1e-8` the patch changes only the rounding regime
    *(Correspondence(f"test_patch_nb_logpmf.py::test_the_patched_kernel_is_cnasters_where_cnasters_p_is_below_one[{mu}]", _nb_logpmf_1d, nb_logpmf._nb_logpmf_1d,
                     lambda mu=mu: (builders.NB_COUNTS, np.full(builders.NB_COUNTS.size, 1000.0), mu, 0.12), within(rtol=1e-9, atol=1e-9), call=out_parameter)
      for mu in (1e-5, 0.3, 1.0, 2.7)),
    # NB #281: assignment, field and likelihood; `single_tumor_prop` delegates (#135), `merge=True` is #59
    *(Correspondence(f"{C}{name}[{case}]", clone_assignment.UPSTREAM, clone_assignment.pipeline_clone_assignment,
                     lambda s=s, c=c, o=o, n=n: (adapters.clone_assignment_arguments(fixtures.spot_clone_field(n_states=s, n_obs=o, n_spots=n * n, n_clones=c), width=n),),
                     bitwise, options, adapters.clone_assignment_call, marks=("cnaster",), config=True)
      for name, case, s, c, o, n, options in (
          ("replacement_assigns_what_upstream_assigns", "5-3", 5, 3, 60, 6, {}),
          ("replacement_assigns_what_upstream_assigns", "3-2", 3, 2, 60, 6, {}),
          ("delegated_and_merging_calls_return_upstreams", "tumour-mixed", 3, 2, 40, 4, {"single_tumor_prop": np.full(16, 0.7)}),
          ("delegated_and_merging_calls_return_upstreams", "merge", 4, 4, 40, 4, {"merge": True}))),
    # NB #269: at one spot, the only width upstream runs
    Correspondence("test_hmm_phased_coded_emission.py::test_the_replacement_matches_upstream_where_upstream_runs", hmm_phased.compute_emission_probability_nb_betabinom_coded,
                   port_phased.compute_emission_probability_nb_betabinom_coded, partial(builders.coded_encoders, n_obs=25, n_spots=1, seed=1), bitwise,
                   call=lambda f, nb, bb, parameters: f(nb, bb, **parameters), config=True),
    # NB #59 item 3: dtype decides the `@njit` specialization
    *(Correspondence(f"test_hmrf_adjacency_patch.py::test_the_triple_is_bitwise_cnasters[{n}]", adapters.cnaster_adjacency_triple, adjacency.adjacency_coo,
                     lambda n=n: (builders.random_graph(np.random.default_rng(11), n * n, (1, 7)),), bitwise) for n in (3, 8, 20)),
    # NB #59 item 1: across segmentations, and `n_clones == n_states`
    *(Correspondence(f"test_hmrf_field_patch.py::test_the_patch_is_bitwise_cnaster[{'-'.join(f'{k}={v}' for k, v in f.items())}]", cnaster.hmrf.compute_loglike_spot_assignment,
                     port_field.compute_loglike_spot_assignment_strided, lambda f=f: (fixtures.spot_clone_field(**f),), bitwise, call=lambda kernel, fixture: fixtures.cnaster_field_of(fixture, kernel))
      for f in FIELDS),
    # NB #59 item 2: against `cnaster`'s producer, not the fixture's emission (#9)
    *(Correspondence(f"test_hmrf_fused_field.py::test_the_fused_field_is_bitwise_the_two_step[{s}-{c}]", fixtures.cnaster_two_step, fixtures.fused_field_of,
                     lambda s=s, c=c: _field(fixtures.spot_clone_field(n_states=s, n_clones=c)), bitwise, tier=tier)
      for s, c, tier in ((7, 3, "merge"), (5, 5, ""), (3, 1, ""))),
    # NB #488: across block edges, fewer bins than a block, one short, one exact, one over
    *(Correspondence(f"test_pseudobulk_patch.py::test_the_blocked_merge_is_upstreams_bitwise[{n}]", merge_pseudobulk_by_index_mix, pseudobulk.merge_pseudobulk_by_index_mix,
                     lambda n=n: (fixtures.pseudobulk_inputs(n, 90, 3, seed=n),), bitwise, call=lambda f, inputs: f(**inputs), marks=("cnaster",))
      for n in (1, 2, 255, 256, 257, 513, 700)),
    # NB #304: where `cnaster` returns, the dev first call and a 12 x 40 band
    *(Correspondence(f"test_rectangular_clones.py::test_where_cnaster_returns_it_returns_the_same[{at}-{n}-{seed}]", cnaster.spatial.initialize_rectangular_clones,
                     spatial.initialize_rectangular_clones, lambda at=at, n=n: (builders.rectangular_coords("rectangular_returns") if at == "returns" else adapters.square_coords(12, 40), n),
                     bitwise, {"random_state": seed}, lambda f, *a, **k: tuple(f(*a, **k))[:2], marks=("cnaster",))
      for at, n, seed in (("returns", 4, 0), ("grid", 4, 1), ("grid", 2, 3), ("grid", 3, 0), ("grid", 1, 0))),
    # NB #278: normal clone, order and every reindexed array
    Correspondence("test_reindex_clones.py::test_the_replacement_reindexes_as_upstream_does", cnaster.hmrf.reindex_clones, reindex.reindex_clones,
                   lambda: (builders.reindex_result(),), bitwise, UNPOSTERIOR, _first),
    Correspondence("test_reindex_clones.py::test_the_deconcatenated_path_is_reindexed_as_upstream_does", cnaster.hmrf.reindex_clones, reindex.reindex_clones,
                   _deconcatenated, bitwise, UNPOSTERIOR, _first, marks=("cnaster",)),
    Correspondence("test_reindex_clones.py::test_a_posterior_is_permuted_with_the_clones", cnaster.hmrf.reindex_clones, reindex.reindex_clones,
                   lambda: (builders.reindex_result(n_clones=3), np.random.default_rng(17).uniform(size=(15, 3))), bitwise,
                   call=lambda f, result, posterior: f(result, posterior=posterior)[1], marks=("cnaster",)),
    # NB #190: graph, weights and dtype; the inter-slice graph is still added
    *(Correspondence(f"{S}sparse_adjacency_is_the_dense_one[{name}]", cnaster.spatial.construct_multislice_lattice_adjacency,
                     spatial.construct_multislice_lattice_adjacency, partial(_adjacency, *shape), bitwise, marks=("preprocessing",))
      for name, shape in {"25x40": (25, 40, False), "50x50": (50, 50, False), "across-slices": (10, 10, True)}.items()),
    *(Correspondence(f"{S}partition_keeps_the_trial_cnaster_keeps[{name}]", cnaster.spatial.best_equal_partition, spatial.best_equal_partition,
                     lambda r=r, c=c: (adapters.square_coords(r, c).astype(float), 3, 3), bitwise, options, marks=("preprocessing",))
      for name, (r, c, options) in PARTITIONS.items()),
    Correspondence("test_lattice_adjacency.py::test_the_swap_reproduces_cnasters_multislice_adjacency_on_square_grids", cnaster.spatial.construct_multislice_lattice_adjacency,
                   spatial.lattice_multislice_adjacency, _multislice, bitwise, {"unit_xsquared": 1, "unit_ysquared": 1}),
    # NB #489: without exposure the call is `cnaster`'s `gmm_init`
    *(Correspondence(f"test_sal_mixture_init.py::test_the_baf_only_and_minor_calls_keep_upstreams_start[{params}-{minor}]", distinct.UPSTREAM,
                     partial(sal_mixture.gmm_init, start=sal_mixture.DEFAULT), lambda params=params: (4, *builders.mixture_draw(400, seed=1), params, np.array([400]), None, None),
                     bitwise, {"random_state": 0, "in_log_space": False, "only_minor": minor}, marks=("cnaster",), config=True)
      for params, minor in (("sp", False), ("smp", True))),
]
"""Every correspondence row; `origin` is unique."""
# fmt: on


def rows(tier: str, referee: str) -> list[Any]:
    """The rows one test runs, as `pytest.param`s carrying their own marks."""
    return [
        pytest.param(
            row,
            id=row.origin.partition("::test_")[2].replace("[", "/").removesuffix("]"),
            marks=[getattr(pytest.mark, m) for m in row.marks],
        )  # fmt: skip
        for row in ROWS
        if (row.tier, row.referee) == (tier, referee)
    ]
