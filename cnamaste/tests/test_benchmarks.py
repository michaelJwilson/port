"""Baselines for the hot kernels, at a gate size and, under `release`, a stress size.

Disabled by default (`--benchmark-disable` runs each body once, as a test);
`pytest --benchmark-enable -m "benchmark or release"` measures. Each kernel
is called once before measuring so the numba compile, which `cache=True`
keeps on disk, is not part of the figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sim.truth import planted

from cnamaste.hmm_nophasing import (
    _dense_bb_logpmf,
    _dense_nb_logpmf,
    get_log_transmat,
    hmm_nophasing,
)
from cnamaste.hmm_phased import hmm_phased

pytestmark = pytest.mark.benchmark

SIZES = {
    "gate": pytest.param({"lattice": (10, 10), "n_obs": 200, "n_states": 4}, id="gate"),
    "stress": pytest.param(
        {"lattice": (50, 60), "n_obs": 2_000, "n_states": 8},
        id="stress",
        marks=pytest.mark.release,
    ),
}
"""Gate: 100 spots by 200 bins. Stress: 3,000 spots by 2,000 bins, the declared
scale's spot count."""


def _instance(size: dict[str, Any]) -> Any:
    return planted(n_clones=3, n_segments=4, **size)


def _warm(function: Any, *args: Any) -> None:
    function(*args)


@pytest.mark.parametrize("size", SIZES.values())
def test_dense_negative_binomial(benchmark: Any, size: dict[str, Any]) -> None:
    truth = _instance(size)
    args = (
        truth.counts_nb.astype(np.float64),
        truth.base_nb_mean,
        truth.log_mu[:, None],
        truth.alphas[:, None],
    )
    _warm(_dense_nb_logpmf, *args)
    out = benchmark(_dense_nb_logpmf, *args)
    assert np.all(np.isfinite(out))
    assert np.all(out <= 0.0)


@pytest.mark.parametrize("size", SIZES.values())
def test_dense_beta_binomial(benchmark: Any, size: dict[str, Any]) -> None:
    truth = _instance(size)
    args = (
        truth.counts_bb.astype(np.float64),
        truth.total_bb_RD.astype(np.float64),
        truth.p_binom[:, None],
        truth.taus[:, None],
    )
    _warm(_dense_bb_logpmf, *args)
    out = benchmark(_dense_bb_logpmf, *args)
    assert np.all(np.isfinite(out))
    assert np.all(out <= 0.0)


def _lattice_args(size: dict[str, Any], phased: bool) -> tuple[Any, ...]:
    truth = _instance(size)
    k = truth.n_states * (2 if phased else 1)
    rng = np.random.default_rng(0)
    # NB clone-stacked, as the HMM sees it: one column per clone.
    log_emission = np.log(rng.uniform(0.05, 1.0, (k, truth.n_obs, truth.n_clones)))
    log_startprob = np.full(truth.n_states, -np.log(truth.n_states))
    return (
        truth.lengths,
        get_log_transmat(truth.n_states, 0.99),
        log_startprob,
        log_emission,
        np.log(truth.switch_prob),
    )


@pytest.mark.parametrize("phased", [False, True], ids=["unphased", "phased"])
@pytest.mark.parametrize("size", SIZES.values())
def test_forward_lattice(benchmark: Any, size: dict[str, Any], phased: bool) -> None:
    forward = (hmm_phased if phased else hmm_nophasing).forward_lattice
    args = _lattice_args(size, phased)
    _warm(forward, *args)
    log_alpha = benchmark(forward, *args)
    assert np.all(np.isfinite(log_alpha))


def _field_args(size: dict[str, Any]) -> tuple[Any, ...]:
    truth = _instance(size)
    return (
        truth.counts_nb.astype(np.float64),
        truth.base_nb_mean,
        truth.unphased_bb().astype(np.float64),
        truth.total_bb_RD.astype(np.float64),
        truth.log_mu,
        truth.alphas,
        truth.p_binom,
        truth.taus,
        np.ascontiguousarray(truth.states.T),
        np.ones(truth.n_spots),
        np.empty((truth.n_spots, truth.n_clones)),
    )


@pytest.mark.parametrize("size", SIZES.values())
def test_spot_clone_field(benchmark: Any, size: dict[str, Any]) -> None:
    """The fused `(n_spots, n_clones)` field, once per outer iteration (#392 stage 1)."""
    from cnamaste.hmrf import fused_spot_clone_field

    args = _field_args(size)
    _warm(fused_spot_clone_field, *args)
    field = benchmark(fused_spot_clone_field, *args)
    assert np.all(np.isfinite(field))


@pytest.mark.parametrize("size", SIZES.values())
def test_label_sweep(benchmark: Any, size: dict[str, Any]) -> None:
    """The ICM from a random labelling on the lattice graph (#392 stage 1)."""
    import scipy.sparse as sp

    from cnamaste.icm_interface import CsrGraph, icm_sweep
    from cnamaste.spatial import construct_multislice_lattice_adjacency

    truth = _instance(size)
    rows, columns = np.unravel_index(np.arange(truth.n_spots), truth.lattice)
    adjacency, _ = construct_multislice_lattice_adjacency(
        np.zeros(truth.n_spots, dtype=np.int64),
        ["S1"],
        np.stack([rows, columns], axis=1).astype(np.float64),
        None,
        maxspots_pooling=1,
        unit_xsquared=1,
        unit_ysquared=1,
    )
    graph = CsrGraph.from_matrix(sp.csr_matrix(adjacency))
    rng = np.random.default_rng(0)
    field = rng.normal(0.0, 1.0, (truth.n_spots, truth.n_clones))
    start = rng.integers(0, truth.n_clones, truth.n_spots)

    def sweep() -> Any:
        np.random.seed(0)  # noqa: NPY002 -- the solver's legacy global stream
        return icm_sweep(field, graph, start.copy(), 1.0, min_clone_spots=0)

    sweep()
    result = benchmark(sweep)
    assert np.isfinite(result.cost)


@pytest.mark.parametrize("size", SIZES.values())
def test_genomic_figure(benchmark: Any, size: dict[str, Any], tmp_path: Any) -> None:
    """`plot_clones_genomic` drawn and written as a PDF (#392 stage 2)."""
    import matplotlib as mpl

    from cnamaste.plot_genomic import plot_clones_genomic
    from cnamaste.utils import write_fig

    mpl.use("Agg")
    truth = _instance(size)
    single_X = np.stack([truth.counts_nb, truth.unphased_bb()], axis=1)
    res = {
        "new_log_mu": truth.log_mu[:, None],
        "new_p_binom": truth.p_binom[:, None],
        "pred_cnv": truth.states.T,
        "new_assignment": truth.labels,
    }
    clone_index = [np.flatnonzero(truth.labels == c) for c in range(truth.n_clones)]
    path = tmp_path / "clones_genomic.pdf"

    def draw() -> None:
        figure = plot_clones_genomic(
            truth.lengths,
            single_X,
            truth.base_nb_mean,
            truth.total_bb_RD,
            res_combine=res,
            clone_index=clone_index,
        )
        write_fig(str(path), figure)

    draw()
    benchmark(draw)
    assert path.stat().st_size > 0


@pytest.mark.parametrize("size", SIZES.values())
def test_library_shifts(benchmark: Any, size: dict[str, Any]) -> None:
    """`log Z_c` per clone, once per emission evaluation under the shift (#392 stage 3)."""
    from cnamaste.hmm_nophasing import shifts

    truth = _instance(size)
    profile = truth.base_nb_mean.sum(axis=1)
    log_lambda = np.tile(np.log(profile / profile.sum()), truth.n_clones)
    args = (
        truth.log_mu,
        truth.states.reshape(-1),
        log_lambda,
        [truth.n_obs] * truth.n_clones,
    )
    _warm(shifts, *args)
    out = benchmark(shifts, *args)
    assert np.all(np.isfinite(out))
