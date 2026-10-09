"""Seeded builders of the arrays `cnaster`'s kernels take, shared across test modules."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from port.patch import lattice
from scipy.sparse import csr_matrix

from tests import TESTS


@dataclass(frozen=True)
class LatticeInputs:
    """What a forward or backward lattice takes, at `cnaster`'s shapes."""

    lengths: np.ndarray
    log_transmat: np.ndarray
    log_startprob: np.ndarray
    log_emission: np.ndarray
    log_sitewise_transmat: np.ndarray
    n_states: int

    @property
    def arguments(self) -> tuple[np.ndarray, ...]:
        """The five positional arguments `cnaster`'s lattices take."""
        return (
            self.lengths,
            self.log_transmat,
            self.log_startprob,
            self.log_emission,
            self.log_sitewise_transmat,
        )


def random_lattice(
    n_states: int,
    lengths: Sequence[int],
    n_spots: int,
    *,
    phased: bool,
    seed: int,
    dirichlet: bool = False,
) -> LatticeInputs:
    """A proper transition and start, drawn emissions, and a site-varying switch kernel."""
    rng = np.random.default_rng(seed)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    n_obs = int(lengths_array.sum())
    shape = (2 * n_states if phased else n_states, n_obs, n_spots)

    if dirichlet:
        transition = rng.dirichlet(np.ones(n_states), n_states)
        start = rng.dirichlet(np.ones(n_states))
        log_emission, high = rng.normal(-5.0, 3.0, shape), 0.3
    else:
        transition = rng.random((n_states, n_states)) + 0.5
        transition /= transition.sum(axis=1, keepdims=True)
        start = rng.random(n_states) + 0.5
        start /= start.sum()
        log_emission, high = rng.normal(-2.0, 1.5, shape), 0.4

    return LatticeInputs(
        lengths=lengths_array,
        log_transmat=np.log(transition),
        log_startprob=np.log(start),
        log_emission=log_emission,
        log_sitewise_transmat=np.log(rng.uniform(1e-4, high, n_obs)),
        n_states=n_states,
    )


def cnaster_lattice(which: str, inputs: LatticeInputs, *, phased: bool) -> np.ndarray:
    """`cnaster`'s `which` lattice, phased or not, on `inputs`."""

    result: np.ndarray = getattr(hmm_phased if phased else hmm_nophasing, which)(
        *inputs.arguments
    )
    return result


def unified_lattice(which: str, inputs: LatticeInputs, *, phased: bool) -> np.ndarray:
    """`port.patch.lattice`'s one recursion for `which`, on `inputs`."""

    result: np.ndarray = getattr(lattice, which)(
        *inputs.arguments, inputs.n_states, phased
    )
    return result


def allele_counts(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    trials: tuple[int, int],
    share: float,
    depth: float | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`X[:, 0] ~ Poisson(depth)`, `X[:, 1] ~ Binomial(total, share)`; `X`, depth, `total`."""
    if isinstance(depth, tuple):
        exposure = rng.integers(*depth, shape).astype(np.float64)
    else:
        exposure = np.full(shape, float(depth))
    total = rng.integers(*trials, shape).astype(np.float64)

    X = np.zeros((shape[0], 2, *shape[1:]))
    X[:, 0] = rng.poisson(exposure)
    X[:, 1] = rng.binomial(total.astype(int), share)
    return X, exposure, total


@dataclass(frozen=True)
class EmissionInputs:
    """Both channels: `(n_obs, ...)` counts and the parameters a fit produces (#278)."""

    single_X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.log_mu.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.single_X.shape[0]), int(self.single_X.shape[2]))

    def columns(self, n_spots: int = 1) -> "EmissionInputs":
        """Per-state parameters tiled to `(n_states, n_spots)`, as `cnaster` takes them."""
        names = ("log_mu", "alphas", "p_binom", "taus")
        tiled = {k: np.tile(getattr(self, k)[:, None], (1, n_spots)) for k in names}
        return replace(self, **tiled)


EMISSION_RANGES = ((-0.35, 0.35), (0.12, 0.55), (0.22, 0.78), (8.0, 28.0))
"""`log_mu`, `alphas`, `p_binom` and `taus`, spaced evenly across states."""


def emission_inputs(
    n_states: int,
    n_obs: int,
    n_spots: int,
    *,
    seed: int,
    exposure: tuple[int, int] = (20, 45),
    trials: tuple[int, int] = (5, 25),
    share: float = 0.42,
    ranges: tuple[tuple[float, float], ...] = EMISSION_RANGES,
) -> EmissionInputs:
    """Counts drawn with repeats, so an encoder deduplicates, and one-dimensional parameters."""
    X, base, total = allele_counts(
        np.random.default_rng(seed), (n_obs, n_spots), trials, share, exposure
    )
    log_mu, alphas, p_binom, taus = (np.linspace(*r, n_states) for r in ranges)
    return EmissionInputs(X, base, total, log_mu, alphas, p_binom, taus)


def cnaster_emission_pair(
    inputs: EmissionInputs, *, phased: bool = False, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """`cnaster`'s `(rdr, baf)` emission on `inputs`, unphased or phased."""

    klass: Any = hmm_phased if phased else hmm_nophasing
    scored: tuple[np.ndarray, np.ndarray] = (
        klass.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            inputs.log_mu,
            inputs.alphas,
            inputs.total_bb_RD,
            inputs.p_binom,
            inputs.taus,
            **kwargs,
        )
    )
    return scored


def random_graph(
    rng: np.random.Generator,
    n_nodes: int,
    degrees: tuple[int, int],
    *,
    weighted: bool = True,
    loops: bool = False,
) -> "csr_matrix":
    """Each node to a drawn number of distinct nodes (itself first, given `loops`), weights U[0.5, 2) or 1."""

    rows, cols, data = [], [], []
    for node in range(n_nodes):
        drawn = rng.choice(n_nodes, size=int(rng.integers(*degrees)), replace=False)
        for neighbour in [node, *drawn.tolist()] if loops else drawn:
            rows.append(node)
            cols.append(int(neighbour))
            data.append(float(rng.uniform(0.5, 2.0)) if weighted else 1.0)

    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def gene_snp_blocks(blocks: Iterable[tuple[int, int, int]]) -> "pd.DataFrame":
    """Per `(contig, start, width)`: a gene at `start`, a SNP in it, a gene ending at `start + width`."""

    rows = []
    for block, (contig, start, width) in enumerate(blocks):
        rows += [
            (contig, start, start + 10, True, block),
            (contig, start + 5, start + 6, False, block),
            (contig, start + width - 10, start + width, True, block),
        ]

    return pd.DataFrame(
        rows, columns=["CHR", "START", "END", "is_interval", "block_id"]
    )


def regular_graph(
    rng: np.random.Generator, n_nodes: int, degree: int, *, weighted: bool = False
) -> "csr_matrix":
    """`degree` edges from each node to nodes drawn with replacement, weights U[0.5, 2) or 1."""

    rows = np.repeat(np.arange(n_nodes), degree)
    cols = rng.integers(0, n_nodes, degree * n_nodes)
    data = rng.uniform(0.5, 2.0, rows.size) if weighted else np.ones(rows.size)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def masked_lattice(
    n_states: int, lengths: tuple[int, ...], spots: int, *, phased: bool, seed: int
) -> tuple[np.ndarray, ...]:
    """A proper transition, a switch kernel that moves, and some `-inf` sites."""
    inputs = random_lattice(
        n_states, lengths, spots, phased=phased, seed=seed, dirichlet=True
    )
    # NB an impossible state at some sites, as a zero-count BAF bin gives `cnaster`.
    inputs.log_emission[0, :: max(inputs.log_emission.shape[1] // 7, 1), 0] = -np.inf
    return inputs.arguments


NB_COUNTS = np.array([0, 1, 7, 42, 300, 1000, 2500], dtype=np.float64)
"""Counts the negative-binomial kernels score at an exposure of 1,000 (#560)."""


def coded_encoders(
    n_obs: int, n_spots: int, seed: int
) -> tuple[Any, Any, dict[str, Any]]:
    """`cnaster`'s two encoders and three states' `(n_states, 1)` parameters (#269)."""
    rng = np.random.default_rng(seed)

    counts = rng.poisson(60, size=(n_obs, n_spots)).astype(float)
    exposure = np.full((n_obs, n_spots), 60.0)
    alleles = rng.binomial(40, 0.4, size=(n_obs, n_spots)).astype(float)
    depth = np.full((n_obs, n_spots), 40.0)

    n_states = 3
    parameters = {
        "log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
        "alphas": np.full((n_states, 1), 0.25),
        "p_binom": rng.uniform(0.2, 0.8, size=(n_states, 1)),
        "taus": np.full((n_states, 1), 30.0),
    }

    return CountEncoder(counts, exposure), CountEncoder(alleles, depth), parameters


def reindex_result(
    n_states: int = 4, n_obs: int = 12, n_clones: int = 3
) -> dict[str, Any]:
    """A fit for `reindex_clones` whose balanced clone is clone 1 (#278)."""
    rng = np.random.default_rng(5)

    # clone 1 is the balanced one, so the reorder has something to do
    p_binom = np.array([[0.2], [0.5], [0.8], [0.35]])[:n_states]
    paths = np.concatenate(
        [
            np.full(n_obs, 0),  # clone 0: p = 0.2, far from balanced
            np.full(n_obs, 1),  # clone 1: p = 0.5, the normal one
            np.full(n_obs, 2),  # clone 2: p = 0.8
        ][:n_clones]
    )

    assignments = np.repeat(np.arange(n_clones), [5, 3, 7][:n_clones])

    return {
        "new_assignment": assignments,
        "pred_cnv": paths,
        "new_p_binom": p_binom,
        "new_log_mu": rng.normal(size=(n_states, 1)),
        "new_alphas": np.full((n_states, 1), 0.25),
        "new_taus": np.full((n_states, 1), 30.0),
        "log_gamma": rng.normal(size=(n_states, n_obs * n_clones)),
    }


#: Planted `(log mu, p)`: neutral, one-copy loss, one-copy gain, copy-neutral LOH.
#: LOH states sit at a small p, as a mixture with normal spots leaves them.
PLANTED = np.array(
    [[0.0, 0.5], [np.log(0.5), 0.05], [np.log(1.5), 1.0 / 3.0], [0.0, 0.05]]
)
WEIGHTS = np.array([0.55, 0.15, 0.15, 0.15])


def mixture_draw(n_obs: int = 3000, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Bins from the model's own family: NB totals over exposure, beta-binomial B counts."""
    rng = np.random.default_rng(seed)
    state = rng.choice(len(PLANTED), size=n_obs, p=WEIGHTS)
    exposure = rng.uniform(200.0, 400.0, n_obs)
    trials = rng.integers(20, 60, n_obs).astype(float)
    mean = exposure * np.exp(PLANTED[state, 0])
    # NB alpha 0.02, a clone pseudobulk's dispersion rather than a spot's.
    size = 50.0
    totals = rng.negative_binomial(size, size / (size + mean))
    p = rng.beta(PLANTED[state, 1] * 1000.0, (1.0 - PLANTED[state, 1]) * 1000.0)
    successes = rng.binomial(trials.astype(int), p)
    X = np.stack([totals, successes], axis=1).astype(float).reshape(n_obs, 2, 1)
    return X, exposure.reshape(-1, 1), trials.reshape(-1, 1)


def rectangular_coords(name: str) -> np.ndarray:
    """The dev run's captured `initialize_rectangular_clones` coordinates (#298)."""
    coords: np.ndarray = np.load(TESTS / "data" / f"{name}.npz")["coords"]

    return coords
