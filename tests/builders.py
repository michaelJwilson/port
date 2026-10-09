"""Seeded builders of the arrays `cnaster`'s kernels take, shared across test modules."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import csr_matrix


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
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased

    result: np.ndarray = getattr(hmm_phased if phased else hmm_nophasing, which)(
        *inputs.arguments
    )
    return result


def unified_lattice(which: str, inputs: LatticeInputs, *, phased: bool) -> np.ndarray:
    """`port.patch.lattice`'s one recursion for `which`, on `inputs`."""
    from port.patch import lattice

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
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased

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


def buffered_emission(
    inputs: EmissionInputs, buffers: tuple[np.ndarray, np.ndarray], phased: bool
) -> None:
    """`port`'s buffered emission on `inputs`, written into `buffers`."""
    from port.sandbox.patch.emission import emission_into

    emission_into(
        inputs.single_X[:, 0, :],
        inputs.base_nb_mean,
        inputs.single_X[:, 1, :],
        inputs.total_bb_RD,
        inputs.log_mu,
        inputs.alphas,
        inputs.p_binom,
        inputs.taus,
        *buffers,
        phased,
    )


def random_graph(
    rng: np.random.Generator,
    n_nodes: int,
    degrees: tuple[int, int],
    *,
    weighted: bool = True,
    loops: bool = False,
) -> "csr_matrix":
    """Each node to a drawn number of distinct nodes (itself first, given `loops`), weights U[0.5, 2) or 1."""
    from scipy.sparse import csr_matrix

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
    import pandas as pd

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
    from scipy.sparse import csr_matrix

    rows = np.repeat(np.arange(n_nodes), degree)
    cols = rng.integers(0, n_nodes, degree * n_nodes)
    data = rng.uniform(0.5, 2.0, rows.size) if weighted else np.ones(rows.size)
    return csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
