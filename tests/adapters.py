"""Conversions from a `snakes_and_ladders` instance to `cnaster`'s arguments.

A constant exposure is exact, absorbed as `log_mu - log(c)`; a zero `total_bb_RD`
silences the beta-binomial channel. Tests pin both.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from cnaster.hmm import compute_copy_state_posterior
from cnaster.hmm_emission import (
    Weighted_BetaBinom_mix,
    betabinom_logpmf_zp,
    compute_bb_ab,
)
from cnaster.hmm_nophasing import get_log_transmat, hmm_nophasing
from cnaster.hmm_phased import hmm_phased
from cnaster.hmm_utils import get_em_solver_params
from cnaster.hmrf_utils import cast_csr, clone_stack_obs
from cnaster.icm import calc_assignment_cost, icm_sweep_deque, unpack_adjacency
from cnaster.phasing import initial_phase_given_partition
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from port.sim.truth import CoreInferenceTruth
from sal.backend import Backend
from sal.emissions import BetaBinomialEmission
from sal.opt.hmm import forward_log_likelihood_from_density
from sal.search.alpha_expansion import alpha_expansion
from sal.search.icm import iterated_conditional_modes
from sal.sim.potts import energy
from scipy.sparse import coo_matrix
from scipy.sparse import eye as sparse_eye
from scipy.special import logsumexp

from tests.fixtures import (
    BetaBinomialChains,
    NegativeBinomialChains,
    PhasedChains,
    PottsLabels,
    SpotCloneField,
    scaled_graph,
)

if TYPE_CHECKING:
    from sal.search.alpha_expansion import ExpansionResult
    from scipy.sparse import csr_matrix

N_CHANNELS = 2
"""`cnaster` packs a count and a success into `single_X`'s middle axis."""

RDR_CHANNEL = 0
BAF_CHANNEL = 1


@dataclass(frozen=True)
class CnasterChainInputs:
    """`cnaster`'s arguments for chains laid end to end, split by `lengths`, and the parameters to score at."""

    single_X: np.ndarray
    lengths: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray
    log_startprob: np.ndarray
    log_transmat: np.ndarray
    log_sitewise_transmat: np.ndarray

    @property
    def n_obs(self) -> int:
        """Positions along the concatenated genomic axis."""
        return int(self.single_X.shape[0])

    @property
    def n_states(self) -> int:
        """`K`, the number of hidden states."""
        return int(self.log_mu.shape[0])


def from_negative_binomial_chains(
    fixture: NegativeBinomialChains,
    *,
    exposure: float = 1.0,
) -> CnasterChainInputs:
    """Lay the fixture's chains out as `cnaster` expects; a constant `exposure` > 0 is absorbed into `log_mu`."""
    if exposure <= 0.0:
        msg = f"exposure must be strictly positive, got {exposure}"
        raise ValueError(msg)

    observations = fixture.dataset.observations
    n_sequences, sequence_length = observations.shape
    n_obs = n_sequences * sequence_length
    n_states = fixture.n_states

    single_X = np.zeros((n_obs, N_CHANNELS, 1), dtype=np.float64)
    single_X[:, RDR_CHANNEL, 0] = observations.reshape(-1)

    # NB zero depth makes the beta-binomial channel inert (pinned by `test_hmm_single_chain`).
    base_nb_mean = np.full((n_obs, 1), exposure, dtype=np.float64)
    total_bb_RD = np.zeros((n_obs, 1), dtype=np.float64)

    log_mu = (np.log(fixture.mean) - np.log(exposure))[:, None]
    alphas = (1.0 / fixture.dispersion)[:, None]

    # NB balanced point, so a leak into the score is visible rather than plausible.
    p_binom = np.full((n_states, 1), 0.5, dtype=np.float64)
    taus = np.full((n_states, 1), 100.0, dtype=np.float64)

    return CnasterChainInputs(
        single_X=single_X,
        lengths=np.full(n_sequences, sequence_length, dtype=int),
        base_nb_mean=base_nb_mean,
        total_bb_RD=total_bb_RD,
        log_mu=log_mu,
        alphas=alphas,
        p_binom=p_binom,
        taus=taus,
        log_startprob=np.log(fixture.dataset.initial),
        log_transmat=np.log(fixture.dataset.transition),
        log_sitewise_transmat=np.zeros(n_obs, dtype=np.float64),
    )


def cnaster_log_emission(inputs: CnasterChainInputs) -> np.ndarray:
    """`cnaster`'s per-state emission, both channels summed, `(n_states, n_obs, 1)`."""

    log_emit_rdr, log_emit_baf = (
        hmm_nophasing.compute_emission_probability_nb_betabinom(
            inputs.single_X,
            inputs.base_nb_mean,
            inputs.log_mu,
            inputs.alphas,
            inputs.total_bb_RD,
            inputs.p_binom,
            inputs.taus,
        )
    )
    emission: np.ndarray = log_emit_rdr + log_emit_baf
    return emission


def cnaster_emission(inputs: CnasterChainInputs) -> np.ndarray:
    """`cnaster_log_emission` of the one spot, shape `(n_states, n_obs)`."""
    return cnaster_log_emission(inputs)[:, :, 0]


def cnaster_lattice_arguments(
    inputs: CnasterChainInputs, emission: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    """The five arguments `cnaster`'s lattices take, at `cnaster_log_emission` unless given."""
    return (
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        cnaster_log_emission(inputs) if emission is None else emission,
        inputs.log_sitewise_transmat,
    )


def cnaster_posterior(
    inputs: CnasterChainInputs, emission: np.ndarray | None = None
) -> np.ndarray:
    """`log gamma` over the concatenated axis via `compute_copy_state_posterior`."""

    arguments = cnaster_lattice_arguments(inputs, emission)
    posterior: np.ndarray = compute_copy_state_posterior(
        hmm_nophasing.forward_lattice(*arguments),
        hmm_nophasing.backward_lattice(*arguments),
    )
    return posterior


def cnaster_total_log_likelihood(inputs: CnasterChainInputs) -> float:
    """The summed forward log-likelihood: `log alpha` at each chain's last position."""

    log_alpha = hmm_nophasing.forward_lattice(*cnaster_lattice_arguments(inputs))

    ends = np.cumsum(inputs.lengths) - 1
    return float(sum(logsumexp(log_alpha[:, end]) for end in ends))


def upstream_chain_densities(
    fixture: NegativeBinomialChains, family: Any = None
) -> list[np.ndarray]:
    """Upstream's per-state log density of each chain, under `family` or the fixture's."""
    family = fixture.family if family is None else family
    return [
        np.asarray(
            family.log_density(torch.as_tensor(row, dtype=torch.float64)), dtype=float
        )
        for row in np.asarray(fixture.dataset.observations)
    ]


def upstream_total_log_likelihood(fixture: NegativeBinomialChains) -> float:
    """The summed forward log-likelihood, from upstream's recursion."""
    density = fixture.family.log_density(
        torch.as_tensor(fixture.dataset.observations, dtype=torch.float64)
    )
    return float(
        forward_log_likelihood_from_density(
            density,
            torch.log(torch.as_tensor(fixture.dataset.initial)),
            torch.log(torch.as_tensor(fixture.dataset.transition)),
        )
    )


@dataclass(frozen=True)
class CnasterPhasedInputs:
    """`cnaster`'s phased-lattice arguments; the emission is an array because `cnaster`'s phased emission raises (issue #9)."""

    log_emission: np.ndarray
    lengths: np.ndarray
    log_startprob: np.ndarray
    log_transmat: np.ndarray
    log_sitewise_transmat: np.ndarray
    penalize_phase_only_on_same_cnv: bool

    @property
    def n_paired_states(self) -> int:
        """`2K`, the number of (copy state, phase) pairs."""
        return int(self.log_emission.shape[0])


def from_phased_chains(
    fixture: PhasedChains,
    *,
    switch: float | None = None,
) -> CnasterPhasedInputs:
    """Lay a phased fixture out for `hmm_phased.forward_lattice`; `switch` overrides the phase kernel."""

    observations = fixture.dataset.observations
    n_sequences, sequence_length = observations.shape
    n_obs = n_sequences * sequence_length

    density = fixture.family.log_density(
        torch.as_tensor(observations, dtype=torch.float64)
    )
    log_emission = density.numpy().reshape(n_obs, fixture.n_paired_states).T[:, :, None]

    effective_switch = fixture.switch if switch is None else switch

    return CnasterPhasedInputs(
        log_emission=log_emission,
        lengths=np.full(n_sequences, sequence_length, dtype=int),
        log_startprob=np.log(fixture.initial),
        log_transmat=np.log(fixture.base_transition),
        log_sitewise_transmat=np.full(n_obs, np.log(effective_switch)),
        penalize_phase_only_on_same_cnv=fixture.penalize_phase_only_on_same_cnv,
    )


def cnaster_phased_total_log_likelihood(inputs: CnasterPhasedInputs) -> float:
    """The summed forward log-likelihood from `cnaster`'s phased lattice."""

    log_alpha = hmm_phased.forward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        inputs.log_emission,
        inputs.log_sitewise_transmat,
        inputs.penalize_phase_only_on_same_cnv,
    )
    ends = np.cumsum(inputs.lengths) - 1
    return float(sum(logsumexp(log_alpha[:, end]) for end in ends))


def upstream_phased_total_log_likelihood(
    fixture: PhasedChains, inputs: CnasterPhasedInputs
) -> float:
    """The same total from upstream, with the paired start the initial halved across phases."""
    n_sequences = inputs.lengths.size
    sequence_length = int(inputs.lengths[0])
    density = torch.as_tensor(
        inputs.log_emission[:, :, 0]
        .T.reshape(n_sequences, sequence_length, fixture.n_paired_states)
        .copy()
    )
    paired_initial = 0.5 * np.concatenate([fixture.initial, fixture.initial])

    return float(
        forward_log_likelihood_from_density(
            density,
            torch.log(torch.as_tensor(paired_initial)),
            torch.log(torch.as_tensor(fixture.combined_transition)),
        )
    )


@dataclass(frozen=True)
class MStepResult:
    """One re-estimation: the fields both `Reestimate` and `OptimizationResult` carry.

    `iterations` is `-1` where unreported and never compared; `seconds` times the solve
    alone.
    """

    alpha: np.ndarray
    beta: np.ndarray
    converged: bool
    iterations: int
    seconds: float

    @property
    def success_probability(self) -> np.ndarray:
        """`alpha / (alpha + beta)`."""
        return np.asarray(self.alpha / (self.alpha + self.beta), dtype=np.float64)

    @property
    def concentration(self) -> np.ndarray:
        """`alpha + beta`."""
        return np.asarray(self.alpha + self.beta, dtype=np.float64)


def upstream_beta_binomial_m_step(
    fixture: BetaBinomialChains, posterior: np.ndarray
) -> MStepResult:
    """`BetaBinomialEmission.reestimate`, the referee, rebuilt at the planted parameters."""

    family = BetaBinomialEmission(
        trials=np.full(fixture.n_states, float(fixture.trials), dtype=np.float64),
        alpha=fixture.alpha,
        beta=fixture.beta,
    )
    observations = torch.as_tensor(np.asarray(fixture.dataset.observations))
    weights = torch.as_tensor(np.asarray(posterior), dtype=torch.float64)

    start = time.perf_counter()
    result = family.reestimate(observations, weights)
    seconds = time.perf_counter() - start

    return MStepResult(
        alpha=np.asarray(result.components.alpha, dtype=np.float64),
        beta=np.asarray(result.components.beta, dtype=np.float64),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        seconds=seconds,
    )


def cnaster_beta_binomial_design(
    fixture: BetaBinomialChains, posterior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(endog, exog, weights, exposure)` for `Weighted_BetaBinom_mix`: one row per `(observation, state)`, C order."""
    observations = np.asarray(fixture.dataset.observations).reshape(-1)
    n_obs, n_states = observations.size, fixture.n_states

    endog = np.repeat(observations.astype(np.float64), n_states)
    exog = np.tile(np.eye(n_states, dtype=np.float64), (n_obs, 1))
    weights = np.asarray(posterior, dtype=np.float64).reshape(-1)
    exposure = np.full(n_obs * n_states, float(fixture.trials), dtype=np.float64)

    return endog, exog, weights, exposure


def cnaster_beta_binomial_m_step(
    fixture: BetaBinomialChains,
    posterior: np.ndarray,
    *,
    shared_dispersion: bool = False,
) -> MStepResult:
    """`Weighted_BetaBinom_mix.fit` with `get_em_solver_params`, as the live caller drives
    it.

    `fit`'s own defaults pass `ftol=None` to `L-BFGS-B`. `shared_dispersion=False`
    matches the upstream shape.
    """

    endog, exog, weights, exposure = cnaster_beta_binomial_design(fixture, posterior)

    model = Weighted_BetaBinom_mix(
        endog, exog, weights, exposure, shared_dispersion=shared_dispersion
    )

    start = time.perf_counter()
    result = model.fit(**get_em_solver_params())
    seconds = time.perf_counter() - start

    params = np.asarray(result.params, dtype=np.float64)
    if shared_dispersion:
        params = np.concatenate(
            [params[:-1], np.full(fixture.n_states, params[-1], dtype=np.float64)]
        )

    alpha, beta = compute_bb_ab(np.eye(fixture.n_states, dtype=np.float64), params)

    return MStepResult(
        alpha=np.asarray(alpha, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        converged=bool(result.converged),
        iterations=int(result.iterations if result.iterations is not None else -1),
        seconds=seconds,
    )


def cnaster_beta_binomial_objective(
    fixture: BetaBinomialChains,
    posterior: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """`cnaster`'s weighted negative log-likelihood at `(alpha, beta)`, through its own `nloglikeobs`."""

    endog, exog, weights, exposure = cnaster_beta_binomial_design(fixture, posterior)

    model = Weighted_BetaBinom_mix(
        endog, exog, weights, exposure, shared_dispersion=False
    )
    model.zero_point = betabinom_logpmf_zp(model.endog, model.exposure)

    concentration = np.asarray(alpha, dtype=np.float64) + np.asarray(
        beta, dtype=np.float64
    )
    params = np.concatenate(
        [np.asarray(alpha, dtype=np.float64) / concentration, concentration]
    )
    return float(model.nloglikeobs(params))


@dataclass(frozen=True)
class CnasterCoreInputs:
    """A planted instance as `cnaster.hmrf.run_core_inference`'s arguments (#4, #14).

    `single_X` is `(n_obs, 2, n_spots)`: channel 0 the total, channel 1 the successes.
    """

    single_X: np.ndarray
    lengths: np.ndarray
    single_base_nb_mean: np.ndarray
    single_total_bb_RD: np.ndarray
    initial_clone_index: list[np.ndarray]
    n_states: int
    log_sitewise_transmat: np.ndarray
    smooth_mat: object
    adjacency_mat: object
    sample_ids: np.ndarray

    def as_kwargs(self) -> dict[str, object]:
        """The keyword form `run_core_inference` takes, so a caller adds only knobs."""
        return {
            "single_X": self.single_X,
            "lengths": self.lengths,
            "single_base_nb_mean": self.single_base_nb_mean,
            "single_total_bb_RD": self.single_total_bb_RD,
            "single_tumor_prop": None,
            "initial_clone_index": self.initial_clone_index,
            "n_states": self.n_states,
            "log_sitewise_transmat": self.log_sitewise_transmat,
            "smooth_mat": self.smooth_mat,
            "adjacency_mat": self.adjacency_mat,
            "sample_ids": self.sample_ids,
        }


def cnaster_valid_counts(
    base: np.ndarray, total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`cnaster.hmrf:262-263`, verbatim: each spot's bins with read depth, and with alleles."""
    return (base > 0).sum(axis=0), (total > 0).sum(axis=0)


def cnaster_adjacency_triple(
    matrix: "csr_matrix",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`cnaster`'s `unpack_adjacency(cast_csr(matrix))`: spots, neighbours, weights."""

    spots, neighbors, weights = unpack_adjacency(cast_csr(matrix))
    return spots, neighbors, weights


def square_coords(rows: int, columns: int) -> np.ndarray:
    """`(row, column)` of each spot of a `rows x columns` lattice, row-major."""
    return np.stack(
        np.unravel_index(np.arange(rows * columns), (rows, columns)), axis=1
    )


def lattice_adjacency(lattice: tuple[int, int]) -> "csr_matrix":
    """Four-neighbour symmetric CSR adjacency with unit weights; `spatial_weight` stays the caller's (#44)."""

    rows, columns = lattice
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if column + 1 < columns:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + columns))

    source = np.array([edge[0] for edge in edges] + [edge[1] for edge in edges])
    target = np.array([edge[1] for edge in edges] + [edge[0] for edge in edges])
    weight = np.ones(source.size)

    return coo_matrix(
        (weight, (source, target)), shape=(rows * columns, rows * columns)
    ).tocsr()


def from_core_inference_truth(truth: CoreInferenceTruth) -> CnasterCoreInputs:
    """Convert a `CoreInferenceTruth` with counts, exposure and trial count handed over as drawn."""

    single_X = np.stack([truth.counts_nb, truth.counts_bb], axis=1)

    return CnasterCoreInputs(
        single_X=single_X,
        lengths=truth.lengths,
        single_base_nb_mean=truth.base_nb_mean,
        single_total_bb_RD=truth.total_bb_RD,
        initial_clone_index=truth.clone_index,
        n_states=truth.n_states,
        log_sitewise_transmat=np.log(truth.switch_prob),
        # Identity smoothing, so the planted truth describes the draw rather than pooled counts.
        smooth_mat=sparse_eye(truth.n_spots, format="csr"),
        adjacency_mat=lattice_adjacency(truth.lattice),
        sample_ids=np.zeros(truth.n_spots, dtype=int),
    )


def cnaster_potts_adjacency(fixture: PottsLabels) -> "csr_matrix":
    """The fixture's graph as `cnaster`'s symmetric CSR, both directions of every edge
    present.

    `calc_assignment_cost` halves its pairwise term to compensate; `spatial_weight`
    stays outside.
    """

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (left, right), j in zip(
        fixture.graph.edges, fixture.graph.coupling, strict=True
    ):
        rows.extend((left, right))
        cols.extend((right, left))
        data.extend((j, j))

    n_nodes = fixture.n_nodes
    matrix = coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    matrix.sort_indices()
    return matrix


def cnaster_potts_coo(
    fixture: PottsLabels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(adj_spots, adj_neighbors, adj_weights)` for `calc_assignment_cost`, derived from the CSR."""

    matrix = cnaster_potts_adjacency(fixture).tocoo()
    return (
        np.asarray(matrix.row, dtype=np.int64),
        np.asarray(matrix.col, dtype=np.int64),
        np.asarray(matrix.data, dtype=np.float64),
    )


def cnaster_assignment_cost(fixture: PottsLabels, labelling: np.ndarray) -> float:
    """`cnaster`'s maximised objective at a labelling, through its own `calc_assignment_cost`."""

    spots, neighbors, weights = cnaster_potts_coo(fixture)

    return float(
        calc_assignment_cost(
            fixture.field,
            spots,
            neighbors,
            weights,
            np.asarray(labelling, dtype=np.int64),
            fixture.spatial_weight,
        )
    )


def upstream_potts_energy(fixture: PottsLabels, labelling: np.ndarray) -> float:
    """Upstream's `energy` at a labelling with `spatial_weight` folded in; the negation of `cnaster`'s cost."""

    return float(
        energy(
            scaled_graph(fixture),
            fixture.field,
            np.asarray(labelling, dtype=np.int64),
        )
    )


def cnaster_sweep(
    field: np.ndarray,
    graph: "csr_matrix",
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    seed: int,
    posterior: np.ndarray | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, int, float]:
    """The fifteen-argument call on a copy of `assignment`, as `hmrf.py:307` makes it."""

    # NB `icm_sweep_deque` calls `np.random.shuffle`, so only the legacy global seed reaches it.
    np.random.seed(seed)  # noqa: NPY002
    labels = assignment.copy()

    niter, cost = icm_sweep_deque(
        single_llf=field,
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.data,
        new_assignment=labels,
        spatial_weight=spatial_weight,
        posterior=posterior,
        **{"min_clone_spots": 0} | kwargs,
    )
    return labels, int(niter), float(cost)


def cnaster_icm_labelling(
    fixture: PottsLabels,
    start: np.ndarray,
    *,
    min_clone_spots: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Run `icm_sweep_deque` on a copy of `start`, seeding the global RNG; returns
    `(labelling, cost, iterations)`.

    `min_clone_spots` defaults to 0, not 200, so the occupancy guard stays slack (as in
    #8).
    """
    assignment, iterations, cost = cnaster_sweep(
        fixture.field,
        cnaster_potts_adjacency(fixture),
        np.asarray(start, dtype=np.int64),
        fixture.spatial_weight,
        seed=seed,
        min_clone_spots=min_clone_spots,
    )
    return assignment, cost, iterations


def upstream_icm(fixture: PottsLabels, graph: Any = None, seed: int = 0) -> Any:
    """`sal`'s ICM on `fixture`'s field from `seed`, over `graph` or the fixture's own."""

    graph = scaled_graph(fixture) if graph is None else graph
    return iterated_conditional_modes(graph, fixture.field, np.random.default_rng(seed))


def upstream_expansion(
    fixture: PottsLabels, graph: Any = None, start: np.ndarray | None = None
) -> "ExpansionResult":
    """`sal`'s alpha expansion on `fixture`'s field, over `graph` or the fixture's own."""

    # NB PYTHON was the default before e0aeb19 made it RUST (#410).
    return alpha_expansion(
        scaled_graph(fixture) if graph is None else graph,
        fixture.field,
        start=None if start is None else np.asarray(start, dtype=np.int64),
        backend=Backend.PYTHON,
    )


def cnaster_initial_phase(truth: CoreInferenceTruth, blocks: Any, t: float) -> Any:
    """`run_cnaster:360`'s `initial_phase_given_partition` call on `blocks`, at `t`."""

    return initial_phase_given_partition(
        blocks.X,
        blocks.lengths,
        # NB BAF only: `run_cnaster` passes a zero `(n_obs, n_spots)` exposure here too.
        np.zeros_like(blocks.total_bb_RD),
        blocks.total_bb_RD,
        None,
        # NB one clone over every spot would wash out the imbalance the vote reads.
        truth.clone_index,
        truth.n_states,
        get_log_transmat(truth.n_states, t),
        np.zeros(blocks.X.shape[0]),
        "sp",
        t,
        0,
        fix_NB_dispersion=False,
        shared_NB_dispersion=True,
        fix_BB_dispersion=False,
        shared_BB_dispersion=True,
        max_iter=100,
        tol=1e-3,
        threshold=0.5,
    )


def range_filter_loop(unique_snp_ids: np.ndarray, ranges: Any) -> np.ndarray:
    """`cnaster.io.load_input_data`'s range filter (`io.py` lines 740-772), transcribed verbatim."""
    num_ranges = ranges.shape[0]
    indicator_filter = np.array([True] * len(unique_snp_ids))
    j = 0

    for i in range(len(unique_snp_ids)):
        this_chr = int(unique_snp_ids[i].split("_")[0])
        this_pos = int(unique_snp_ids[i].split("_")[1])

        while j < num_ranges and (
            (ranges.Chr.to_numpy()[j] < this_chr)
            or (
                (ranges.Chr.to_numpy()[j] == this_chr)
                and (ranges.End.to_numpy()[j] <= this_pos)
            )
        ):
            j += 1

        if (
            j < num_ranges
            and (ranges.Chr.to_numpy()[j] == this_chr)
            and (ranges.Start.to_numpy()[j] <= this_pos)
            and (ranges.End.to_numpy()[j] > this_pos)
        ):
            indicator_filter[i] = False

    return indicator_filter


def drawn(figure: Any, *, colours: bool = True) -> list[np.ndarray]:
    """Every scatter point (with face colours unless `colours` is off) and segment a figure drew, in order."""
    out: list[np.ndarray] = []

    for axis in figure.axes:
        for collection in axis.collections:
            offsets = np.asarray(collection.get_offsets())

            if offsets.size:
                out.append(offsets)
                if colours:
                    out.append(np.asarray(collection.get_facecolors()))

            segments = getattr(collection, "get_segments", None)

            if segments is not None:
                out.extend(np.asarray(segment) for segment in segments())

    return out


def grid_adjacency(n_spots: int, width: int) -> Any:
    """A four-neighbour grid, as `construct_multislice_lattice_adjacency` builds."""

    rows: list[int] = []
    columns: list[int] = []

    for spot in range(n_spots):
        row, column = divmod(spot, width)

        for neighbour_row, neighbour_column in (
            (row, column + 1),
            (row + 1, column),
        ):
            neighbour = neighbour_row * width + neighbour_column

            if neighbour_column < width and neighbour < n_spots:
                rows.extend((spot, neighbour))
                columns.extend((neighbour, spot))

    data = np.ones(len(rows))

    return coo_matrix((data, (rows, columns)), shape=(n_spots, n_spots)).tocsr()


CLONE_ASSIGNMENT_POSITIONAL = ("single_X", "single_base_nb_mean", "single_total_bb_RD", "res", "pred",
                               "adjacency_mat", "prev_assignment", "sample_ids", "spatial_weight")  # fmt: skip
"""`pipeline_clone_assignment`'s positional arguments, in order."""


def clone_assignment_call(
    function: Any, arguments: dict[str, Any], **keywords: Any
) -> Any:
    """`function` called as `run_core_inference` calls it, on a copy of `prev_assignment`."""

    given = arguments | {"prev_assignment": arguments["prev_assignment"].copy()}
    given |= {k: keywords.pop(k) for k in CLONE_ASSIGNMENT_POSITIONAL if k in keywords}
    positional = (given[name] for name in CLONE_ASSIGNMENT_POSITIONAL)
    return function(*positional, **{"hmmclass": hmm_nophasing} | keywords)


def clone_assignment_arguments(fixture: SpotCloneField, width: int) -> dict[str, Any]:
    """`pipeline_clone_assignment`'s arguments at `(n_states, 1)`, built once so two arms cannot differ (#278)."""
    n_obs, n_spots = fixture.counts_nb.shape

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = fixture.counts_nb
    single_X[:, 1, :] = fixture.counts_bb

    generator = np.random.default_rng(fixture.seed)

    return {
        "single_X": single_X,
        "single_base_nb_mean": fixture.base_nb_mean,
        "single_total_bb_RD": fixture.total_bb_RD,
        "res": {
            "new_log_mu": fixture.log_mu.reshape(-1, 1),
            "new_alphas": fixture.alphas.reshape(-1, 1),
            "new_p_binom": fixture.p_binom.reshape(-1, 1),
            "new_taus": fixture.taus.reshape(-1, 1),
        },
        "pred": fixture.pred.T.reshape(-1),
        "adjacency_mat": grid_adjacency(n_spots, width),
        "prev_assignment": generator.integers(0, fixture.n_clones, size=n_spots).astype(
            np.int64
        ),
        "sample_ids": np.zeros(n_spots, dtype=np.int64),
        "spatial_weight": 1.5,
    }


@dataclass(frozen=True)
class Stacked:
    """The pseudobulk, clone-stacked as `cnaster`'s `clone_stack_obs` stacks it."""

    X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    lengths: np.ndarray
    sitewise: np.ndarray


def stacked_clones(truth: CoreInferenceTruth) -> Stacked:
    """Aggregate to pseudobulk and stack the clones along the genomic axis."""

    counts = np.stack([truth.counts_nb, truth.counts_bb], axis=1)
    X, base, total, _ = merge_pseudobulk_by_index_mix(
        counts, truth.base_nb_mean, truth.total_bb_RD, truth.clone_index
    )
    stack_X, stack_base, stack_total, lengths, sitewise, _ = clone_stack_obs(
        X, base, total, truth.lengths, np.zeros((truth.n_obs, 2)), None
    )
    return Stacked(stack_X, stack_base, stack_total, lengths, sitewise)
