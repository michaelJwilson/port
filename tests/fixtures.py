"""Seeded fixtures for port's tests, built on `snakes_and_ladders` simulators.

Truth is planted, never recovered from the data; each builder returns its seed.
"""

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import cnaster.hmm_nophasing as upstream
import numpy as np
import pandas as pd
import pytest
import torch
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import hmm_nophasing
from cnaster.hmrf import compute_loglike_spot_assignment, run_core_inference
from port.patch.hmm_nophasing import hmm_nophasing as port_hmm_nophasing
from port.patch.hmrf.fused_field import fused_spot_clone_field
from port.patch.hmrf.tabulated_field import tabulated_spot_clone_field
from port.patch.icm.interface import CsrGraph
from port.pipeline import patched, with_attributes
from port.sim.draw import (
    DrawManifest,
    extended,
    from_document,
    merged_tables,
)
from port.sim.fixtures import SIM_ROOT
from port.sim.truth import (
    DEFAULT_SEED,
    CoreInferenceTruth,
    core_inference_truth,
)
from sal.emissions import (
    BetaBinomialEmission,
    NegativeBinomialEmission,
)
from sal.enumeration import enumerated_optimum
from sal.sim.graph import BoundaryCondition, PottsGraph, lattice_graph
from sal.sim.hmm import (
    HmmParams,
    SimulatedHmmDataset,
    simulate_sequences,
)
from sal.sim.potts import energy
from scipy.sparse import eye as sparse_eye

from tests.builders import allele_counts


def tiers(gate: object, stress: object) -> list[Any]:
    """A benchmark's two sizes: `gate`, and `stress` under `release`."""
    return [
        pytest.param(gate, id="gate"),
        pytest.param(stress, id="stress", marks=pytest.mark.release),
    ]


@dataclass(frozen=True)
class NegativeBinomialChains:
    """Independent negative binomial chains with the parameters that drew them."""

    dataset: SimulatedHmmDataset
    family: NegativeBinomialEmission
    mean: np.ndarray
    dispersion: np.ndarray
    seed: int

    @property
    def n_states(self) -> int:
        """`K`, the number of hidden states."""
        return int(self.mean.shape[0])

    @property
    def n_sequences(self) -> int:
        """The number of independent chains."""
        return int(self.dataset.observations.shape[0])

    @property
    def sequence_length(self) -> int:
        """`L`, the length of every chain."""
        return int(self.dataset.observations.shape[1])


def negative_binomial_chains(
    *,
    n_states: int = 3,
    sequence_length: int = 60,
    n_sequences: int = 4,
    separation: float = 2.5,
    base_mean: float = 10.0,
    dispersion: float = 8.0,
    self_transition: float = 0.8,
    drift: float = 0.5,
    seed: int = DEFAULT_SEED,
) -> NegativeBinomialChains:
    """Draw `n_sequences` chains of length `sequence_length`.

    `separation` is the ratio between adjacent state means; `drift` splits
    off-diagonal mass between higher and lower states, symmetric (cnaster's
    circulant) at 0.5. Raises `ValueError` if `separation < 1` or
    `self_transition` is outside `(0, 1)`.
    """
    if separation < 1.0:
        msg = f"separation must be at least one, got {separation}"
        raise ValueError(msg)
    if not 0.0 < self_transition < 1.0:
        msg = f"self_transition must lie strictly in (0, 1), got {self_transition}"
        raise ValueError(msg)
    if not 0.0 < drift < 1.0:
        msg = f"drift must lie strictly in (0, 1), got {drift}"
        raise ValueError(msg)

    mean = base_mean * separation ** np.arange(n_states, dtype=np.float64)
    dispersions = np.full(n_states, dispersion, dtype=np.float64)
    family = NegativeBinomialEmission(dispersion=dispersions, mean=mean)

    transition = drift_transition(n_states, self_transition, drift)
    initial = np.full(n_states, 1.0 / n_states, dtype=np.float64)

    dataset = simulate_sequences(
        HmmParams(
            n_states=n_states,
            lengths=(sequence_length,) * n_sequences,
            initial=initial,
            transition=transition,
            emissions=family,
            seed=seed,
            tolerance=1e-8,
        )
    )
    return NegativeBinomialChains(
        dataset=dataset,
        family=family,
        mean=mean,
        dispersion=dispersions,
        seed=seed,
    )


def circulant_transition(n_states: int, self_transition: float) -> np.ndarray:
    """`t` on the diagonal, the remainder spread evenly; pinned equal to cnaster's `get_log_transmat`."""
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)

    off = (1.0 - self_transition) / (n_states - 1)
    transition = np.full((n_states, n_states), off, dtype=np.float64)
    np.fill_diagonal(transition, self_transition)
    return transition


def drift_transition(n_states: int, self_transition: float, drift: float) -> np.ndarray:
    """`t` on the diagonal, off-diagonal mass split by direction; `drift = 0.5` is
    circulant.

        Asymmetry makes a transposed transition observable.
    """
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)

    transition = np.zeros((n_states, n_states), dtype=np.float64)
    off_mass = 1.0 - self_transition

    for state in range(n_states):
        higher = np.arange(state + 1, n_states)
        lower = np.arange(state)

        weight_up = drift * higher.size
        weight_down = (1.0 - drift) * lower.size
        total_weight = weight_up + weight_down

        if higher.size:
            transition[state, higher] = (
                off_mass * weight_up / total_weight / higher.size
            )
        if lower.size:
            transition[state, lower] = (
                off_mass * weight_down / total_weight / lower.size
            )

        transition[state, state] = self_transition

    return transition


@dataclass(frozen=True)
class PhasedChains:
    """Chains over a paired `2K` (copy state, phase) space with a constant phase kernel.

    The emission is not mirrored across phase, so a mis-blocked transition is
    observable.
    """

    dataset: SimulatedHmmDataset
    family: NegativeBinomialEmission
    base_transition: np.ndarray
    combined_transition: np.ndarray
    initial: np.ndarray
    switch: float
    penalize_phase_only_on_same_cnv: bool
    seed: int

    @property
    def n_copy_states(self) -> int:
        """`K`, the number of copy states."""
        return int(self.base_transition.shape[0])

    @property
    def n_paired_states(self) -> int:
        """`2K`, the number of (copy state, phase) pairs."""
        return 2 * self.n_copy_states

    @property
    def n_sequences(self) -> int:
        """The number of independent chains."""
        return int(self.dataset.observations.shape[0])

    @property
    def sequence_length(self) -> int:
        """`L`, the length of every chain."""
        return int(self.dataset.observations.shape[1])


def phased_combined_transition(
    base_transition: np.ndarray,
    switch: float,
    *,
    penalize_phase_only_on_same_cnv: bool,
) -> np.ndarray:
    """Assemble the `2K x 2K` transfer matrix from a `K x K` base and a phase kernel.

    Independent of cnaster's `update_combined_transmat`; a test pins them
    equal. Raises `ValueError` if `switch` is outside `(0, 1)`.
    """
    if not 0.0 < switch < 1.0:
        msg = f"switch must lie strictly in (0, 1), got {switch}"
        raise ValueError(msg)

    n_copy_states = base_transition.shape[0]
    stay = 1.0 - switch

    if not penalize_phase_only_on_same_cnv:
        kernel = np.array([[stay, switch], [switch, stay]], dtype=np.float64)
        return np.kron(kernel, base_transition)

    combined = 0.5 * np.tile(base_transition, (2, 2))
    for state in range(n_copy_states):
        conserved = base_transition[state, state]
        other = state + n_copy_states
        combined[state, state] = stay * conserved
        combined[state, other] = switch * conserved
        combined[other, state] = switch * conserved
        combined[other, other] = stay * conserved
    return combined


def phased_chains(
    *,
    n_copy_states: int = 3,
    sequence_length: int = 50,
    n_sequences: int = 3,
    separation: float = 1.6,
    base_mean: float = 10.0,
    dispersion: float = 8.0,
    self_transition: float = 0.8,
    drift: float = 0.3,
    switch: float = 0.15,
    penalize_phase_only_on_same_cnv: bool = False,
    seed: int = DEFAULT_SEED,
) -> PhasedChains:
    """Draw chains over the `2K` paired space; `drift` defaults off 0.5 so the base is asymmetric."""
    base_transition = drift_transition(n_copy_states, self_transition, drift)
    combined = phased_combined_transition(
        base_transition,
        switch,
        penalize_phase_only_on_same_cnv=penalize_phase_only_on_same_cnv,
    )

    n_paired = 2 * n_copy_states
    mean = base_mean * separation ** np.arange(n_paired, dtype=np.float64)
    dispersions = np.full(n_paired, dispersion, dtype=np.float64)
    family = NegativeBinomialEmission(dispersion=dispersions, mean=mean)

    initial = np.full(n_copy_states, 1.0 / n_copy_states, dtype=np.float64)
    paired_initial = 0.5 * np.concatenate([initial, initial])

    dataset = simulate_sequences(
        HmmParams(
            n_states=n_paired,
            lengths=(sequence_length,) * n_sequences,
            initial=paired_initial,
            transition=combined,
            emissions=family,
            seed=seed,
            tolerance=1e-8,
        )
    )
    return PhasedChains(
        dataset=dataset,
        family=family,
        base_transition=base_transition,
        combined_transition=combined,
        initial=initial,
        switch=switch,
        penalize_phase_only_on_same_cnv=penalize_phase_only_on_same_cnv,
        seed=seed,
    )


@dataclass(frozen=True)
class BetaBinomialChains:
    """Beta-binomial chains at a constant number of trials, with their truth (#24)."""

    dataset: SimulatedHmmDataset
    family: BetaBinomialEmission
    alpha: np.ndarray
    beta: np.ndarray
    trials: int
    seed: int

    @property
    def n_states(self) -> int:
        """`K`, the number of hidden states."""
        return int(self.alpha.shape[0])

    @property
    def success_probability(self) -> np.ndarray:
        """`alpha / (alpha + beta)`, cnaster's `p`."""
        return np.asarray(self.alpha / (self.alpha + self.beta), dtype=np.float64)

    @property
    def concentration(self) -> np.ndarray:
        """`alpha + beta`, `cnaster`'s `tau`."""
        return np.asarray(self.alpha + self.beta, dtype=np.float64)


def beta_binomial_chains(
    *,
    n_states: int = 3,
    sequence_length: int = 400,
    n_sequences: int = 6,
    trials: int = 40,
    success_probability: np.ndarray | None = None,
    concentration: float = 16.0,
    self_transition: float = 0.7,
    drift: float = 0.3,
    seed: int = DEFAULT_SEED,
) -> BetaBinomialChains:
    """Draw `n_sequences` chains of beta-binomial counts.

    Raises `ValueError` if `concentration` or `trials` is not positive, or
    `success_probability` leaves `(0, 1)` or has the wrong length.
    """
    if concentration <= 0.0:
        msg = f"concentration must be positive, got {concentration}"
        raise ValueError(msg)
    if trials <= 0:
        msg = f"trials must be positive, got {trials}"
        raise ValueError(msg)

    if success_probability is None:
        success_probability = (1.0 + np.arange(n_states, dtype=np.float64)) / (
            n_states + 1.0
        )
    success_probability = np.asarray(success_probability, dtype=np.float64)

    if success_probability.shape != (n_states,):
        msg = (
            f"success_probability must have shape ({n_states},), "
            f"got {success_probability.shape}"
        )
        raise ValueError(msg)
    if not np.all((success_probability > 0.0) & (success_probability < 1.0)):
        msg = f"success_probability must lie strictly in (0, 1), got {success_probability}"
        raise ValueError(msg)

    alpha = success_probability * concentration
    beta = (1.0 - success_probability) * concentration
    family = BetaBinomialEmission(
        trials=np.full(n_states, float(trials), dtype=np.float64),
        alpha=alpha,
        beta=beta,
    )

    transition = drift_transition(n_states, self_transition, drift)
    initial = np.full(n_states, 1.0 / n_states, dtype=np.float64)

    dataset = simulate_sequences(
        HmmParams(
            n_states=n_states,
            lengths=(sequence_length,) * n_sequences,
            initial=initial,
            transition=transition,
            emissions=family,
            seed=seed,
            tolerance=1e-8,
        )
    )
    return BetaBinomialChains(
        dataset=dataset,
        family=family,
        alpha=alpha,
        beta=beta,
        trials=trials,
        seed=seed,
    )


def planted_posterior(
    fixture: BetaBinomialChains, *, smoothing: float = 0.0
) -> np.ndarray:
    """The planted posterior an M step is handed, shape `(n_sequences, sequence_length,
    n_states)`.

        `smoothing = 0` is the truth's indicator; above zero it mixes with the
        uniform. Raises `ValueError` unless `smoothing` is in `[0, 1)`.
    """
    if not 0.0 <= smoothing < 1.0:
        msg = f"smoothing must lie in [0, 1), got {smoothing}"
        raise ValueError(msg)

    states = np.asarray(fixture.dataset.states)
    n_states = fixture.n_states

    indicator = np.zeros((*states.shape, n_states), dtype=np.float64)
    np.put_along_axis(indicator, states[..., None], 1.0, axis=-1)

    return (1.0 - smoothing) * indicator + smoothing / n_states


@dataclass(frozen=True)
class SpotCloneField:
    """Inputs to cnaster's `compute_loglike_spot_assignment` from a known model (#59).

    Emissions are `(n_states, n_obs, n_spots)` log-densities; `pred` is drawn
    from a Markov chain so `segments` controls the profile's fragmentation.
    """

    log_emission_rdr: np.ndarray
    log_emission_baf: np.ndarray
    pred: np.ndarray
    segments: int
    seed: int
    counts_nb: np.ndarray
    base_nb_mean: np.ndarray
    counts_bb: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray

    @property
    def n_states(self) -> int:
        """`K`."""
        return int(self.log_emission_rdr.shape[0])

    @property
    def emission_gigabytes(self) -> float:
        """Both emission channels together."""
        return 2.0 * self.log_emission_rdr.nbytes / 1e9

    @property
    def n_obs(self) -> int:
        """`G`, genomic bins."""
        return int(self.log_emission_rdr.shape[1])

    @property
    def n_spots(self) -> int:
        """`N`."""
        return int(self.log_emission_rdr.shape[2])

    @property
    def n_clones(self) -> int:
        """`M`."""
        return int(self.pred.shape[1])

    @property
    def megabytes(self) -> float:
        """One emission channel's footprint."""
        return self.log_emission_rdr.nbytes / 1e6

    def spot_major(self) -> tuple[np.ndarray, np.ndarray]:
        """The same emissions as `(n_states, n_spots, n_obs)`, contiguous; kept for reproducing #59."""
        return (
            np.ascontiguousarray(self.log_emission_rdr.transpose(0, 2, 1)),
            np.ascontiguousarray(self.log_emission_baf.transpose(0, 2, 1)),
        )


def spot_clone_field(
    *,
    n_states: int = 5,
    n_obs: int = 240,
    n_spots: int = 160,
    n_clones: int = 3,
    self_transition: float = 0.99,
    trials: int = 30,
    seed: int = DEFAULT_SEED,
) -> SpotCloneField:
    """Build the boundary's inputs; `self_transition` sets `segments`.

    Raises `ValueError` if `n_clones > n_states` or any extent is below one.
    """
    if n_clones < 1 or n_states < 1 or n_obs < 1 or n_spots < 1:
        msg = "every extent must be at least one"
        raise ValueError(msg)

    profiles = negative_binomial_chains(
        n_states=n_states,
        sequence_length=n_obs,
        n_sequences=n_clones,
        self_transition=self_transition,
        seed=seed,
    )
    pred = np.ascontiguousarray(np.asarray(profiles.dataset.states, dtype=np.int64).T)

    # NB observations drawn from the family, so the emission is a log-density.
    counts = negative_binomial_chains(
        n_states=n_states,
        sequence_length=n_obs,
        n_sequences=n_spots,
        self_transition=self_transition,
        seed=seed + 1,
    )
    observations = np.asarray(counts.dataset.observations, dtype=np.int64).T

    allele = beta_binomial_chains(
        n_states=n_states,
        sequence_length=n_obs,
        n_sequences=n_spots,
        trials=trials,
        seed=seed + 2,
    )
    successes = np.asarray(allele.dataset.observations, dtype=np.int64).T

    rdr = np.ascontiguousarray(
        np.asarray(
            profiles.family.log_density(torch.as_tensor(observations)), dtype=np.float64
        ).transpose(2, 0, 1)
    )
    baf = np.ascontiguousarray(
        np.asarray(
            allele.family.log_density(torch.as_tensor(successes)), dtype=np.float64
        ).transpose(2, 0, 1)
    )

    return SpotCloneField(
        log_emission_rdr=rdr,
        log_emission_baf=baf,
        pred=pred,
        segments=int(1 + np.sum(pred[1:, 0] != pred[:-1, 0])),
        seed=seed,
        counts_nb=observations.astype(np.float64),
        base_nb_mean=np.ones_like(observations, dtype=np.float64),
        counts_bb=successes.astype(np.float64),
        total_bb_RD=np.full(successes.shape, float(trials), dtype=np.float64),
        log_mu=np.log(profiles.mean),
        alphas=1.0 / profiles.dispersion,
        p_binom=allele.success_probability,
        taus=allele.concentration,
    )


@dataclass(frozen=True)
class PottsLabels:
    """A Potts labelling problem with its planted labelling (#40).

    `labels` is what the field was drawn around, not the optimum.
    """

    graph: PottsGraph
    labels: np.ndarray
    field: np.ndarray
    coupling: float
    spatial_weight: float
    shape: tuple[int, ...]
    seed: int

    @property
    def n_nodes(self) -> int:
        """The number of sites."""
        return int(self.field.shape[0])

    @property
    def n_clones(self) -> int:
        """The number of labels."""
        return int(self.field.shape[1])

    @property
    def edge_coupling(self) -> float:
        """`spatial_weight * J`, the product both implementations score with."""
        return self.spatial_weight * self.coupling


def potts_labels(
    *,
    shape: tuple[int, ...] = (6, 6),
    n_clones: int = 3,
    coupling: float = 1.0,
    spatial_weight: float = 1.0,
    signal: float = 2.0,
    noise: float = 1.0,
    boundary: BoundaryCondition | None = None,
    seed: int = DEFAULT_SEED,
) -> PottsLabels:
    """Plant contiguous label domains on a lattice and draw a Gaussian field around
    them.

        Contiguous slabs, since single-site descent fails on large regions. Raises
        `ValueError` if `n_clones < 2`, `noise < 0`, or the lattice has fewer
        sites than clones.
    """

    if n_clones < 2:
        msg = f"n_clones must be at least two, got {n_clones}"
        raise ValueError(msg)
    if noise < 0.0:
        msg = f"noise must not be negative, got {noise}"
        raise ValueError(msg)

    n_nodes = int(np.prod(shape))
    if n_nodes < n_clones:
        msg = f"shape {shape} has {n_nodes} sites, fewer than n_clones={n_clones}"
        raise ValueError(msg)

    if boundary is None:
        boundary = BoundaryCondition.OPEN

    graph = lattice_graph(tuple(shape), boundary, coupling)

    # NB contiguous slabs along the first axis, sized as evenly as the extent
    #    allows, so every clone is present and the domains are large.
    first_axis = shape[0]
    per_node = n_nodes // first_axis
    slab_of_row = np.floor_divide(np.arange(first_axis) * n_clones, first_axis)
    labels = np.repeat(slab_of_row, per_node).astype(np.int64)

    rng = np.random.default_rng(seed)
    field = rng.normal(loc=0.0, scale=noise, size=(n_nodes, n_clones))
    field[np.arange(n_nodes), labels] += signal

    return PottsLabels(
        graph=graph,
        labels=labels,
        field=field,
        coupling=coupling,
        spatial_weight=spatial_weight,
        shape=tuple(shape),
        seed=seed,
    )


def enumerate_minimum_energy(fixture: PottsLabels) -> tuple[np.ndarray, float]:
    """The exact Potts energy minimiser by exhaustive search; returns the labelling and its energy."""

    graph = scaled_graph(fixture)

    def negated(configuration: tuple[int, ...]) -> float:
        labelling = np.array(configuration[::-1], dtype=np.int64)
        return -energy(graph, fixture.field, labelling)

    optimum, best = enumerated_optimum(fixture.n_clones, fixture.n_nodes, negated)
    return np.array(optimum[::-1], dtype=np.int64), -best


def scaled_graph(fixture: PottsLabels) -> PottsGraph:
    """The fixture's graph with `spatial_weight` folded into the coupling, once."""

    return PottsGraph(
        n_nodes=fixture.graph.n_nodes,
        edges=fixture.graph.edges,
        coupling=tuple(fixture.spatial_weight * j for j in fixture.graph.coupling),
    )


def synthetic_ranges(
    n_snps: int, n_ranges: int, seed: int = 11
) -> tuple[np.ndarray, Any]:
    """Sorted SNP ids in cnaster's `{chr}_{pos}_{ref}_{alt}` text form, and filter ranges."""

    rng = np.random.default_rng(seed)

    chromosomes = rng.integers(1, 23, n_snps)
    positions = rng.integers(0, 250_000_000, n_snps)
    order = np.lexsort((positions, chromosomes))
    snp_ids = np.array(
        [f"{chromosomes[k]}_{positions[k]}_A_T" for k in order], dtype=object
    )

    range_chromosomes = rng.integers(1, 23, n_ranges)
    range_starts = rng.integers(0, 250_000_000, n_ranges)
    range_order = np.lexsort((range_starts, range_chromosomes))

    ranges = pd.DataFrame(
        {
            "Chr": range_chromosomes[range_order],
            "Start": range_starts[range_order],
            "End": range_starts[range_order] + 2_000_000,
        }
    )

    return snp_ids, ranges


def genomic_plot_instance(
    seed: int = 17, n_states: int = 4, n_obs: int = 24, n_spots: int = 9
) -> dict[str, Any]:
    """`plot_clones_genomic`'s arguments and a fit result: 24 bins, 9 spots, 3 clones."""

    rng = np.random.default_rng(seed)
    n_clones = 3
    X, _, total = allele_counts(rng, (n_obs, n_spots), (20, 80), 0.45, 150)

    return {
        "arguments": (
            np.array([n_obs]),
            X,
            rng.uniform(100.0, 200.0, size=(n_obs, n_spots)),
            total,
        ),
        "result": {
            "new_assignment": np.tile(np.arange(n_clones), n_spots // n_clones),
            "pred_cnv": rng.integers(0, n_states, size=(n_obs, n_clones)),
            "new_log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
            "new_p_binom": rng.uniform(0.15, 0.85, size=(n_states, 1)),
        },
        "rng": rng,
    }


def integer_copies(rng: np.random.Generator, n_obs: int, n_clones: int) -> Any:
    """A `df_cnv` of major and minor copies per clone, the first four bins diploid."""

    frame: dict[str, np.ndarray] = {"CHR": np.ones(n_obs, dtype=int)}

    for clone in range(n_clones):
        major = rng.integers(1, 4, size=n_obs)
        minor = rng.integers(0, 2, size=n_obs)
        major[:4], minor[:4] = 1, 1
        frame[f"clone{clone} A"] = major
        frame[f"clone{clone} B"] = minor

    return pd.DataFrame(frame)


def fused_field_arguments(
    fixture: SpotCloneField, weight: np.ndarray
) -> tuple[np.ndarray, ...]:
    """The fused and tabulated field kernels' arguments on `fixture`, but `out`."""
    return (
        fixture.counts_nb,
        fixture.base_nb_mean,
        fixture.counts_bb,
        fixture.total_bb_RD,
        fixture.log_mu,
        fixture.alphas,
        fixture.p_binom,
        fixture.taus,
        fixture.pred,
        weight,
    )


def fused_field_of(
    fixture: SpotCloneField, weight: np.ndarray, *, tabulated: bool = False
) -> np.ndarray:
    """`port`'s fused (or tabulated) spot-by-clone field on `fixture`, weighted by `weight`."""

    # NB `Any`: `numba` dispatchers, whose stubs take no positional unpacking.
    kernel: Any = tabulated_spot_clone_field if tabulated else fused_spot_clone_field
    field: np.ndarray = kernel(
        *fused_field_arguments(fixture, weight),
        np.empty((fixture.n_spots, fixture.n_clones)),
    )
    return field


def two_step_field_of(
    fixture: SpotCloneField,
    valid_nb: np.ndarray,
    valid_bb: np.ndarray,
    kernel: Any,
    *,
    smooth: bool = True,
    swaps: Any = None,
) -> np.ndarray:
    """`cnaster`'s dense producer (under `swaps`), then `kernel`'s field: what the fused kernel replaces."""

    # NB `_dense_*_logpmf` indexes `[i, 0]`; the fused kernel takes `(n_states,)` (#278).
    with nullcontext() if swaps is None else patched(swaps):
        rdr = upstream._dense_nb_logpmf(
            fixture.counts_nb,
            fixture.base_nb_mean,
            fixture.log_mu[:, None],
            fixture.alphas[:, None],
        )
        baf = upstream._dense_bb_logpmf(
            fixture.counts_bb,
            fixture.total_bb_RD,
            fixture.p_binom[:, None],
            fixture.taus[:, None],
        )

    neighbourhood: dict[str, np.ndarray] = {}
    if smooth:
        identity = sparse_eye(fixture.n_spots, format="csr")
        neighbourhood = {
            "smooth_indices": identity.indices,
            "smooth_indptr": identity.indptr,
        }

    field: np.ndarray = kernel(
        fixture.n_spots,
        valid_nb,
        valid_bb,
        np.empty(0),
        False,
        rdr,
        baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
        **neighbourhood,
    )
    return field


def cnaster_field_of(fixture: SpotCloneField, kernel: Any = None) -> np.ndarray:
    """`cnaster`'s spot-by-clone field on `fixture`, unweighted, or `kernel`'s in its place."""

    kernel = compute_loglike_spot_assignment if kernel is None else kernel
    field: np.ndarray = kernel(
        fixture.n_spots,
        np.ones(fixture.n_spots),
        np.ones(fixture.n_spots),
        np.empty(0),
        False,
        fixture.log_emission_rdr,
        fixture.log_emission_baf,
        fixture.pred,
        fixture.n_obs,
        fixture.n_clones,
    )
    return field


def partition_ari(planted: np.ndarray, fitted: np.ndarray) -> float:
    """Adjusted Rand index of two partitions, written here so the referee is independent of cnaster."""

    table = np.zeros((int(planted.max()) + 1, int(fitted.max()) + 1), dtype=np.int64)
    np.add.at(table, (planted, fitted), 1)

    pairs = sum(comb(int(n), 2) for n in table.ravel())
    by_planted = sum(comb(int(n), 2) for n in table.sum(axis=1))
    by_fitted = sum(comb(int(n), 2) for n in table.sum(axis=0))
    total = comb(int(planted.size), 2)

    expected = by_planted * by_fitted / total
    maximum = 0.5 * (by_planted + by_fitted)
    return float((pairs - expected) / (maximum - expected))


def run_planted_core_inference(truth: CoreInferenceTruth, **kwargs: object) -> Any:
    from tests.adapters import from_core_inference_truth

    with warnings.catch_warnings():
        # `scipy` rejects the `ftol` the shipped solver options pass (#46); the
        # warning is that defect firing on the live path, not this test's.
        warnings.simplefilter("ignore")
        return run_core_inference(
            **from_core_inference_truth(truth).as_kwargs(),
            hmmclass=hmm_nophasing,
            **kwargs,
        )


def two_clone_stacked_instance(seed: int = 4) -> dict[str, Any]:
    """Two clones of 30 bins, stacked as `clone_stack_obs` stacks them."""
    rng = np.random.default_rng(seed)
    n_obs, n_clones = 30, 2
    n_segments = n_obs * n_clones

    base = rng.uniform(40.0, 80.0, (n_segments, 1))
    states = rng.integers(0, 2, n_segments)
    means = base[:, 0] * np.array([1.0, 2.0])[states]

    X = np.zeros((n_segments, 2, 1))
    X[:, 0, 0] = rng.poisson(means)
    X[:, 1, 0] = rng.binomial(20, np.array([0.5, 0.25])[states])

    return {
        "X": X,
        "lengths": np.array([n_obs] * n_clones),
        "base": base,
        "total": np.full((n_segments, 1), 20.0),
        "normal_lambda": base[:n_obs, 0] / base[:n_obs, 0].sum(),
        "clone_lengths": np.array([n_obs] * n_clones),
    }


def pseudobulk_inputs(
    n_obs: int, n_spots: int, n_clones: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_clones, n_spots)
    return {
        "single_X": rng.poisson(3.0, (n_obs, 2, n_spots)).astype(np.float64),
        # NB not integral, so the order of the sum decides its last bit.
        "single_base_nb_mean": rng.gamma(2.0, 0.37, (n_obs, n_spots)),
        "single_total_bb_RD": rng.poisson(5.0, (n_obs, n_spots)).astype(np.float64),
        "clone_index": [np.flatnonzero(labels == k) for k in range(n_clones)],
    }


def recombination_map(path: Path, contigs: range, rate: float = 1.0) -> Path:
    """A map at `rate` cM/Mb with a jitter, markers every 500 kb to 60 Mb."""
    rng = np.random.default_rng(11)
    rows = []

    for contig in contigs:
        positions = np.arange(0, 60_000_001, 500_000)
        steps = rng.uniform(0.5, 1.5, positions.size - 1) * rate * 0.5
        cm = np.concatenate(([0.0], np.cumsum(steps)))
        rows += [
            {"chrom": f"chr{contig}", "pos": int(p), "pos_cm": float(c)}
            for p, c in zip(positions, cm, strict=True)
        ]

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


END_TO_END_LATTICE = (25, 40)
"""Rows and columns. A thousand spots, which is `icm_sweep_deque`'s floor times five."""


def end_to_end_truth(**overrides: Any) -> CoreInferenceTruth:
    """Two clones and three states over 40 bins and `END_TO_END_LATTICE`, seed 11."""
    settings: dict[str, Any] = {"n_clones": 2, "n_states": 3, "lattice": END_TO_END_LATTICE,
                                "n_obs": 40, "n_segments": 3, "seed": 11}  # fmt: skip
    return core_inference_truth(**settings | overrides)


def divergent_clone_instance(
    n_states: int = 3, n_clones: int = 3, per_clone: int = 8, seed: int = 41
) -> dict[str, Any]:
    """A clone-stacked instance whose clones decode to different states.

    `per_clone > n_clones`, so indexing by clone reads clone zero's block
    silently rather than raising.
    """

    generator = np.random.default_rng(seed)
    n_segments = n_clones * per_clone

    X, exposure, trials = allele_counts(
        generator, (n_segments,), (10, 40), 0.4, (20, 60)
    )
    observed, successes = X[:, 0], X[:, 1]

    # NB one state per clone, so the shifts are distinct by construction.
    decode = np.repeat(np.arange(n_clones) % n_states, per_clone).astype(np.int64)

    return {
        "nbEncoder": CountEncoder(observed.reshape(-1, 1), exposure.reshape(-1, 1)),
        "bbEncoder": CountEncoder(successes.reshape(-1, 1), trials.reshape(-1, 1)),
        "log_mu": generator.normal(0.0, 0.3, size=(n_states, 1)),
        "alphas": np.full((n_states, 1), 0.2),
        "p_binom": generator.uniform(0.2, 0.8, size=(n_states, 1)),
        "taus": np.full((n_states, 1), 25.0),
        "normal_log_lambda": generator.normal(0.0, 0.1, size=n_segments),
        "clone_lengths": np.full(n_clones, per_clone, dtype=np.int64),
        "decode": decode,
        "n_states": n_states,
        "n_clones": n_clones,
        "per_clone": per_clone,
    }


def shifted_emission_call(
    model: Any, instance: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    scored: tuple[np.ndarray, np.ndarray]
    scored = model.compute_emission_probability_nb_betabinom_coded(
        instance["nbEncoder"],
        instance["bbEncoder"],
        instance["log_mu"],
        instance["alphas"],
        instance["p_binom"],
        instance["taus"],
        normal_log_lambda=instance["normal_log_lambda"],
        clone_lengths=instance["clone_lengths"],
    )
    return scored


def shifted_replacement(
    instance: dict[str, Any], *, shifted: bool = False, kernels: str = "cnaster"
) -> Any:
    """The drop-in, carrying the decode the shift is taken at (#517)."""

    model = with_attributes(
        port_hmm_nophasing, apply_logmu_shift=shifted, emission_kernels=kernels
    )()
    model.state_posteriors = np.eye(instance["n_states"])[instance["decode"]].T

    return model


def planted_blocky_field(
    side: int, n_states: int, seed: int, beta: float
) -> tuple[np.ndarray, CsrGraph, np.ndarray, float]:
    """A field with a planted blocky labelling, plus a 4-neighbour lattice."""
    from tests.adapters import lattice_adjacency

    rng = np.random.default_rng(seed)
    n = side * side

    blocks = np.zeros((side, side), dtype=np.int64)
    blocks[: side // 2, : side // 2] = 1 % n_states
    blocks[side // 2 :, : side // 2] = 2 % n_states
    blocks[: side // 2, side // 2 :] = 3 % n_states
    planted = blocks.ravel()

    # NB weak, noisy field, so the coupling decides the boundaries.
    field = rng.normal(0.0, 1.0, size=(n, n_states))
    field[np.arange(n), planted] += 0.6

    graph = CsrGraph.from_matrix(lattice_adjacency((side, side)).sorted_indices())

    return field, graph, planted, beta


SIM_MANIFESTS = SIM_ROOT / "manifests"


SMALL_ARRAY = {"array": {"rows": 20, "columns": 20}}


def draw_manifest(name: str, overrides: dict[str, Any] | None = None) -> DrawManifest:
    document = merged_tables(
        extended(SIM_MANIFESTS / f"{name}.toml"), SMALL_ARRAY | (overrides or {})
    )
    return from_document(document, SIM_MANIFESTS)
