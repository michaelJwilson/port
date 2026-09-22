"""Fixtures for port's tests, declared with `snakes_and_ladders` simulators.

Truth is planted here and never recovered from the data: a test that reads a
parameter back out of the fixture it generated is checking arithmetic, not
recovery. Every builder takes a seed and returns it alongside the instance,
so a failure names the draw that produced it.

The builders cover one rung of issue #14's ladder each. This module carries
the first: a single chain, no spatial layer and no factored state space,
which is the one rung whose correspondence with `cnaster` is exact today.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from snakes_and_ladders.ragged import MINIMUM_LENGTH, Ragged
from snakes_and_ladders.sim.hmm import (
    HmmParams,
    SimulatedHmmDataset,
    simulate_sequences,
)

if TYPE_CHECKING:
    from snakes_and_ladders.sim.graph import BoundaryCondition, PottsGraph

DEFAULT_SEED = 11
"""The seed every builder defaults to, so a bare call is reproducible."""


@dataclass(frozen=True)
class NegativeBinomialChains:
    """Independent negative binomial chains, with the parameters that drew them.

    Parameters
    ----------
    dataset : SimulatedHmmDataset
        The draw: planted `states`, `observations`, and the `initial` and
        `transition` they were drawn under.
    family : NegativeBinomialEmission
        The emission the observations came from, one mean and dispersion per
        state.
    mean, dispersion : np.ndarray
        The family's parameters, shape `(n_states,)`, carried separately
        because the adapter needs them as numbers rather than as a family.
    seed : int
        The draw's seed.
    """

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

    Parameters
    ----------
    separation : float
        The ratio between adjacent state means, so state `k` has mean
        `base_mean * separation ** k`. It is the axis every recovery claim
        has to be stated against: at wide separation every method agrees and
        the comparison is empty, and at `separation = 1` the states are not
        identified at all.
    self_transition : float
        The diagonal; the remainder is spread over the other states.
    drift : float
        How the off-diagonal mass splits between higher and lower states.
        At `0.5` the transition is circulant and **symmetric**, which is the
        form `cnaster.hmm_nophasing.get_log_transmat` builds — and a
        symmetric transition cannot distinguish a row convention from a
        column one, so a fixture that only ever uses it leaves a transpose
        between the two implementations undetectable. Away from `0.5` it
        can.

    Raises
    ------
    ValueError
        If `separation` is below one, where the state means are not ordered,
        or `self_transition` is not strictly inside `(0, 1)`, where a row of
        the transition is a point mass.
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
    """`t` on the diagonal, the remainder spread evenly off it.

    Stated here rather than imported so the fixture and `cnaster`'s
    `get_log_transmat` are compared as two constructions of one matrix; a
    test pins them equal.
    """
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)

    off = (1.0 - self_transition) / (n_states - 1)
    transition = np.full((n_states, n_states), off, dtype=np.float64)
    np.fill_diagonal(transition, self_transition)
    return transition


def place_events(
    lengths: np.ndarray,
    n_states: int,
    *,
    rng: np.random.Generator,
    events: tuple[int, int],
    event_bins: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """A neutral genome with copy-number events placed on it.

    The state path is state zero -- diploid, balanced, and **emitted** like any
    other state -- everywhere except where an event is placed. Each event takes
    a contiguous run of bins inside one chromosome and gives it a single
    non-neutral state.

    This is the generative model the shipped configuration's own instance names
    describe: `numcnas3.3_cnasize5e7_ploidy2_random4` is a count of events and
    an event size, not a transition rate.

    **The path is piecewise constant, not a draw from the transition the HMM
    fits**, and that is the trade #120 records. A Markov chain visiting ten
    states uniformly leaves the genome a tenth neutral, which is
    `find_diploid_balanced_state`'s own threshold, so whether the normal state
    is a candidate comes down to the seed. A real genome is mostly neutral, and
    a piecewise-constant path is what that looks like. The cost is that the
    planted path is a *special case* of the fitted model rather than a draw
    from it -- consistent with a very sticky chain, and not drawn from one.

    Events are confined within a chromosome: a copy-number event does not span
    a centromere-to-centromere boundary, and `lengths` is where the recursion
    restarts.

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int, int, int]]]
        The state path, and the events as `(chromosome, offset, extent,
        state)`. The events are returned rather than left implicit because the
        path cannot be read back into them: two events on adjacent chromosomes
        that draw the same state abut, and an abutment is indistinguishable
        from a crossing by looking at the path.

    Raises
    ------
    ValueError
        If `n_states` leaves no non-neutral state to place.
    """
    if n_states < 2:
        msg = f"an event needs a state other than the neutral one, got {n_states}"
        raise ValueError(msg)

    path = np.zeros(int(np.sum(lengths)), dtype=np.int64)
    edges = np.concatenate(([0], np.cumsum(lengths)))
    placed: list[tuple[int, int, int, int]] = []

    for _ in range(int(rng.integers(*events))):
        chromosome = int(rng.integers(lengths.size))
        start, stop = int(edges[chromosome]), int(edges[chromosome + 1])

        extent = min(int(rng.integers(*event_bins)), stop - start)
        offset = int(rng.integers(start, stop - extent + 1))
        state = int(rng.integers(1, n_states))

        path[offset : offset + extent] = state
        placed.append((chromosome, offset, extent, state))

    return path, placed


def drift_transition(n_states: int, self_transition: float, drift: float) -> np.ndarray:
    """`t` on the diagonal, the remainder split by direction.

    `drift` is the share of the off-diagonal mass going to higher states,
    weighted by how many states lie on each side, so `drift = 0.5` spreads
    it evenly over every other state and reproduces
    :func:`circulant_transition` exactly. Any other value gives an
    asymmetric matrix.

    Asymmetry is what makes a transposed transition observable: against a
    symmetric one the two row conventions score identically and the
    comparison is empty. Weighting by group size is what keeps `0.5` the
    circulant case at every `n_states` -- splitting the mass in half between
    the sides instead would over-weight whichever side has fewer states, and
    the default fixture would stop being the one `cnaster` builds.

    A row with neighbours on one side only sends all of its off-diagonal
    mass that way, so every row sums to one at every `drift`.
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
    """Chains over a paired state space, with the truth that drew them.

    `cnaster`'s phased model factors a state into a copy state and a phase,
    so the space is `2K` and the transfer matrix is assembled from a `K x K`
    base and a two-element kernel. Here the kernel is **constant along the
    chain**, which is what lets one matrix stand for the whole of it and is
    the only regime `snakes_and_ladders` can express: its recursion takes a
    single transition, so a kernel that varies by position has no upstream
    form until the structured transfer matrix lands.

    Parameters
    ----------
    dataset : SimulatedHmmDataset
        The draw over the `2K` paired states.
    family : NegativeBinomialEmission
        The emission over paired states. Its means are **not** mirrored
        across the phase: `cnaster` pairs states that share a copy state, but
        the lattice takes the emission as data, and a phase-degenerate
        emission would hide a transposed or mis-blocked transition the same
        way a symmetric transition hides a transposed one.
    base_transition : np.ndarray
        The `K x K` base, shape `(n_copy_states, n_copy_states)`.
    combined_transition : np.ndarray
        The assembled `2K x 2K` matrix the draw used.
    initial : np.ndarray
        The copy-state initial distribution, shape `(n_copy_states,)`. The
        paired start is this halved across the two phases, which is what
        `hmm_phased.forward_lattice` builds internally.
    switch : float
        The constant probability of changing phase between positions.
    penalize_phase_only_on_same_cnv : bool
        Which assembly the matrix used.
    seed : int
        The draw's seed.
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
    """Assemble the `2K x 2K` transfer matrix from a base and a phase kernel.

    Stated here independently of `cnaster.hmm_phased.update_combined_transmat`
    so the two constructions are compared rather than one trusted; a test
    pins them equal for both assemblies.

    With `penalize_phase_only_on_same_cnv` false the kernel multiplies every
    transition, which is the Kronecker product of the kernel and the base.
    With it true the kernel applies only where the copy state is conserved
    and the rest of the mass splits evenly between phases, on the reading
    that the phase is uninformative across a change of copy state. Both are
    row stochastic, and a test pins that too.

    Raises
    ------
    ValueError
        If `switch` is not strictly inside `(0, 1)`, where the phase either
        never changes or always does and the kernel is not a parameter the
        data could move.
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
    """Draw chains over the `2K` paired state space at a constant phase kernel.

    `drift` defaults away from `0.5` so the base is asymmetric. A symmetric
    base makes the assembled matrix symmetric too, and a transposed transfer
    matrix then scores identically -- measured, not assumed: the transpose
    shifts the total by 0.14 at `drift = 0.3` and by exactly zero at `0.5`.
    Unlike the single-chain fixture there is nothing to lose by it, since no
    claim here rests on the base being the circulant one `cnaster` builds.
    """
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
    """Chains of successes out of a fixed number of trials, and their truth.

    The rung issue #24 needs. The M step is the one part of the fit that is
    itself an optimization on both sides, so what validates it is a draw
    whose `(alpha, beta)` are known and whose posterior can be set rather
    than inferred: given a posterior, re-estimation is a self-contained
    problem two implementations can be handed independently.

    Parameters
    ----------
    dataset : SimulatedHmmDataset
        The draw: planted `states`, `observations`, and the `initial` and
        `transition` they were drawn under.
    family : BetaBinomialEmission
        The emission the observations came from, one `(alpha, beta)` per
        state at a common number of trials.
    alpha, beta : np.ndarray
        The family's parameters, shape `(n_states,)`.
    trials : int
        The number of trials every observation was drawn at. Constant along
        the chain for the reason `adapters` gives for exposure: `cnaster`
        carries one exposure per observation and the upstream family one per
        state, so only a constant is the same problem on both sides.
    seed : int
        The draw's seed.
    """

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
        """`alpha / (alpha + beta)`, the mean of each state's beta.

        The parameterization both implementations share: `cnaster` carries
        `(p, tau)` and reads `a = p tau`, `b = (1 - p) tau` in
        `hmm_emission.compute_bb_ab`, so `p` is this and `tau` the sum.
        """
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

    Parameters
    ----------
    success_probability : np.ndarray, optional
        One `p` per state. Defaults to `n_states` values spread evenly
        inside `(0, 1)`, which keeps the states separated without putting
        any of them against the `(EPSILON, 1 - EPSILON)` bound
        `Weighted_BetaBinom_mix.get_bounds` imposes -- a state at the bound
        is a boundary case rather than a comparison, and issue #24 asks for
        that one separately.
    concentration : float
        `alpha + beta`, shared across states. Shared rather than per-state
        because it is the regime where `cnaster`'s `shared_dispersion`
        branch and its per-state branch describe one problem, so a test may
        use either and say which.
    trials : int
        Constant, for the reason the class docstring gives.

    Raises
    ------
    ValueError
        If `concentration` or `trials` is not positive, or a supplied
        `success_probability` leaves `(0, 1)` or does not have one entry per
        state -- each of which is a family the draw cannot be taken from
        rather than a hard case.
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
    """The posterior an M step is handed, built from the planted states.

    Shape `(n_sequences, sequence_length, n_states)`. At `smoothing = 0` this
    is the indicator of the truth, so the M step it drives is the
    complete-data maximum likelihood and its answer is comparable with the
    planted parameters. Above zero it is mixed with the uniform, which is
    what an EM posterior looks like and what gives **every** state weight at
    **every** observation -- a state whose weight is exactly zero somewhere
    is not a harder problem, it is a smaller one, and it would let a
    per-state solve that ignores its weights pass.

    Planting the posterior rather than running a forward-backward pass is
    deliberate: it is the input to the step under test, so inferring it here
    would make a defect in the lattice look like a defect in the M step.

    Raises
    ------
    ValueError
        If `smoothing` is not in `[0, 1)`; at one the posterior is uniform
        and carries no information about the states at all.
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
class CoreInferenceTruth:
    """A planted instance of the model `cnaster.hmrf.run_core_inference` fits.

    Issue #4, and the decision #66 records: each `(segment, spot)` is **one
    negative binomial draw** for the total channel and **one beta-binomial
    draw** for the success channel, independently, through
    `snakes_and_ladders`' own families. Not a hierarchical draw whose marginals
    come out the same shape -- those are indistinguishable per spot and are a
    different joint, which is why the form is recorded rather than left to
    whatever a sampler happens to do.

    Parameters
    ----------
    labels : np.ndarray
        Clone of each spot, shape `(n_spots,)`. Contiguous bands, which is the
        smooth labelling with the fewest boundary edges: the Potts prior is
        ferromagnetic, so a labelling with no large regions would make the
        prior fight the truth and the test would measure that fight.
    states : np.ndarray
        Copy state per clone and segment, shape `(n_clones, n_obs)`, drawn from
        the circulant chain upstream declares.
    counts_nb, counts_bb : np.ndarray
        The drawn counts, shape `(n_obs, n_spots)`. `cnaster` reads them as
        `single_X[:, 0, :]` and `single_X[:, 1, :]`.
    base_nb_mean, total_bb_RD : np.ndarray
        The exposure the total is scored against and the trial count the
        successes are out of, shape `(n_obs, n_spots)`. **Data, not
        parameters**: `cnaster` conditions on both and never fits them, so any
        array here is a valid instance of its likelihood.
    log_mu, alphas, p_binom, taus : np.ndarray
        The planted emission parameters, shape `(n_states,)`, in `cnaster`'s
        own parameterization. The mapping upstream is by identity, never by
        fitting: `dispersion = 1 / alpha`, and `alpha, beta = p * tau,
        (1 - p) * tau` is `_bb_logpmf_1d` verbatim.
    lengths : np.ndarray
        Segment extents summing to `n_obs`. `cnaster` restarts the chain at
        each, so the count of them is a property of the problem and not a
        detail (#67).
    switch_prob : np.ndarray
        Per-site phase-switch probability, shape `(n_obs,)`. `cnaster` takes
        its log as `log_sitewise_transmat` and derives the complement.
    """

    labels: np.ndarray
    states: np.ndarray
    counts_nb: np.ndarray
    counts_bb: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray
    lengths: np.ndarray
    events: tuple[tuple[tuple[int, int, int, int], ...], ...]
    switch_prob: np.ndarray
    lattice: tuple[int, int]
    self_transition: float
    seed: int

    @property
    def n_clones(self) -> int:
        """`M`."""
        return int(self.states.shape[0])

    @property
    def n_states(self) -> int:
        """`K`."""
        return int(self.log_mu.shape[0])

    @property
    def n_obs(self) -> int:
        """`G`, genomic segments."""
        return int(self.states.shape[1])

    @property
    def n_spots(self) -> int:
        """`S`."""
        return int(self.labels.shape[0])

    @property
    def clone_index(self) -> list[np.ndarray]:
        """Spot indices per clone, which is `initial_clone_index`'s shape."""
        return [np.flatnonzero(self.labels == c) for c in range(self.n_clones)]

    @property
    def ragged(self) -> Ragged:
        """The planted genome as upstream's batch shape.

        Constructing it is the check, not a conversion for its own sake:
        `Ragged` refuses a `lengths` that does not tile the array and a
        chromosome under two bins, so a fixture that got its own segmentation
        wrong fails where the shape is declared rather than inside a fit.
        """
        return Ragged(
            values=self.counts_nb, lengths=tuple(int(x) for x in self.lengths)
        )

    def stacked_lengths(self, n_clones: int | None = None) -> np.ndarray:
        """`lengths` as the clone-stacked HMM sees it.

        `cnaster` stacks clones along the genomic axis and tiles the
        segmentation with them (`hmrf_utils.py:51`), so the fit runs over
        `n_clones * n_segments` chains rather than `n_segments`. Ragged
        chromosomes make that the shape upstream's `Ragged` carries and the
        rectangular route cannot.
        """
        return np.tile(self.lengths, self.n_clones if n_clones is None else n_clones)

    @property
    def emission_gigabytes(self) -> float:
        """What `cnaster` allocates per outer iteration for both channels.

        `(n_states, n_obs, n_spots)` twice, which is what decides whether an
        end-to-end run fits before anything else does.
        """
        return 2.0 * self.n_states * self.n_obs * self.n_spots * 8 / 1e9


def _emission_families(
    log_mu: np.ndarray, alphas: np.ndarray, p_binom: np.ndarray, taus: np.ndarray
) -> EmissionFamily:
    """`cnaster`'s parameters as upstream's two-channel family.

    The identity mapping, stated once so no test re-derives it.
    """
    from snakes_and_ladders.emissions import (
        BetaBinomialEmission,
        NegativeBinomialEmission,
    )
    from snakes_and_ladders.sim.count_pairs import IndependentCountPair

    return IndependentCountPair(
        NegativeBinomialEmission(
            torch.as_tensor(1.0 / alphas, dtype=torch.float64),
            torch.as_tensor(np.exp(log_mu), dtype=torch.float64),
        ),
        BetaBinomialEmission(
            # Placeholder trials: the covariate overrides them per observation,
            # and a family must still declare a positive count to be built.
            torch.ones(p_binom.shape[0], dtype=torch.float64),
            torch.as_tensor(p_binom * taus, dtype=torch.float64),
            torch.as_tensor((1.0 - p_binom) * taus, dtype=torch.float64),
        ),
    )


def weierstrass_exposure(
    n_obs: int,
    n_spots: int,
    *,
    low: float,
    high: float,
    a: float = 0.5,
    b: float = 7.0,
    terms: int = 12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A strictly positive exposure that varies violently along the bin axis.

    `W(x) = sum_k a^k cos(b^k pi x)` -- continuous everywhere, differentiable
    nowhere for `ab > 1` -- rescaled into `[low, high]` and multiplied by a
    per-spot library size.

    **Along the bin axis, deliberately.** `merge_pseudobulk_by_index_mix` sums
    the exposure across a clone's spots, so variation in the spot direction
    averages out before the fit ever sees it and variation in the bin
    direction survives intact. An exposure drawn i.i.d. over both axes is
    therefore a weaker fixture than it looks.

    **Strictly positive, and that is load-bearing.** `_nb_logpmf_1d` scores a
    non-positive rate as `out[i] = 0.0` -- log-density zero, probability one
    -- rather than excluding it. A Weierstrass function oscillates about its
    mean and is negative half the time, so an unshifted one would silently
    score half the genome as certain.

    Why it is worth the trouble: a **constant** exposure is absorbed into the
    emission as `log_mu - log(c)`, so `log_mu` is not separately identifiable
    from it. A varying one breaks that absorption, which is what makes a
    planted `log_mu` a thing a fit can be wrong about.

    Raises
    ------
    ValueError
        If `low` is not positive, or `a * b <= 1`, where the function is
        differentiable and the fixture is merely a smooth ripple.
    """
    if low <= 0.0:
        msg = f"the exposure must be strictly positive, got low={low}"
        raise ValueError(msg)
    if a * b <= 1.0:
        msg = f"a*b must exceed 1 for the construction to bite, got {a * b}"
        raise ValueError(msg)

    x = np.linspace(0.0, 1.0, n_obs, endpoint=False)
    walk = np.zeros(n_obs)
    for k in range(terms):
        walk += a**k * np.cos(b**k * np.pi * x)

    span = walk.max() - walk.min()
    shaped = low + (high - low) * (walk - walk.min()) / span

    generator = rng if rng is not None else np.random.default_rng(DEFAULT_SEED)
    library = generator.uniform(0.75, 1.25, n_spots)

    return np.outer(shaped, library)


MINIMUM_SEGMENT = MINIMUM_LENGTH
"""The shortest chromosome the fixture may plant, which is upstream's floor.

Aliased rather than re-stated so that a change upstream is a change here, and
so the reason travels with it: one position carries no transition.
"""


def ragged_lengths(
    n_obs: int,
    n_segments: int,
    *,
    rng: np.random.Generator,
    concentration: float = 2.0,
) -> np.ndarray:
    """A partition of `n_obs` into `n_segments` unequal parts.

    Chromosomes are not the same size, and until #667 landed `Ragged` there was
    no upstream shape that could say so -- `np.full(n_segments, n_obs //
    n_segments)` was the whole of this function, and it forced `n_segments` to
    divide `n_obs`.

    Dirichlet weights over a floor of `MINIMUM_SEGMENT`, with the floor's
    residual handed out to the largest fractional parts, so the result is
    integral and sums exactly. `concentration` sets the spread: smaller is more
    unequal, and 2.0 puts the dev instance's extremes about 3x apart.

    The floor is upstream's and not this repository's: a segment of one
    position is an initial distribution and no transition, and `Ragged` refuses
    it where the shape is declared.

    Raises
    ------
    ValueError
        If the floor alone exceeds `n_obs`, where no partition exists.
    """
    if n_segments * MINIMUM_SEGMENT > n_obs:
        msg = (
            f"{n_segments} segments of at least {MINIMUM_SEGMENT} do not fit "
            f"in {n_obs} observations"
        )
        raise ValueError(msg)

    free = n_obs - n_segments * MINIMUM_SEGMENT
    weights = rng.dirichlet(np.full(n_segments, concentration))

    exact = weights * free
    lengths = np.floor(exact).astype(int)
    residual = free - int(lengths.sum())
    if residual:
        # NB largest fractional parts first, so the rounding is a rule rather
        #    than an artefact of the order the segments happen to be in.
        order = np.argsort(exact - lengths)[::-1]
        lengths[order[:residual]] += 1

    return lengths + MINIMUM_SEGMENT


def core_inference_truth(
    *,
    n_clones: int = 3,
    n_states: int = 4,
    lattice: tuple[int, int] = (12, 10),
    n_obs: int = 240,
    n_segments: int = 4,
    segmentation: str = "ragged",
    events: tuple[int, int] = (3, 8),
    event_bins: tuple[int, int] | None = None,
    self_transition: float = 0.99,
    exposure: str = "weierstrass",
    depth: tuple[float, float] = (0.5, 3.0),
    reads: tuple[int, int] = (10, 60),
    switch: tuple[float, float] = (0.01, 0.20),
    seed: int = DEFAULT_SEED,
) -> CoreInferenceTruth:
    """Plant an instance, drawing every count through upstream's families.

    One stream per spot, `default_rng([seed, spot])`, so the draw is a function
    of the seed and the spot alone and not of the order they are visited in.
    That is what lets a reduced fixture be a prefix of a larger one rather than
    a different dataset.

    Parameters
    ----------
    exposure : str
        How `base_nb_mean` is planted. `"weierstrass"` varies it violently
        along the **bin** axis, which is the axis that survives the pseudobulk
        and the one a fit has to divide out; `"uniform"` draws it i.i.d. over
        both axes, which averages out per state and is the weaker fixture;
        `"constant"` makes it one number, under which `log_mu` is absorbed as
        `log_mu - log(c)` and is not separately identifiable.

        The three are kept because the contrast between them is a measurement
        -- `tests/test_exposure_fixture.py` reports what each costs the fit --
        and because a goodness-of-fit test needs one distribution per state,
        which only `"constant"` gives.

    Raises
    ------
    ValueError
        If `n_clones` exceeds the lattice's rows, where a band would be empty,
        the segments do not partition `n_obs`, `n_states < 2`, or `exposure`
        names no mode.
    """
    rows, columns = lattice
    if n_clones > rows:
        msg = f"{n_clones} bands over {rows} rows leaves one empty"
        raise ValueError(msg)
    if n_states < 2:
        msg = f"a chain needs at least two states, got {n_states}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    n_spots = rows * columns

    # State zero is diploid and balanced: `mu = 1`, `p = 0.5`. It is planted
    # rather than left to chance because two stages of `run_cnaster` require
    # one to exist -- `find_diploid_balanced_state` raises "No candidate
    # diploid balanced state found!" without it, and the normal-spot path
    # tests every bin against a beta-binomial with `p` forced to 0.5 (#106).
    # A fixture with no normal state asks both for something it never planted.
    #
    # The rest spread across a decade of expression and above balance:
    # `run_core_inference` calls `gmm_init` with `only_minor=False` because,
    # as `cnaster`'s own comment says, with no phasing the states have to sit
    # at or above 0.5. A state planted below it is asking the initializer for
    # something the model does not carry.
    log_mu = np.concatenate(([0.0], np.log(np.linspace(1.5, 5.0, n_states - 1))))
    alphas = np.full(n_states, 1.0 / 6.0)
    p_binom = np.concatenate(([0.5], np.linspace(0.58, 0.88, n_states - 1)))
    taus = np.full(n_states, 30.0)

    row_of = np.arange(n_spots) // columns
    labels = np.minimum(row_of * n_clones // rows, n_clones - 1).astype(np.int64)

    if segmentation == "ragged":
        lengths = ragged_lengths(n_obs, n_segments, rng=rng)
    elif segmentation == "equal":
        if n_obs % n_segments:
            msg = f"{n_segments} equal segments do not partition {n_obs}"
            raise ValueError(msg)
        lengths = np.full(n_segments, n_obs // n_segments, dtype=int)
    else:
        msg = f"unknown segmentation {segmentation!r}"
        raise ValueError(msg)

    # NB a neutral genome with events placed on it, rather than a chain
    #    visiting every state equally (#120). Ten states visited uniformly
    #    leave the genome a tenth neutral, which is
    #    `find_diploid_balanced_state`'s own threshold, so whether the normal
    #    state is a candidate at all came down to the seed. Events are
    #    confined within a chromosome, which is also what makes `lengths` the
    #    truth rather than a label: the recursion restarts there.
    # NB the event size scales with the genome, so the neutral backbone
    #    survives at any `n_obs`. Absolute sizes suit a real genome, where a
    #    bin is a fixed number of bases -- but a fixture's `n_obs` is a budget
    #    rather than a length, and events of five to forty bins that leave a
    #    thousand-bin genome 89 per cent neutral bury a sixty-bin one.
    extent = event_bins or (max(2, n_obs // 200), max(3, n_obs // 25))
    placements = [
        place_events(lengths, n_states, rng=rng, events=events, event_bins=extent)
        for _ in range(n_clones)
    ]
    states = np.stack([path for path, _ in placements])
    placed_events = tuple(tuple(placed) for _, placed in placements)

    if exposure == "constant":
        base_nb_mean = np.full((n_obs, n_spots), float(depth[0]))
    elif exposure == "uniform":
        base_nb_mean = rng.uniform(*depth, (n_obs, n_spots))
    elif exposure == "weierstrass":
        base_nb_mean = weierstrass_exposure(
            n_obs, n_spots, low=depth[0], high=depth[1], rng=rng
        )
    else:
        msg = f"unknown exposure {exposure!r}"
        raise ValueError(msg)
    total_bb_RD = rng.integers(*reads, (n_obs, n_spots)).astype(np.float64)
    switch_prob = rng.uniform(*switch, n_obs)

    family = _emission_families(log_mu, alphas, p_binom, taus)
    counts_nb = np.empty((n_obs, n_spots), dtype=np.float64)
    counts_bb = np.empty((n_obs, n_spots), dtype=np.float64)

    for spot in range(n_spots):
        covariate = np.stack([base_nb_mean[:, spot], total_bb_RD[:, spot]], axis=-1)
        drawn = family.sample(
            states[labels[spot]],
            np.random.default_rng([seed, spot]),
            covariate=torch.as_tensor(covariate),
        )
        counts_nb[:, spot] = drawn[..., 0]
        counts_bb[:, spot] = drawn[..., 1]

    return CoreInferenceTruth(
        labels=labels,
        states=states,
        counts_nb=counts_nb,
        counts_bb=counts_bb,
        base_nb_mean=base_nb_mean,
        total_bb_RD=total_bb_RD,
        log_mu=log_mu,
        alphas=alphas,
        p_binom=p_binom,
        taus=taus,
        lengths=lengths,
        events=placed_events,
        switch_prob=switch_prob,
        lattice=lattice,
        self_transition=self_transition,
        seed=seed,
    )


def critical_instance(**overrides: object) -> CoreInferenceTruth:
    """The instance the early gate runs on: the smallest that is still a run.

    `M = K = 2`, `G = 1,000`, `S = 500` -- two clones over a `20 x 25` lattice,
    250 spots each, which clears `icm_sweep_deque`'s floor of 200 (#81) with
    the least room to spare. Two states is one event state against the
    neutral one, so a fit that cannot separate them has nothing else to
    confuse them with, and a failure here is structural rather than
    statistical.

    The point is the budget. `critical` gates first and gates everything
    (upstream's rule, mirrored in `pyproject.toml`), so the instance it fits
    end to end has to cost seconds, not the 16 s `dev_instance` costs or the
    minutes `key_instance` would. Reduce further and the labelling stops
    being recoverable at all -- one clone under the floor is merged away
    before the solver runs.

    `events=(6, 10)` rather than the default `(3, 8)`, and the reason is
    measured: at `K = 2` the one event state is the weakest the law plants
    (`mu = 1.5`, `p = 0.58`), and at the default's 13.5 per cent occupancy
    the solver merges the two clones into one. At 18.1 per cent it recovers
    the labelling exactly, in about a second warm; deeper reads or a third
    state do the same, and more events is the change that keeps `K = 2`.
    """
    settings: dict[str, object] = {
        "n_clones": 2,
        "n_states": 2,
        "lattice": (20, 25),
        "n_obs": 1_000,
        "n_segments": 4,
        "events": (6, 10),
    }
    settings.update(overrides)
    return core_inference_truth(**settings)  # type: ignore[arg-type]


def dev_instance(**overrides: object) -> CoreInferenceTruth:
    """The instance to develop against: small enough to fail fast.

    `K = 10` as the key instance has, a tenth of its `G`, and **four** clones
    rather than ten. Four because of the floor, not taste: `icm_sweep_deque`
    merges any clone under 200 spots and does not expose the threshold (#81),
    so ten clones cannot exist below `S = 2,000` and a development instance
    that small would measure the merge rather than the model. Four over 1,600
    spots leaves 400 each.

    **Square, and that is the point (#137).** The lattice was `(10, 100)` -- a
    ten-to-one strip whose four bands were 2.5 rows thick -- which contradicted
    the reason `CoreInferenceTruth` gives for planting bands at all. Bands are
    the labelling with the fewest boundary edges *for a given lattice*, and
    that lattice then maximised boundary among the factorizations available:
    300 edges at a perimeter-to-area of 1.20, against **120 edges at 0.30**
    here. A spot's four neighbours are now the same distance away in both
    directions, so the spatial plots read as a section rather than a ribbon.

    The point is wall time, and squaring costs some: `S` rises from 1,000 to
    1,600, so the instance is about 1.6x its old size. An error found in 30 s
    is still an error found; the same error at 310 s is a reason to stop
    looking, and that is the comparison that matters.
    """
    settings: dict[str, object] = {
        "n_clones": 4,
        "n_states": 10,
        "lattice": (40, 40),
        "n_obs": 1_000,
        "n_segments": 10,
    }
    settings.update(overrides)
    return core_inference_truth(**settings)  # type: ignore[arg-type]


def key_instance(**overrides: object) -> CoreInferenceTruth:
    """The instance final validation and benchmarking are reported at.

    `M = K = 10`, `G = 10,000`, `S = 5,000` -- the declared scale (#87),
    unreduced. It builds here in 16.6 s at 2.09 GB and every planted parameter
    is recovered from it.

    **The inference on it does not fit**, and that is #90 rather than a reason
    to redefine the instance. `cnaster` materializes `(n_states, n_obs,
    n_spots)` twice per outer iteration -- 8.00 GB here, against 15 GB
    available -- and measured, one outer iteration at `M = K = 10`:

    | `G` | `S` | emission | wall | peak RSS | clones out |
    | ---: | ---: | ---: | ---: | ---: | ---: |
    | 750 | 2,000 | 0.24 GB | 21.6 s | 1.19 GB | 10 |
    | 1,500 | 2,000 | 0.48 GB | 38.6 s | 1.55 GB | 10 |
    | 4,000 | 2,000 | 1.28 GB | 90.9 s | 2.77 GB | 10 |
    | 10,000 | 2,000 | 3.20 GB | 222.5 s | 5.68 GB | 10 |
    | 10,000 | 3,000 | 4.80 GB | 310.3 s | 7.98 GB | 10 |
    | **10,000** | **5,000** | **8.00 GB** | — | ~18 GB projected | — |

    Peak tracks the emission at about 1.6x plus a fixed 0.5 GB, so the array
    is the whole story and the last row is an interpolation rather than a
    guess. #61 is the one written patch that removes the array rather than
    reading it faster.
    """
    settings: dict[str, object] = {
        "n_clones": 10,
        "n_states": 10,
        "lattice": (50, 100),
        "n_obs": 10_000,
        "n_segments": 10,
    }
    settings.update(overrides)
    return core_inference_truth(**settings)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SpotCloneField:
    """The inputs to `cnaster`'s HMM/spatial boundary, from a known model.

    Issue #59. `compute_loglike_spot_assignment` reduces a per-state emission
    over a clone's decoded profile into a `(n_spots, n_clones)` field. What it
    costs depends on two things the array shapes do not carry, so both are
    planted here rather than drawn arbitrarily.

    Parameters
    ----------
    log_emission_rdr, log_emission_baf : np.ndarray
        Shape `(n_states, n_obs, n_spots)`, which is the layout `cnaster`
        produces and states at `hmrf.py:244`. Real log-densities, scored from
        the fixture's own families rather than filled with noise.
    pred : np.ndarray
        The decoded copy-state profile per clone, shape `(n_obs, n_clones)`.
        **Drawn from a Markov chain, not uniformly.** A real profile is
        piecewise constant because the HMM's self-transition is near one, and
        a uniform draw maximises the spread of state indices in the inner
        loop -- so it measures a access pattern no run produces. `segments`
        below is the knob.
    counts_nb, base_nb_mean, counts_bb, total_bb_RD : np.ndarray
        The raw `(n_obs, n_spots)` inputs the emissions were scored from, and
        the per-state parameters beside them. Carried because issue #59 item
        2 fuses the emission into the field, so a patch for it needs what the
        emission was built from rather than the emission itself.
    log_mu, alphas, p_binom, taus : np.ndarray
        `(n_states,)`. The shape a fit produces and the only one the patched
        kernels read (#278). `cnaster`'s own two-step indexes `[i, 0]`, so a
        test using it as referee adds the column at that call site.
    segments : int
        Runs in clone zero's profile. The data-dependence `CLAUDE.md` names:
        a near-constant profile and a fragmented one are different problems
        at identical shapes, and a ratio from one is not a ratio from the
        other.
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
        """Both emission channels together, which is what issue #59 item 2 removes."""
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
        """One emission channel's footprint, which is what decides the size."""
        return self.log_emission_rdr.nbytes / 1e6

    def spot_major(self) -> tuple[np.ndarray, np.ndarray]:
        """The same emissions as `(n_states, n_spots, n_obs)`, contiguous.

        Kept for the record rather than used: issue #59 item 1 landed as a
        loop reorder at `cnaster`'s own layout, which measured 36.1 ms against
        the transposed kernel's 61.8 ms at `(7, 3000, 5000)` with seven
        clones, and needs nothing from the producer. A transposed layout fixes
        the contiguity and leaves the scalar reduction; reordering fixes both.

        Retained so that comparison stays reproducible, and because the
        transpose is the obvious patch a reader will propose.
        """
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
    """Build the boundary's inputs, with the profile's segmentation controlled.

    Parameters
    ----------
    self_transition : float
        The chain's diagonal, which sets `segments`. `cnaster` runs at
        `1 - 1e-4`, giving one segment in three thousand bins; a real profile
        has more structure than that, so the default here is lower and a
        benchmark sweeps it.

    Raises
    ------
    ValueError
        If `n_clones` exceeds `n_states`, where a clone's profile could not
        be drawn from the state space, or any extent is below one.
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

    # NB observations per (bin, spot), drawn from the family so the emission
    #    is a log-density rather than noise shaped like one.
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

    import torch

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


MAX_ENUMERABLE_LABELLINGS = 200_000


@dataclass(frozen=True)
class PottsLabels:
    """A spatial labelling problem, and the truth that built it.

    The rung issue #40 needs. A label solver takes a unary field and a graph
    and returns a labelling, so what validates it is an instance whose graph
    is declared and whose optimum is computable rather than approximable.

    Parameters
    ----------
    graph : PottsGraph
        The lattice, carrying its edges once each and a coupling per edge.
        `cnaster` describes the same object as a symmetric CSR adjacency; the
        adapter converts, and a test pins the two equal.
    labels : np.ndarray
        The planted labelling, shape `(n_nodes,)`. **Not the optimum.** It is
        what the field was drawn around, and at a low signal-to-noise the
        energy-minimising labelling is a different one -- which is the
        difference between a recovery claim and an optimality claim, and the
        reason the two are asserted separately.
    field : np.ndarray
        The unary, shape `(n_nodes, n_clones)`. `cnaster`'s
        `single_llf`, upstream's `field_values`, and a per-spot per-clone log
        likelihood in the application.
    coupling : float
        `J`, uniform over the edges.
    spatial_weight : float
        `cnaster` multiplies its edge weights by this and upstream folds it
        into the coupling, so the two carry the same product differently.
        Held separately here so a test can vary one and hold the other.
    shape : tuple[int, ...]
        The lattice extent. Carried because a failure naming a node index is
        unreadable without it.
    seed : int
        The draw's seed.
    """

    graph: "PottsGraph"
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
    boundary: "BoundaryCondition | None" = None,
    seed: int = DEFAULT_SEED,
) -> PottsLabels:
    """Plant contiguous label domains on a lattice and draw a field around them.

    The field is **planted, not inferred**, for the reason
    :func:`planted_posterior` gives: it is the input to the step under test,
    so deriving it from an emission would make a defect in the emission look
    like a defect in the solver. The emission-derived field belongs to issue
    #4, which is where the whole inference is refereed.

    The domains are contiguous slabs along the first axis rather than an
    independent draw per node. That is the regime the paper's solver argument
    is about: `wolff.tex` claims single-site descent freezes large same-label
    regions into spurious disjoint clusters, and a labelling with no large
    regions cannot exhibit it. An i.i.d. labelling would make every solver
    agree and the comparison empty.

    Parameters
    ----------
    signal : float
        How far the planted label's field entry sits above the others, before
        noise. The axis every claim here is stated against: at `signal` far
        above `noise` the unary decides alone and the coupling is decoration,
        and at zero the field is pure noise and no method can recover
        anything.
    noise : float
        Standard deviation of the Gaussian added to every entry. `signal /
        noise` is the quantity to report, not either alone.
    coupling, spatial_weight : float
        Their product is what both implementations score with. At zero the
        problem separates per node and every solver must return the field's
        argmax, which is the degenerate case a test uses to check the harness
        before it checks a solver.

    Raises
    ------
    ValueError
        If `n_clones` is below two, where there is nothing to label; if
        `noise` is negative; or if `shape` has fewer sites than clones, where
        a planted labelling cannot use them all.
    """
    from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

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
    """The exact minimiser of the Potts energy, by exhaustive search.

    The referee `CLAUDE.md` names first among independent sources, and the
    one this rung can actually have: a label solver returns the best of what
    it visited, and without the true optimum there is no way to tell a good
    search from a lucky one.

    Returns the labelling and its energy, under upstream's sign convention
    (`energy` is minimised). Cost is `n_clones ** n_nodes`, so this is for
    fixtures built to be enumerable and nothing else.

    Raises
    ------
    ValueError
        If the search space exceeds `MAX_ENUMERABLE_LABELLINGS`. Refusing is
        the point: an enumeration that silently takes an hour is a test that
        will be deleted rather than fixed, and a fixture too large to
        enumerate needs a different referee rather than more patience.
    """
    from snakes_and_ladders.sim.potts import energy

    n_nodes, n_clones = fixture.n_nodes, fixture.n_clones
    total = n_clones**n_nodes

    if total > MAX_ENUMERABLE_LABELLINGS:
        msg = (
            f"{n_clones}**{n_nodes} = {total} labellings exceeds "
            f"{MAX_ENUMERABLE_LABELLINGS}; use a smaller fixture"
        )
        raise ValueError(msg)

    graph = _scaled_graph(fixture)

    best_labelling, best_energy = None, np.inf
    for index in range(total):
        labelling = np.array(
            [(index // n_clones**position) % n_clones for position in range(n_nodes)],
            dtype=np.int64,
        )
        value = energy(graph, fixture.field, labelling)
        if value < best_energy:
            best_labelling, best_energy = labelling, value

    assert best_labelling is not None
    return best_labelling, float(best_energy)


def _scaled_graph(fixture: PottsLabels) -> "PottsGraph":
    """The fixture's graph with `spatial_weight` folded into the coupling.

    `cnaster` carries the weight outside the adjacency and multiplies at
    scoring time; upstream carries one number per edge. Folding here is what
    makes the two score the same objective, and doing it in one place is what
    stops a test folding it twice.
    """
    from snakes_and_ladders.sim.graph import PottsGraph

    return PottsGraph(
        n_nodes=fixture.graph.n_nodes,
        edges=fixture.graph.edges,
        coupling=tuple(fixture.spatial_weight * j for j in fixture.graph.coupling),
    )
