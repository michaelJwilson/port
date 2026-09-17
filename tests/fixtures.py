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

import numpy as np
import torch
from snakes_and_ladders.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from snakes_and_ladders.sim.hmm import (
    HmmParams,
    SimulatedHmmDataset,
    simulate_sequences,
)

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
            sequence_length=sequence_length,
            n_sequences=n_sequences,
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
            sequence_length=sequence_length,
            n_sequences=n_sequences,
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
            sequence_length=sequence_length,
            n_sequences=n_sequences,
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


def core_inference_truth(
    *,
    n_clones: int = 3,
    n_states: int = 4,
    lattice: tuple[int, int] = (12, 10),
    n_obs: int = 240,
    n_segments: int = 4,
    self_transition: float = 0.99,
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

    Raises
    ------
    ValueError
        If `n_clones` exceeds the lattice's rows, where a band would be empty,
        or the segments do not partition `n_obs`, or `n_states < 2`.
    """
    rows, columns = lattice
    if n_clones > rows:
        msg = f"{n_clones} bands over {rows} rows leaves one empty"
        raise ValueError(msg)
    if n_obs % n_segments:
        msg = f"{n_segments} segments do not partition {n_obs} observations"
        raise ValueError(msg)
    if n_states < 2:
        msg = f"a chain needs at least two states, got {n_states}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    n_spots = rows * columns

    # Spread across a decade of expression and either side of balance, so the
    # states are separable by the data rather than by their index.
    log_mu = np.log(np.linspace(0.5, 5.0, n_states))
    alphas = np.full(n_states, 1.0 / 6.0)
    # Above balance: `run_core_inference` calls `gmm_init` with
    # `only_minor=False` because, as its own comment says, with no phasing the
    # states have to sit above 0.5. A fixture planted below it is asking the
    # initializer for something the model does not carry.
    p_binom = np.linspace(0.52, 0.88, n_states)
    taus = np.full(n_states, 30.0)

    row_of = np.arange(n_spots) // columns
    labels = np.minimum(row_of * n_clones // rows, n_clones - 1).astype(np.int64)

    transition = circulant_transition(n_states, self_transition)
    states = np.empty((n_clones, n_obs), dtype=np.int64)
    for clone in range(n_clones):
        states[clone, 0] = rng.integers(n_states)
        for obs in range(1, n_obs):
            states[clone, obs] = rng.choice(
                n_states, p=transition[states[clone, obs - 1]]
            )

    base_nb_mean = rng.uniform(*depth, (n_obs, n_spots))
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
        lengths=np.full(n_segments, n_obs // n_segments, dtype=int),
        switch_prob=switch_prob,
        lattice=lattice,
        self_transition=self_transition,
        seed=seed,
    )


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
        log_mu=np.log(profiles.mean).reshape(-1, 1),
        alphas=(1.0 / profiles.dispersion).reshape(-1, 1),
        p_binom=allele.success_probability.reshape(-1, 1),
        taus=allele.concentration.reshape(-1, 1),
    )
