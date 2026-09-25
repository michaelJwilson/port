"""A planted instance of the model `cnamaste` fits, drawn with `numpy` alone.

Each `(bin, spot)` is one negative binomial draw for read depth and one
beta-binomial draw for the B allele, independently, in the parameterization
`cnamaste.hmm_nophasing` scores: `r = 1 / alpha`, `p = 1 / (1 + alpha * m)`
with `m = exposure * mu`, and `Beta(p_binom * tau, (1 - p_binom) * tau)`.
Nothing here is imported from the package under test, so a defect there
cannot also sit in its referee.

Clones are row bands on the lattice: the Potts prior is ferromagnetic, and a
labelling with no large regions would make the prior fight the truth. Copy
states follow a sticky chain restarted at each segment (chromosome), and the
B allele's haplotype flips between bins with the planted switch
probability, which is what the phased model integrates over.

Every spot draws from its own stream, `default_rng([seed, spot])`, so a draw
is a function of the seed and the spot and not of the order spots are
visited in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 11
"""The seed every fixture takes unless it says otherwise."""

STATES = (
    (1.0, 0.5),
    (0.5, 0.88),
    (1.5, 0.66),
    (2.0, 0.75),
    (1.0, 0.88),
    (2.5, 0.8),
    (3.0, 0.66),
    (1.5, 0.88),
)
"""`(mu, p_binom)` per state, state 0 neutral: off the integer lattice, and
separated in at least one channel from every other state by more than the
per-bin noise at the default depth."""

COPY_LATTICE = ((1, 1), (1, 0), (2, 1), (2, 0), (3, 1), (2, 2), (3, 2), (4, 1))
"""`(A, B)` allele copies per state for `copy_lattice=True`, state 0 diploid.
`mu = (A + B) / 2` and `p_binom = A / (A + B)`, so an integer decoder has an
exact answer to recover."""

LOH_P = 1.0 - 1.0e-5
"""`p_binom` for a state with no B allele: the beta-binomial needs `p < 1`."""

ALPHA = 0.01
"""Negative binomial over-dispersion, one value for every state."""

TAU = 100.0
"""Beta-binomial concentration, one value for every state."""

NORMAL_SHARE = 0.25
"""The least share of spots the normal clone holds, when there is one."""


@dataclass(frozen=True)
class Truth:
    """What was planted and what was drawn from it.

    Parameters
    ----------
    labels : np.ndarray
        Clone of each spot, `(n_spots,)`.
    states : np.ndarray
        Copy state per clone and bin, `(n_clones, n_obs)`.
    counts_nb, counts_bb : np.ndarray
        Drawn read depth and B-allele counts, `(n_obs, n_spots)`, the second
        on the drawn haplotype of each bin.
    base_nb_mean, total_bb_RD : np.ndarray
        Exposure and allele trials, `(n_obs, n_spots)`. Data the model
        conditions on, not parameters it fits.
    log_mu, alphas, p_binom, taus : np.ndarray
        Emission parameters per state, `(n_states,)`.
    lengths : np.ndarray
        Bins per segment, summing to `n_obs`.
    phase : np.ndarray
        Haplotype per bin, `(n_obs,)`: 1 where the B count is `total - B`.
    switch_prob : np.ndarray
        Per-bin probability the haplotype flips from the previous bin.
    copies : np.ndarray | None
        `(n_states, 2)` allele copies when planted on `COPY_LATTICE`.
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
    phase: np.ndarray
    switch_prob: np.ndarray
    lattice: tuple[int, int]
    self_transition: float
    seed: int
    copies: np.ndarray | None = None

    @property
    def n_clones(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.log_mu.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_spots(self) -> int:
        return int(self.labels.shape[0])

    def spot_states(self) -> np.ndarray:
        """The state each `(bin, spot)` was drawn under, `(n_obs, n_spots)`."""
        spot_states: np.ndarray = self.states[self.labels].T
        return spot_states

    def expected_nb(self) -> np.ndarray:
        """`E[counts_nb]`, `(n_obs, n_spots)`."""
        expected: np.ndarray = self.base_nb_mean * np.exp(
            self.log_mu[self.spot_states()]
        )
        return expected

    def expected_b(self) -> np.ndarray:
        """`E[B]` on haplotype H0, before the phase flip, `(n_obs, n_spots)`."""
        expected: np.ndarray = self.total_bb_RD * self.p_binom[self.spot_states()]
        return expected

    def unphased_bb(self) -> np.ndarray:
        """`counts_bb` with the drawn phase undone."""
        flip = self.phase[:, None].astype(bool)
        return np.where(flip, self.total_bb_RD - self.counts_bb, self.counts_bb)


def segment_lengths(
    n_obs: int, n_segments: int, rng: np.random.Generator
) -> np.ndarray:
    """Uneven segment extents of at least two bins, summing to `n_obs`."""
    if n_obs < 2 * n_segments:
        msg = f"{n_obs} bins cannot make {n_segments} segments of two or more"
        raise ValueError(msg)

    weights = rng.uniform(0.5, 1.5, n_segments)
    spare = n_obs - 2 * n_segments
    lengths = 2 + np.floor(spare * weights / weights.sum()).astype(np.int64)
    lengths[: n_obs - int(lengths.sum())] += 1
    return lengths


def band_labels(
    lattice: tuple[int, int], n_clones: int, normal_clone: bool
) -> np.ndarray:
    """Clone per spot as row bands, spots row major on the lattice."""
    rows, _ = np.unravel_index(np.arange(lattice[0] * lattice[1]), lattice)
    labels = (rows * n_clones) // lattice[0]

    if normal_clone and n_clones > 1:
        share = np.mean(labels == 0)
        if share < NORMAL_SHARE:
            msg = f"the normal clone holds {share:.2f} of spots, under {NORMAL_SHARE}"
            raise ValueError(msg)

    return labels.astype(np.int64)


def state_paths(
    n_clones: int,
    n_states: int,
    lengths: np.ndarray,
    self_transition: float,
    normal_clone: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """A sticky chain per clone, restarted at each segment in the neutral state.

    Redrawn until every tumour clone carries an aberration and no two clones
    share a path, so the clones are identifiable from their copy numbers.
    """
    n_obs = int(lengths.sum())

    for _ in range(1_000):
        states = np.zeros((n_clones, n_obs), dtype=np.int64)

        for clone in range(n_clones):
            if normal_clone and clone == 0:
                continue

            start = 0
            for length in lengths:
                current = 0
                for site in range(start, start + int(length)):
                    if n_states > 1 and site > start and rng.random() > self_transition:
                        current = int(
                            rng.choice([k for k in range(n_states) if k != current])
                        )
                    states[clone, site] = current
                start += int(length)

        tumour = states[1:] if normal_clone else states
        # NB one state has no aberration to carry: the null instance.
        aberrant = n_states == 1 or bool(np.all(np.any(tumour != 0, axis=1)))
        distinct = n_states == 1 or len({row.tobytes() for row in states}) == n_clones

        if aberrant and distinct:
            return states

    msg = "no identifiable clone paths in 1,000 draws; lower self_transition"
    raise RuntimeError(msg)


def emission_parameters(
    n_states: int, copy_lattice: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """`(log_mu, alphas, p_binom, taus, copies)` for the first `n_states` states."""
    if copy_lattice:
        copies = np.array(COPY_LATTICE[:n_states], dtype=np.int64)
        total = copies.sum(axis=1)
        mu = total / 2.0
        p_binom = np.where(copies[:, 1] == 0, LOH_P, copies[:, 0] / total)
    else:
        copies = None
        mu = np.array([m for m, _ in STATES[:n_states]])
        p_binom = np.array([p for _, p in STATES[:n_states]])

    if len(mu) < n_states:
        msg = f"{n_states} states requested, {len(mu)} planted"
        raise ValueError(msg)

    return (
        np.log(mu),
        np.full(n_states, ALPHA),
        p_binom.astype(np.float64),
        np.full(n_states, TAU),
        copies,
    )


def planted(
    *,
    n_clones: int = 3,
    n_states: int = 4,
    lattice: tuple[int, int] = (12, 10),
    n_obs: int = 120,
    n_segments: int = 4,
    self_transition: float = 0.95,
    depth: tuple[float, float] = (0.5, 3.0),
    exposure_scale: float = 20.0,
    reads: tuple[int, int] = (10, 60),
    switch: tuple[float, float] = (0.01, 0.20),
    normal_clone: bool = True,
    copy_lattice: bool = False,
    seed: int = DEFAULT_SEED,
) -> Truth:
    """Plant an instance and draw its counts.

    `depth` is the per-spot library factor and `exposure_scale` the mean
    count of a neutral bin at factor one; the per-bin profile is log-normal
    with unit mean, so the exposure is not a function of spot alone.
    """
    rng = np.random.default_rng(seed)

    lengths = segment_lengths(n_obs, n_segments, rng)
    labels = band_labels(lattice, n_clones, normal_clone)
    states = state_paths(
        n_clones, n_states, lengths, self_transition, normal_clone, rng
    )
    log_mu, alphas, p_binom, taus, copies = emission_parameters(n_states, copy_lattice)

    profile = rng.lognormal(0.0, 0.3, n_obs)
    profile /= profile.mean()

    switch_prob = rng.uniform(*switch, n_obs)
    flips = rng.random(n_obs) < switch_prob
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    flips[starts] = False
    phase = (np.cumsum(flips) % 2).astype(np.int64)

    n_spots = labels.shape[0]
    base_nb_mean = np.empty((n_obs, n_spots))
    total_bb_RD = np.empty((n_obs, n_spots), dtype=np.int64)
    counts_nb = np.empty((n_obs, n_spots), dtype=np.int64)
    counts_bb = np.empty((n_obs, n_spots), dtype=np.int64)

    for spot in range(n_spots):
        stream = np.random.default_rng([seed, spot])
        path = states[labels[spot]]

        base = exposure_scale * stream.uniform(*depth) * profile
        mean = base * np.exp(log_mu[path])
        r = 1.0 / alphas[path]
        counts_nb[:, spot] = stream.negative_binomial(r, r / (r + mean))

        trials = stream.integers(reads[0], reads[1] + 1, n_obs)
        p = p_binom[path]
        tau = taus[path]
        b = stream.binomial(trials, stream.beta(p * tau, (1.0 - p) * tau))

        base_nb_mean[:, spot] = base
        total_bb_RD[:, spot] = trials
        counts_bb[:, spot] = np.where(phase == 1, trials - b, b)

    return Truth(
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
        phase=phase,
        switch_prob=switch_prob,
        lattice=lattice,
        self_transition=self_transition,
        seed=seed,
        copies=copies,
    )
