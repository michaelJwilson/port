"""The planted truth of the in-memory fixtures: `CoreInferenceTruth` and its builders (T- #673 G6).

Truth is planted here and never recovered from the data: a test that reads a
parameter back out of the fixture it generated is checking arithmetic, not
recovery. Every builder takes a seed and returns it alongside the instance,
so a failure names the draw that produced it.

`core_inference_truth` plants clones, copy states and segments on a lattice;
`dev_instance`, `critical_instance`, `calicost_instance` and `key_instance`
are the named instances the audits and the ledger run (`dev` is `07b82e92`,
`tests.metrics.fixture_hash`). Moved from `tests/fixtures.py`, which keeps
the chain and Potts builders only unit tests read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sal.emissions import (
    BetaBinomialEmission,
    EmissionFamily,
    NegativeBinomialEmission,
)
from sal.ragged import Ragged

if TYPE_CHECKING:
    pass


DEFAULT_SEED = 11
"""The seed every builder defaults to, so a bare call is reproducible."""


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
    mirrored: tuple[tuple[int, int, int, int], ...] = ()
    """`(clone, mirror, offset, extent)`: bins where `mirror` carries `clone`'s
    LOH with the other allele lost, `loh=True` only."""

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
    from sal.sim.count_pairs import IndependentCountPair

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


MINIMUM_SEGMENT = 2
"""The shortest chromosome the fixture may plant: two bins, so each carries a transition.

Port's own since sal #1233 lowered `Ragged`'s floor to one position. The
alias it replaces moved with upstream, and that moved the dev instance's
partition and hash (07b82e92 to aa63209e) with no change here; a fixture's
draw is this repository's to state.
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

    The floor is `MINIMUM_SEGMENT`, this repository's: `Ragged` admits a
    segment of one position since sal #1233, and a planted chromosome keeps
    at least one transition.

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


NORMAL_SHARE = 0.3
"""The least share of spots the normal clone holds, when there is one (#298)."""


def clone_bands(
    rows: int, columns: int, n_clones: int, *, normal_clone: bool = True
) -> np.ndarray:
    """Clone label per spot: row bands, clone 0 at least `NORMAL_SHARE`.

    Clone 0 takes `ceil(0.3 * rows)` rows or its equal share, whichever is
    more, and the rest split the remaining rows as evenly as the rows
    allow. A single clone takes every row; it cannot also be normal, so it
    carries events as before.
    """
    row = np.arange(rows * columns) // columns

    if not normal_clone:
        equal: np.ndarray = np.minimum(row * n_clones // rows, n_clones - 1)

        return equal.astype(np.int64)

    if n_clones == 1:
        return np.zeros(rows * columns, dtype=np.int64)

    normal_rows = max(int(np.ceil(NORMAL_SHARE * rows)), rows // n_clones)
    rest = rows - normal_rows

    if rest < n_clones - 1:
        msg = (
            f"{rows} rows leave {rest} after the normal clone's {normal_rows}, "
            f"fewer than the {n_clones - 1} other clones"
        )
        raise ValueError(msg)

    others = 1 + (row - normal_rows) * (n_clones - 1) // rest
    labels: np.ndarray = np.where(row < normal_rows, 0, others).astype(np.int64)

    return labels


def clone_quadrants(
    rows: int, columns: int, n_clones: int, *, normal_clone: bool = True
) -> np.ndarray:
    """Clone label per spot: an axis-aligned `p x p` grid of rectangles (#347).

    `p = ceil(sqrt(n_clones))`, block `b` holding clone `b % n_clones`, the
    layout CalicoST's `rectangle_initialize_initial_clone` draws. Clone 0's
    block is the top-left, widened to `sqrt(NORMAL_SHARE)` of each side so
    it holds at least `NORMAL_SHARE` of the spots; the other splits are even.
    Every clone is a rectangle, and so is the union of two side-adjacent
    ones, which is what keeps CalicoST's initializer off its
    non-terminating case (`port.scripts.run_calicost.terminating`).
    """
    p = int(np.ceil(np.sqrt(n_clones)))
    first = np.sqrt(NORMAL_SHARE) if normal_clone and n_clones > 1 else 1.0 / p

    def cuts(extent: int) -> np.ndarray:
        rest = np.linspace(first, 1.0, p)[1:-1] if p > 2 else np.array([])
        edges = np.concatenate(([first], rest))[: p - 1] if p > 1 else np.array([])
        return np.rint(edges * extent).astype(int)

    row = np.arange(rows * columns) // columns
    column = np.arange(rows * columns) % columns
    block = np.searchsorted(cuts(rows), row, side="right") * p + np.searchsorted(
        cuts(columns), column, side="right"
    )
    labels: np.ndarray = (block % n_clones).astype(np.int64)

    return labels


def balanced_clone(truth: CoreInferenceTruth) -> int:
    """Which clone the fixture planted at the balanced state in most bins."""
    return int(
        np.argmax(
            [np.mean(truth.states[clone] == 0) for clone in range(truth.n_clones)]
        )
    )


COPY_LATTICE: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 4),
    (2, 4),
    (1, 5),
    (2, 2),
    (3, 3),
)
"""`(A, B)` allele copies for `copy_lattice=True`, the diploid normal first.

`mu = (A + B) / 2` against a diploid normal and `p = B / (A + B)`, at or
above balance as the unphased initializer requires, and every total within
`cnaster`'s `max_total_copy = 6`. No `A = 0` state here: `p = 1` is a
degenerate beta-binomial, and `LOH_STATES` carries LOH held off it. The two balanced amplifications come last, so a fixture of
seven or fewer states is identifiable from BAF as well as RDR.
"""


LOH_STATES: tuple[tuple[int, int], ...] = ((0, 2), (2, 0), (0, 1), (1, 0))
"""`(A, B)` states `loh=True` appends: copy-neutral LOH and hemizygous
deletion, each with its mirror, the other allele lost."""


LOH_EPSILON = 1e-5
"""How far an LOH state's `p` sits from 0 or 1: `p = 1` is a degenerate
beta-binomial, and `1 - 1e-5` at `tau = 30` gives the lost allele a beta
shape of `3e-4`, so a draw is all one allele to within the tolerance
`tests/test_loh_fixture.py` states."""


def spot_counts(
    family: Any,
    spot_states: np.ndarray,
    base_nb_mean: np.ndarray,
    total_bb_RD: np.ndarray,
    entropy: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """`(n_obs, n_spots)` count and success draws, spot `s` from `default_rng([*entropy, s])`.

    `spot_states[s]` is spot `s`'s state per bin; the covariate is its
    exposure and trials. The one draw `core_inference_truth` (`entropy =
    (seed,)`) and `realize` (`(genome_seed, seed)`) share (#749 WP9).
    """
    n_obs, n_spots = base_nb_mean.shape
    counts_nb = np.empty((n_obs, n_spots), dtype=np.float64)
    counts_bb = np.empty((n_obs, n_spots), dtype=np.float64)

    for spot in range(n_spots):
        covariate = np.stack([base_nb_mean[:, spot], total_bb_RD[:, spot]], axis=-1)
        drawn = family.sample(
            spot_states[spot],
            np.random.default_rng([*entropy, spot]),
            covariate=torch.as_tensor(covariate),
        )
        counts_nb[:, spot] = drawn[..., 0]
        counts_bb[:, spot] = drawn[..., 1]

    return counts_nb, counts_bb


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
    normal_clone: bool = True,
    labelling: str = "bands",
    copy_lattice: bool = False,
    loh: bool = False,
) -> CoreInferenceTruth:
    """Plant an instance, drawing every count through upstream's families.

    One stream per spot, `default_rng([seed, spot])`, so the draw is a function
    of the seed and the spot alone and not of the order they are visited in.
    That is what lets a reduced fixture be a prefix of a larger one rather than
    a different dataset.

    Parameters
    ----------
    labelling : str
        How spots are labelled with clones. `"bands"` lays them in row
        bands, the default; `"quadrants"` in an axis-aligned grid of
        rectangles (`clone_quadrants`, #347).
    copy_lattice : bool
        Plant integer allele copies, `COPY_LATTICE`, instead of the default
        grid of `mu` in `[1.5, 5]` and `p` in `[0.58, 0.88]`. The default grid
        is off the integer lattice -- `mu = 1.5` at `p = 0.58` is no `(A, B)`
        -- so it cannot referee an integer copy decoder, and three of its
        states exceed `cnaster`'s `max_total_copy = 6` (#313). Off by
        default, so every existing fixture draws what it drew.
    loh : bool
        With `copy_lattice`, append `LOH_STATES` after the `n_states` lattice
        states and plant them **mirrored**: each tumor clone carries a
        copy-neutral LOH and a hemizygous deletion, and the next tumor clone
        carries the same bins with the other allele lost, so the phase of
        the LOH flips between the two clones while `mu` does not. `p` is
        held `LOH_EPSILON` from 0 and 1. The events are drawn after every
        other draw, so the other clones' events are unchanged by it.
    normal_clone : bool
        Clone 0 all state 0 and at least `NORMAL_SHARE` of the spots (#298),
        which `cnaster`'s baseline needs. On by default at every size;
        `False` restores equal bands with events in every clone, for the
        tests whose subject is that layout.
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
        the segments do not partition `n_obs`, `n_states < 2`, `exposure`
        names no mode, or `loh` is asked without `copy_lattice` or of fewer
        than two tumor clones, which leaves no clone to mirror.
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
    if copy_lattice:
        if n_states > len(COPY_LATTICE):
            msg = f"the copy lattice has {len(COPY_LATTICE)} states, not {n_states}"
            raise ValueError(msg)

        table = COPY_LATTICE[:n_states] + (LOH_STATES if loh else ())
        copies = np.asarray(table, dtype=np.float64)
        log_mu = np.log(copies.sum(axis=1) / 2.0)
        p_binom = np.clip(
            copies[:, 1] / copies.sum(axis=1), LOH_EPSILON, 1.0 - LOH_EPSILON
        )
    elif loh:
        msg = "loh plants integer (A, B) states, so it needs copy_lattice"
        raise ValueError(msg)
    else:
        log_mu = np.concatenate(([0.0], np.log(np.linspace(1.5, 5.0, n_states - 1))))
        p_binom = np.concatenate(([0.5], np.linspace(0.58, 0.88, n_states - 1)))

    alphas = np.full(log_mu.size, 1.0 / 6.0)
    taus = np.full(log_mu.size, 30.0)

    if labelling == "bands":
        labels = clone_bands(rows, columns, n_clones, normal_clone=normal_clone)
    elif labelling == "quadrants":
        labels = clone_quadrants(rows, columns, n_clones, normal_clone=normal_clone)
    else:
        msg = f"unknown labelling {labelling!r}"
        raise ValueError(msg)

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

    # NB clone 0 is normal (#298): `cnaster` builds its baseline from normal
    #    spots (`determine_normal_baseline`), and a fixture with none hands
    #    it spots that share the events it divides out. Its path is drawn and
    #    then replaced rather than skipped, so every other clone's events are
    #    the ones the same seed drew before.
    if normal_clone and n_clones > 1:
        states[0] = 0
        placed_events = ((), *placed_events[1:])

    mirrored: tuple[tuple[int, int, int, int], ...] = ()
    if loh:
        states, placed_events, mirrored = _mirror_loh(
            states,
            placed_events,
            lengths,
            first=n_states,
            tumor=list(range(1 if normal_clone else 0, n_clones)),
            event_bins=extent,
            rng=np.random.default_rng([seed, 0, 1]),  # no spot stream is three long
        )

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

    counts_nb, counts_bb = spot_counts(
        _emission_families(log_mu, alphas, p_binom, taus),
        states[labels],
        base_nb_mean,
        total_bb_RD,
        (seed,),
    )

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
        mirrored=mirrored,
    )


def _mirror_loh(
    states: np.ndarray,
    placed_events: tuple[tuple[tuple[int, int, int, int], ...], ...],
    lengths: np.ndarray,
    *,
    first: int,
    tumor: list[int],
    event_bins: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[int, int, int, int], ...], ...],
    tuple[tuple[int, int, int, int], ...],
]:
    """Each tumor clone's two LOH events, and the next one's mirror of them.

    `first` is the index of `LOH_STATES[0]`; a state and its mirror are
    adjacent there, `first + 2 j` and `first + 2 j + 1`. The mirror is the
    next tumor clone, cyclically, so every tumor clone carries both phases.
    Events sit inside one chromosome, as `place_events`' do, and overwrite
    what is under them in both clones, but never each other: a pair drawn
    over an earlier pair's bins is redrawn, so every pair `mirrored` records
    is intact in the path.
    """
    if len(tumor) < 2:
        msg = f"mirrored LOH needs two tumor clones, got {len(tumor)}"
        raise ValueError(msg)

    states = states.copy()
    events = [list(placed) for placed in placed_events]
    edges = np.concatenate(([0], np.cumsum(lengths)))
    taken = np.zeros(states.shape[1], dtype=bool)
    mirrored = []

    for i, clone in enumerate(tumor):
        mirror = tumor[(i + 1) % len(tumor)]

        for j in range(len(LOH_STATES) // 2):
            for _ in range(1_000):
                chromosome = int(rng.integers(lengths.size))
                start, stop = int(edges[chromosome]), int(edges[chromosome + 1])
                extent = min(int(rng.integers(*event_bins)), stop - start)
                offset = int(rng.integers(start, stop - extent + 1))
                if not taken[offset : offset + extent].any():
                    break
            else:
                msg = "no room left for a disjoint mirrored LOH event"
                raise ValueError(msg)

            taken[offset : offset + extent] = True
            state = first + 2 * j

            states[clone, offset : offset + extent] = state
            states[mirror, offset : offset + extent] = state + 1
            events[clone].append((chromosome, offset, extent, state))
            events[mirror].append((chromosome, offset, extent, state + 1))
            mirrored.append((clone, mirror, offset, extent))

    return states, tuple(tuple(e) for e in events), tuple(mirrored)


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


def calicost_instance(**overrides: object) -> CoreInferenceTruth:
    """The instance `run_calicost` is compared with `port` on (#347).

    **Size:**

    | | |
    | --- | --- |
    | spots | 1,600, a `40 x 40` square lattice |
    | clones | 4, planted as quadrants: 484 (normal), 396, 396, 324 spots |
    | bins | 1,000 over 10 ragged chromosomes (69 to 182 bins) |
    | states | 10 planted, 8 of them used; `mu` 1 to 5, `p` 0.5 to 0.88 |
    | altered bins | 0, 67, 92 and 64 of 1,000 per clone |
    | seed | 11 |

    That is `dev_instance` with one change, the spatial layout: the genome,
    the states and every clone's path are its own, drawn from the same
    stream. Only which spot carries which clone differs, and so the clone
    sizes (dev's bands are 480, 400, 360, 360).

    **Why the layout changes.** CalicoST splits each BAF clone by read depth
    from an initial layout of `ceil(sqrt(n_clones_rdr))^2` rectangles, and at
    `n_clones_rdr = 4` its retry loop cannot exit when one holds under 5 per
    cent of the clone's spots (`port.scripts.run_calicost.terminating`). The
    dev instance's row bands leave BAF clones on which it does not. Planted
    as rectangles, the BAF clones are rectangles too, and the comparison with
    `port` runs at `n_clones_rdr = 4`.
    """
    settings: dict[str, object] = {
        "n_clones": 4,
        "n_states": 10,
        "lattice": (40, 40),
        "n_obs": 1_000,
        "n_segments": 10,
        "labelling": "quadrants",
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


def fixture_hash(truth: Any) -> str:
    """A digest of the data a fixture built, not of the code that built it.

    Every field of the `CoreInferenceTruth`, by dtype, shape and bytes: a run
    reproduces only where the same arrays still come out, whatever changed in
    between.
    """
    digest = hashlib.sha256()
    for field in fields(truth):
        value = getattr(truth, field.name)
        digest.update(field.name.encode())
        if isinstance(value, np.ndarray):
            digest.update(f"{value.dtype.str}{value.shape}".encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:8]
