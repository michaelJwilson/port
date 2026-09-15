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
from snakes_and_ladders.emissions import NegativeBinomialEmission
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
