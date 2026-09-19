"""`cnaster`'s four live emission entry points, against each other and against scipy.

**Step 1 of #205's plan: the referee the refactor is measured against.** Two
classes carry four callable emission paths between them and a fifth is kept
as a string literal; `tests/test_emission_consistency.py` pins two of the
four against each other, and nothing pins the phased pair to the unphased
one at all. A refactor cannot be shown to preserve a relation nobody wrote
down.

Three of the claims here are `patch` -- two implementations agreeing, with
neither designated right, which is #9's open question. The fourth is an
`oracle`: **the switch term is the allele swap**, and `scipy` is what says
so. That one is worth more than the others, because every phased emission in
the pipeline is built on it and a sign error there is invisible to any test
that compares `cnaster` with itself.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import scipy.stats

SWAP_TOLERANCE = 1.0e-12
"""How far the switch term may sit from `scipy` on the swapped allele.

Realized 6.0e-14 over three states and twelve distinct `(k, n)` pairs. The
two compute the same quantity by different groupings of log-gammas, so this
is reassociation rather than a different formula, and a departure at 1e-12
is a defect rather than noise.
"""


@dataclass(frozen=True)
class EmissionInputs:
    """Both channels live, which is what the phased claims need.

    `negative_binomial_chains` plants the read-depth channel alone, so its
    `total_bb_RD` is zero and every beta-binomial score on it is `0.0` --
    under which the switched half of the phased emission equals the
    unswitched half and the relations below hold vacuously. A fixture that
    makes a claim true by carrying no data is the coverage theatre
    `CLAUDE.md` forbids, so the allele channel is drawn here.

    Seeded, and synthetic rather than planted, because what is asserted is an
    algebraic relation between two implementations and not the recovery of
    anything: no generative truth is involved on either side.
    """

    single_X: np.ndarray
    base_nb_mean: np.ndarray
    total_bb_RD: np.ndarray
    log_mu: np.ndarray
    alphas: np.ndarray
    p_binom: np.ndarray
    taus: np.ndarray


def _inputs(n_states: int, *, n_obs: int = 60, n_spots: int = 1) -> EmissionInputs:
    """Counts and parameters at `cnaster`'s shapes, with repeats to deduplicate."""
    generator = np.random.default_rng(17)

    exposure = generator.integers(20, 60, (n_obs, n_spots)).astype(np.float64)
    trials = generator.integers(5, 40, (n_obs, n_spots)).astype(np.float64)

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = generator.poisson(exposure)
    single_X[:, 1, :] = generator.binomial(trials.astype(int), 0.45)

    def column(values: np.ndarray) -> np.ndarray:
        """One value per state, repeated across the spot axis."""
        return np.tile(np.asarray(values)[:, None], (1, n_spots))

    return EmissionInputs(
        single_X=single_X,
        base_nb_mean=exposure,
        total_bb_RD=trials,
        log_mu=column(np.linspace(-0.4, 0.4, n_states)),
        alphas=column(np.linspace(0.1, 0.6, n_states)),
        p_binom=column(np.linspace(0.2, 0.83, n_states)),
        taus=column(np.linspace(7.5, 30.0, n_states)),
    )


def _phased_emission(
    inputs: "EmissionInputs", *, clone_stack: bool = False
) -> "tuple[Any, Any]":
    from cnaster.hmm_phased import hmm_phased

    scored: tuple[Any, Any] = hmm_phased.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        inputs.log_mu,
        inputs.alphas,
        inputs.total_bb_RD,
        inputs.p_binom,
        inputs.taus,
        clone_stack=clone_stack,
    )
    return scored


def _unphased_emission(inputs: "EmissionInputs") -> "tuple[Any, Any]":
    from cnaster.hmm_nophasing import hmm_nophasing

    scored: tuple[Any, Any] = hmm_nophasing.compute_emission_probability_nb_betabinom(
        inputs.single_X,
        inputs.base_nb_mean,
        inputs.log_mu,
        inputs.alphas,
        inputs.total_bb_RD,
        inputs.p_binom,
        inputs.taus,
    )
    return scored


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_switch_term_is_the_allele_swap(n_states: int) -> None:
    """`_switch_betabinom_1d` scores the B allele as `scipy` scores `n - k`.

    The phased model's second half of states is the same copy state with the
    haplotypes exchanged, so its emission must be the beta-binomial evaluated
    at the complementary count. `cnaster` reaches it by adding four
    log-gammas to the unswitched score rather than by re-evaluating, which is
    the same identity written for speed -- and is the one place a sign error
    would leave every phased emission wrong and every `cnaster`-against-
    `cnaster` comparison green.
    """
    from cnaster.hmm_nophasing import _bb_logpmf_1d
    from cnaster.hmm_phased import _switch_betabinom_1d

    generator = np.random.default_rng(11)
    successes = generator.integers(0, 40, 12).astype(np.float64)
    totals = successes + generator.integers(1, 40, 12).astype(np.float64)

    p_binom = np.linspace(0.2, 0.83, n_states)
    taus = np.linspace(7.5, 30.0, n_states)

    unswitched = np.zeros((n_states, successes.size))
    for state in range(n_states):
        _bb_logpmf_1d(successes, totals, p_binom[state], taus[state], unswitched[state])

    switched = _switch_betabinom_1d(unswitched, successes, totals, p_binom, taus)

    expected = np.vstack(
        [
            scipy.stats.betabinom.logpmf(
                totals - successes,
                totals,
                p_binom[state] * taus[state],
                (1.0 - p_binom[state]) * taus[state],
            )
            for state in range(n_states)
        ]
    )

    realized = float(np.abs(switched - expected).max())
    assert realized < SWAP_TOLERANCE, (
        f"switch term departs from scipy by {realized:.3g}"
    )


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3])
def test_the_phased_wrapper_is_the_phased_coded_path(n_states: int) -> None:
    """`hmm_phased`'s dense entry point builds encoders and calls the other one.

    Pinned rather than read, because the two are separate public names and a
    refactor that keeps only one has to know they were never different.
    """
    from cnaster.count_encoder import CountEncoder
    from cnaster.hmm_phased import hmm_phased

    inputs = _inputs(n_states)

    wrapper_rdr, wrapper_baf = _phased_emission(inputs)
    coded_rdr, coded_baf = hmm_phased.compute_emission_probability_nb_betabinom_coded(
        CountEncoder(inputs.single_X[:, 0, :], inputs.base_nb_mean),
        CountEncoder(inputs.single_X[:, 1, :], inputs.total_bb_RD),
        inputs.log_mu,
        inputs.alphas,
        inputs.p_binom,
        inputs.taus,
        clone_stack=False,
    )

    np.testing.assert_array_equal(wrapper_rdr, coded_rdr)
    np.testing.assert_array_equal(wrapper_baf, coded_baf)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_phased_read_depth_is_the_unphased_one_twice(n_states: int) -> None:
    """Phasing exchanges haplotypes, so it cannot move the read depth.

    Both halves of the phased RDR must be the unphased RDR. This crosses the
    two implementations -- the unphased dense kernels against the phased
    deduplicated path -- so it is the relation that says the state doubling
    is a relabelling of the BAF channel alone.
    """
    inputs = _inputs(n_states)

    unphased_rdr, _ = _unphased_emission(inputs)
    phased_rdr, _ = _phased_emission(inputs)

    assert phased_rdr.shape[0] == 2 * n_states

    np.testing.assert_array_equal(phased_rdr[:n_states], unphased_rdr)
    np.testing.assert_array_equal(phased_rdr[n_states:], unphased_rdr)


@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize("n_states", [1, 3, 5])
def test_the_phased_allele_channel_opens_with_the_unphased_one(n_states: int) -> None:
    """The first half of the phased BAF is the unphased BAF, unswitched.

    Together with the switch oracle above this pins the whole phased
    emission: the first half is what the unphased path computes, the second
    half is the allele swap of it, and nothing else is in the array.
    """
    inputs = _inputs(n_states)

    _, unphased_baf = _unphased_emission(inputs)
    _, phased_baf = _phased_emission(inputs)

    np.testing.assert_array_equal(phased_baf[:n_states], unphased_baf)

    # The switched half is a different score, or the doubling buys nothing
    # and the relation above holds because the channel is empty.
    assert np.abs(unphased_baf).max() > 0.0, "the allele channel carries no data"
    assert not np.array_equal(phased_baf[n_states:], unphased_baf)


EVIDENCE_TOLERANCE = 1.0e-12
"""How far the two recursions may sit apart on an instance where they agree.

Realized 3.6e-15 over two chains and three states. Both sum the same terms;
the phased one sums them over a doubled state space with a per-position
combined matrix, so the gap is reassociation and nothing else.
"""


@pytest.mark.analytic
@pytest.mark.parametrize("switch", [0.0, 1.0e-12, 1.0e-8])
@pytest.mark.parametrize("n_states", [2, 3, 5])
def test_phasing_a_phase_free_emission_changes_no_evidence(
    n_states: int, switch: float
) -> None:
    """With no switch to make, the doubled chain scores what the single one does.

    The model's own identity, and the only place the two recursions can be
    compared at all: `hmm_phased` overrides both lattices rather than
    parameterizing `hmm_nophasing`'s, so nothing else relates them. Give both
    phases the same emission and forbid the switch, and the doubling is two
    independent copies of one chain, each entered with probability one half
    -- so the marginal is unchanged.

    This is what step 3 of #205 has to preserve when the two become one call
    with the state space as an argument. Parametrized over a switch that is
    exactly zero and two that are merely small, because `-inf` and a tiny
    finite log are different arithmetic and only one of them is what the
    pipeline passes.

    `hmm_nophasing.forward_lattice` takes a `log_sitewise_transmat` it never
    reads -- a parameter carried for signature compatibility with the
    override, which is the seam this refactor removes.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from cnaster.hmm_phased import hmm_phased
    from scipy.special import logsumexp

    generator = np.random.default_rng(3)
    lengths = np.array([25, 15])
    n_obs = int(lengths.sum())

    emission = generator.normal(size=(n_states, n_obs, 1))
    paired = np.vstack([emission, emission])

    log_startprob = np.log(np.full(n_states, 1.0 / n_states))
    self_transition = 0.9
    off = (1.0 - self_transition) / (n_states - 1)
    log_transmat = np.log(
        np.full((n_states, n_states), off) + np.eye(n_states) * (self_transition - off)
    )

    sitewise = np.full(n_obs, np.log(switch) if switch > 0.0 else -np.inf)

    def evidence(alpha: np.ndarray) -> float:
        return float(sum(logsumexp(alpha[:, end - 1]) for end in np.cumsum(lengths)))

    unphased = evidence(
        hmm_nophasing.forward_lattice(
            lengths, log_transmat, log_startprob, emission, sitewise
        )
    )
    phased = evidence(
        hmm_phased.forward_lattice(
            lengths, log_transmat, log_startprob, paired, sitewise
        )
    )

    assert abs(phased - unphased) < EVIDENCE_TOLERANCE, (
        f"phased evidence {phased:.12f} against unphased {unphased:.12f}"
    )
