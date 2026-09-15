"""Conversions from a `snakes_and_ladders` instance to `cnaster`'s arguments.

The adapter is the reusable half of every test in this repository: the
assertions are short once the conversion exists, and a second test of the
same entry point extends this module rather than restating it.

**The correspondence this module rests on.** `cnaster` conditions on
`base_nb_mean` and `total_bb_RD` as data and never fits them, so any arrays
supplied here are a valid instance of its likelihood. Two consequences are
load-bearing and each is pinned by a test rather than assumed:

* A *constant* exposure is exact, absorbed as `log_mu - log(c)`, because
  `cnaster` forms `lam = exposure * mu` per observation while the upstream
  family carries one mean per state. Exposure that varies along a chain has
  no upstream form today and is out of scope here.
* A zero `total_bb_RD` makes the beta-binomial channel contribute exactly
  zero, which is what isolates the count channel.
"""

from dataclasses import dataclass

import numpy as np

from tests.fixtures import NegativeBinomialChains

N_CHANNELS = 2
"""`cnaster` packs a count and a success into `single_X`'s middle axis."""

RDR_CHANNEL = 0
BAF_CHANNEL = 1


@dataclass(frozen=True)
class CnasterChainInputs:
    """`cnaster`'s arguments for a chain, and the parameters to score it at.

    The chains are laid end to end along the genomic axis and `lengths`
    restarts the recursion at each boundary, which is how `cnaster` carries
    more than one sequence through one lattice.
    """

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
    """Lay the fixture's chains out as `cnaster` expects them.

    Parameters
    ----------
    exposure : float
        The constant `base_nb_mean`. Absorbed into `log_mu` so the scored
        model is unchanged; varying it is what the upstream count family
        cannot yet express, so it is a scalar here and not an array.

    Raises
    ------
    ValueError
        If `exposure` is not strictly positive, where the mean is undefined.
    """
    if exposure <= 0.0:
        msg = f"exposure must be strictly positive, got {exposure}"
        raise ValueError(msg)

    observations = fixture.dataset.observations
    n_sequences, sequence_length = observations.shape
    n_obs = n_sequences * sequence_length
    n_states = fixture.n_states

    single_X = np.zeros((n_obs, N_CHANNELS, 1), dtype=np.float64)
    single_X[:, RDR_CHANNEL, 0] = observations.reshape(-1)

    # NB the beta-binomial channel is inert at zero depth, which is what
    #    leaves the count channel alone under test; `test_hmm_single_chain`
    #    pins that rather than trusting it.
    base_nb_mean = np.full((n_obs, 1), exposure, dtype=np.float64)
    total_bb_RD = np.zeros((n_obs, 1), dtype=np.float64)

    # NB `lam = exposure * exp(log_mu)`, so a constant exposure moves into
    #    log_mu and the mean each state scores at is unchanged.
    log_mu = (np.log(fixture.mean) - np.log(exposure))[:, None]
    alphas = (1.0 / fixture.dispersion)[:, None]

    # NB unused where the depth is zero; declared at the balanced point so a
    #    leak into the score would be visible rather than plausible.
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


def cnaster_emission(inputs: CnasterChainInputs) -> np.ndarray:
    """`cnaster`'s per-state emission score, shape `(n_states, n_obs)`.

    Both channels summed, as `pipeline_baum_welch` sums them before the
    recursion, and the trailing spot axis dropped: one spot is what a chain
    with no spatial layer has.
    """
    from cnaster.hmm_nophasing import hmm_nophasing

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
    emission: np.ndarray = (log_emit_rdr + log_emit_baf)[:, :, 0]
    return emission


def cnaster_total_log_likelihood(inputs: CnasterChainInputs) -> float:
    """The summed forward log-likelihood over the fixture's chains.

    `forward_lattice` returns `log alpha` over the concatenated axis, so the
    total is the marginal at each chain's last position, summed: the chains
    are independent and the recursion restarts at every boundary `lengths`
    declares.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from scipy.special import logsumexp

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
    log_alpha = hmm_nophasing.forward_lattice(
        inputs.lengths,
        inputs.log_transmat,
        inputs.log_startprob,
        log_emit_rdr + log_emit_baf,
        inputs.log_sitewise_transmat,
    )

    ends = np.cumsum(inputs.lengths) - 1
    return float(sum(logsumexp(log_alpha[:, end]) for end in ends))
