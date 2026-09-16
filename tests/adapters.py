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
from snakes_and_ladders.emissions import BetaBinomialEmission

from tests.fixtures import BetaBinomialChains, NegativeBinomialChains, PhasedChains

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


@dataclass(frozen=True)
class CnasterPhasedInputs:
    """`cnaster`'s arguments for the phased lattice.

    `hmm_phased.forward_lattice` takes the `K x K` base and assembles the
    `2K x 2K` matrix per position from the sitewise kernel, so the base and
    the kernel are carried separately here rather than pre-combined.

    The emission arrives as an array. That is the lattice's own contract,
    and it is also what keeps this rung testable: `cnaster`'s phased emission
    raises before it returns (issue #9), so supplying the scores directly is
    what separates the transfer matrix from a defect below it.
    """

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
    """Lay a phased fixture out as `hmm_phased.forward_lattice` expects it.

    Parameters
    ----------
    switch : float | None
        Overrides the fixture's constant phase kernel, so a test can show
        the lattice reads it. `None` keeps the one the draw used, which is
        the only value the upstream comparison is valid at: a kernel that
        varies by position, or differs from the one that generated the data,
        is not the matrix upstream was handed.
    """
    import torch

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
    from cnaster.hmm_phased import hmm_phased
    from scipy.special import logsumexp

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


@dataclass(frozen=True)
class MStepResult:
    """One re-estimation, at the parameterization both sides share.

    `snakes_and_ladders` returns a `Reestimate` carrying what its inner
    solve had to report; `cnaster` returns an `OptimizationResult` carrying
    a different set. This is the intersection, so a comparison reads as one
    table rather than two.

    Parameters
    ----------
    alpha, beta : np.ndarray
        The re-estimated pair, shape `(n_states,)`.
    converged : bool
        Whether the inner solve settled. Both sides report it; neither
        reports it the same way, which is why it is carried rather than
        asserted inside the adapter.
    iterations : int
        Iterations the solve took, or `-1` where the implementation does not
        say. Never compared -- the two solve by different methods, so the
        counts are not commensurate and only their finiteness means anything.
    seconds : float
        Wall time for the call, for the benchmark. Measured here so the
        comparison times the same span on both sides: the solve and nothing
        around it.
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
    """`BetaBinomialEmission.reestimate`, as the referee.

    The family is rebuilt at the fixture's planted parameters rather than
    reused, so the starting point is stated here and a caller cannot leave a
    previous step's answer in it.
    """
    import time

    import torch

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
        alpha=np.asarray(result.emissions.alpha, dtype=np.float64),
        beta=np.asarray(result.emissions.beta, dtype=np.float64),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        seconds=seconds,
    )


def cnaster_beta_binomial_design(
    fixture: BetaBinomialChains, posterior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(endog, exog, weights, exposure)` for `Weighted_BetaBinom_mix`.

    `cnaster` takes the M step as a *regression*: one row per
    `(observation, state)` pair, the state written into a one-hot `exog`, and
    the posterior for that pair as the row's weight. The upstream family
    takes the same problem as an array of observations and a posterior with
    a trailing state axis. Flattening one into the other is the whole of the
    correspondence, and it is exact rather than approximate -- the two
    objectives are the same sum, written with the state index in a different
    place.

    The row order is `(observation, state)`, so `weights` is the posterior
    read in C order and needs no permutation. Stated because getting it
    wrong permutes the states rather than failing: the fit would still
    converge, to the wrong assignment.
    """
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
    """`Weighted_BetaBinom_mix.fit`, driven as the live caller drives it.

    The solver options come from `cnaster.hmm_utils.get_em_solver_params`,
    which is what `normal_spot.normal_baf_bin_filter` splats into `fit` on
    the `run_cnaster` path. Passing them rather than relying on `fit`'s own
    defaults is not a convenience: `fit` reads `kwargs.get("ftol", None)` and
    hands `None` to `L-BFGS-B`, which divides by it. The live route never
    hits that because it always supplies the settings, and neither does this.

    Parameters
    ----------
    shared_dispersion : bool
        `False` gives one `tau` per state, which is the upstream family's
        shape and the only branch the comparison can be exact on. `cnaster`
        defaults to `True`; the live caller fits one state, where the two
        branches are the same parameter, so neither is the more live of the
        two.
    """
    import time

    from cnaster.hmm_emission import Weighted_BetaBinom_mix, compute_bb_ab
    from cnaster.hmm_utils import get_em_solver_params

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
    """`cnaster`'s weighted negative log-likelihood at a given `(alpha, beta)`.

    The objective its M step minimises, evaluated through `cnaster`'s own
    `nloglikeobs` so a monotonicity claim is made against the function that
    was optimized rather than against a restatement of it.
    """
    from cnaster.hmm_emission import Weighted_BetaBinom_mix, betabinom_logpmf_zp

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
