"""`cnaster.hmm_nophasing` at the pin, its four log-pmf kernels `port`'s (T- #670 PR4).

`_nb_logpmf_1d` and `_dense_nb_logpmf` come from `cnamaste.nb_logpmf` (#560),
`_bb_logpmf_1d` and `_dense_bb_logpmf` from `cnamaste.bb_logpmf` (#561): log
space, so a vanishing mean cannot score a count at probability 1, and exact
at a large concentration. Bound here by import, so `hmm_phased` and every
compiled caller read them under their `cnaster` names. `cnaster`'s
`nbinom_logpmf_numba` and `betabinom_logpmf_numba`, which only its kernels
called, are gone with them.

**T- #670 PR5: the fit is `port`'s `hmm_nophasing`** (row 26, #276, #433),
below the marked seam. `cnaster`'s class is kept as `_cnaster_hmm_nophasing`,
which `port`'s subclasses and `hmm_phased` keeps as its base, as `port`'s
does. Two options, class attributes as in `port`: `analytic_gradient`, on,
the M step's gradient in closed form (#433, #244); `apply_logmu_shift`, off
as in `cnaster`, the per-clone `log Z_c` folded into the depth channel
(#276). `port`'s third, `emission_kernels`, moved in by T- #670 PR6b once
`cnamaste` declared `sal`: `"cnaster"` by default, or `"sal"`, which scores
the coded emission with `sal`'s dense tables (`cnamaste.dense_emission`,
#425), to a tolerance rather than bitwise.

`_run_optimization_pipeline` no longer takes `tumor_prop` (#135): it never
read it, so a proportion handed in was fitted under no mixture with nothing
said. `hmm.pipeline_baum_welch` warns where one is given.
"""

import pprint
import time
from math import exp, lgamma, log

import numpy as np
import scipy.optimize
import scipy.special
from numba import njit, prange
from scipy.optimize import OptimizeResult

from cnamaste.bb_logpmf import _bb_logpmf_1d, _dense_bb_logpmf
from cnamaste.config import start_time
from cnamaste.count_encoder import CountEncoder
from cnamaste.logger import get_logger
from cnamaste.nb_logpmf import _dense_nb_logpmf, _nb_logpmf_1d

# NB what `port`'s `hmm_nophasing` below imports (T- #670 PR5).
from typing import Any, NamedTuple

from cnamaste.clone_paths import state_vector
from cnamaste.config import get_global_config
from cnamaste.gradient import EmGradient, analytic_bfgs
from cnamaste.logmu_shift import shifts as logmu_shifts

logger = get_logger(__name__, start_time=start_time)


@njit(cache=True, inline="always", fastmath=False, error_model="numpy")
def np_sum_ax_squeeze(arr, axis=0):
    return np.sum(arr, axis=axis)


@njit
def numba_logsumexp(a):
    a_max = np.max(a)
    if np.isinf(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def get_log_transmat(n_states, t):
    if n_states > 1:
        transmat = np.ones((n_states, n_states)) * (1.0 - t) / (n_states - 1)
        np.fill_diagonal(transmat, t)
        log_transmat = np.log(transmat)
    else:
        log_transmat = np.zeros((1, 1))

    return log_transmat


@njit(nogil=True, cache=True, parallel=False, error_model="numpy")
def compute_logmu_shifts(log_mus, copy_states, normal_log_lambda, clone_lengths):
    # NB per-clone shift in log_mu due to (clone) library normalization, used to
    #    debias inferred mus; assumes clones concatenate along the genomic axis.
    """
    return scipy.special.logsumexp(
        log_mu[clone_copy_states, :] + normal_log_lambda.reshape(-1, 1),
        axis=0,
    )
    """
    n_clones = len(clone_lengths)
    n_segments = len(copy_states)
    
    logmu_shifts = np.empty(n_segments, dtype=np.float64)
    
    start_idx = 0
    
    for c in range(n_clones):
        clone_len = clone_lengths[c]
        
        max_val = -np.inf

        for i in range(clone_len):
            idx = start_idx + i
            state = copy_states[idx]
            val = log_mus[state] + normal_log_lambda[idx]
            
            if val > max_val:
                max_val = val

        if np.isinf(max_val):
            shift_val = max_val
        else:
            sum_exp = 0.0
            for i in range(clone_len):
                idx = start_idx + i
                state = copy_states[idx]
                val = log_mus[state] + normal_log_lambda[idx]
                
                sum_exp += np.exp(val - max_val)
                
            shift_val = max_val + np.log(sum_exp)
            
        logmu_shifts[start_idx : start_idx + clone_len] = shift_val
            
        start_idx += clone_len
        
    return logmu_shifts

class _cnaster_hmm_nophasing:
    def __init__(self, params="stmp", t=1 - 1e-4):
        self.params = params
        self.t = t

        # NB small alpha tend to Poisson. 0.1 ->
        # NB large dispersions tend to Binomial, flat landscape, initialize just before.  30 -> 1_000
        # self.n_states = n_states
        # self.default_log_mu = np.linspace(-0.1, 0.1, n_states)
        # self.default_p_binom = np.linspace(0.05, 0.45, n_states)
        # self.default_alphas = 0.5 * np.ones(n_states)
        # self.default_taus = 1_000 * np.ones(n_states)
        # self.default_log_startprob = np.log(np.ones(n_states) / n_states)

    # TODO call coded
    @staticmethod
    def compute_emission_probability_nb_betabinom(
        X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
    ):
        # X is shape (n_obs, 2, n_spots). Split into NB (index 0) and BB (index 1) arrays
        log_emit_rdr = _dense_nb_logpmf(X[:, 0, :], base_nb_mean, log_mu, alphas)
        log_emit_baf = _dense_bb_logpmf(X[:, 1, :], total_bb_RD, p_binom, taus)

        return log_emit_rdr, log_emit_baf

    """
    @staticmethod
    def compute_emission_probability_nb_betabinom_coded(
        nbEncoder, bbEncoder, log_mu, alphas, p_binom, taus
    ):
        # TODO assumes called on each clone independently.
        n_states = log_mu.shape[0]

        nb_endog = nbEncoder.get_unique_obs(0)
        nb_exposure = nbEncoder.get_unique_total(0)

        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        log_emit_rdr_uniq = np.zeros((n_states, len(nb_endog)))
        log_emit_baf_uniq = np.zeros((n_states, len(bb_endog)))

        for i in range(n_states):
            log_emit_rdr_uniq[i, :] = _nb_logpmf_1d(
                nb_endog, nb_exposure, exp(log_mu[i, 0]), alphas[i, 0]
            )
            log_emit_baf_uniq[i, :] = _bb_logpmf_1d(
                bb_endog, bb_exposure, p_binom[i, 0], taus[i, 0]
            )

        log_emit_rdr = nbEncoder.decode_array(log_emit_rdr_uniq, 0)
        log_emit_baf = bbEncoder.decode_array(log_emit_baf_uniq, 0)

        return log_emit_rdr, log_emit_baf
    """

    def compute_emission_probability_nb_betabinom_coded(
        self,
        nbEncoder,
        bbEncoder,
        log_mu,
        alphas,
        p_binom,
        taus,
        clone_stack=True,
        scratch_rdr=None,
        scratch_baf=None,
        normal_log_lambda=None,
        clone_lengths=None,
    ):
        n_states = log_mu.shape[0]
        n_spots = nbEncoder.n_spots

        assert bbEncoder.n_spots == n_spots

        log_emit_rdr_list, log_emit_baf_list = [], []

        # NB typically, n_spot is unity as clones are concatenataed along the genomic axis.
        for s in range(n_spots):
            nb_endog = nbEncoder.get_unique_obs(s)
            nb_exposure = nbEncoder.get_unique_total(s)

            bb_endog = bbEncoder.get_unique_obs(s)
            bb_exposure = bbEncoder.get_unique_total(s)

            log_emit_rdr_uniq = (
                scratch_rdr[s] if scratch_rdr else np.zeros((n_states, len(nb_endog)))
            )
            log_emit_baf_uniq = (
                scratch_baf[s] if scratch_baf else np.zeros((n_states, len(bb_endog)))
            )

            for i in range(n_states):
                if normal_log_lambda is not None:
                    # log_gamma = self.get_state_posteriors()
                    # copy_states = self.get_copy_states(log_gamma, includes_phased=False)

                    # NB clone concatenated
                    # logmu_shifts = compute_logmu_shifts(log_mu, copy_states, normal_log_lambda, clone_lengths)
                    logger.warning("logmu_shifts are not currently supported.")
                    

                # TODO fold in logmu_shifts; assumed concatenated (repeated) along the genomic axis.
                _nb_logpmf_1d(
                    nb_endog,
                    nb_exposure,
                    exp(log_mu[i, s]),
                    alphas[i, s],
                    log_emit_rdr_uniq[i, :],
                )
                _bb_logpmf_1d(
                    bb_endog,
                    bb_exposure,
                    p_binom[i, s],
                    taus[i, s],
                    log_emit_baf_uniq[i, :],
                )

            log_emit_rdr_list.append(nbEncoder.decode_array(log_emit_rdr_uniq, s))
            log_emit_baf_list.append(bbEncoder.decode_array(log_emit_baf_uniq, s))

        if clone_stack:
            # NB concatenate clones along the genomic axis (n_states, total_obs) -> (n_states, total_obs, 1)
            log_emit_rdr = np.concatenate(
                log_emit_rdr_list, axis=1
            )
            log_emit_baf = np.concatenate(
                log_emit_baf_list, axis=1
            )
        else:
            # NB (n_states, n_obs, n_spots)
            log_emit_rdr = np.stack(log_emit_rdr_list, axis=2)
            log_emit_baf = np.stack(log_emit_baf_list, axis=2)

        return log_emit_rdr, log_emit_baf

    @staticmethod
    @njit
    def forward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        """
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.

        Input
            log_emission: n_states * n_observations * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states.  Transition probability.
            log_startprob: n_states. Start probability after log transformation.
        Output
            log_alpha: size n_states * n_observations. log alpha[j, t] = log P(o_1, ... o_t, q_t = j | lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]

        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"

        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"

        log_alpha = np.zeros((n_states, n_obs))
        buf = np.zeros(n_states)
        cumlen = 0

        for le in lengths:
            # NB initialize with start_prob and emission of first obs. for each item of lengths,
            #    e.g. contig.  Treats last axis (spots/clones) as iid (TBC).
            log_alpha[:, cumlen] = log_startprob + np_sum_ax_squeeze(
                log_emission[:, cumlen, :], axis=1
            )

            for t in np.arange(1, le):
                for j in np.arange(n_states):
                    for i in np.arange(n_states):
                        buf[i] = log_alpha[i, (cumlen + t - 1)] + log_transmat[i, j]

                    log_alpha[j, (cumlen + t)] = numba_logsumexp(buf) + np.sum(
                        log_emission[j, (cumlen + t), :]
                    )

            cumlen += le

        return log_alpha

    @staticmethod
    @njit
    def backward_lattice(
        lengths,
        log_transmat,
        log_startprob,
        log_emission,
        log_sitewise_transmat,
    ):
        """
        Note that n_states is the CNV states, and there are n_states of paired states for (CNV, phasing) pairs.

        Input
            X: size n_observations * n_components * n_spots.
            lengths: sum of lengths = n_observations.
            log_transmat: n_states * n_states. Transition probability after log transformation.
            log_startprob: n_states. Start probability after log transformation.
            log_emission: n_states * n_observations * n_spots. Log probability.
        Output
            log_beta: (n_states * n_observations). log beta[i, t] = log P(o_{t+1}, ..., o_T | q_t = i, lambda).
        """
        n_obs = log_emission.shape[1]
        n_states = log_emission.shape[0]
        assert (
            np.sum(lengths) == n_obs
        ), "Sum of lengths must be equal to the first dimension of X!"
        assert (
            len(log_startprob) == n_states
        ), "Length of startprob_ must be equal to the first dimension of log_transmat!"

        log_beta = np.zeros((n_states, n_obs))
        buf = np.zeros(n_states)
        cumlen = 0
        for le in lengths:
            log_beta[:, (cumlen + le - 1)] = 0

            for t in np.arange(le - 2, -1, -1):
                for i in np.arange(n_states):
                    for j in np.arange(n_states):
                        buf[j] = (
                            log_beta[j, (cumlen + t + 1)]
                            + log_transmat[i, j]
                            + np.sum(log_emission[j, (cumlen + t + 1), :])
                        )
                    log_beta[i, (cumlen + t)] = numba_logsumexp(buf)
            cumlen += le
        return log_beta

    # TODO rename get_log_state_posteriors.
    def get_state_posteriors(
        self, lengths, log_transmat, log_startprob, log_emission, log_sitewise_transmat
    ):
        log_alpha = self.forward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        log_beta = self.backward_lattice(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        # NB log_gamma (n_states * n_observations), potentially concatenated by clone.
        log_gamma = log_alpha + log_beta

        if np.any(np.sum(log_gamma, axis=0) == 0):
            logger.error("Sum of posterior probability is zero for some observations!")
            raise RuntimeError()

        # NB normalize across states for each observation.
        log_gamma -= scipy.special.logsumexp(log_gamma, axis=0)

        return log_gamma
    
    @staticmethod
    def get_copy_states(log_gamma, includes_phased=False):
        n_states = log_gamma.shape[0]

        if includes_phased:
            return np.argmax(log_gamma, axis=0) % (n_states // 2)
        else:
            return np.argmax(log_gamma, axis=0)

    # TODO define self.n_states
    def get_initial_params(
        self,
        n_states,
        n_spots,
        init_log_mu=None,  # DEPRECATE
        init_p_binom=None,  # DEPRECATE
        init_alphas=None,  # DEPRECATE
        init_taus=None,  # DEPRECATE
    ):
        # TODO use self.default_log_mu on class instance
        log_mu = (
            np.vstack([np.linspace(-0.1, 0.1, n_states) for _ in range(n_spots)]).T
            if init_log_mu is None
            else init_log_mu
        )

        # TODO define self.default_p_binom on class instance
        p_binom = (
            np.vstack([np.linspace(0.05, 0.45, n_states) for _ in range(n_spots)]).T
            if init_p_binom is None
            else init_p_binom
        )

        # NB small alpha tend to Poisson. 0.1 ->
        alphas = (
            0.5 * np.ones((n_states, n_spots)) if init_alphas is None else init_alphas
        )

        # TODO define ...
        # NB large dispersions tend to Binomial, flat landscape, initialize just before.  30 -> 1_000
        taus = 1_000 * np.ones((n_states, n_spots)) if init_taus is None else init_taus

        # NB initialize start probability and emission probability
        log_startprob = np.log(np.ones(n_states) / n_states)

        """
        # TODO definse self.trans_mat on class instance
        if n_states > 1:
            transmat = np.ones((n_states, n_states)) * (1.0 - self.t) / (n_states - 1)
            np.fill_diagonal(transmat, self.t)
            log_transmat = np.log(transmat)
        else:
            log_transmat = np.zeros((1, 1))
        """
        log_transmat = get_log_transmat(n_states, self.t)

        return log_mu, p_binom, alphas, taus, log_startprob, log_transmat

    def get_bounds(
        self,
        n_states,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
        max_alpha=1_000.0,
        min_alpha=1e-6,
        max_tau=5_000.0,
        min_tau=1e-4,
    ):
        """
        Dynamically constructs the bounds list of (min, max) tuples to exactly
        match the flattened optimization vector generated by pack_params.
        """
        bounds = []

        if "s" in self.params:
            # Unconstrained because they pass through a Softmax upon unpack
            bounds.extend([(None, None)] * n_states)

        if optimize_nb and "m" in self.params:
            bounds.extend([(None, None)] * n_states)

        if "p" in self.params:
            if use_logit:
                # Logit constraint automatically bounds domain to (0, 1) upon unpack
                bounds.extend([(None, None)] * n_states)
            else:
                bounds.extend([(1e-6, 1.0 - 1e-6)] * n_states)

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            alpha_bnds = (float(np.log(min_alpha)), float(np.log(max_alpha)))
            if shared_NB_dispersion:
                bounds.append(alpha_bnds)
            else:
                bounds.extend([alpha_bnds] * n_states)

        if "p" in self.params and not fix_BB_dispersion:
            tau_bnds = (float(np.log(min_tau)), float(np.log(max_tau)))

            if shared_BB_dispersion:
                bounds.append(tau_bnds)
            else:
                bounds.extend([tau_bnds] * n_states)

        return bounds

    def pack_params(
        self,
        log_startprob,
        log_mu,
        p_binom,
        alphas,
        taus,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        """
        Defines an optimization vector given canonical parameterization & runtime settings.
        """
        # TODO parameter block, dispersion block, parameter block, dispersion block, etc.
        params_list = []
        if "s" in self.params:
            params_list.append(log_startprob.flatten())

        # TODO assert m not in self.params if optimize_nb is False on class instance,
        #      drop branch clause.
        if optimize_nb and "m" in self.params:
            params_list.append(log_mu.flatten())

        if "p" in self.params:
            params_list.append(
                scipy.special.logit(p_binom.flatten())
                if use_logit
                else p_binom.flatten()
            )

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                params_list.append(np.array([np.log(alphas.flatten()[0])]))
            else:
                params_list.append(np.log(alphas.flatten()))

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                params_list.append(np.array([np.log(taus.flatten()[0])]))
            else:
                params_list.append(np.log(taus.flatten()))

        return np.concatenate(params_list) if params_list else np.array([])

    def unpack_params(
        self,
        x,
        n_states,
        log_startprob_init,
        log_mu_init,
        p_binom_init,
        alphas_init,
        taus_init,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        """
        Reconstruct canonical parameterization given optimization vector & runtime settings.
        """
        idx = 0

        if "s" in self.params:
            raw_startprob = x[idx : idx + n_states]
            log_startprob = raw_startprob - scipy.special.logsumexp(raw_startprob)
            idx += n_states
        else:
            log_startprob = log_startprob_init

        # NB returns default log_mu if not optimizing NB mean.
        if optimize_nb and "m" in self.params:
            log_mu = x[idx : idx + n_states].reshape(n_states, 1)
            idx += n_states
        else:
            log_mu = log_mu_init

        if "p" in self.params:
            if use_logit:
                p_binom = scipy.special.expit(
                    x[idx : idx + n_states].reshape(n_states, 1)
                )
            else:
                p_binom = np.clip(
                    x[idx : idx + n_states].reshape(n_states, 1), 1e-6, 1 - 1e-6
                )
            idx += n_states
        else:
            p_binom = p_binom_init

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                val = np.exp(x[idx])
                alphas = np.full((n_states, 1), val)
                idx += 1
            else:
                alphas = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            alphas = alphas_init

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                val = np.exp(x[idx])
                taus = np.full((n_states, 1), val)
                idx += 1
            else:
                taus = np.exp(x[idx : idx + n_states]).reshape(n_states, 1)
                idx += n_states
        else:
            taus = taus_init

        return log_startprob, log_mu, p_binom, alphas, taus

    # WARNING
    def unpack_param_errors(
        self,
        x,
        hess_inv,
        n_states,
        optimize_nb=True,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        use_logit=True,
    ):
        idx = 0
        parameter_errors_diag = np.sqrt(np.clip(np.diag(hess_inv), a_min=0, a_max=None))

        if "s" in self.params:
            raw_startprob = x[idx : idx + n_states]
            cov_raw = hess_inv[idx : idx + n_states, idx : idx + n_states]

            p_start = scipy.special.softmax(raw_startprob)

            J = np.eye(n_states) - np.outer(np.ones(n_states), p_start)

            cov_transformed = J @ cov_raw @ J.T

            log_startprob_err = np.sqrt(
                np.clip(np.diag(cov_transformed), a_min=0, a_max=None)
            )
            idx += n_states
        else:
            log_startprob_err = None

        if optimize_nb and "m" in self.params:
            log_mu_err = parameter_errors_diag[idx : idx + n_states].reshape(
                n_states, 1
            )
            idx += n_states
        else:
            log_mu_err = None

        if "p" in self.params:
            raw_p_err = parameter_errors_diag[idx : idx + n_states].reshape(n_states, 1)
            if use_logit:
                p_binom_val = scipy.special.expit(
                    x[idx : idx + n_states].reshape(n_states, 1)
                )
                p_binom_err = p_binom_val * (1 - p_binom_val) * raw_p_err
            else:
                p_binom_err = raw_p_err
            idx += n_states
        else:
            p_binom_err = None

        if optimize_nb and "m" in self.params and not fix_NB_dispersion:
            if shared_NB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]

                alpha_val = np.exp(val_raw)
                alpha_err = alpha_val * val_err
                alphas_err = np.full((n_states, 1), alpha_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )

                alphas_val = np.exp(val_raw)
                alphas_err = alphas_val * val_err
                idx += n_states
        else:
            alphas_err = None

        if "p" in self.params and not fix_BB_dispersion:
            if shared_BB_dispersion:
                val_raw = x[idx]
                val_err = parameter_errors_diag[idx]

                tau_val = np.exp(val_raw)
                tau_err = tau_val * val_err
                taus_err = np.full((n_states, 1), tau_err)
                idx += 1
            else:
                val_raw = x[idx : idx + n_states].reshape(n_states, 1)
                val_err = parameter_errors_diag[idx : idx + n_states].reshape(
                    n_states, 1
                )

                taus_val = np.exp(val_raw)
                taus_err = taus_val * val_err
                idx += n_states
        else:
            taus_err = None

        return log_startprob_err, log_mu_err, p_binom_err, alphas_err, taus_err

    def optimize(self, *args, **kwargs):
        return self.run_baum_welch_nb_bb(*args, **kwargs)

    def run_baum_welch_nb_bb(self, *args, **kwargs):
        return self._run_optimization_pipeline("em", *args, **kwargs)

    def run_marg_likelihood_nb_bb(self, *args, **kwargs):
        return self._run_optimization_pipeline("marginal", *args, **kwargs)

    def _run_optimization_pipeline(
        self,
        mode,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat=None,
        fix_NB_dispersion=False,
        shared_NB_dispersion=False,
        fix_BB_dispersion=False,
        shared_BB_dispersion=False,
        is_diag=False, # TODO
        init_log_mu=None,
        init_p_binom=None,
        init_alphas=None,
        init_taus=None,
        max_iter=1_000,
        max_rdr=5.0,
        tol=1e-4,
        use_logit=True,
        propagate_errors=False,
        optimizer="BFGS",
        clone_lengths=None,
        normal_lambda=None,
        log_gamma=None, # TODO
        # **kwargs,
    ):
        _, n_comp, n_spots = X.shape
        assert (
            n_spots == 1
        ), "Currently expects (a) clone(s) concatenated along the genomic axis."
        assert n_comp == 2

        base_nb_mean = base_nb_mean.copy()
        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0
                base_nb_mean[est_rdr > max_rdr] = 0.0

        optimize_nb = np.any(base_nb_mean > 0)
        normal_log_lambda = None
        
        if "m" in self.params:
            assert (
                optimize_nb
            ), "Cannot optimize negative binomial if normal baseline is not defined."

            normal_log_lambda = np.log(normal_lambda) if normal_lambda is not None else None
            assert clone_lengths is not None

        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

        logger.info(
            f"Encoders built. Medians: NB={np.median(nbEncoder.total_count):.4f} ({nbEncoder.compression_rate:.2%} comp), "
            f"BB={np.median(bbEncoder.total_count):.4f} ({bbEncoder.compression_rate:.2%} comp)."
        )

        (log_mu, p_binom, alphas, taus, log_startprob, log_transmat) = (
            self.get_initial_params(
                n_states, n_spots, init_log_mu, init_p_binom, init_alphas, init_taus
            )
        )

        # kwargs_str = pprint.pformat(kwargs, indent=2) if kwargs else "{}"
        logger.info(
            f"--- hmm initialized ({mode.upper()}) ---\n"
            # f"kwargs:\n{kwargs_str}\n"
            f"log_mu:\n{np.array2string(log_mu, precision=4, suppress_small=True)}\n"
            f"p_binom:\n{np.array2string(p_binom, precision=4, suppress_small=True)}\n"
            f"alphas:\n{np.array2string(alphas, precision=4, suppress_small=True)}\n"
            f"taus:\n{np.array2string(taus, precision=2, suppress_small=True)}\n"
            "--------------------------------"
        )

        x0 = self.pack_params(
            log_startprob,
            log_mu,
            p_binom,
            alphas,
            taus,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        scratch_rdr = [
            np.zeros((n_states, len(nbEncoder.get_unique_obs(s))))
            for s in range(n_spots)
        ]
        scratch_baf = [
            np.zeros((n_states, len(bbEncoder.get_unique_obs(s))))
            for s in range(n_spots)
        ]

        if mode == "em":
            self.log_emissions, self.state_posteriors = None, None
            self.log_startprob = log_startprob
            self.iterations = 0

            def callback(intermediate_result: OptimizeResult = None):
                if (self.iterations > 0) and (self.iterations % 2 != 0):
                    self.iterations += 1
                    return
                self.state_posteriors = np.exp(
                    self.get_state_posteriors(
                        lengths,
                        log_transmat,
                        self.log_startprob,
                        self.log_emissions,
                        log_sitewise_transmat,
                    )
                )
                self.iterations += 1

            def cost_fn(params):
                _, this_log_mu, this_p_binom, this_alphas, this_taus = (
                    self.unpack_params(
                        params,
                        n_states,
                        log_startprob,
                        log_mu,
                        p_binom,
                        alphas,
                        taus,
                        optimize_nb=optimize_nb,
                        fix_NB_dispersion=fix_NB_dispersion,
                        shared_NB_dispersion=shared_NB_dispersion,
                        fix_BB_dispersion=fix_BB_dispersion,
                        shared_BB_dispersion=shared_BB_dispersion,
                        use_logit=use_logit,
                    )
                )
                log_emission_rdr, log_emission_baf = (
                    self.compute_emission_probability_nb_betabinom_coded(
                        nbEncoder,
                        bbEncoder,
                        this_log_mu,
                        this_alphas,
                        this_p_binom,
                        this_taus,
                        scratch_rdr=scratch_rdr,
                        scratch_baf=scratch_baf,
                        normal_log_lambda=normal_log_lambda,
                        clone_lengths=clone_lengths,
                    )
                )
                self.log_emissions = (log_emission_rdr + log_emission_baf)[
                    :, :, np.newaxis
                ]

                if self.state_posteriors is None:
                    callback()
                return -np.sum(self.state_posteriors * self.log_emissions[..., 0])

        elif mode == "marginal":
            callback = None

            def cost_fn(params):
                (
                    this_log_startprob,
                    this_log_mu,
                    this_p_binom,
                    this_alphas,
                    this_taus,
                ) = self.unpack_params(
                    params,
                    n_states,
                    log_startprob,
                    log_mu,
                    p_binom,
                    alphas,
                    taus,
                    optimize_nb=optimize_nb,
                    fix_NB_dispersion=fix_NB_dispersion,
                    shared_NB_dispersion=shared_NB_dispersion,
                    fix_BB_dispersion=fix_BB_dispersion,
                    shared_BB_dispersion=shared_BB_dispersion,
                    use_logit=use_logit,
                )
                log_emission_rdr, log_emission_baf = (
                    self.compute_emission_probability_nb_betabinom_coded(
                        nbEncoder,
                        bbEncoder,
                        this_log_mu,
                        this_alphas,
                        this_p_binom,
                        this_taus,
                        scratch_rdr=scratch_rdr,
                        scratch_baf=scratch_baf,
                    )
                )
                log_emissions = (log_emission_rdr + log_emission_baf)[:, :, np.newaxis]
                log_alpha = self.forward_lattice(
                    lengths,
                    log_transmat,
                    this_log_startprob,
                    log_emissions,
                    log_sitewise_transmat,
                )

                curr, total_nll = 0, 0
                for le in lengths:
                    total_nll += -numba_logsumexp(log_alpha[:, curr + le - 1])
                    curr += le
                return total_nll

        else:
            raise ValueError(f"Unknown optimization mode: {mode}")

        options = {
            "maxiter": max_iter,
            "ftol": 1e-6,
            "gtol": 1e-5,
            "disp": False,
        }

        # options = options | kwargs.get("options", {})

        bounds = self.get_bounds(
            n_states,
            optimize_nb=optimize_nb,
            fix_NB_dispersion=fix_NB_dispersion,
            shared_NB_dispersion=shared_NB_dispersion,
            fix_BB_dispersion=fix_BB_dispersion,
            shared_BB_dispersion=shared_BB_dispersion,
            use_logit=use_logit,
        )

        start_time_opt = time.time()
        logger.info(
            f"Starting {mode} optimization with {optimizer}. Initial cost={cost_fn(x0):.6e}"
        )

        res = scipy.optimize.minimize(
            cost_fn,
            x0,
            method=optimizer,
            bounds=None,
            callback=callback,
            options=options,
        )

        logger.info(
            f"Optimization complete: {time.time() - start_time_opt:.2f}s | "
            f"{res.nit} iter | converged={res.success} | negative ln. likelihood={res.fun:.6e}"
        )

        final_log_startprob, final_log_mu, final_p_binom, final_alphas, final_taus = (
            self.unpack_params(
                res.x,
                n_states,
                log_startprob,
                log_mu,
                p_binom,
                alphas,
                taus,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
        )

        if propagate_errors:
            _, log_mu_err, p_binom_err, alphas_err, taus_err = self.unpack_param_errors(
                x=res.x,
                hess_inv=res.hess_inv,
                n_states=n_states,
                optimize_nb=optimize_nb,
                fix_NB_dispersion=fix_NB_dispersion,
                shared_NB_dispersion=shared_NB_dispersion,
                fix_BB_dispersion=fix_BB_dispersion,
                shared_BB_dispersion=shared_BB_dispersion,
                use_logit=use_logit,
            )
            param_errors = {
                "new_log_mu_err": log_mu_err,
                "new_alphas_err": alphas_err,
                "new_p_binom_err": p_binom_err,
                "new_taus_err": taus_err,
                "new_log_startprob_err": None,
            }
        else:
            param_errors = {}

        # TODO call coded
        log_emission_rdr, log_emission_baf = (
            self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                final_log_mu,
                final_alphas,
                total_bb_RD,
                final_p_binom,
                final_taus,
            )
        )
        log_emission = log_emission_rdr + log_emission_baf
        log_gamma = self.get_state_posteriors(
            lengths,
            log_transmat,
            final_log_startprob,
            log_emission,
            log_sitewise_transmat,
        )
        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))

        log_lines = [
            f"--- Final HMM State ({self.__class__.__name__}) ---",
            f"p_binom:\n{np.array2string(final_p_binom, precision=3, suppress_small=True)}",
            f"taus:\n{np.array2string(final_taus, formatter={'float_kind': lambda x: f'{x:.3e}'})}",
        ]
        if optimize_nb:
            log_lines.extend(
                [
                    f"log_mu:\n{np.array2string(final_log_mu, precision=3, suppress_small=True)}",
                    f"alphas:\n{np.array2string(final_alphas, precision=3, suppress_small=True)}",
                ]
            )
        log_lines.extend(
            [
                f"State posteriors:\n{np.array2string(state_prior, formatter={'float_kind': lambda x: f'{x:.4e}'})}",
                f"Max updates (tol={tol:.6e}): mu={np.max(np.abs(np.exp(final_log_mu) - np.exp(log_mu))):.6e}",
            ]
        )
        logger.info("\n".join(log_lines))

        return {
            "new_log_mu": final_log_mu,
            "new_alphas": final_alphas,
            "new_p_binom": final_p_binom,
            "new_taus": final_taus,
            "new_log_startprob": final_log_startprob,
            "new_log_transmat": log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -res.fun,
            "n_states": n_states,
        } | param_errors


# --- `port.patch.hmm_nophasing.shifted_emission`, moved in by T- #670 PR5 -----


class _Triples(NamedTuple):
    """The genome-wide `(clone, obs, total)` compression, and how to undo it.

    `bounds[c]:bounds[c + 1]` is clone `c`'s block of unique rows, contiguous
    because the clone index is the **first** column and `np.unique` sorts
    lexicographically. That is what lets one shift be applied per block with
    a scalar rather than per entry with a gather.
    """

    obs: np.ndarray
    total: np.ndarray
    inverse: np.ndarray
    bounds: np.ndarray


def _clone_major(
    channel: Any, lengths: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """One clone-stacked channel, contiguous, and the clone each entry is in.

    `CountEncoder` keeps a view of the clone-stacked `X`, so one channel walks
    at a stride of two elements; this is the contiguous clone-major copy.
    Lengths that do not tile the rows are refused, as `port`'s
    `sal.ragged.Ragged` refuses them.
    """
    values = np.ascontiguousarray(np.asarray(channel).reshape(-1))
    counts = np.asarray(lengths, dtype=np.int64)

    if counts.size == 0 or np.any(counts < 0) or int(counts.sum()) != values.size:
        msg = f"clone lengths {lengths} do not tile the {values.size} rows"
        raise ValueError(msg)

    clones = np.repeat(np.arange(counts.size, dtype=np.int64), counts)

    return values, clones


def _triples(
    obs_count: np.ndarray, total_count: np.ndarray, lengths: tuple[int, ...]
) -> _Triples:
    """Compress `(clone, obs, total)` once over the whole genome.

    **The clone index is what lets a shared pair carry two rates.**
    `CountEncoder` compresses `(obs, total)` genome-wide, so two segments in
    different clones sharing a pair collapse to one entry -- and under the
    shift they need different rates, which one entry cannot hold. Adding the
    clone to the key separates exactly those and nothing else: the unique
    rows are, per clone, that clone's unique pairs.

    The alternative is one `CountEncoder` per clone, which is what #276
    deferred at **1.69x**. This keeps one pass and one mapping, so the
    duplication it admits is the minimum the shift requires rather than a
    re-encoding of the genome per clone.

    Rounding follows `CountEncoder.construct_unique_encoding`: non-integer
    counts are rounded to the configured decimals before the compare, so two
    entries that upstream would collapse are not separated here by a float
    the encoder never looked at.
    """
    obs, clones = _clone_major(obs_count, lengths)
    total, _ = _clone_major(total_count, lengths)

    counts = np.column_stack([clones.astype(np.float64), obs, total])

    if not np.issubdtype(total.dtype, np.integer):
        counts = counts.round(decimals=get_global_config().hmm.compression_decimals)

    unique, inverse = np.unique(counts, axis=0, return_inverse=True)

    return _Triples(
        obs=np.ascontiguousarray(unique[:, 1]),
        total=np.ascontiguousarray(unique[:, 2]),
        inverse=inverse.reshape(-1),
        bounds=np.searchsorted(unique[:, 0], np.arange(len(lengths) + 1)),
    )


def _current(lengths: tuple[int, ...], n_segments: int) -> tuple[int, ...]:
    """The clone lengths of the sequence actually being fitted.

    **`cnaster` passes stale ones once clones merge.** `hmrf.py:564` sets
    `clone_lengths` from the initial pseudobulk, before the loop, and never
    updates it; the HMRF then merges clones, so on #292's genome it still
    says six clones of 300 bins while the fit is over three -- 1,800 against
    900. Every clone carries the whole genome, so the length is the one bin
    count and the current number of clones is the decode's size over it.
    Anything that does not divide is refused rather than guessed.
    """
    if int(sum(lengths)) == n_segments:
        return lengths

    if lengths and len(set(lengths)) == 1 and n_segments % lengths[0] == 0:
        return (lengths[0],) * (n_segments // lengths[0])

    msg = f"clone lengths {lengths} do not tile the {n_segments} segments decoded"
    raise ValueError(msg)


def _stacked(normal_log_lambda: Any, lengths: tuple[int, ...]) -> np.ndarray:
    """`log lambda` over the clone-stacked sequence the decode indexes.

    **`cnaster` passes it per genome bin.** `hmrf.py:476` builds
    `normal_lambda` by summing the baseline over spots, so it has one entry
    per bin, while the decode and the reduction walk `sum(lengths)` stacked
    segments. The reduction is a `numba` loop without bounds checks, so the
    short array would be read past its end rather than refused. Every clone
    shares the one normal profile, so the stacked form is the per-bin one
    repeated clone after clone.
    """
    values = np.asarray(normal_log_lambda, dtype=np.float64).reshape(-1)
    total = int(sum(lengths))

    if values.size == total:
        return values

    if lengths and all(length == values.size for length in lengths):
        return np.tile(values, len(lengths))

    msg = (
        f"normal_log_lambda has {values.size} entries; expected one per genome "
        f"bin ({lengths[0] if lengths else 0}) or per stacked segment ({total})"
    )
    raise ValueError(msg)


class hmm_nophasing(_cnaster_hmm_nophasing):
    """`port.patch.hmm_nophasing:hmm_nophasing` (#276, #433): `cnaster`'s
    class, with the analytic gradient and, when the flag is set, the shift.

    A subclass of `_cnaster_hmm_nophasing`, as `port`'s is of `cnaster`'s:
    every method it does not override is `cnaster`'s by inheritance.
    """

    # NB the options are class attributes rather than keywords, as in `port`:
    #    the caller is `optimize_params` inside `hmrf`, and a keyword would
    #    have to be threaded through it. Set on a subclass, never on this one.

    apply_logmu_shift: bool = False
    """Off, as `cnaster` is; `port`'s `SHIFT_SWAPS` row binds it on (#276)."""

    analytic_gradient: bool = True
    """On: the M step's gradient in closed form (#433, #244).

    **A stated departure from `cnaster`**, whose BFGS differences `cost_fn`,
    one call per coordinate, as this class does with the attribute off.
    """

    emission_kernels: str = "cnaster"
    """`cnaster` (default) or `sal`: which kernels score the coded emission (#425, T- #670 PR6b).

    `port`'s `run_cnaster_port` binds `sal` where the shift is. Not a name
    rebind, because the compiled kernels call `_nb_logpmf_1d` as a global and
    a Python function in its place breaks their compilation.
    """

    def _clone_triples(self, encoder: Any, lengths: tuple[int, ...]) -> _Triples:
        """`(clone, obs, total)` compressed once over the whole genome.

        One `np.unique`, not one per clone. Built once per
        `(encoder, lengths)` and cached, keyed on the encoder's identity
        **and holding a reference to it**, so an id cannot be reused by a
        later object while the entry is live. `optimize_params` builds the
        encoder once per fit, so this is one build per fit rather than one
        per optimizer iteration.
        """
        cache: dict[tuple[int, tuple[int, ...]], tuple[Any, _Triples]]
        cache = getattr(self, "_triple_cache", None) or {}
        self._triple_cache = cache

        key = (id(encoder), lengths)

        if key not in cache:
            cache[key] = (
                encoder,
                _triples(encoder.obs_count, encoder.total_count, lengths),
            )

        return cache[key][1]

    def _distinct(self) -> dict[Any, Any]:
        """sal's distinct-count cache for this instance's dense emission (#702).

        Held on the instance as `_triple_cache` is: one fit scores the same
        observations every iteration, and sal verifies each hit against the
        values, so a stale entry is a miss rather than a wrong score.
        """
        cache: dict[Any, Any] = getattr(self, "_distinct_cache", None) or {}
        self._distinct_cache = cache
        return cache

    def _decode(self) -> np.ndarray | None:
        """The hard decode the shift is taken at, or `None` if unavailable.

        Upstream's commented-out block reads `self.get_state_posteriors()`
        and then `self.get_copy_states(log_gamma)`; this reads the posteriors
        the instance is carrying and applies upstream's own `get_copy_states`
        to them. Returning `None` rather than a default is deliberate -- a
        shift computed from a decode nobody supplied is a number nobody asked
        for.
        """
        posteriors = getattr(self, "state_posteriors", None)

        if posteriors is None:
            return None

        decoded: np.ndarray = self.get_copy_states(np.asarray(posteriors))

        return decoded

    _row_shift: np.ndarray | None = None
    """The last shifted fit's shift, one entry per clone-stacked segment.

    **Class state, and deliberately so.** `hmm.py:155` rescores the fit
    through `hmmclass.compute_emission_probability_nb_betabinom`, a static
    method called on the class with no argument that could carry the shift.
    The pipeline is sequential, the call follows `optimize` directly, and the
    override applies it only to an input of exactly this length, so a stale
    value cannot reach a different problem unnoticed.
    """

    @classmethod
    def compute_emission_probability_nb_betabinom(
        cls,
        X: np.ndarray,
        base_nb_mean: np.ndarray,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        total_bb_RD: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's dense emission, with the last fit's shift applied.

        The shift enters the negative binomial through its mean,
        `base * exp(log_mu - shift)`, which is `base * exp(-shift)` against
        the unshifted `exp(log_mu)`. So it is applied to the exposure and the
        rest is upstream's kernel unchanged.
        """
        shift = hmm_nophasing._row_shift

        if (
            cls.apply_logmu_shift
            and shift is not None
            and shift.size == np.asarray(X).shape[0]
        ):
            # NB **recentred, because the rates have no scale.** The shifted
            #    likelihood is flat along `mu -> c mu`, and the fit wanders
            #    along it: measured, `log mu` and the shift both near -7,024 on
            #    #292's realization 3. `exp(-shift) * exp(log_mu)` is then
            #    `inf * 0`. Taking a common `c` off both leaves the product --
            #    the emission -- unchanged and each factor near one.
            centre = float(np.mean(shift))
            base_nb_mean = np.asarray(base_nb_mean) * np.exp(-(shift - centre))[:, None]
            log_mu = np.asarray(log_mu) - centre

        scored: tuple[np.ndarray, np.ndarray]
        scored = _cnaster_hmm_nophasing.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        return scored

    def optimize(
        self,
        X: np.ndarray,
        lengths: np.ndarray,
        n_states: int,
        base_nb_mean: np.ndarray,
        total_bb_RD: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upstream's fit, then a shifted decode.

        Inside the fit the shift is already applied: the M step's objective
        and the E step's posteriors both come from the coded emission below.
        What upstream does after it is not shifted -- `hmm_nophasing.py:1085`
        rescored the fit through the dense emission for `log_gamma` -- so
        that is redone here with the shift, and the shift is recorded for
        `hmm.py:155`, which rescores it once more.

        The rates are returned as fitted. The shifted mean `base * mu / sum
        lambda mu` is unchanged by `mu -> c mu`, so their scale is arbitrary
        here; `port.patch.hmrf.run_core_inference` (PR7) pins it once, after the
        whole optimization.
        """
        hmm_nophasing._row_shift = None

        # NB the M step's gradient in closed form, through `minimize`'s
        #    callable `method` (#433); positional extras leave the fit as is,
        #    since the settings the gradient reads would then be unnamed.
        #    BFGS, as `cnaster` runs it, whatever `hmm.solver` states:
        #    `configured_method` honours it and is not installed (#448).
        if (
            self.analytic_gradient
            and not args
            and kwargs.get("optimizer", "BFGS") == "BFGS"
        ):
            kwargs["optimizer"] = analytic_bfgs(
                EmGradient.for_fit(
                    self, X, n_states, base_nb_mean, total_bb_RD, **kwargs
                )
            )

        res: dict[str, Any] = super().optimize(
            X, lengths, n_states, base_nb_mean, total_bb_RD, *args, **kwargs
        )

        normal_lambda = kwargs.get("normal_lambda")
        clone_lengths = kwargs.get("clone_lengths")
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or "m" not in self.params
            or normal_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            return res

        rates = state_vector(res["new_log_mu"])

        n_segments = int(np.asarray(X).shape[0])
        current = _current(
            tuple(int(length) for length in np.asarray(clone_lengths)), n_segments
        )

        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            _stacked(np.log(np.asarray(normal_lambda, dtype=np.float64)), current),
            np.asarray(current, dtype=np.int64),
        )
        hmm_nophasing._row_shift = np.repeat(shifts, current)

        log_emission_rdr, log_emission_baf = (
            self.compute_emission_probability_nb_betabinom(
                X,
                base_nb_mean,
                res["new_log_mu"],
                res["new_alphas"],
                total_bb_RD,
                res["new_p_binom"],
                res["new_taus"],
            )
        )
        log_gamma = self.get_state_posteriors(
            lengths,
            res["new_log_transmat"],
            res["new_log_startprob"],
            log_emission_rdr + log_emission_baf,
            kwargs.get("log_sitewise_transmat"),
        )

        res["log_gamma"] = log_gamma
        res["pred_cnv"] = np.argmax(log_gamma, axis=0)

        return res

    def compute_emission_probability_nb_betabinom_coded(
        self,
        nbEncoder: Any,
        bbEncoder: Any,
        log_mu: np.ndarray,
        alphas: np.ndarray,
        p_binom: np.ndarray,
        taus: np.ndarray,
        clone_stack: bool = True,
        scratch_rdr: Any = None,
        scratch_baf: Any = None,
        normal_log_lambda: Any = None,
        clone_lengths: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Upstream's emission, with `log_mu` debiased per clone.

        Hands the call on unchanged unless **all** of the flag, the exposures,
        the clone lengths and a decode are present. A shift needs every one of
        them, and computing it from a default would be a number nobody asked
        for -- which is also upstream's own reason for warning rather than
        guessing at `hmm_nophasing.py:279`.
        """
        decode = self._decode()

        if (
            not self.apply_logmu_shift
            or normal_log_lambda is None
            or clone_lengths is None
            or decode is None
        ):
            if self.apply_logmu_shift:
                # NB expected with the shift on: the BAF stage has no
                #    exposure, and a first iteration has no decode yet. Said
                #    at debug, and the exposure withheld from upstream so it
                #    does not warn "not currently supported" for a shift that
                #    applies from the next call (#362).
                logger.debug(
                    "logmu shift not applied on this call: %s",
                    ", ".join(
                        name
                        for name, missing in (
                            ("no exposure", normal_log_lambda is None),
                            ("no clone lengths", clone_lengths is None),
                            ("no decode yet", decode is None),
                        )
                        if missing
                    ),
                )
                normal_log_lambda = None
            if self.emission_kernels == "sal":
                from cnamaste.dense_emission import coded_emission

                return coded_emission(
                    nbEncoder,
                    bbEncoder,
                    log_mu,
                    alphas,
                    p_binom,
                    taus,
                    clone_stack=clone_stack,
                    distinct=self._distinct(),
                )

            unshifted: tuple[np.ndarray, np.ndarray]
            unshifted = super().compute_emission_probability_nb_betabinom_coded(
                nbEncoder,
                bbEncoder,
                log_mu,
                alphas,
                p_binom,
                taus,
                clone_stack=clone_stack,
                scratch_rdr=scratch_rdr,
                scratch_baf=scratch_baf,
                normal_log_lambda=normal_log_lambda,
                clone_lengths=clone_lengths,
            )
            return unshifted

        if nbEncoder.n_spots != 1 or bbEncoder.n_spots != 1:
            msg = (
                f"one spot only: got {nbEncoder.n_spots} and {bbEncoder.n_spots}. "
                "The shift is per clone along the genomic axis (#276)."
            )
            raise ValueError(msg)

        # NB `(n_states,)`, normalized once at the edge. The fit returns
        #    `(n_states, 1)` and nothing else can reach here (#278), so the
        #    column is dropped rather than indexed at four call sites below.
        rates = state_vector(log_mu)
        dispersions = state_vector(alphas)
        probabilities = state_vector(p_binom)
        concentrations = state_vector(taus)

        n_states = rates.shape[0]
        lengths = _current(
            tuple(int(length) for length in np.asarray(clone_lengths)),
            int(np.asarray(decode).size),
        )

        # NB **once per call, not once per state.** Upstream's commented-out
        #    call sits inside `for i in range(n_states)` at
        #    `hmm_nophasing.py:275-279`, so folding it in as written would
        #    recompute the whole reduction `n_states` times over an
        #    `n_segments` array for a quantity that does not depend on the
        #    state. That is the efficiency here; the reduction itself is
        #    upstream's own loop, kept (`logmu_shift`).
        #
        #    `(n_clones,)`, so it is indexed by clone. Upstream's shape is
        #    `(n_segments,)` and indexing *that* by clone is silently wrong.
        shifts = logmu_shifts(
            rates,
            np.asarray(decode, dtype=np.int64),
            _stacked(normal_log_lambda, lengths),
            np.asarray(lengths, dtype=np.int64),
        )

        # NB the allele channel is untouched by the shift, so it keeps the
        #    whole-genome encoder and upstream's path.
        bb_endog = bbEncoder.get_unique_obs(0)
        bb_exposure = bbEncoder.get_unique_total(0)

        baf_uniq = (
            scratch_baf[0] if scratch_baf else np.zeros((n_states, len(bb_endog)))
        )

        if self.emission_kernels == "sal":
            from cnamaste.dense_emission import bb_states

            baf_uniq[:] = bb_states(
                bb_endog,
                bb_exposure,
                probabilities,
                concentrations,
                distinct=self._distinct(),
            )
        else:
            for state in range(n_states):
                _bb_logpmf_1d(
                    bb_endog,
                    bb_exposure,
                    probabilities[state],
                    concentrations[state],
                    baf_uniq[state, :],
                )

        log_emit_baf = bbEncoder.decode_array(baf_uniq, 0)

        # NB scored once per unique `(clone, obs, total)`, then decoded by a
        #    single gather. The clone's block is contiguous, so its shift is
        #    a scalar the kernel already takes -- no per-entry rate array and
        #    no second mapping.
        triples = self._clone_triples(nbEncoder, lengths)
        rdr_uniq = np.zeros((n_states, triples.obs.size))

        for clone in range(len(lengths)):
            first, last = int(triples.bounds[clone]), int(triples.bounds[clone + 1])

            if first == last:
                continue

            if self.emission_kernels == "sal":
                from cnamaste.dense_emission import nb_states

                rdr_uniq[:, first:last] = nb_states(
                    triples.obs[first:last],
                    triples.total[first:last],
                    np.exp(rates - shifts[clone]),
                    dispersions,
                    distinct=self._distinct(),
                )
                continue

            for state in range(n_states):
                _nb_logpmf_1d(
                    triples.obs[first:last],
                    triples.total[first:last],
                    exp(rates[state] - shifts[clone]),
                    dispersions[state],
                    rdr_uniq[state, first:last],
                )

        log_emit_rdr = rdr_uniq[:, triples.inverse]

        if clone_stack:
            return log_emit_rdr, log_emit_baf

        return log_emit_rdr[:, :, None], log_emit_baf[:, :, None]


def shifted(model: Any) -> bool:
    """Whether `model`, a class or an instance, fits with the shift applied.

    `False` for `cnaster`'s class, which has no such option: the one gate
    the clone assignment, the core inference and the gradient read (#517).
    """
    return bool(getattr(model, "apply_logmu_shift", False))


def release() -> None:
    """Drop the last fit's shift; `run_cnamaste` calls this on return (#517).

    Keyed by size, so a later fit of the same shape would read this one's.
    """
    hmm_nophasing._row_shift = None
