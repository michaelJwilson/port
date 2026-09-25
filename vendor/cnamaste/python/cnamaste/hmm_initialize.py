import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# from cnamaste.hmm_nophasing import hmm_nophasing
from cnamaste.config import get_global_config, start_time
from cnamaste.hmm_phased import hmm_phased
from cnamaste.hmrf_utils import clone_stack_obs
from cnamaste.logger import get_logger

logger = get_logger(__name__, start_time=start_time)


def fit_dispersions_mle(X, base_nb_mean, total_bb_RD, log_mu, p_binom, n_states):
    """
    Given a fixed set of cluster centers (log_mu, p_binom), fits the global
    overdispersion parameters (alpha, tau) by maximizing the log-likelihood.
    """

    def neg_log_likelihood(params):
        # Exponentiate to ensure strictly positive dispersions during optimization
        alpha_val = np.exp(params[0])
        tau_val = np.exp(params[1])

        alphas = alpha_val * np.ones((n_states, 1))
        taus = tau_val * np.ones((n_states, 1))

        lnlike_rdr, lnlike_baf = hmm_phased.compute_emission_probability_nb_betabinom(
            X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus
        )

        lnlike = lnlike_rdr + lnlike_baf
        lnlike[~np.isfinite(lnlike)] = -1e10  # Penalize invalid states safely

        # Log-sum-exp or Max over states. For fast burn-in, Max (hard assignment) is stable
        best_lnlike = np.max(lnlike, axis=0)
        return -np.sum(best_lnlike)

    # Initial guess: slight overdispersion
    init_params = [np.log(0.05), np.log(100.0)]

    res = minimize(
        neg_log_likelihood,
        init_params,
        method="L-BFGS-B",
        bounds=[(np.log(1e-4), np.log(1.0)), (np.log(10.0), np.log(10000.0))],
    )

    return np.exp(res.x[0]), np.exp(res.x[1])


def cna_mixture_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    lengths,
    log_transmat,
    log_sitewise_transmat,
    max_iter=100,
    anneal=False,
    max_rdr=np.inf,
    random_state=42,
    in_log_space=False,
    only_minor=False,
):
    """
    cna-mixture++: adaptive burn-in mirroring k-means++.

    1. Starts at Poisson/Binomial limits.
    2. Samples centers proportional to -ln(P) under the emission model.
    3. Fits new dispersions given the complete set of centers.
    4. Feeds back the best dispersions to sample better centers iteratively.
    5. Evaluates the posteriors across the phase-expanded space to return the most populated states.
    """
    assert not in_log_space

    np.random.seed(random_state)

    known_normal = np.any(base_nb_mean > 0.0)

    current_alpha, current_tau = 0.1, 1_000
    best_solution, best_solution_lnlike = None, -np.inf
    final_lnlike_matrix = None

    logger.info(f"Starting cna-mixture++ (max_iter={max_iter}, states={n_states})")

    (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        _,
        _,
        _,
    ) = clone_stack_obs(X, base_nb_mean, total_bb_RD, None, None, None)

    num_segments, _, _ = clone_stack_X.shape
    flat_idx = np.arange(num_segments)

    for iteration in range(max_iter):
        log_mu = np.array([0.0 if known_normal else np.nan]).reshape((1, 1))
        p_binom = np.array([0.5]).reshape((1, 1))

        while len(log_mu) < n_states:
            alphas = current_alpha * np.ones((len(log_mu), 1))
            taus = current_tau * np.ones((len(log_mu), 1))

            lnlike_rdr, lnlike_baf = (
                hmm_phased.compute_emission_probability_nb_betabinom(
                    clone_stack_X,
                    clone_stack_base_nb_mean,
                    log_mu,
                    alphas,
                    clone_stack_total_bb_RD,
                    p_binom,
                    taus,
                )
            )

            lnlike = lnlike_rdr + lnlike_baf
            lnlike = np.nan_to_num(lnlike, nan=-1e6, posinf=1e6, neginf=-1e6)

            best_lnlike = np.max(lnlike, axis=0)

            # NB shift relative to global best fit to guarantee non-negative probabilities
            # rel_lnlike = best_lnlike - np.max(best_lnlike)
            # ps = -rel_lnlike

            ps = -best_lnlike

            if known_normal:
                ps[clone_stack_base_nb_mean.ravel() == 0.0] = 0.0

            # NB a lot more marginally disfavored data.
            if anneal:
                thres = np.percentile(ps, 100.0 * 1.0 - (len(log_mu) / (n_states - 1)))
                ps[ps < thres] = 0.0

            ps_sum = ps.sum()
            if ps_sum > 0:
                ps /= ps_sum
            else:
                ps = np.ones_like(ps) / len(ps)

            ps = ps.ravel()

            logger.debug(f"Solving for ps={ps}")

            sample_ln_rdr, sample_baf = np.inf, np.inf

            while (
                ((known_normal and ~np.isfinite(sample_ln_rdr)))
                or ~np.isfinite(sample_baf)
                or sample_ln_rdr > np.log(max_rdr)
            ):
                # NB clones are concatenated across the genomic axis.
                sample_idx = np.random.choice(flat_idx, p=ps)

                with np.errstate(divide="ignore", invalid="ignore"):
                    sample_ln_rdr = (
                        np.log(
                            clone_stack_X[sample_idx, 0, 0]
                            / clone_stack_base_nb_mean[sample_idx, 0]
                        )
                        if known_normal
                        else 0.0
                    )

                    sample_baf = (
                        clone_stack_X[sample_idx, 1, 0]
                        / clone_stack_total_bb_RD[sample_idx, 0]
                    )

                    logger.debug(
                        f"Drawn new sample with log_mu={sample_ln_rdr:.4f}, p_binom={sample_baf:.4f}"
                    )

            log_mu = np.vstack([log_mu, [[sample_ln_rdr]]])
            p_binom = np.vstack([p_binom, [[sample_baf]]])

            logger.debug(
                f"Found new center: log_mu={sample_ln_rdr:.4f}, p_binom={sample_baf:.4f}"
            )

        new_alpha, new_tau = fit_dispersions_mle(
            clone_stack_X,
            clone_stack_base_nb_mean,
            clone_stack_total_bb_RD,
            log_mu,
            p_binom,
            n_states,
        )

        logger.debug(f"Found new dispersions: alpha={new_alpha:.4f}, tau={new_tau:.2f}")

        fit_alphas = new_alpha * np.ones((n_states, 1))
        fit_taus = new_tau * np.ones((n_states, 1))

        lnlike_rdr, lnlike_baf = hmm_phased.compute_emission_probability_nb_betabinom(
            clone_stack_X,
            clone_stack_base_nb_mean,
            log_mu,
            fit_alphas,
            clone_stack_total_bb_RD,
            p_binom,
            fit_taus,
        )

        final_lnlike = lnlike_rdr + lnlike_baf
        final_lnlike = np.nan_to_num(final_lnlike, nan=-1e6, posinf=1e6, neginf=-1e6)

        total_lnlike = np.sum(np.max(final_lnlike, axis=0))

        if total_lnlike > best_solution_lnlike:
            best_solution = [log_mu, fit_alphas, p_binom, fit_taus]
            best_solution_lnlike = total_lnlike
            final_lnlike_matrix = final_lnlike

            logger.info(
                f"Iteration {iteration}: new best ln. likelihood = {total_lnlike:.2f} | alpha = {new_alpha:.4f}, tau = {new_tau:.2f}"
            )
            current_alpha = new_alpha
            current_tau = new_tau

    log_mu, alphas, p_binom, taus = best_solution

    if only_minor:
        p_binom = np.where(p_binom > 0.5, 1.0 - p_binom, p_binom)
    else:
        lnlike_rdr, lnlike_baf = hmm_phased.compute_emission_probability_nb_betabinom(
            clone_stack_X,
            clone_stack_base_nb_mean,
            log_mu,
            alphas,
            clone_stack_total_bb_RD,
            p_binom,
            taus,
        )

        log_emission = lnlike_rdr + lnlike_baf
        log_startprob = np.full(n_states, -np.log(n_states))

        # NB (n_states, n_states) according to phase flip.
        #
        # TODO HACK FINAL hmm_phased
        log_gamma = hmm_phased.get_state_posteriors(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        # TODO check normalization.
        posteriors = np.exp(log_gamma)

        sum_axes = tuple(range(1, posteriors.ndim))
        component_weights = np.sum(posteriors, axis=sum_axes)

        full_log_mu = np.vstack([log_mu, log_mu])
        full_p_binom = np.vstack([p_binom, 1.0 - p_binom])

        if len(component_weights) == len(full_log_mu):
            top_indices = np.argsort(component_weights)[-n_states:][::-1]

            log_mu = full_log_mu[top_indices]
            p_binom = full_p_binom[top_indices]

            logger.info(
                f"Selected top {n_states} states from phase-expanded space via HMM posteriors."
            )
        else:
            logger.warning(
                "Emission likelihood shape does not match 2 * n_states. Skipping phase selection."
            )

    if not known_normal:
        log_mu, alphas = None, None

    log_mu_str = log_mu.ravel() if log_mu is not None else "None"
    alpha_str = f"{alphas[0,0]:.4f}" if alphas is not None else "None"

    logger.info(
        f"cna-mixture++ converged (max. ln. likelihood = {best_solution_lnlike:.6e}):\n"
        f"log_mu={log_mu_str},\nalpha={alpha_str},\np_binom={p_binom.ravel()},\ntau={taus[0,0]:.2f}"
    )

    return log_mu, p_binom, alphas, taus


# TODO FINAL utilize state posteriors to determine most populated
def gmm_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    lengths,
    log_transmat,
    log_sitewise_transmat,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    mirrored_baf_augmentation=True,
):
    logger.info(
        f"Initializing HMM emission with GMM (only_minor={only_minor}, log_space={in_log_space}, mirrored_baf={mirrored_baf_augmentation})."
    )

    X_gmm_rdr, X_gmm_baf = None, None
    n_samples = X.shape[2]

    # ---------------------------------------------------------
    # 1. RDR Processing
    # ---------------------------------------------------------
    if "m" in params:
        rdr_ratio = X[:, 0, :] / base_nb_mean

        if in_log_space:
            rdr_ratio = np.clip(rdr_ratio, a_min=1e-6, a_max=None)
            X_gmm_rdr = np.log(rdr_ratio)
        else:
            X_gmm_rdr = rdr_ratio

        valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)

        if not np.any(valid):
            logger.error(
                f"No valid RDR data given sum(base_nb_mean)={np.sum(base_nb_mean)}"
            )
            raise RuntimeError("No valid RDR data.")

        if in_log_space:
            offset = np.median(X_gmm_rdr[valid])
            low_percentile = np.percentile(X_gmm_rdr[valid], 1)
            high_percentile = np.percentile(X_gmm_rdr[valid], 99)
            scale_factor = high_percentile - low_percentile
        else:
            offset = 0
            scale_factor = np.percentile(X_gmm_rdr[valid], 99)

        if scale_factor < 1e-6:
            scale_factor = 1.0

        logger.info(
            f"RDR offset (median) and scale (percentile): {offset:.4f}, {scale_factor:.4f}"
        )

        X_gmm_rdr = (X_gmm_rdr - offset) / scale_factor

    # ---------------------------------------------------------
    # 2. BAF Processing
    # ---------------------------------------------------------
    if "p" in params:
        X_gmm_baf = X[:, 1, :] / total_bb_RD

        # Assuming get_global_config is available. If not, replace with direct floats (e.g., 0.05, 0.95)
        config = get_global_config().hmm
        min_binom = float(config.gmm_min_binom_prob)
        max_binom = float(config.gmm_max_binom_prob)

        clipped_mask = (X_gmm_baf < min_binom) | (X_gmm_baf > max_binom)
        logger.warning(
            f"Clipping {100. * np.mean(clipped_mask):.4f}% of BAF values to [{min_binom}, {max_binom}]."
        )

        X_gmm_baf = np.clip(X_gmm_baf, min_binom, max_binom)

    # ---------------------------------------------------------
    # 3. Concatenation & NaN Patching
    # ---------------------------------------------------------
    if ("m" in params) and ("p" in params):
        X_gmm_original = np.hstack([X_gmm_rdr, X_gmm_baf])
    else:
        X_gmm_original = X_gmm_rdr if "m" in params else X_gmm_baf

    nan_mask = np.isnan(X_gmm_original)
    num_nans = nan_mask.sum()

    if num_nans > 0:
        X_gmm_original = pd.DataFrame(X_gmm_original).ffill().bfill().to_numpy()
        logger.info(f"Patched {num_nans} nan values via ffill/bfill.")

    valid_rows = ~np.isnan(X_gmm_original).any(axis=1) & ~np.isinf(X_gmm_original).any(
        axis=1
    )
    X_gmm_original = X_gmm_original[valid_rows, :]
    logger.info(f"Retained {np.mean(valid_rows):.4%} of samples after patching.")

    # ---------------------------------------------------------
    # 4. Mirrored BAF Augmentation & 2K Fit
    # ---------------------------------------------------------
    max_iter = get_global_config().hmm.gmm_maxiter
    is_augmented = mirrored_baf_augmentation and ("p" in params)

    n_components_fit = (2 * n_states) if is_augmented else n_states
    X_gmm_fit = X_gmm_original

    if is_augmented:
        logger.info("Applying BAF data augmentation to robustly model phase switching.")
        X_gmm_flipped = X_gmm_original.copy()

        if "m" in params:
            X_gmm_flipped[:, n_samples:] = 1.0 - X_gmm_flipped[:, n_samples:]
        else:
            X_gmm_flipped = 1.0 - X_gmm_flipped

        X_gmm_fit = np.vstack([X_gmm_original, X_gmm_flipped])

    gmm = GaussianMixture(
        n_components=n_components_fit,
        max_iter=max_iter,
        random_state=random_state,
        n_init=3,
        reg_covar=1e-4,
    ).fit(X_gmm_fit)

    logger.info(
        f"GMM Fit (k={n_components_fit}): score={gmm.score(X_gmm_fit):.6f}, converged={gmm.converged_}, iterations={gmm.n_iter_}"
    )

    # ---------------------------------------------------------
    # 5. Parameter Extraction & Posterior Reduction
    # ---------------------------------------------------------
    rdr_means, gmm_p_binom = None, None

    if is_augmented:
        logger.info(
            f"Reducing {n_components_fit} states to {n_states} based on un-augmented data posteriors."
        )

        # 5A. E-Step on Original Data: Calculate Responsibilities (N x 2K)
        posteriors = gmm.predict_proba(X_gmm_original)

        # Total mass assigned to each of the 2K components by the original data
        component_weights = posteriors.sum(axis=0)

        if not only_minor:
            # Simply select the top K components with the most data mass, regardless of phase pairing
            logger.info(
                "only_minor=False: Selecting top K populated components, allowing mixed phases."
            )

            # Sort indices by descending weight and slice top K
            top_k_indices = np.argsort(component_weights)[-n_states:][::-1]

            if "m" in params:
                rdr_means = gmm.means_[top_k_indices, :n_samples]
            if "p" in params:
                gmm_p_binom = (
                    gmm.means_[top_k_indices, n_samples:]
                    if ("m" in params)
                    else gmm.means_[top_k_indices, :]
                )

        else:
            # 5B. Extract and Fold Means for standard grouping
            if "m" in params:
                rdr_raw = gmm.means_[:, :n_samples]
            if "p" in params:
                baf_raw = gmm.means_[:, n_samples:] if ("m" in params) else gmm.means_
                baf_folded = np.where(baf_raw > 0.5, 1.0 - baf_raw, baf_raw)

            # 5C. Group the 2K components into K symmetric pairs
            folded_for_clustering = (
                np.hstack([rdr_raw, baf_folded])
                if ("m" in params and "p" in params)
                else (rdr_raw if "m" in params else baf_folded)
            )
            group_labels = KMeans(
                n_clusters=n_states, n_init=10, random_state=random_state
            ).fit_predict(folded_for_clustering)

            # 5D. Merge components weighted by their E-step posteriors
            rdr_means = np.zeros((n_states, n_samples)) if "m" in params else None
            gmm_p_binom = np.zeros((n_states, n_samples)) if "p" in params else None

            for k in range(n_states):
                mask = group_labels == k
                weights_in_group = component_weights[mask]
                weight_sum = weights_in_group.sum()

                if weight_sum > 1e-9:
                    if "m" in params:
                        rdr_means[k] = np.average(
                            rdr_raw[mask], axis=0, weights=weights_in_group
                        )
                    if "p" in params:
                        # Force into [0.0, 0.5] space using weighted average of folded means
                        gmm_p_binom[k] = np.average(
                            baf_folded[mask], axis=0, weights=weights_in_group
                        )
                else:
                    if "m" in params:
                        rdr_means[k] = np.mean(rdr_raw[mask], axis=0)
                    if "p" in params:
                        gmm_p_binom[k] = np.mean(baf_folded[mask], axis=0)

    else:
        # Standard Extraction (No Augmentation)
        if "m" in params:
            rdr_means = gmm.means_[:, :n_samples] if ("p" in params) else gmm.means_
        if "p" in params:
            gmm_p_binom = gmm.means_[:, n_samples:] if ("m" in params) else gmm.means_
            if only_minor:
                gmm_p_binom = np.where(
                    gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom
                )

    # ---------------------------------------------------------
    # 6. Final Inverse Transforms
    # ---------------------------------------------------------
    gmm_log_mu = None

    if "m" in params:
        mu_recovered = rdr_means * scale_factor + offset
        gmm_log_mu = mu_recovered if in_log_space else np.log(mu_recovered)

    if "p" in params:
        if np.any(gmm_p_binom > 0.5):
            logger.warning(
                f"GMM initialized p_binom > 0.5: {gmm_p_binom[gmm_p_binom > 0.5]}"
            )

    logger.debug(f"Solved for GMM initialized parameters:\n{gmm_log_mu}\n{gmm_p_binom}")

    return gmm_log_mu, gmm_p_binom, None, None


"""
def gmm_init(
    n_states,
    X,
    base_nb_mean,
    total_bb_RD,
    params,
    lengths,
    log_transmat,
    log_sitewise_transmat,
    random_state=None,
    in_log_space=True,
    only_minor=True,
    mirrored_baf_augmentation=True,
):
    logger.info(
        f"Initializing HMM emission with GMM (only_minor={only_minor}, log_space={in_log_space}, mirrored_baf={mirrored_baf_augmentation})."
    )

    n_explore = n_states if only_minor else (2 * n_states)
    X_gmm_rdr, X_gmm_baf = None, None
    n_samples = X.shape[2]

    # ---------------------------------------------------------
    # 1. RDR Processing
    # ---------------------------------------------------------
    if "m" in params:
        rdr_ratio = X[:, 0, :] / base_nb_mean

        if in_log_space:
            rdr_ratio = np.clip(rdr_ratio, a_min=1e-6, a_max=None)
            X_gmm_rdr = np.log(rdr_ratio)
        else:
            X_gmm_rdr = rdr_ratio

        valid = ~np.isnan(X_gmm_rdr) & ~np.isinf(X_gmm_rdr)
        if not np.any(valid):
            raise RuntimeError("No valid RDR data.")

        if in_log_space:
            offset = np.median(X_gmm_rdr[valid])
            scale_factor = np.percentile(X_gmm_rdr[valid], 99) - np.percentile(X_gmm_rdr[valid], 1)
        else:
            offset = 0
            scale_factor = np.percentile(X_gmm_rdr[valid], 99)

        scale_factor = max(scale_factor, 1.0)
        X_gmm_rdr = (X_gmm_rdr - offset) / scale_factor

    # ---------------------------------------------------------
    # 2. BAF Processing
    # ---------------------------------------------------------
    if "p" in params:
        X_gmm_baf = X[:, 1, :] / total_bb_RD
        config = get_global_config().hmm
        min_binom, max_binom = float(config.gmm_min_binom_prob), float(config.gmm_max_binom_prob)

        X_gmm_baf = np.clip(X_gmm_baf, min_binom, max_binom)

    # ---------------------------------------------------------
    # 3. Concatenation & NaN Patching
    # ---------------------------------------------------------
    if ("m" in params) and ("p" in params):
        X_gmm_original = np.hstack([X_gmm_rdr, X_gmm_baf])
    else:
        X_gmm_original = X_gmm_rdr if "m" in params else X_gmm_baf

    if np.isnan(X_gmm_original).sum() > 0:
        X_gmm_original = pd.DataFrame(X_gmm_original).ffill().bfill().to_numpy()

    valid_rows = ~np.isnan(X_gmm_original).any(axis=1) & ~np.isinf(X_gmm_original).any(axis=1)
    X_gmm_original = X_gmm_original[valid_rows, :]

    # ---------------------------------------------------------
    # 4. GMM Fit (with Optional Data Augmentation)
    # ---------------------------------------------------------
    max_iter = get_global_config().hmm.gmm_maxiter
    is_augmented = mirrored_baf_augmentation and ("p" in params)
    n_components_fit = (2 * n_explore) if is_augmented else n_explore
    
    X_gmm_fit = X_gmm_original
    if is_augmented:
        X_gmm_flipped = X_gmm_original.copy()
        if "m" in params:
            X_gmm_flipped[:, n_samples:] = 1.0 - X_gmm_flipped[:, n_samples:]
        else:
            X_gmm_flipped = 1.0 - X_gmm_flipped
        X_gmm_fit = np.vstack([X_gmm_original, X_gmm_flipped])

    gmm = GaussianMixture(
        n_components=n_components_fit, max_iter=max_iter, random_state=random_state, n_init=3, reg_covar=1e-4
    ).fit(X_gmm_fit)

    # ---------------------------------------------------------
    # 5. Parameter Extraction & Base-State Reduction
    # ---------------------------------------------------------
    rdr_means, gmm_p_binom = None, None

    if is_augmented:
        posteriors = gmm.predict_proba(X_gmm_original)
        component_weights = posteriors.sum(axis=0)

        if "m" in params: rdr_raw = gmm.means_[:, :n_samples]
        if "p" in params:
            baf_raw = gmm.means_[:, n_samples:] if ("m" in params) else gmm.means_
            baf_folded = np.where(baf_raw > 0.5, 1.0 - baf_raw, baf_raw)

        folded_for_clustering = np.hstack([rdr_raw, baf_folded]) if ("m" in params and "p" in params) else (rdr_raw if "m" in params else baf_folded)
        group_labels = KMeans(n_clusters=n_explore, n_init=10, random_state=random_state).fit_predict(folded_for_clustering)

        rdr_means = np.zeros((n_explore, n_samples)) if "m" in params else None
        gmm_p_binom = np.zeros((n_explore, n_samples)) if "p" in params else None

        for k in range(n_explore):
            mask = group_labels == k
            w = component_weights[mask]
            if w.sum() > 1e-9:
                if "m" in params: rdr_means[k] = np.average(rdr_raw[mask], axis=0, weights=w)
                if "p" in params: gmm_p_binom[k] = np.average(baf_folded[mask], axis=0, weights=w)
            else:
                if "m" in params: rdr_means[k] = np.mean(rdr_raw[mask], axis=0)
                if "p" in params: gmm_p_binom[k] = np.mean(baf_folded[mask], axis=0)
    else:
        if "m" in params: rdr_means = gmm.means_[:, :n_samples] if ("p" in params) else gmm.means_
        if "p" in params: gmm_p_binom = gmm.means_[:, n_samples:] if ("m" in params) else gmm.means_

    # ---------------------------------------------------------
    # 6. Final Inverse Transforms (and Safe Defaults)
    # ---------------------------------------------------------
    if "m" in params:
        mu_recovered = rdr_means * scale_factor + offset
        gmm_log_mu = mu_recovered if in_log_space else np.log(mu_recovered)
        gmm_log_mu = gmm_log_mu.reshape(-1, 1)
    else:
        # Dummy array to allow emission evaluation to proceed safely
        gmm_log_mu = np.zeros((n_explore, 1))
        
    if "p" in params:
        gmm_p_binom = gmm_p_binom.reshape(-1, 1)
    else:
        # Dummy array to allow emission evaluation to proceed safely
        gmm_p_binom = np.full((n_explore, 1), 0.5)

    # ---------------------------------------------------------
    # 7. HMM Posterior Ranking
    # ---------------------------------------------------------
    # Guaranteed construction of stacked inputs required for Steps 7 & 8
    (
        clone_stack_X,
        clone_stack_base_nb_mean,
        clone_stack_total_bb_RD,
        _, _, _,
    ) = clone_stack_obs(X, base_nb_mean, total_bb_RD, None, None, None)

    if only_minor:
        gmm_p_binom = np.where(gmm_p_binom > 0.5, 1.0 - gmm_p_binom, gmm_p_binom)
    else:
        alphas = 0.1 * np.ones((n_explore, 1))
        taus = 1_000.0 * np.ones((n_explore, 1))

        lnlike_rdr, lnlike_baf = hmm_sitewise.compute_emission_probability_nb_betabinom(
            clone_stack_X, clone_stack_base_nb_mean, gmm_log_mu, alphas, clone_stack_total_bb_RD, gmm_p_binom, taus
        )
        
        if "m" not in params: lnlike_rdr = 0.0
        if "p" not in params: lnlike_baf = 0.0
        
        log_emission = lnlike_rdr + lnlike_baf
        
        log_startprob = np.full(n_explore, -np.log(n_explore)) 

        log_gamma = hmm_sitewise.get_state_posteriors(
            lengths,
            log_transmat,
            log_startprob,
            log_emission,
            log_sitewise_transmat,
        )

        posteriors = np.exp(log_gamma)
        sum_axes = tuple(range(1, posteriors.ndim))
        component_weights = np.sum(posteriors, axis=sum_axes)
        
        full_log_mu = np.vstack([gmm_log_mu, gmm_log_mu])
        full_p_binom = np.vstack([gmm_p_binom, 1.0 - gmm_p_binom])
        
        if len(component_weights) == len(full_log_mu):
            top_indices = np.argsort(component_weights)[-n_states:][::-1]
            gmm_log_mu = full_log_mu[top_indices]
            gmm_p_binom = full_p_binom[top_indices]

    # ---------------------------------------------------------
    # 8. Fit Dispersions (MLE)
    # ---------------------------------------------------------
    logger.info("Fitting dispersions (alpha, tau) for the finalized GMM states via MLE.")
    
    new_alpha, new_tau = fit_dispersions_mle(
        clone_stack_X, 
        clone_stack_base_nb_mean, 
        clone_stack_total_bb_RD, 
        gmm_log_mu, 
        gmm_p_binom, 
        n_states
    )

    alphas = new_alpha * np.ones((n_states, 1))
    taus = new_tau * np.ones((n_states, 1))

    logger.info(
        f"GMM init converged. Final MLE dispersions: alpha={new_alpha:.4f}, tau={new_tau:.2f}"
    )

    return gmm_log_mu, gmm_p_binom, alphas, taus
    """
