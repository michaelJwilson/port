"""`_run_optimization_pipeline`, with the spot axis gone (#259 stage 3).

**Upstream asserts the axis is a singleton and then carries it anyway.**
`optimize_params` opens with

    assert X.shape[-1] == 1, "Currently expects multiple clone to be
                              concatenated along the genomic axis."

and below it: `_, _, n_spots = X.shape`, two scratch **lists** comprehended
over `range(n_spots)`, `get_initial_params(n_states, n_spots, ...)` building
`(n_states, n_spots)` parameters, and an emission that loops the same range.
Every one is a no-op a reader must check, and `unpack_params` already
reshapes to `(n_states, 1)` -- so the optimizer's own view of the parameters
has been single-spot all along and only the scaffolding around it was not.

This is that function with the scaffolding removed. The body is upstream's
otherwise, statement for statement, and it is written out here rather than
wrapped because a `super()` call cannot reach inside a closure to change what
`cost_fn` builds.

## What moves, which is nothing

Two scratch buffers instead of two one-element lists, `(n_states, 1)`
parameters instead of `(n_states, n_spots)`, and no `range(n_spots)`.
`tests/test_optimization_pipeline.py` is the referee: upstream's pipeline and
this one on the same fixture, every returned array bitwise.

**The final emission now takes the coded path**, which is upstream's own
`# TODO call coded` at the point it calls the dense one. Measured bitwise
against it on a `(300, 2, 1)` fixture, so the TODO is spent rather than
inherited -- and it matters beyond tidiness: with the shift on, the emission
the fit optimized and the emission the posterior is read from have to be the
same one.

## What it refuses

More than one column, by upstream's own assert, so the removal is checked
where the code relies on it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.optimize
from cnaster.config import start_time
from cnaster.count_encoder import CountEncoder
from cnaster.hmm_nophasing import get_log_transmat, numba_logsumexp
from cnaster.logger import get_logger

from port.patch.hmm_parameters import Parameters

__all__ = ["OptimizationPipeline"]

logger = get_logger(__name__, start_time=start_time)


class OptimizationPipeline:
    """`_run_optimization_pipeline` without the spot axis.

    A mixin on the patched `hmm_nophasing`. It calls `self.` throughout, so
    the emission it optimizes is whichever the class installs -- shifted or
    not -- and the posterior it reads back is the same one.

    The attributes below are `cnaster`'s, declared rather than reached for:
    a mixin that says what it needs from its host is one a type checker can
    read, and the alternative was an `# type: ignore[attr-defined]` on every
    line that touched one.
    """

    params: str
    """Which blocks the optimizer varies -- `"s"`, `"t"`, `"m"`, `"p"`."""

    t: float
    """The transition matrix's self-transition weight."""

    log_emissions: np.ndarray | None
    """Set per call by the objective; read by the EM callback."""

    state_posteriors: np.ndarray | None
    """Set per EM callback; read by the objective. `gamma`, not in log space."""

    log_startprob: np.ndarray
    """Held across the M step, because the callback needs the one it started at."""

    iterations: int
    """How many times the callback has been entered."""

    apply_logmu_shift: bool
    compute_emission_probability_nb_betabinom_coded: Any
    get_state_posteriors: Any
    forward_lattice: Any
    pack_params: Any
    unpack_params: Any

    def _run_optimization_pipeline(
        self,
        mode: str,
        X: np.ndarray,
        lengths: Any,
        n_states: int,
        base_nb_mean: np.ndarray,
        total_bb_RD: np.ndarray,
        log_sitewise_transmat: Any = None,
        tumor_prop: Any = None,
        fix_NB_dispersion: bool = False,
        shared_NB_dispersion: bool = False,
        fix_BB_dispersion: bool = False,
        shared_BB_dispersion: bool = False,
        is_diag: bool = False,
        init_log_mu: Any = None,
        init_p_binom: Any = None,
        init_alphas: Any = None,
        init_taus: Any = None,
        max_iter: int = 1_000,
        max_rdr: float = 5.0,
        tol: float = 1e-4,
        use_logit: bool = True,
        optimizer: str = "BFGS",
        num_segments_clones: Any = None,
        normal_lambda: Any = None,
        log_gamma: Any = None,
        clone_lengths: Any = None,
        propagate_errors: Any = None,
    ) -> dict[str, Any]:
        """Upstream's, with `n_spots` spent rather than carried."""
        del tumor_prop, is_diag, log_gamma, propagate_errors

        if clone_lengths is not None and num_segments_clones is None:
            num_segments_clones = clone_lengths

        if X.shape[-1] != 1:
            msg = (
                f"expected one column, got {X.shape[-1]}. `optimize_params` "
                'asserts "Currently expects multiple clone to be concatenated '
                'along the genomic axis"; this body relies on that rather '
                "than looping (#259 stage 3)."
            )
            raise ValueError(msg)

        base_nb_mean = base_nb_mean.copy()

        if max_rdr is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                est_rdr = X[:, 0, :] / base_nb_mean
                est_rdr[np.isnan(est_rdr)] = 0.0
                base_nb_mean[est_rdr > max_rdr] = 0.0

        optimize_nb = bool(np.any(base_nb_mean > 0))
        normal_log_lambda = None

        if "m" in self.params:
            assert optimize_nb, (
                "Cannot optimize negative binomial if normal baseline is not defined."
            )

            normal_log_lambda = (
                np.log(normal_lambda) if normal_lambda is not None else None
            )

        nbEncoder = CountEncoder(X[:, 0, :], base_nb_mean)
        bbEncoder = CountEncoder(X[:, 1, :], total_bb_RD)

        logger.info(
            f"Encoders built. Medians: "
            f"NB={np.median(nbEncoder.total_count):.4f} "
            f"({nbEncoder.compression_rate:.2%} comp), "
            f"BB={np.median(bbEncoder.total_count):.4f} "
            f"({bbEncoder.compression_rate:.2%} comp)."
        )

        (log_mu, p_binom, alphas, taus, log_startprob, log_transmat) = (
            self.get_initial_params(
                n_states, 1, init_log_mu, init_p_binom, init_alphas, init_taus
            )
        )

        logger.info(
            f"--- hmm initialized ({mode.upper()}) ---\n"
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

        # NB one buffer per channel, not one per spot. Reused across every
        #    call the optimizer makes, which is what they are for.
        scratch_rdr = [np.zeros((n_states, len(nbEncoder.get_unique_obs(0))))]
        scratch_baf = [np.zeros((n_states, len(bbEncoder.get_unique_obs(0))))]

        def emission(
            this_log_mu: np.ndarray,
            this_alphas: np.ndarray,
            this_p_binom: np.ndarray,
            this_taus: np.ndarray,
        ) -> np.ndarray:
            """`(n_states, n_obs, 1)`, through whichever emission is installed."""
            rdr, baf = self.compute_emission_probability_nb_betabinom_coded(
                nbEncoder,
                bbEncoder,
                this_log_mu,
                this_alphas,
                this_p_binom,
                this_taus,
                scratch_rdr=scratch_rdr,
                scratch_baf=scratch_baf,
                normal_log_lambda=normal_log_lambda,
                num_segments_clones=num_segments_clones,
            )
            out: np.ndarray = (rdr + baf)[:, :, np.newaxis]
            return out

        def unpack(params: np.ndarray) -> Parameters:
            """The optimization vector, as named parameters."""
            return Parameters.of(
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

        callback: Any

        if mode == "em":
            self.log_emissions, self.state_posteriors = None, None
            self.log_startprob = log_startprob
            self.iterations = 0

            def callback(intermediate_result: Any = None) -> None:  # noqa: ARG001
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

            def cost_fn(params: np.ndarray) -> float:
                fitted = unpack(params)

                self.log_emissions = emission(
                    fitted.log_mu, fitted.alphas, fitted.p_binom, fitted.taus
                )

                if self.state_posteriors is None:
                    callback()

                return -float(
                    np.sum(self.state_posteriors * self.log_emissions[..., 0])
                )

        elif mode == "marginal":
            callback = None

            def cost_fn(params: np.ndarray) -> float:
                fitted = unpack(params)

                log_emissions = emission(
                    fitted.log_mu, fitted.alphas, fitted.p_binom, fitted.taus
                )
                log_alpha = self.forward_lattice(
                    lengths,
                    log_transmat,
                    fitted.log_startprob,
                    log_emissions,
                    log_sitewise_transmat,
                )

                curr, total_nll = 0, 0.0

                for le in lengths:
                    total_nll += -numba_logsumexp(log_alpha[:, curr + le - 1])
                    curr += le

                return float(total_nll)

        else:
            msg = f"Unknown optimization mode: {mode}"
            raise ValueError(msg)

        options = {"maxiter": max_iter, "ftol": 1e-6, "gtol": 1e-5, "disp": False}

        start_time_opt = time.time()
        logger.info(
            f"Starting {mode} optimization with {optimizer}. "
            f"Initial cost={cost_fn(x0):.6e}"
        )

        res = scipy.optimize.minimize(
            cost_fn,
            x0,
            method=optimizer,
            jac=None,
            bounds=None,
            callback=callback,
            options=options,
        )

        logger.info(
            f"Optimization complete: {time.time() - start_time_opt:.2f}s | "
            f"{res.nit} iter | converged={res.success} | "
            f"negative ln. likelihood={res.fun:.6e}"
        )

        final = unpack(res.x)

        # NB upstream calls the dense emission here under `# TODO call coded`.
        #    The two agree bitwise, and taking the coded path is what makes
        #    the posterior below read from the emission the fit optimized --
        #    which matters once a shift is installed on one and not the other.
        # NB three-dimensional, as the lattice wants: `forward_lattice` and
        #    `backward_lattice` index `log_emission[:, t, :]`, so the trailing
        #    axis is not decoration.
        log_emission = emission(final.log_mu, final.alphas, final.p_binom, final.taus)

        log_gamma = self.get_state_posteriors(
            lengths,
            log_transmat,
            final.log_startprob,
            log_emission,
            log_sitewise_transmat,
        )
        state_prior = np.sum(np.exp(log_gamma), axis=1) / np.sum(np.exp(log_gamma))

        log_lines = [
            f"--- Final HMM State ({self.__class__.__name__}) ---",
            f"p_binom:\n"
            f"{np.array2string(final.p_binom, precision=3, suppress_small=True)}",
            f"taus:\n"
            f"{np.array2string(final.taus, formatter={'float_kind': lambda x: f'{x:.3e}'})}",
        ]

        if optimize_nb:
            log_lines.extend(
                [
                    f"log_mu:\n"
                    f"{np.array2string(final.log_mu, precision=3, suppress_small=True)}",
                    f"alphas:\n"
                    f"{np.array2string(final.alphas, precision=3, suppress_small=True)}",
                ]
            )

        log_lines.extend(
            [
                f"State posteriors:\n"
                f"{np.array2string(state_prior, formatter={'float_kind': lambda x: f'{x:.4e}'})}",
                f"Max updates (tol={tol:.6e}): "
                f"mu={np.max(np.abs(np.exp(final.log_mu) - np.exp(log_mu))):.6e}",
            ]
        )

        logger.info("\n".join(log_lines))

        return {
            "new_log_mu": final.log_mu,
            "new_alphas": final.alphas,
            "new_p_binom": final.p_binom,
            "new_taus": final.taus,
            "new_log_startprob": final.log_startprob,
            "new_log_transmat": log_transmat,
            "log_gamma": log_gamma,
            "pred_cnv": np.argmax(log_gamma, axis=0),
            "llf": -res.fun,
            "n_states": n_states,
        }

    def get_initial_params(
        self,
        n_states: int,
        n_spots: int = 1,
        init_log_mu: Any = None,
        init_p_binom: Any = None,
        init_alphas: Any = None,
        init_taus: Any = None,
    ) -> tuple[np.ndarray, ...]:
        """Upstream's defaults, as `(n_states, 1)` rather than a `vstack`.

        `np.vstack([linspace(...) for _ in range(1)]).T` is
        `linspace(...)[:, None]`; the comprehension is the spot axis showing
        through. `n_spots` is kept in the signature because `optimize_params`
        passes it positionally, and refused when it is not 1.
        """
        if n_spots != 1:
            msg = f"expected one column, got {n_spots} (#259 stage 3)."
            raise ValueError(msg)

        log_mu = (
            np.linspace(-0.1, 0.1, n_states)[:, None]
            if init_log_mu is None
            else init_log_mu
        )
        p_binom = (
            np.linspace(0.05, 0.45, n_states)[:, None]
            if init_p_binom is None
            else init_p_binom
        )
        alphas = 0.5 * np.ones((n_states, 1)) if init_alphas is None else init_alphas
        taus = 1_000 * np.ones((n_states, 1)) if init_taus is None else init_taus

        log_startprob = np.log(np.ones(n_states) / n_states)
        log_transmat = get_log_transmat(n_states, self.t)

        return log_mu, p_binom, alphas, taus, log_startprob, log_transmat
