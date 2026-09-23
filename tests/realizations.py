"""Realizations of one planted genome, each fitted by `run_cnaster_port` (#291).

The genome -- segments, events, clone bands, exposure, depth -- is planted
once by `tests.fixtures.core_inference_truth`. A **realization** redraws only
the counts, through the same families and the same per-spot streams the
fixture uses, keyed on a realization seed instead of the genome's. So the
scatter across realizations is the sampling variation of one experiment,
which is what a fit's stated error claims to describe.

Each realization is run through `port.scripts.run_cnaster.main`, in process,
with `cnaster.scripts.run_cnaster.run_core_inference` wrapped so that its
inputs and its result are kept. The objective the fit maximized is then
rebuilt from those inputs in `jax` and differentiated at the fit by
`port.extensions.parameter_errors`: nothing is re-fitted.

Run as `python -m tests.realizations` to write the figure.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from port.extensions import copy_errors as _copy_errors
from port.extensions.copy_errors import Captured
from scipy.optimize import linear_sum_assignment

from tests.fixtures import CoreInferenceTruth, _emission_families, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

GENOME = {
    "n_clones": 3,
    "n_states": 3,
    "lattice": (30, 40),
    "n_obs": 300,
    "n_segments": 3,
    "events": (3, 5),
    "event_bins": (20, 40),
    "reads": (10, 31),
    "seed": 12,
}
"""The planted genome the figure is drawn for, before clone 0 is neutralized.

Three bands of 400 spots, over `icm_sweep_deque`'s 200-spot merge (#81).
Events of 20 to 40 bins, so each event state holds enough of the genome to be
estimated: at the fixture's default extent over 120 bins they are 2 to 4 bins
and the fit invents states instead. Seed 12 because it plants all three
states -- seed 11 plants no bin of state 1, which is a figure of two states.
"""

EXPOSURE = 100.0
"""Expected normal-coverage counts per `(segment, spot)`: `mu = 1` draws 100
in expectation. The fixture's `depth` draws about 1.75, which leaves a
thousand spots a thin BAF and RDR signal each; the planted exposure is
rescaled to this mean, keeping its shape along the genome. Allele trials are
`reads = (10, 31)`, uniform on 10 to 30, 20 in expectation."""

PLANTED_MU = (1.0, 1.5, 3.0)
"""The rates planted, relative to normal coverage, in place of the fixture's
`linspace(1.5, 5, K - 1)`: the highest state at 3 rather than 5."""

GENOME_DRAW = 2**31
"""The realization stream the planted genome's own counts come from."""

RUN = {"max_iter_outer": 3, "max_iter": 200}
"""The pipeline's iteration budgets. `max_iter` is the HMM's and decides
whether the returned point is an optimum; the Newton decrement reported for
the realization with errors says whether it was."""


def planted_genome(
    genome: dict[str, Any] | None = None, *, lattice: bool = False
) -> CoreInferenceTruth:
    """`core_inference_truth`, planted from the model.

    **The model:** `<u_gn> = lambda_g T_n mu_{s_n(g)} / sum_g' lambda_g' mu_{s_n(g')}`,
    with `lambda_g` the normal profile over the genome, normalized to one,
    and `T_n` spot `n`'s coverage. The normalizer `Z_n` is the spot's own
    `sum lambda mu`, so a spot's expected total is `T_n` whatever its copy
    states. `cnaster` builds its baseline as `lambda_g T_n` from the counts
    (`normal_spot.py:162`), which is this model's denominator-free part.

    `lambda_g` is the fixture's exposure averaged over spots, which keeps its
    variation along the genome, and `T_n` its per-spot total rescaled so a
    segment carries `EXPOSURE` counts in expectation. The planted
    `base_nb_mean` is then `lambda_g T_n / Z_n`, the factor `realize` draws
    `mu` against.

    Clone 0 is the fixture's normal clone (#298), which
    `determine_normal_baseline` needs: without one, at the fixture's default
    rates, a planted `(5, 0.88)` came back as `mu = 0.92, p = 0.12`.

    `lattice` plants `tests.fixtures.COPY_LATTICE` instead of `PLANTED_MU` and
    the fixture's allele fractions, so every state is an integer `(A, B)`
    with `2 mu = A + B` (#353); the genome is otherwise the same.
    """
    # NB clone 0 is planted normal by the fixture itself (#298).
    truth = core_inference_truth(**(genome or GENOME), copy_lattice=lattice)
    states = truth.states

    log_mu = (
        np.asarray(truth.log_mu, dtype=np.float64)
        if lattice
        else np.log(np.asarray(PLANTED_MU[: truth.log_mu.size], dtype=np.float64))
    )
    mu = np.exp(log_mu)

    profile = truth.base_nb_mean.mean(axis=1)
    profile = profile / profile.sum()

    coverage = truth.base_nb_mean.sum(axis=0)
    coverage = coverage * (EXPOSURE * profile.size / coverage.mean())

    normalizer = (profile[:, None] * mu[states[truth.labels]].T).sum(axis=0)
    base_nb_mean = profile[:, None] * coverage[None, :] / normalizer[None, :]

    truth = dataclasses.replace(
        truth, log_mu=log_mu, base_nb_mean=base_nb_mean, states=states
    )

    # NB the counts are redrawn so they follow the new path; the stream is one
    #    no realization index reaches, so the genome's own draw is not also
    #    one of its realizations.
    return realize(truth, GENOME_DRAW)


def normalizers(truth: CoreInferenceTruth) -> np.ndarray:
    """`Z_c = sum_g lambda_g mu_{s_c(g)}` per clone, under the planted profile."""
    profile = truth.base_nb_mean.mean(axis=1)
    profile = profile / profile.sum()
    mu = np.exp(truth.log_mu)
    z: np.ndarray = (profile[None, :] * mu[truth.states]).sum(axis=1)

    return z


def realize(truth: CoreInferenceTruth, seed: int) -> CoreInferenceTruth:
    """The same genome, with its counts redrawn from `seed`.

    `core_inference_truth` draws spot `s` from `default_rng([genome_seed, s])`;
    this draws it from `default_rng([genome_seed, seed, s])`, so no
    realization shares a stream with the genome's own draw or with another.
    """
    family = _emission_families(truth.log_mu, truth.alphas, truth.p_binom, truth.taus)
    counts_nb = np.empty_like(truth.counts_nb)
    counts_bb = np.empty_like(truth.counts_bb)

    for spot in range(truth.counts_nb.shape[1]):
        covariate = np.stack(
            [truth.base_nb_mean[:, spot], truth.total_bb_RD[:, spot]], axis=-1
        )
        drawn = family.sample(
            truth.states[truth.labels[spot]],
            np.random.default_rng([truth.seed, seed, spot]),
            covariate=torch.as_tensor(covariate),
        )
        counts_nb[:, spot] = drawn[..., 0]
        counts_bb[:, spot] = drawn[..., 1]

    return dataclasses.replace(truth, counts_nb=counts_nb, counts_bb=counts_bb)


def run(truth: CoreInferenceTruth, root: Path) -> Captured:
    """Write one realization's inputs and run `run_cnaster_port` on them.

    The capture wraps **`port`'s** `run_core_inference`, the one the default
    shift table installs, so what is kept is the pinned result `run_cnaster`
    hands to integer copy. Wrapping `cnaster`'s binding instead would hide it
    from the swap, which rebinds only names still bound to upstream, and the
    pin would never run.
    """
    import port.patch.hmrf as patch
    from port.scripts.run_cnaster import main

    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config = write_run_cnaster_config(written, truth, **RUN)

    kept: list[Captured] = []
    original = patch.run_core_inference

    def keep(
        single_X: Any, lengths: Any, base: Any, total: Any, *rest: Any, **kw: Any
    ) -> Any:
        result = original(single_X, lengths, base, total, *rest, **kw)

        # NB the BAF-only stage calls it too, with `params="sp"`; the copy
        #    state fit this plots is the one that also fits `mu`.
        if kw.get("params") != "smp":
            return result

        kept.append(
            Captured(
                np.array(single_X, dtype=np.float64),
                np.asarray(lengths, dtype=np.int64),
                np.array(base, dtype=np.float64),
                np.array(total, dtype=np.float64),
                result,
            )
        )
        return result

    patch.run_core_inference = keep

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main([str(config), "--no-figures"])
    finally:
        patch.run_core_inference = original

    if len(kept) != 1:
        msg = f"run_core_inference was called {len(kept)} times, expected once"
        raise RuntimeError(msg)

    return kept[0]


class Fit(NamedTuple):
    """One realization's fitted `(mu, p)` per planted state, in truth order.

    `covariance` is `(n_states, 2, 2)` in `(mu, p)` and `None` where it
    was not computed; `decrement` is the Newton decrement `g' S g` at the
    returned point, which says whether it is an optimum of the objective the
    covariance is the curvature of.
    """

    mu: np.ndarray
    p: np.ndarray
    covariance: np.ndarray | None
    decrement: float | None
    truth_covariance: np.ndarray | None = None
    """The same objective's covariance at the **planted** parameters, on this
    realization's data: the error bars the truth would carry."""


_column = _copy_errors._column
pseudobulk = _copy_errors.pseudobulk


def match_states(truth: CoreInferenceTruth, captured: Captured) -> np.ndarray:
    """`order[k]` is the fitted state whose responsibility is closest to `k`'s.

    Planted state `k`'s occupancy is its indicator over every `(bin, spot)`;
    fitted state `j`'s responsibility is its posterior `gamma_j` there, read
    at the clone the fit assigned the spot to. The distance is
    `sum (gamma_j - 1[s = k])^2`, and the one-to-one assignment minimizing
    the total is taken.

    Neither by index nor by value: the pipeline's labels are its own, and
    sorting by `mu` would pair states by the value under test. The posterior
    rather than the decoded path, so a state the fit is unsure of counts as
    unsure rather than as its argmax.
    """
    result = captured.result
    gamma = np.exp(np.asarray(result["log_gamma"], dtype=np.float64))
    assignment = np.asarray(result["new_assignment"], dtype=np.int64)
    n_states = truth.log_mu.size

    responsibility = gamma[:, :, assignment]
    planted = truth.states[truth.labels].T

    if responsibility.shape[1:] != planted.shape:
        msg = f"responsibility {responsibility.shape} is not over {planted.shape}"
        raise ValueError(msg)

    occupancy = (planted[None] == np.arange(n_states)[:, None, None]).astype(float)
    distance = ((occupancy[:, None] - responsibility[None]) ** 2).sum(axis=(2, 3))

    rows, columns = linear_sum_assignment(distance)
    order: np.ndarray = np.asarray(columns)[np.argsort(rows)]

    return order


def planted_mu(truth: CoreInferenceTruth) -> np.ndarray:
    """The planted `mu`, as planted.

    The fixture draws `counts = exposure * mu`, so `mu` is UMIs relative to
    normal coverage, which is what the fit's `exp(log_mu)` estimates against
    the pipeline's normal baseline. Neither side is shifted: `logmu_shift`
    maps `mu` to `mu`, a different quantity, and a comparison that
    shifted one side would be comparing two.
    """
    mu: np.ndarray = np.exp(truth.log_mu)

    return mu


def planted_minor(truth: CoreInferenceTruth) -> np.ndarray:
    """The planted allele fraction, folded to the minor one."""
    minor: np.ndarray = np.minimum(truth.p_binom, 1.0 - truth.p_binom)

    return minor


_covariance = _copy_errors.pinned_covariance


def fitted(truth: CoreInferenceTruth, captured: Captured, *, errors: bool) -> Fit:
    """The fit's `(mu, p)` per planted state, and its covariance if asked.

    `mu` is `exp(new_log_mu)` as the pipeline returns it: fitted with the
    per-clone shift and pinned so the balanced, lowest-`mu` state is 1
    (`port.patch.hmrf.run_core_inference`), which is the planted convention.

    **The covariance is taken in the pinned coordinates.** The shifted
    likelihood is flat along `mu -> c mu`, so its information over every
    `log mu` is singular; fixing the neutral `log mu` at 0 and differentiating
    in the rest is the pin, applied to the parameters rather than after them.
    So the other states' errors are those of their ratio to the neutral one,
    and the neutral `mu` carries none: it is 1 by construction.

    The objective is the one the HMM maximized, rebuilt in `jax`: per clone,
    the rates `log mu_k - log sum_g lambda_g mu_{s_c(g)}` over the fit's own
    decoded path, with `lambda` built as `hmrf.py:476` builds it. The path is
    held fixed, so the shift's derivative is through the rates alone. Over
    `(log mu_k != neutral, logit p_k, log alpha, log tau)`, dispersions
    shared as configured and transitions held at the fit.

    **At the truth too.** The same objective on the same data, evaluated at
    the planted `(mu, p)` placed in the fit's state labels, gives the
    covariance the truth would carry. The planted neutral state is the one
    pinned there, the fit's dispersions stand in for the pseudobulk's (the
    planted `alpha = 1/6` is per spot, not per sum over hundreds of spots),
    and each planted `p` takes the allele convention its fitted state has.
    """
    result = captured.result
    log_mu = _column(result["new_log_mu"])
    p_binom = _column(result["new_p_binom"])
    alpha = float(_column(result["new_alphas"])[0])
    tau = float(_column(result["new_taus"])[0])
    n_states = log_mu.size

    mu = np.exp(log_mu)

    # NB phasing makes the allele label arbitrary, so `p` is reported folded
    #    to the minor fraction, as `planted_minor` folds the truth. Folding is
    #    `p -> 1 - p` where it applies, which flips the sign of the covariance
    #    between `mu` and `p` and leaves the variances alone.
    flipped = p_binom > 0.5
    minor = np.where(flipped, 1.0 - p_binom, p_binom)

    order = match_states(truth, captured)

    if not errors:
        return Fit(mu[order], minor[order], None, None)

    # NB the pinned coordinates and the shift's Jacobian are
    #    `port.extensions.copy_errors`', which `--copy-errors` also uses (#353).
    pinned = _copy_errors.pinned_errors(captured)
    covariance, decrement = pinned.covariance, pinned.decrement

    # NB the truth in the fit's labels: planted state `k` is fitted state
    #    `order[k]`, and its `p` is written in that state's orientation.
    planted_log_mu = np.empty(n_states)
    planted_log_mu[order] = truth.log_mu - truth.log_mu[0]
    minor_truth = np.minimum(truth.p_binom, 1.0 - truth.p_binom)
    planted_minor_in_fit = np.empty(n_states)
    planted_minor_in_fit[order] = minor_truth
    planted_p = np.where(flipped, 1.0 - planted_minor_in_fit, planted_minor_in_fit)

    free_truth = np.array([k for k in range(n_states) if k != order[0]])

    truth_covariance, _ = _covariance(
        _copy_errors.pinned_objective(captured, free_truth),
        _copy_errors.coordinates(planted_log_mu, planted_p, free_truth, alpha, tau),
        free_truth,
        np.exp(planted_log_mu),
        planted_p,
        flipped,
    )

    return Fit(
        mu[order],
        minor[order],
        covariance[order],
        decrement,
        truth_covariance[order],
    )


def fit_one(genome: dict[str, Any] | None, index: int, root: Path, errors: bool) -> Fit:
    """Plant, realize, run and fit one realization. The worker's whole job."""
    truth = planted_genome(genome)
    realization = realize(truth, index)
    captured = run(realization, root / f"realization-{index}")

    if captured.single_X.shape[0] != truth.states.shape[1]:
        msg = f"{captured.single_X.shape[0]} bins survived of {truth.states.shape[1]}"
        raise ValueError(msg)

    return fitted(realization, captured, errors=errors)


def chosen(n_realizations: int, seed: int) -> int:
    """The realization drawn to carry the error bars, uniformly from `seed`.

    Drawn rather than fixed at 0, so the realization whose errors are shown
    is not a choice anyone made; seeded, so the figure reproduces. The index
    is printed and saved beside the figure.
    """
    return int(np.random.default_rng(seed).integers(n_realizations))


def realizations(
    n_realizations: int,
    root: Path,
    genome: dict[str, Any] | None = None,
    jobs: int = 1,
    single: int = 0,
) -> tuple[CoreInferenceTruth, list[Fit]]:
    """Plant the genome, run each realization, and fit `single` with errors.

    **One fresh process per realization.** Run back to back in one process,
    the fourth of eight stalled for over twenty minutes after its genomic
    figure, while the same realization alone completes in 35 s. A worker
    peaks at 4.7 GB resident -- three at once were killed by the memory
    cgroup -- so a process that exits after one run is what returns the
    memory. `jobs` above 1 is for a host with room for that many.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = [
            pool.submit(fit_one, genome, index, root, index == single)
            for index in range(n_realizations)
        ]
        fits = [future.result() for future in futures]

    return planted_genome(genome), fits


class Summary(NamedTuple):
    """The figure's claims as numbers, per planted state.

    `bias` is the single realization's distance from truth in its own
    standard errors, `(mu, p)`. `spread` is the other realizations'
    standard deviation over the single realization's stated one: 1 where the
    stated error describes the scatter, above 1 where it understates it.
    """

    bias: np.ndarray
    spread: np.ndarray


def summarize(
    truth: CoreInferenceTruth, fits: Sequence[Fit], index: int = 0
) -> Summary:
    single = fits[index]
    if single.covariance is None:
        msg = f"realization {index} carries no covariance"
        raise ValueError(msg)

    sigma = np.sqrt(np.stack([single.covariance[:, 0, 0], single.covariance[:, 1, 1]]))
    planted = np.stack([planted_mu(truth), planted_minor(truth)])
    estimate = np.stack([single.mu, single.p])

    others = np.stack(
        [np.stack([fit.mu, fit.p]) for at, fit in enumerate(fits) if at != index]
    )

    # NB the pinned state's `mu` has no error -- it is 1 by construction -- so
    #    its standardized entries are undefined and reported as `nan` rather
    #    than as an infinite bias.
    sigma = np.where(sigma > 0.0, sigma, np.nan)

    return Summary(
        bias=((estimate - planted) / sigma).T,
        spread=(others.std(axis=0, ddof=1) / sigma).T,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=GENOME["seed"],
        help="draws which realization carries the error bars",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/plots/realizations.png")
    )
    arguments = parser.parse_args(argv)

    from port.extensions.realization_plot import plot_realizations

    index = chosen(arguments.realizations, arguments.seed)

    with tempfile.TemporaryDirectory() as scratch:
        truth, fits = realizations(
            arguments.realizations, Path(scratch), jobs=arguments.jobs, single=index
        )

    single = fits[index]
    assert single.covariance is not None

    others = [(fit.mu, fit.p) for at, fit in enumerate(fits) if at != index]
    labels = [
        f"planted ({np.exp(mu):g}, {p:g})"
        for mu, p in zip(truth.log_mu, planted_minor(truth), strict=True)
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    # NB two figures, the same points: the errors on the realization drawn,
    #    and the errors on the truth, from the same likelihood at the planted
    #    parameters on that realization's data.
    figure = plot_realizations(
        planted=(planted_mu(truth), planted_minor(truth)),
        single=(single.mu, single.p, single.covariance),
        others=others,
        labels=labels,
    )
    figure.savefig(arguments.output, dpi=150)

    at_truth = arguments.output.with_name(
        f"{arguments.output.stem}_truth{arguments.output.suffix}"
    )
    figure = plot_realizations(
        planted=(planted_mu(truth), planted_minor(truth)),
        single=(single.mu, single.p, None),
        others=others,
        labels=labels,
        planted_covariance=single.truth_covariance,
    )
    figure.savefig(at_truth, dpi=150)

    summary = summarize(truth, fits, index)
    np.savez(
        arguments.output.with_suffix(".npz"),
        planted_mu=planted_mu(truth),
        planted_p=planted_minor(truth),
        mu=np.stack([fit.mu for fit in fits]),
        p=np.stack([fit.p for fit in fits]),
        single=index,
        covariance=single.covariance,
        truth_covariance=np.full((1,), np.nan)
        if single.truth_covariance is None
        else single.truth_covariance,
        decrement=np.nan if single.decrement is None else single.decrement,
        bias=summary.bias,
        spread=summary.spread,
    )

    print(
        f"wrote {arguments.output} and {at_truth}; realization {index} carries "
        "the errors, "
        f"Newton decrement {single.decrement:.2e}"
    )
    for state in range(truth.log_mu.size):
        print(
            f"state {state}: bias {summary.bias[state].round(2)} sigma, "
            f"spread {summary.spread[state].round(2)} x stated"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
