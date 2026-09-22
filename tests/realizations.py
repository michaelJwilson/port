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


def planted_genome(genome: dict[str, Any] | None = None) -> CoreInferenceTruth:
    """`core_inference_truth`, with clone 0 made neutral.

    **The pipeline needs normal spots, and the fixture plants none.** Every
    clone carries events, so `determine_normal_baseline` builds its baseline
    from spots that share them and divides them out: at the fixture's
    default rates a planted `(5, 0.88)` came back as `mu = 0.92, p = 0.12`. One clone with every bin in state 0
    gives the baseline what it assumes it has.
    """
    truth = core_inference_truth(**(genome or GENOME))
    states = truth.states.copy()
    states[0] = 0

    log_mu = np.log(np.asarray(PLANTED_MU[: truth.log_mu.size], dtype=np.float64))
    base_nb_mean = truth.base_nb_mean * (EXPOSURE / truth.base_nb_mean.mean())
    truth = dataclasses.replace(truth, log_mu=log_mu, base_nb_mean=base_nb_mean)

    # NB the counts are redrawn so they follow the new path; the stream is one
    #    no realization index reaches, so the genome's own draw is not also
    #    one of its realizations.
    return realize(dataclasses.replace(truth, states=states), GENOME_DRAW)


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


class Captured(NamedTuple):
    """What `run_core_inference` was given, and what it returned."""

    single_X: np.ndarray
    lengths: np.ndarray
    single_base_nb_mean: np.ndarray
    single_total_bb_RD: np.ndarray
    result: Any


def run(truth: CoreInferenceTruth, root: Path) -> Captured:
    """Write one realization's inputs and run `run_cnaster_port` on them."""
    import cnaster.scripts.run_cnaster as pipeline
    from port.scripts.run_cnaster import main

    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config = write_run_cnaster_config(written, truth, **RUN)

    kept: list[Captured] = []
    original = pipeline.run_core_inference

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

    pipeline.run_core_inference = keep

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main([str(config), "--no-figures"])
    finally:
        pipeline.run_core_inference = original

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


def _column(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def pseudobulk(captured: Captured) -> dict[str, np.ndarray]:
    """The clone-summed inputs the HMM scored, stacked clone after clone.

    `run_core_inference` sums each clone's spots and concatenates the clones
    along the genome (`clone_stack_obs`), so the objective is one sequence of
    `n_clones * n_obs` with `lengths` tiled. The assignment is the fit's own.
    """
    assignment = np.asarray(captured.result["new_assignment"], dtype=np.int64)
    clones = np.unique(assignment)

    def summed(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [values[..., assignment == c].sum(axis=-1) for c in clones]
        )

    return {
        "counts_nb": summed(captured.single_X[:, 0, :]),
        "counts_bb": summed(captured.single_X[:, 1, :]),
        "base_nb_mean": summed(captured.single_base_nb_mean),
        "total_bb_RD": summed(captured.single_total_bb_RD),
        "lengths": np.tile(captured.lengths, clones.size),
        "n_clones": np.asarray(clones.size),
    }


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


def fitted(truth: CoreInferenceTruth, captured: Captured, *, errors: bool) -> Fit:
    """The fit's `(mu, p)` per planted state, and its covariance if asked.

    `mu` is `exp(new_log_mu)`, unshifted, as `planted_mu` is.

    The covariance is the inverse observed information of the objective the
    HMM maximized, over `(log mu_k, logit p_k, log alpha, log tau)` with the
    two dispersions shared as the configuration shares them, and transitions
    held at the fit: they are not what is plotted. `(log mu, logit p)` is
    taken to `(mu, p)` by the delta method.
    """
    from port.extensions.parameter_errors import parameter_errors

    result = captured.result
    log_mu = _column(result["new_log_mu"])
    p_binom = _column(result["new_p_binom"])
    alpha = float(_column(result["new_alphas"])[0])
    tau = float(_column(result["new_taus"])[0])
    n_states = log_mu.size

    inputs = pseudobulk(captured)
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

    import jax
    import jax.numpy as jnp
    from port.extensions.jax_hmm import emission, marginal_negative_log_likelihood

    log_startprob = _column(result["new_log_startprob"])
    log_transmat = np.asarray(result["new_log_transmat"], dtype=np.float64)

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        log_emission = emission(
            theta[:n_states],
            jnp.full(n_states, jnp.exp(theta[-2])),
            jax.nn.sigmoid(theta[n_states : 2 * n_states]),
            jnp.full(n_states, jnp.exp(theta[-1])),
            inputs["counts_nb"],
            inputs["base_nb_mean"],
            inputs["counts_bb"],
            inputs["total_bb_RD"],
        )

        return marginal_negative_log_likelihood(
            log_emission, log_startprob, log_transmat, inputs["lengths"]
        )

    theta = np.concatenate(
        [log_mu, np.log(p_binom / (1.0 - p_binom)), [np.log(alpha), np.log(tau)]]
    )
    estimate = parameter_errors(objective, theta)

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(theta)))
    decrement = float(gradient @ estimate.covariance @ gradient)

    rates = estimate.covariance[:n_states, :n_states]
    slope = p_binom * (1.0 - p_binom)
    alleles = estimate.covariance[n_states : 2 * n_states, n_states : 2 * n_states]

    covariance = np.zeros((n_states, 2, 2))

    for k in range(n_states):
        cross = estimate.covariance[k, n_states + k] * mu[k] * slope[k]
        cross = -cross if flipped[k] else cross
        covariance[k] = [
            [rates[k, k] * mu[k] ** 2, cross],
            [cross, alleles[k, k] * slope[k] ** 2],
        ]

    return Fit(mu[order], minor[order], covariance[order], decrement)


def fit_one(genome: dict[str, Any] | None, index: int, root: Path, errors: bool) -> Fit:
    """Plant, realize, run and fit one realization. The worker's whole job."""
    truth = planted_genome(genome)
    realization = realize(truth, index)
    captured = run(realization, root / f"realization-{index}")

    if captured.single_X.shape[0] != truth.states.shape[1]:
        msg = f"{captured.single_X.shape[0]} bins survived of {truth.states.shape[1]}"
        raise ValueError(msg)

    return fitted(realization, captured, errors=errors)


def realizations(
    n_realizations: int,
    root: Path,
    genome: dict[str, Any] | None = None,
    jobs: int = 1,
) -> tuple[CoreInferenceTruth, list[Fit]]:
    """Plant the genome, run each realization, and fit the first with errors.

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
            pool.submit(fit_one, genome, index, root, index == 0)
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


def summarize(truth: CoreInferenceTruth, fits: Sequence[Fit]) -> Summary:
    single = fits[0]
    if single.covariance is None:
        msg = "the first realization carries no covariance"
        raise ValueError(msg)

    sigma = np.sqrt(np.stack([single.covariance[:, 0, 0], single.covariance[:, 1, 1]]))
    planted = np.stack([planted_mu(truth), planted_minor(truth)])
    estimate = np.stack([single.mu, single.p])

    others = np.stack([np.stack([fit.mu, fit.p]) for fit in fits[1:]])

    return Summary(
        bias=((estimate - planted) / sigma).T,
        spread=(others.std(axis=0, ddof=1) / sigma).T,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/plots/realizations.png")
    )
    arguments = parser.parse_args(argv)

    from port.extensions.realization_plot import plot_realizations

    with tempfile.TemporaryDirectory() as scratch:
        truth, fits = realizations(
            arguments.realizations, Path(scratch), jobs=arguments.jobs
        )

    single = fits[0]
    assert single.covariance is not None

    figure = plot_realizations(
        planted=(planted_mu(truth), planted_minor(truth)),
        single=(single.mu, single.p, single.covariance),
        others=[(fit.mu, fit.p) for fit in fits[1:]],
        labels=[
            f"planted ({np.exp(mu):g}, {p:g})"
            for mu, p in zip(truth.log_mu, planted_minor(truth), strict=True)
        ],
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)

    summary = summarize(truth, fits)
    np.savez(
        arguments.output.with_suffix(".npz"),
        planted_mu=planted_mu(truth),
        planted_p=planted_minor(truth),
        mu=np.stack([fit.mu for fit in fits]),
        p=np.stack([fit.p for fit in fits]),
        covariance=single.covariance,
        decrement=np.nan if single.decrement is None else single.decrement,
        bias=summary.bias,
        spread=summary.spread,
    )

    print(f"wrote {arguments.output}; Newton decrement {single.decrement:.2e}")
    for state in range(truth.log_mu.size):
        print(
            f"state {state}: bias {summary.bias[state].round(2)} sigma, "
            f"spread {summary.spread[state].round(2)} x stated"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
