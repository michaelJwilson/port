"""#489: `sal`'s count-pair mixture starts on the read-depth + BAF HMM call, under the covariate.

Two steps, each a subcommand of `python -m tests.studies.hmm_starts`:

`capture SAMPLE OUT`
    One `--sal --hmm-start none --no-plots` arm, pickling every call of the
    HMM initializer (`distinct.gmm_init`) as `OUT/initNNN.pkl`.
`per-call CONFIG OUT.pkl INIT.pkl [START ...]`
    The call as `sal`'s `MixtureInstance`, conditioned on exposure and trials
    (`port.patch.hmm_initialize.sal_mixture.instance_of`). Each start is a
    `sal.search.mixture_starts.TimedStart` -- the start, then `sal`'s EM
    polish under 160 s -- through `sal.opt.budget.compare`, 3 seeds, 4
    workers, as `sal`'s `docs/nb/emission_mixture_starts.ipynb` runs them.
    Port's `distinct` start is registered beside `sal`'s and run in-process
    (the pool's workers start fresh). `CONFIG` is the run's `config.yaml`,
    whose `hmm` section `distinct` reads.

Its figure, `docs/plots/studies/hmm_starts.png`, is retired: #540's
`docs/nb/copy_state_starts.ipynb` draws runtime against gap for every start
on both stages.

A pseudobulk of inferred clones has no generating truth, so the reference is
the highest log-likelihood any trial reached and every gap is >= 0.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = [0, 1, 2]
WORKERS = 4
BUDGET_SECONDS = 160
STARTS = (
    "distinct",
    "prior",
    "data",
    "kmeans++",
    "emission++",
    "gaussian-em",
    "objective",
    "perturbed",
    "restart",
    "quantile",
    "burn-in",
    "hmc",
    "anneal",
    "gibbs-anneal",
    "tempering",
    "emission++x5+em",
    "kmeans++x5+em",
    "datax5+em",
    "annealx5+em",
)
_CALL: dict[str, Any] = {}


def capture(sample_name: str, out: Path) -> None:
    """Run `--sal --hmm-start none` on the sample, pickling each initializer call."""
    import matplotlib as mpl

    mpl.use("Agg")
    from port.patch.hmm_initialize import distinct
    from port.qa.audit import audit_sample
    from port.sim.fixtures import load_simulated

    out.mkdir(parents=True, exist_ok=True)
    real = distinct.gmm_init
    names = ("n_states", "X", "base_nb_mean", "total_bb_RD", "params", "lengths")
    count = [0]

    def capturing(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        record: dict[str, Any] = dict(zip(names, args, strict=False))
        record |= {k: kwargs.get(k) for k in ("random_state", "only_minor")}
        returned = real(*args, **kwargs)
        record["returned"] = list(returned)
        with (out / f"init{count[0]:03d}.pkl").open("wb") as fh:
            pickle.dump(record, fh, protocol=5)
        return returned

    distinct.gmm_init = capturing

    try:
        path = Path(sample_name)
        sample = (
            load_simulated(path.name, path.parent)
            if path.is_absolute()
            else load_simulated(sample_name)
        )
        audit_sample(sample, ["--sal", "--hmm-start", "none", "--no-plots"])
    finally:
        distinct.gmm_init = real


def _distinct_seeding(instance: Any, rng: np.random.Generator) -> Any:
    """port's `distinct.gmm_init` (#348) as a `sal` start."""
    from port.patch.hmm_initialize import distinct
    from sal.search.projection import Seeding

    call = _CALL["call"]
    log_mu, p_binom, _, _ = distinct.gmm_init(
        int(call["n_states"]),
        call["X"],
        call["base_nb_mean"],
        call["total_bb_RD"],
        call["params"],
        call["lengths"],
        None,
        None,
        random_state=int(rng.integers(2**31)),
        in_log_space=False,
        only_minor=False,
    )
    rates = np.column_stack([np.exp(np.ravel(log_mu)), np.ravel(p_binom)])
    return Seeding(instance.at(rates), 1.0)


def per_call(config: Path, out: Path, init: Path, names: list[str]) -> None:
    """Every start on the call, into `out` as a pickle of trials and refusals."""
    import yaml
    from cnaster.config import YAMLConfig, set_global_config
    from port.patch.hmm_initialize.sal_mixture import instance_of
    from sal.cost import Cost
    from sal.opt.budget import Budget, compare
    from sal.search.mixture_starts import (
        DETERMINISTIC,
        TimedStart,
    )
    from sal.search.mixture_starts import (
        STARTS as SAL_STARTS,
    )

    set_global_config(YAMLConfig(yaml.safe_load(config.read_text())))
    with init.open("rb") as fh:
        _CALL["call"] = pickle.load(fh)
    SAL_STARTS["distinct"] = _distinct_seeding
    call = _CALL["call"]
    instance = instance_of(
        np.asarray(call["X"]),
        np.asarray(call["base_nb_mean"]),
        np.asarray(call["total_bb_RD"]),
        int(call["n_states"]),
    )
    budget = Budget(Cost.SECONDS, BUDGET_SECONDS)
    trials: dict[str, list[Any]] = {}
    refused: dict[str, str] = {}
    seconds: dict[str, float] = {}

    for name in names or list(STARTS):
        seeds = [0] if name in DETERMINISTIC or name == "distinct" else SEEDS
        opened = time.perf_counter()
        try:
            if name == "distinct":
                trials[name] = [
                    TimedStart(name)(
                        instance, budget, np.random.default_rng([s, 0])
                    ).detail
                    for s in seeds
                ]
            else:
                comparison = compare(
                    {name: TimedStart(name)}, [instance], budget, seeds, workers=WORKERS
                )
                trials[name] = [outcome.detail for outcome in comparison.outcomes]
        except Exception as error:  # noqa: BLE001 -- a refusal is a result here
            refused[name] = f"{type(error).__name__}: {error}"
        seconds[name] = time.perf_counter() - opened
        print(
            name,
            "refused" if name in refused else "ran",
            round(seconds[name], 1),
            file=sys.stderr,
        )

    with out.open("wb") as fh:
        pickle.dump(
            {
                "trials": trials,
                "refused": refused,
                "seconds": seconds,
                "n": instance.n_samples,
            },
            fh,
            protocol=5,
        )


def main(argv: list[str] | None = None) -> None:
    """The three subcommands."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("capture")
    one.add_argument("sample")
    one.add_argument("out", type=Path)
    two = sub.add_parser("per-call")
    two.add_argument("config", type=Path)
    two.add_argument("out", type=Path)
    two.add_argument("init", type=Path)
    two.add_argument("starts", nargs="*")
    arguments = parser.parse_args(argv)

    if arguments.command == "capture":
        capture(arguments.sample, arguments.out)
    else:
        per_call(arguments.config, arguments.out, arguments.init, arguments.starts)


if __name__ == "__main__":
    main()
