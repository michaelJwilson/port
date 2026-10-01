"""#540's arms: every copy-state start, and what masking, smoothing, outliers and RDR do to the best few.

Each arm is a list of `Job`s; each job is one start on one stage at one seed,
through `port.sandbox.extensions.copy_starts`, and returns one row of results. Jobs
run in forked workers, which inherit the captured calls.

- `starts`: every start on every stage it takes.
- `no-covariate`: the shortlist's covariate-reading `sal` starts on counts alone.
- `mask-seed`, `mask-fit`: the shortlist seeded, then also fitted, on the
  rows a mask keeps; every result is polished and scored on the whole call.
- `smooth`: the shortlist seeded on rows summed over a window along the genome.
- `outlier`: the shortlist on a call with 1% or 5% of its rows replaced,
  masked and not; scored on that call, read against the clean call's states.
- `rdr-seeds-baf`: the BAF-only stage seeded from the BAF + RDR call's fit,
  and from its read-depth quantiles alone.
"""

from __future__ import annotations

import multiprocessing
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, NamedTuple

import numpy as np

SHORTLIST = (
    "kmeans++x5+em",
    "emission++x5+em",
    "datax5+em",
    "kmeans++",
    "distinct",
    "lattice",
    "rdr-quantiles",
)
"""#489's three polished best-of-five starts, its best single start, port's, and the lattice and read-depth starts."""

MASKS = {
    "rdrbaf": (
        "rdr-top-1%",
        "rdr-top-5%",
        "zero-counts",
        "baf-se-0.2",
        "baf-se-0.15",
        "baf-se-0.1",
    ),
    "baf": ("zero-counts", "baf-se-0.2", "baf-se-0.15", "baf-se-0.1"),
}
WINDOWS: tuple[dict[str, Any], ...] = (
    {"segments": 3},
    {"segments": 5},
    {"segments": 9},
    {"bp": 1e6},
    {"bp": 5e6},
    {"bp": 1e7},
)
OUTLIERS = (("rdr", 0.01), ("rdr", 0.05), ("baf", 0.01), ("baf", 0.05))
OUTLIER_MASK = {"rdr": "rdr-top-5%", "baf": "baf-se-0.15"}
OUTLIER_SEED = 540
"""One corruption per (kind, fraction), so every start sees the same call."""

_CALLS: dict[str, Any] = {}


class Job(NamedTuple):
    arm: str
    variant: str
    stage: str
    start: str
    seed: int


def _window_name(window: dict[str, Any]) -> str:
    if "segments" in window:
        return f"{int(window['segments'])} segments"
    return f"{window['bp'] / 1e6:g} Mb"


def _jobs(arm: str, seeds: list[int]) -> list[Job]:
    from port.sandbox.extensions.copy_starts import starts

    jobs: list[Job] = []
    stages = [s for s in ("baf", "rdrbaf") if s in _CALLS]

    def each(
        variant: str, stage: str, names: Any, *, covariate_only: bool = False
    ) -> None:
        for name in names:
            row = starts()[name]
            if stage not in row.stages or (
                covariate_only and (row.source != "sal" or not row.covariate)
            ):
                continue
            for seed in seeds if row.stochastic else seeds[:1]:
                jobs.append(Job(arm, variant, stage, name, seed))

    for stage in stages:
        if arm == "starts":
            each("", stage, list(starts()))
        elif arm == "no-covariate":
            each("", stage, SHORTLIST, covariate_only=True)
        elif arm in ("mask-seed", "mask-fit"):
            for mask in MASKS[stage]:
                each(mask, stage, SHORTLIST)
        elif arm == "smooth":
            for window in WINDOWS:
                each(_window_name(window), stage, SHORTLIST)
        elif arm == "outlier":
            for kind, fraction in OUTLIERS:
                if kind == "rdr" and stage == "baf":
                    continue
                for masked in ("", f" masked {OUTLIER_MASK[kind]}"):
                    each(f"{kind} {fraction:.0%}{masked}", stage, SHORTLIST)
        elif arm == "rdr-seeds-baf" and stage == "baf" and "rdrbaf" in _CALLS:
            each("from the BAF + RDR fit", stage, SHORTLIST)
            for seed in seeds[:1]:
                jobs.append(
                    Job(arm, "from RDR quantiles", stage, "rdr-quantiles", seed)
                )
    return jobs


def _corrupt(call: Any, variant: str) -> tuple[Any, Any]:
    from port.sandbox.extensions.copy_starts import corrupted

    kind, rest = variant.split(" ", 1)
    fraction = float(rest.split("%")[0]) / 100.0
    return corrupted(
        call, fraction, kind, np.random.default_rng([OUTLIER_SEED, len(variant)])
    )


def _run(job: Job, seconds: float) -> dict[str, Any]:
    from port.sandbox.extensions import copy_starts as cs

    call = _CALLS[job.stage]
    rng = np.random.default_rng([job.seed, 540])
    scored = call
    try:
        if job.arm == "starts":
            result = cs.run_start(job.start, call, rng, seconds=seconds)
        elif job.arm == "no-covariate":
            result = cs.run_start(
                job.start, call, rng, covariate=False, seconds=seconds
            )
        elif job.arm == "mask-seed":
            result = cs.run_start(
                job.start,
                call,
                rng,
                seed_on=cs.masked(call, job.variant),
                seconds=seconds,
            )
        elif job.arm == "mask-fit":
            kept = cs.masked(call, job.variant)
            result = cs.run_start(
                job.start, call, rng, seed_on=kept, fit_on=kept, seconds=seconds
            )
        elif job.arm == "smooth":
            window = next(w for w in WINDOWS if _window_name(w) == job.variant)
            result = cs.run_start(
                job.start,
                call,
                rng,
                seed_on=cs.smoothed(call, **window),
                seconds=seconds,
            )
        elif job.arm == "outlier":
            scored, _ = _corrupt(call, job.variant.split(" masked")[0])
            seed_on = (
                cs.masked(scored, job.variant.split("masked ")[1])
                if "masked" in job.variant
                else None
            )
            result = cs.run_start(
                job.start, scored, rng, seed_on=seed_on, seconds=seconds
            )
        elif job.arm == "rdr-seeds-baf":
            rdrbaf = _CALLS["rdrbaf"]
            opened = time.perf_counter()
            if job.variant == "from RDR quantiles":
                p = cs.rdr_quantile_states(rdrbaf)[1]
            else:
                p = cs.run_start(job.start, rdrbaf, rng, seconds=seconds / 2.0).p_binom
            p = _to_k(p, call.n_states)
            result = cs.polish_states(
                job.start, call, np.zeros(p.size), p, seconds=seconds / 2.0,
                handover=time.perf_counter() - opened,
            )  # fmt: skip
        else:
            msg = f"no arm {job.arm}"
            raise ValueError(msg)
    except Exception as error:  # noqa: BLE001 -- a refusal is a result here
        return {**job._asdict(), "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(limit=3)}  # fmt: skip

    states = cs.planted_states(call)
    return {
        **job._asdict(),
        "error": None,
        "log_likelihood": result.log_likelihood,
        "seconds": result.seconds,
        "handover": result.handover,
        "log_mu": result.log_mu,
        "p_binom": result.p_binom,
        "found": cs.found(result, states),
        "scored_on": "corrupted" if scored is not call else "call",
    }


def _to_k(p: Any, k: int) -> np.ndarray:
    """`p` as `k` states: sorted, then evenly subsampled, or padded at 0.5."""
    ordered: np.ndarray = np.sort(np.asarray(p, dtype=np.float64))
    if ordered.size >= k:
        picked: np.ndarray = ordered[
            np.round(np.linspace(0, ordered.size - 1, k)).astype(int)
        ]
        return picked
    padded: np.ndarray = np.concatenate([ordered, np.full(k - ordered.size, 0.5)])
    return padded


def _init(calls: dict[str, Any]) -> None:
    _CALLS.update(calls)


def run_arms(
    calls: dict[str, Any],
    arms: list[str],
    *,
    seeds: list[int],
    workers: int,
    seconds: float,
    progress: Callable[[str], None] = print,
    only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Every job of every arm, in `workers` forked processes; rows, planted states and settings.

    `only` keeps the jobs of those starts alone: how a start added later
    joins an earlier run's rows.
    """
    from port.sandbox.extensions.copy_starts import planted_states

    _init(calls)
    jobs = [
        job
        for arm in arms
        for job in _jobs(arm, seeds)
        if only is None or job.start in only
    ]
    progress(f"{len(jobs)} jobs over {arms}")
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("fork")
    opened = time.perf_counter()
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_init, initargs=(calls,)
    ) as pool:
        futures = [pool.submit(_run, job, seconds) for job in jobs]
        for k, future in enumerate(futures, start=1):
            rows.append(future.result())
            if k % 20 == 0 or k == len(futures):
                progress(
                    f"{k}/{len(futures)} jobs, {time.perf_counter() - opened:.0f} s"
                )
    return {
        "rows": rows,
        "planted": {stage: planted_states(call) for stage, call in calls.items()},
        "settings": {
            "seeds": seeds,
            "workers": workers,
            "seconds": seconds,
            "shortlist": SHORTLIST,
            "arms": arms,
        },  # fmt: skip
    }


ARMS = (
    "starts",
    "no-covariate",
    "mask-seed",
    "mask-fit",
    "smooth",
    "outlier",
    "rdr-seeds-baf",
)
