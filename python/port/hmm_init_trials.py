"""Run an HMM initializer several times and report what it actually does.

**#229.** `cnaster`'s initializers are stochastic and their variance has
never been measured. `gmm_init` defaults `random_state=None`, fits with
`n_init=3` and reduces with `KMeans(n_init=10)`; `cna_mixture_init` restarts
up to a hundred times keeping the best log-likelihood. So "which initializer
is better" is not a question the code can currently answer, and a single
run's numbers are exactly what `CLAUDE.md` forbids reporting for a
stochastic procedure.

`HMMInit` is the harness that makes it answerable. It runs a backend `n`
times under independent seeds and records, per run: wall time, the referee
score, the returned parameters, the seed, and the filter record. `__str__`
reports the **spread** -- median and range -- because that is the statement
a stochastic procedure supports.

## The score is not the fitter's own

A `GaussianMixture`'s likelihood is Gaussian in a standardized space; an
emission-family mixture's is negative binomial times beta-binomial on raw
counts. **Those are not comparable numbers**, so ranking backends by "their"
likelihood ranks the spaces rather than the fits. Every candidate is scored
by one referee supplied by the caller, and `best` reads that.

## Selecting the best inflates it

`best` takes the maximum over `n` runs, which is in-sample maximization, so
the winning score is biased upward by construction. `Trials.n` is reported
beside it for that reason, and the verdict on *which initializer is better*
belongs to #230, which judges on recovery of planted truth instead.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIRRORS: tuple[str, ...] = ()
"""`port`'s own trial recorder; `cnaster` runs an initializer once and reports nothing.

The `cnaster` module this stands in for, or `()` where it stands in for
none (#250). Declared rather than inferred: a reader holding a `cnaster`
module open should be able to find `port`'s answer to it, and
`tests/test_module_correspondence.py` reads this to check that every swap
row lands in a module that admits to its target."""

__all__ = ["HMMInit", "Trial", "Trials"]

Initializer = Callable[..., Any]
Scorer = Callable[[Any], float]


@dataclass(frozen=True)
class Trial:
    """One run of one backend."""

    backend: str
    seed: int
    seconds: float
    score: float
    result: Any = field(repr=False)
    record: Any = field(default=None, repr=False)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _spread(values: Sequence[float]) -> str:
    """Median and range, or the single value when there is one run."""
    if not values:
        return "--"

    if len(values) == 1:
        return f"{values[0]:.4g}"

    return f"{statistics.median(values):.4g} [{min(values):.4g}, {max(values):.4g}]"


@dataclass(frozen=True)
class Trials:
    """Every run of every backend, and what they say together."""

    trials: tuple[Trial, ...]

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def completed(self) -> tuple[Trial, ...]:
        return tuple(trial for trial in self.trials if trial.ok)

    def backends(self) -> tuple[str, ...]:
        seen = dict.fromkeys(trial.backend for trial in self.trials)

        return tuple(seen)

    def for_backend(self, backend: str) -> tuple[Trial, ...]:
        return tuple(t for t in self.completed if t.backend == backend)

    def best(self) -> Trial | None:
        """The highest-scoring completed run, by the referee score.

        Biased upward by the selection itself; `n` is reported beside it in
        `__str__` so the number is never read as a fit quality.
        """
        completed = self.completed

        return max(completed, key=lambda t: t.score) if completed else None

    def __str__(self) -> str:
        if not self.trials:
            return "HMMInit: no runs"

        lines = [
            f"HMMInit: {self.n} runs over {len(self.backends())} backend(s), "
            f"{len(self.completed)} completed"
        ]

        for backend in self.backends():
            runs = self.for_backend(backend)
            failed = sum(1 for t in self.trials if t.backend == backend and not t.ok)

            if not runs:
                lines.append(f"  {backend}: 0/{failed} completed")
                continue

            lines.append(
                f"  {backend}: score {_spread([t.score for t in runs])}"
                f"  seconds {_spread([t.seconds for t in runs])}"
                + (f"  failed {failed}" if failed else "")
            )

            if runs[0].record is not None:
                lines.append(f"    filter: {runs[0].record}")

        winner = self.best()

        if winner is not None:
            lines.append(
                f"  best: {winner.backend} at {winner.score:.6g} "
                f"(max over n={len(self.completed)}, so biased upward; "
                "see #230 for which backend is better)"
            )

        return "\n".join(lines)


@dataclass
class HMMInit:
    """Several initializers, run repeatedly, scored by one referee.

    `backends` maps a name to a callable taking `seed` and returning
    whatever the caller's `scorer` can score. Nothing here knows what an
    initializer returns, which is what lets the same harness carry
    `cnaster`'s two and `snakes_and_ladders`' two (#229 stages 4 and 5).
    """

    backends: dict[str, Initializer]
    scorer: Scorer
    n_runs: int = 1
    base_seed: int = 0
    record_of: Callable[[Any], Any] | None = None

    def run(self, *args: Any, **kwargs: Any) -> Trials:
        """Run every backend `n_runs` times, and collect."""
        if self.n_runs < 1:
            msg = f"n_runs must be at least 1, got {self.n_runs}"
            raise ValueError(msg)

        trials: list[Trial] = []

        for name, backend in self.backends.items():
            for index in range(self.n_runs):
                seed = self.base_seed + index
                started = time.perf_counter()

                try:
                    result = backend(*args, seed=seed, **kwargs)
                except Exception as error:
                    # NB a backend that raises is a result, not an abort. #230
                    #    compares recovery *rates*, so a run that failed has
                    #    to survive as a failure rather than stop the sweep.
                    trials.append(
                        Trial(
                            backend=name,
                            seed=seed,
                            seconds=time.perf_counter() - started,
                            score=-np.inf,
                            result=None,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                    continue

                seconds = time.perf_counter() - started

                trials.append(
                    Trial(
                        backend=name,
                        seed=seed,
                        seconds=seconds,
                        score=float(self.scorer(result)),
                        result=result,
                        record=self.record_of(result)
                        if self.record_of is not None
                        else None,
                    )
                )

        return Trials(tuple(trials))
