"""`HMMInit` reports the spread, and survives a backend that raises.

**#229.** The harness exists because `cnaster`'s initializers are stochastic
and nothing measures their variance. These are `infra`: they say the harness
reports what it claims to, not anything about `cnaster`.

The failure case is the one worth pinning. #230 compares backends on how
*often* they recover every state, so a run that raises has to survive as a
recorded failure rather than aborting the sweep -- otherwise the metric that
matters cannot be computed at all.
"""

from __future__ import annotations

import pytest
from port.extensions.hmm_init_trials import HMMInit


@pytest.mark.infra
def test_it_records_every_run_of_every_backend() -> None:
    harness = HMMInit(
        backends={
            "rising": lambda seed: float(seed),
            "falling": lambda seed: -float(seed),
        },
        scorer=float,
        n_runs=3,
    )
    trials = harness.run()

    assert trials.n == 6
    assert len(trials.completed) == 6
    assert trials.backends() == ("rising", "falling")
    assert [t.seed for t in trials.for_backend("rising")] == [0, 1, 2]


@pytest.mark.infra
def test_best_reads_the_referee_score_not_the_backends_own() -> None:
    """A backend cannot win by reporting a bigger number in its own units."""

    # NB named rather than lambdas because the harness passes `seed` as a
    #    keyword, so the parameter cannot be renamed away to quiet the linter.
    def smaller(seed: int) -> float:
        del seed

        return 1.0

    def larger(seed: int) -> float:
        del seed

        return 2.0

    harness = HMMInit(
        backends={"a": smaller, "b": larger},
        scorer=lambda result: -result,
        n_runs=1,
    )
    best = harness.run().best()

    assert best is not None
    assert best.backend == "a", "best must follow the scorer, not the value"


@pytest.mark.infra
def test_a_backend_that_raises_is_recorded_rather_than_fatal() -> None:
    """#230 compares recovery rates, so a failure is data."""

    def explodes(seed: int) -> float:
        if seed == 1:
            msg = "planted"
            raise RuntimeError(msg)

        return float(seed)

    harness = HMMInit(backends={"flaky": explodes}, scorer=float, n_runs=3)
    trials = harness.run()

    assert trials.n == 3
    assert len(trials.completed) == 2

    failed = [t for t in trials.trials if not t.ok]

    assert len(failed) == 1
    assert failed[0].error is not None
    assert "planted" in failed[0].error
    assert "failed 1" in str(trials)


@pytest.mark.infra
def test_str_reports_a_spread_and_says_the_best_is_biased() -> None:
    """A single run's number is what CLAUDE.md forbids for a stochastic method."""
    harness = HMMInit(
        backends={"varied": lambda seed: float(seed)}, scorer=float, n_runs=4
    )
    rendered = str(harness.run())

    assert "score" in rendered
    assert "[0, 3]" in rendered, f"no range in:\n{rendered}"
    assert "biased upward" in rendered
    assert "n=4" in rendered


@pytest.mark.infra
def test_zero_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        HMMInit(backends={}, scorer=float, n_runs=0).run()
