"""`HMMInit` reports the spread and records a raising backend as a failure (#229, #230).

`infra`: checks the harness, not `cnaster`.
"""

from __future__ import annotations

import pytest
from port.sandbox.extensions.hmm_init_trials import HMMInit


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
    """`best` ranks by the referee score, not a backend's own units."""

    # NB named functions: the harness passes `seed` as a keyword.
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
    """A raising backend is recorded as a failure, not fatal (#230)."""

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
    """`str` reports a spread and flags the best as biased."""
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
