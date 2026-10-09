"""`sal`'s count-pair mixture start for the read-depth + BAF stage (#489)."""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

import numpy as np
import port.patch.hmrf.core_inference as core
import pytest
import sal.search.mixture_starts as starts
from port.extensions import copy_starts
from port.patch.hmm_initialize import distinct, sal_mixture
from port.patch.hmm_initialize.distinct import UPSTREAM
from port.patch.hmm_initialize.sal_mixture import DEFAULT, gmm_init
from port.patch.hmm_initialize.sal_mixture import _call as call_of
from sal.search.mixture_starts import lookup, polish

from tests import ROOT

#: Planted `(log mu, p)`: neutral, one-copy loss, one-copy gain, copy-neutral LOH.
#: LOH states sit at a small p, as a mixture with normal spots leaves them.
PLANTED = np.array(
    [[0.0, 0.5], [np.log(0.5), 0.05], [np.log(1.5), 1.0 / 3.0], [0.0, 0.05]]
)
WEIGHTS = np.array([0.55, 0.15, 0.15, 0.15])


def _draw(n_obs: int = 3000, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Bins from the model's own family: NB totals over exposure, beta-binomial B counts."""
    rng = np.random.default_rng(seed)
    state = rng.choice(len(PLANTED), size=n_obs, p=WEIGHTS)
    exposure = rng.uniform(200.0, 400.0, n_obs)
    trials = rng.integers(20, 60, n_obs).astype(float)
    mean = exposure * np.exp(PLANTED[state, 0])
    # NB alpha 0.02, a clone pseudobulk's dispersion rather than a spot's.
    size = 50.0
    totals = rng.negative_binomial(size, size / (size + mean))
    p = rng.beta(PLANTED[state, 1] * 1000.0, (1.0 - PLANTED[state, 1]) * 1000.0)
    successes = rng.binomial(trials.astype(int), p)
    X = np.stack([totals, successes], axis=1).astype(float).reshape(n_obs, 2, 1)
    return X, exposure.reshape(-1, 1), trials.reshape(-1, 1)


def _fitted(start: str) -> np.ndarray:
    """`gmm_init`'s states under `start` on `_draw`'s bins, `(log mu, p)` per row."""

    X, base, trials = _draw()
    log_mu, p_binom, _, _ = gmm_init(
        len(PLANTED),
        X,
        base,
        trials,
        "smp",
        None,
        None,
        None,
        random_state=0,
        only_minor=False,
        start=start,
    )
    return np.column_stack([log_mu.ravel(), p_binom.ravel()])


def _near(fitted: np.ndarray, planted: np.ndarray) -> bool:
    distance = np.abs(fitted - planted)
    return bool(((distance[:, 0] < 0.05) & (distance[:, 1] < 0.03)).any())


@pytest.mark.oracle
def test_the_lattice_start_recovers_the_planted_states() -> None:
    """Every planted state is fitted to 0.05 in log mu and 0.03 in p, against the drawing parameters."""
    fitted = _fitted("lattice")

    for planted in PLANTED:
        assert _near(fitted, planted), (planted, fitted)


@pytest.mark.bug
def test_the_default_start_merges_the_loss_into_copy_neutral_loh() -> None:
    """`kmeans++x5+em` fits no state at the one-copy loss, merging it with LOH (#471, #547)."""

    fitted = _fitted(DEFAULT)
    loss, neutral_loh = PLANTED[1], PLANTED[3]

    assert not _near(fitted, loss), fitted
    assert not _near(fitted, neutral_loh), fitted
    assert _near(fitted, PLANTED[0]), fitted
    assert _near(fitted, PLANTED[2]), fitted


@pytest.mark.infra
def test_the_start_is_handed_over_only_under_its_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `hmm_start`, `sal_mixture.gmm_init` takes precedence over `distinct`'s."""

    seen: list[Any] = []
    monkeypatch.setattr(
        core, "UPSTREAM", lambda *_, **k: seen.append(k.get("hmm_initializer"))
    )
    parameters = inspect.signature(core.run_core_inference).parameters.values()
    blanks = [None] * sum(
        p.default is inspect.Parameter.empty
        and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in parameters
    )

    core.run_core_inference(*blanks, distinct_init=True)
    core.run_core_inference(*blanks, distinct_init=True, hmm_start=sal_mixture.DEFAULT)

    assert seen[0] is distinct.gmm_init
    assert seen[1].func is sal_mixture.gmm_init
    assert seen[1].keywords == {
        "start": sal_mixture.DEFAULT,
        "distinct": True,
    }


@pytest.mark.patch
@pytest.mark.cnaster
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize(("params", "only_minor"), [("sp", False), ("smp", True)])
def test_the_baf_only_and_minor_calls_keep_upstreams_start(
    params: str, only_minor: bool
) -> None:
    """Without exposure, the call equals cnaster's `hmm_initialize.gmm_init`, bitwise."""

    X, base, trials = _draw(400, seed=1)
    arguments = (4, X, base, trials, params, np.array([400]), None, None)
    keywords = {"random_state": 0, "in_log_space": False, "only_minor": only_minor}

    theirs = UPSTREAM(*arguments, **keywords)

    ours = gmm_init(*arguments, **keywords, start=DEFAULT)

    for mine, their in zip(ours, theirs, strict=True):
        if their is None:
            assert mine is None
        else:
            np.testing.assert_array_equal(mine, their)


def _call(n_obs: int = 3000, seed: int = 0) -> Any:
    """`_draw`'s bins as the start's `CopyCall`, read-depth + BAF stage."""

    X, base, trials = _draw(n_obs, seed)
    arguments = {
        "X": X,
        "base_nb_mean": base,
        "total_bb_RD": trials,
        "n_states": len(PLANTED),
    }
    return call_of(arguments, "rdrbaf")


def _refusing(
    monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator, refused: set[int]
) -> None:
    """`sal`'s seeding, raising as its M step does on the streams `rng` spawns at `refused`, keyed by stream."""

    inner = starts.lookup
    chosen: Any = inner(sal_mixture.DEFAULT)
    states = [
        stream.bit_generator.state
        for index, stream in enumerate(copy.deepcopy(rng).spawn(chosen.n))
        if index in refused
    ]

    def lookup(name: str) -> Any:
        found = inner(name)

        if name != chosen.name:
            return found

        def seeding(instance: Any, generator: np.random.Generator) -> Any:
            if generator.bit_generator.state in states:
                msg = "mean and weight must be positive, got 0.0 and 1e-41"
                raise ValueError(msg)

            return found(instance, generator)

        return seeding

    monkeypatch.setattr(starts, "lookup", lookup)


@pytest.mark.oracle
def test_a_refused_seeding_is_dropped_and_the_best_survivor_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One of five seedings refused: the start is the best of the other four run alone (T- #596)."""

    call = _call()
    held = copy_starts.instance(call)
    chosen: Any = lookup(sal_mixture.DEFAULT)
    seconds = sal_mixture.POLISH_SECONDS
    streams = np.random.default_rng([0, 0]).spawn(chosen.n)
    alone = [
        polish(
            held,
            lookup(chosen.name)(held, streams[index]).components,
            seconds=seconds / 2.0,
            tolerance=1e-6,
        )
        for index in (0, 1, 3, 4)
    ]
    finals = [float(fit.log_likelihoods[-1]) for fit in alone]
    best: Any = alone[int(np.argmax(finals))].components

    _refusing(monkeypatch, np.random.default_rng([0, 0]), {2})

    with caplog.at_level(logging.WARNING, logger="port.extensions.copy_starts"):
        kept: Any = copy_starts._seeded(
            sal_mixture.DEFAULT, call, held, np.random.default_rng([0, 0]), seconds, {}
        )

    np.testing.assert_allclose(
        np.asarray(kept.total.mean), np.asarray(best.total.mean), rtol=1e-9
    )
    np.testing.assert_allclose(np.asarray(kept.rate), np.asarray(best.rate), rtol=1e-9)
    assert [r.getMessage() for r in caplog.records] == []


@pytest.mark.smoke
def test_the_start_fails_only_when_every_seeding_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every seeding refused: sal's best-of raises, naming the start, rather than returning nothing."""

    call = _call(600, seed=2)
    held = copy_starts.instance(call)
    _refusing(monkeypatch, np.random.default_rng(0), set(range(5)))

    with pytest.raises(ValueError, match="every one of 5 seedings"):
        copy_starts._seeded(
            sal_mixture.DEFAULT, call, held, np.random.default_rng(0), 10.0, {}
        )


REFUSED = "tests/data/sal_seeding_refused_hard.npz"
"""The read-depth + BAF start's `gmm_init` call on CalicoST hard (`8797710b`): 10,448 bins, 7 states (T- #596)."""


@pytest.mark.release
@pytest.mark.patch
def test_sal_survives_the_call_it_refused_on_hard_with_the_filter_off() -> None:
    """On the T- #596 call, port's start is `sal`'s `polished` best-of on the same stream (T- #632)."""

    saved = np.load(ROOT / REFUSED)
    arguments = {
        "X": saved["X"],
        "base_nb_mean": saved["base"],
        "total_bb_RD": saved["total"],
        "n_states": int(saved["n_states"]),
    }
    call = sal_mixture._call(arguments, "rdrbaf")
    held = copy_starts.instance(call)
    chosen: Any = lookup(sal_mixture.DEFAULT)
    seconds = sal_mixture.POLISH_SECONDS
    rng = [int(saved["random_state"]), 0]

    _, best = chosen.polished(
        held,
        np.random.default_rng(rng),
        seconds=seconds / 2.0,
        passes=None,
        tolerance=1e-6,
    )
    kept: Any = copy_starts._seeded(
        sal_mixture.DEFAULT, call, held, np.random.default_rng(rng), seconds, {}
    )

    assert np.asarray(kept.rate).size == int(saved["n_states"])
    # NB both polish under a wall-clock budget, so to the sibling test's 1e-9.
    np.testing.assert_allclose(
        np.asarray(kept.rate), np.asarray(best.components.rate), rtol=1e-9
    )
