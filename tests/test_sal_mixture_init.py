"""`sal`'s count-pair mixture start for the read-depth + BAF stage (#489)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

#: Planted `(log mu, p)`: neutral, a one-copy loss (LOH), a one-copy gain,
#: copy-neutral LOH. The LOH states sit at a small p, not 0, as a mixture
#: with normal spots leaves them.
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
    from port.patch.hmm_initialize.sal_mixture import gmm_init

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
    """Every planted state is a fitted one, to 0.05 in log mu and 0.03 in p.

    Referee: the parameters the bins were drawn at. The exposure and trials
    vary per bin, so a start that ignored the covariate would read the
    depth's spread as states.
    """
    fitted = _fitted("lattice")

    for planted in PLANTED:
        assert _near(fitted, planted), (planted, fitted)


@pytest.mark.bug
def test_the_default_start_merges_the_loss_into_copy_neutral_loh() -> None:
    """`kmeans++x5+em`, `--sal`'s, fits no state at the one-copy loss: #471.

    Seeded in `sal`'s rate space (#547), its polish merges the loss (log mu
    -0.69, p 0.05) with copy-neutral LOH (0, 0.05) into one state near
    (-0.28, 0.05) and splits neutral in two; the lattice, from the same
    bins, fits all four. This pins the defect: it fails once the default
    start separates them, and #471 closes with it.
    """
    from port.patch.hmm_initialize.sal_mixture import DEFAULT

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
    import inspect

    import port.patch.hmrf.core_inference as core
    from port.patch.hmm_initialize import distinct, sal_mixture

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
        "baf_start": None,
    }


@pytest.mark.patch
@pytest.mark.cnaster
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize(("params", "only_minor"), [("sp", False), ("smp", True)])
def test_the_baf_only_and_minor_calls_keep_upstreams_start(
    params: str, only_minor: bool
) -> None:
    """Without exposure to condition on, the call is `cnaster`'s `gmm_init`, bitwise.

    Referee: `cnaster.hmm_initialize.gmm_init` on the same arguments.
    """
    from port.patch.hmm_initialize.distinct import UPSTREAM
    from port.patch.hmm_initialize.sal_mixture import DEFAULT, gmm_init

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
    from port.patch.hmm_initialize.sal_mixture import _call as call_of

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
    """`sal`'s seeding, raising as its M step does on the streams `rng` spawns at `refused`.

    Keyed by stream, not by call order, so a seeding is refused wherever it
    runs: in `sal`'s best-of or in port's rerun of the survivors.
    """
    import sal.search.mixture_starts as starts
    from port.patch.hmm_initialize import sal_mixture

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
    """T- #596: one seeding of five refused, the start is the best of the other four.

    Referee: those four seedings run alone, each on the stream `sal`'s
    best-of spawns for it and polished to convergence, the best by final
    log-likelihood. Since sal #1136 `sal`'s best-of skips the refused
    seeding itself, so port's fallback logs nothing (at sal `3ad4b04` it
    logged "seeding 2 dropped"; T- #632 PR A).
    """
    import logging

    from port.extensions import copy_starts
    from port.patch.hmm_initialize import sal_mixture
    from sal.search.mixture_starts import lookup, polish

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
    """Every seeding refused: the start raises, naming the start, rather than returning nothing."""
    from port.extensions import copy_starts
    from port.patch.hmm_initialize import sal_mixture

    call = _call(600, seed=2)
    held = copy_starts.instance(call)
    _refusing(monkeypatch, np.random.default_rng(0), set(range(5)))

    with pytest.raises(ValueError, match="refused every seeding"):
        copy_starts._seeded(
            sal_mixture.DEFAULT, call, held, np.random.default_rng(0), 10.0, {}
        )


REFUSED = "tests/data/sal_seeding_refused_hard.npz"
"""The read-depth + BAF start's call on CalicoST hard (`8797710b`) with the
outlier filter off (T- #596): `X`, `base`, `total`, `n_states` and
`random_state` as `gmm_init` received them, 10,448 bins and 7 states."""


@pytest.mark.release
@pytest.mark.bug
def test_sal_refuses_one_seeding_of_hard_with_the_filter_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """**T- #596:** `sal`'s best-of raises on the call; port's start drops seeding 1 and returns.

    `sal`'s beta-binomial M step refuses a component EM collapsed to weight
    1e-17 at a weighted trial count of 1.02. Fails when `sal` survives the
    call, and port's guard can go.
    """
    import logging
    from pathlib import Path

    from port.extensions import copy_starts
    from port.patch.hmm_initialize import sal_mixture
    from sal.search.mixture_starts import lookup

    saved = np.load(Path(__file__).parents[1] / REFUSED)
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

    with pytest.raises(ValueError, match="trials must be >= 2 and weight positive"):
        chosen.polished(
            held,
            np.random.default_rng(rng),
            seconds=seconds / 2.0,
            passes=None,
            tolerance=1e-6,
        )

    with caplog.at_level(logging.WARNING, logger="port.extensions.copy_starts"):
        kept: Any = copy_starts._seeded(
            sal_mixture.DEFAULT, call, held, np.random.default_rng(rng), seconds, {}
        )

    assert [r.getMessage().split(":")[1] for r in caplog.records] == [
        " seeding 1 dropped"
    ]
    assert np.asarray(kept.rate).size == int(saved["n_states"])
