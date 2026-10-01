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


@pytest.mark.oracle
def test_the_start_recovers_the_planted_states() -> None:
    """Every planted state is a fitted one, to 0.05 in log mu and 0.03 in p.

    Referee: the parameters the bins were drawn at. The exposure and trials
    vary per bin, so a start that ignored the covariate would read the
    depth's spread as states.
    """
    from port.patch.hmm_initialize.sal_mixture import DEFAULT, gmm_init

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
        start=DEFAULT,
    )

    fitted = np.column_stack([log_mu.ravel(), p_binom.ravel()])

    for planted in PLANTED:
        distance = np.abs(fitted - planted)
        assert ((distance[:, 0] < 0.05) & (distance[:, 1] < 0.03)).any(), (
            planted,
            fitted,
        )


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
    assert seen[1].keywords == {"start": sal_mixture.DEFAULT, "distinct": True}


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T- #596: one seeding of five refused, the start is the best of the other four.

    Referee: those four seedings run alone, each on the stream `sal`'s
    best-of spawns for it and polished to convergence, the best by final
    log-likelihood.
    """
    from port.patch.hmm_initialize import sal_mixture
    from sal.search.mixture_starts import lookup, polish

    X, base, trials = _draw()
    instance = sal_mixture.instance_of(X, base, trials, len(PLANTED))
    chosen: Any = lookup(sal_mixture.DEFAULT)
    streams = np.random.default_rng([0, 0]).spawn(chosen.n)
    alone = [
        polish(
            instance,
            lookup(chosen.name)(instance, streams[index]).components,
            seconds=sal_mixture.POLISH_SECONDS,
            tolerance=1e-6,
        )
        for index in (0, 1, 3, 4)
    ]
    finals = [float(fit.log_likelihoods[-1]) for fit in alone]
    best: Any = alone[int(np.argmax(finals))].components

    before = len(sal_mixture.dropped())
    _refusing(monkeypatch, np.random.default_rng([0, 0]), {2})
    log_mu, p_binom = sal_mixture.fitted(
        instance, sal_mixture.DEFAULT, np.random.default_rng([0, 0])
    )

    np.testing.assert_allclose(
        log_mu, np.log(np.asarray(best.total.mean).reshape(-1)), rtol=1e-9
    )
    np.testing.assert_allclose(p_binom, np.asarray(best.rate).reshape(-1), rtol=1e-9)
    assert [index for index, _ in sal_mixture.dropped()[before:]] == [2]


@pytest.mark.smoke
def test_the_start_fails_only_when_every_seeding_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every seeding refused: the start raises, naming the start, rather than returning nothing."""
    from port.patch.hmm_initialize import sal_mixture

    X, base, trials = _draw(600, seed=2)
    instance = sal_mixture.instance_of(X, base, trials, len(PLANTED))
    _refusing(monkeypatch, np.random.default_rng(0), set(range(5)))

    with pytest.raises(ValueError, match="refused every seeding"):
        sal_mixture.fitted(instance, sal_mixture.DEFAULT, np.random.default_rng(0))


REFUSED = "tests/data/sal_seeding_refused_hard.npz"
"""The read-depth + BAF start's call on CalicoST hard (`1ae26365`) with the
outlier filter off (T- #596): `X`, `base`, `total` and `n_states` as
`instance_of` received them, 13,688 bins and 7 states."""


@pytest.mark.release
@pytest.mark.bug
def test_sal_refuses_one_seeding_of_hard_with_the_filter_off() -> None:
    """**T- #596:** `sal`'s best-of raises on the call; port's start drops seeding 2 and returns.

    `sal`'s dispersion M step refuses a component EM collapsed to weight
    1e-41. Fails when `sal` survives the call, and port's guard can go.
    """
    from pathlib import Path

    from port.patch.hmm_initialize import sal_mixture
    from sal.search.mixture_starts import lookup

    call = np.load(Path(__file__).parents[1] / REFUSED)
    instance = sal_mixture.instance_of(
        call["X"], call["base"], call["total"], int(call["n_states"])
    )
    chosen: Any = lookup(sal_mixture.DEFAULT)

    with pytest.raises(ValueError, match="mean and weight must be positive"):
        chosen.polished(
            instance,
            np.random.default_rng([0, 0]),
            seconds=sal_mixture.POLISH_SECONDS,
            passes=None,
            tolerance=1e-6,
        )

    before = len(sal_mixture.dropped())
    log_mu, p_binom = sal_mixture.fitted(
        instance, sal_mixture.DEFAULT, np.random.default_rng([0, 0])
    )

    assert [index for index, _ in sal_mixture.dropped()[before:]] == [2]
    assert log_mu.shape == p_binom.shape == (7,)
