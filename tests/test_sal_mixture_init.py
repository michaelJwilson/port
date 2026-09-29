"""`sal`'s count-pair mixture start for the read-depth + BAF stage (#489)."""

from __future__ import annotations

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
