"""`sal`'s count-pair mixture start for the read-depth + BAF stage (#489)."""

from __future__ import annotations

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
    from port.patch.hmm_initialize.sal_mixture import sal_mixture

    X, base, trials = _draw()

    with sal_mixture():
        from port.patch.hmm_initialize.sal_mixture import gmm_init

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
        )

    fitted = np.column_stack([log_mu.ravel(), p_binom.ravel()])

    for planted in PLANTED:
        distance = np.abs(fitted - planted)
        assert ((distance[:, 0] < 0.05) & (distance[:, 1] < 0.03)).any(), (
            planted,
            fitted,
        )


@pytest.mark.infra
def test_the_start_is_handed_over_only_while_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed, `sal_mixture.gmm_init` takes precedence over `distinct`'s."""
    import port.patch.hmrf.core_inference as core
    from port.patch.hmm_initialize import distinct, sal_mixture

    seen: list[object] = []
    monkeypatch.setattr(
        core, "UPSTREAM", lambda *_, **k: seen.append(k.get("hmm_initializer"))
    )

    with distinct.distinct_init():
        core.run_core_inference()
        with sal_mixture.sal_mixture():
            core.run_core_inference()

    assert seen == [distinct.gmm_init, sal_mixture.gmm_init]


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
    from port.patch.hmm_initialize.sal_mixture import gmm_init, sal_mixture

    X, base, trials = _draw(400, seed=1)
    arguments = (4, X, base, trials, params, np.array([400]), None, None)
    keywords = {"random_state": 0, "in_log_space": False, "only_minor": only_minor}

    theirs = UPSTREAM(*arguments, **keywords)

    with sal_mixture():
        ours = gmm_init(*arguments, **keywords)

    for mine, their in zip(ours, theirs, strict=True):
        if their is None:
            assert mine is None
        else:
            np.testing.assert_array_equal(mine, their)
