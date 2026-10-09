"""The distinct-component GMM initializer against analytic merges and `cnaster`'s `gmm_init` (#348)."""

from __future__ import annotations

import inspect

import cnaster.hmm_initialize as upstream
import numpy as np
import port.patch.hmrf.core_inference as core
import pytest
from port.patch.hmm_initialize import distinct
from port.patch.hmm_initialize.distinct import distinct_weights, gmm_init


@pytest.mark.analytic
def test_duplicates_merge_into_the_heavier_and_distinct_ones_stay() -> None:
    """Two normal slices a tenth of a sigma apart are one; an event is not."""

    means = np.array([[0.0, 0.5], [0.02, 0.5], [0.5, 0.2]])
    covariances = np.stack([np.eye(2) * 0.04] * 3)
    posteriors = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8]])

    merged = distinct_weights(means, covariances, posteriors)

    np.testing.assert_allclose(merged.sum(axis=1), posteriors.sum(axis=1))
    np.testing.assert_allclose(merged[:, 0], posteriors[:, 0] + posteriors[:, 1])
    np.testing.assert_array_equal(merged[:, 1], 0.0)
    np.testing.assert_array_equal(merged[:, 2], posteriors[:, 2])


@pytest.mark.analytic
def test_a_mirror_image_at_one_half_is_the_same_component() -> None:
    """At `p = 0.5` the mirror is the same point; at 0.2 it is 0.8, distinct."""

    means = np.array([[0.0, 0.5], [0.0, 0.5], [0.4, 0.2], [0.4, 0.8]])
    covariances = np.stack([np.eye(2) * 0.01] * 4)
    posteriors = np.full((2, 4), 0.25)

    merged = distinct_weights(means, covariances, posteriors)

    assert (merged.sum(axis=0) > 0).sum() == 3


@pytest.mark.smoke
def test_the_initializer_is_handed_over_only_under_its_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    monkeypatch.setattr(
        core, "UPSTREAM", lambda *_, **k: seen.append(k.get("hmm_initializer"))
    )
    parameters = inspect.signature(core.run_core_inference).parameters.values()
    blanks = [None] * sum(
        p.default is inspect.Parameter.empty
        and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for p in parameters
    )

    core.run_core_inference(*blanks)
    core.run_core_inference(*blanks, distinct_init=True)

    assert seen == [None, gmm_init]


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize(
    ("seed", "patches", "options"),
    [
        (3, {}, {"only_minor": True}),
        (4, {"RADIUS": 0.0}, {"in_log_space": False, "only_minor": False}),
    ],
    ids=["only-minor", "mixed-phase-at-radius-zero"],
)
def test_the_initializer_is_upstreams_bitwise(
    seed: int,
    patches: dict[str, float],
    options: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`only_minor=True` is not changed, and at `RADIUS = 0` `only_minor=False` is upstream's: the same parameters, bit for bit."""

    for name, value in patches.items():
        monkeypatch.setattr(distinct, name, value)
    rng = np.random.default_rng(seed)
    n_obs = 300
    base = np.full((n_obs, 1), 60.0)
    total = np.full((n_obs, 1), 40.0)
    X = np.stack(
        [rng.poisson(60.0, (n_obs, 1)), rng.binomial(40, 0.3, (n_obs, 1))], axis=1
    ).astype(float)
    arguments = (4, X, base, total, "smp", np.array([n_obs]), None, None)

    ours = distinct.gmm_init(*arguments, random_state=0, **options)
    theirs = upstream.gmm_init(*arguments, random_state=0, **options)

    for mine, upstreams in zip(ours, theirs, strict=True):
        if upstreams is None:
            assert mine is None
        else:
            np.testing.assert_array_equal(mine, upstreams)
