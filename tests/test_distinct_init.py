"""The GMM initializer chooses among distinct components (#348).

On `calicost_instance`, `cnaster`'s top-`K`-by-mass selection started the
read-depth fit from six slices of the normal cluster and two event states;
choosing among distinct components started it from three and five, and the
integer copy-state ARI went from 0.896 to 0.997 (CalicoST 0.999). Pinned:

- mirror images and near-duplicates are merged into the heavier, mass and
  all, and distinct components are kept (`analytic`);
- `port.patch.hmrf.run_core_inference` hands the initializer to `cnaster`
  only while `distinct_init()` is active (`infra`).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.analytic
def test_duplicates_merge_into_the_heavier_and_distinct_ones_stay() -> None:
    """Two normal slices a tenth of a sigma apart are one; an event is not."""
    from port.patch.hmm_initialize.distinct import distinct_weights

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
    from port.patch.hmm_initialize.distinct import distinct_weights

    means = np.array([[0.0, 0.5], [0.0, 0.5], [0.4, 0.2], [0.4, 0.8]])
    covariances = np.stack([np.eye(2) * 0.01] * 4)
    posteriors = np.full((2, 4), 0.25)

    merged = distinct_weights(means, covariances, posteriors)

    assert (merged.sum(axis=0) > 0).sum() == 3


@pytest.mark.infra
def test_the_initializer_is_handed_over_only_while_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import port.patch.hmrf.core_inference as core
    from port.patch.hmm_initialize.distinct import distinct_init, gmm_init

    seen: list[object] = []
    monkeypatch.setattr(
        core, "UPSTREAM", lambda *_, **k: seen.append(k.get("hmm_initializer"))
    )

    core.run_core_inference()
    with distinct_init():
        core.run_core_inference()

    assert seen == [None, gmm_init]


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_initializer_is_upstreams_bitwise_with_only_minor() -> None:
    """`only_minor=True` is not changed: the same parameters, bit for bit."""
    import cnaster.hmm_initialize as upstream
    from port.patch.hmm_initialize.distinct import gmm_init

    rng = np.random.default_rng(3)
    n_obs = 300
    base = np.full((n_obs, 1), 60.0)
    total = np.full((n_obs, 1), 40.0)
    X = np.stack(
        [rng.poisson(60.0, (n_obs, 1)), rng.binomial(40, 0.3, (n_obs, 1))], axis=1
    ).astype(float)
    arguments = (4, X, base, total, "smp", np.array([n_obs]), None, None)

    ours = gmm_init(*arguments, random_state=0, only_minor=True)
    theirs = upstream.gmm_init(*arguments, random_state=0, only_minor=True)

    for mine, upstreams in zip(ours, theirs, strict=True):
        if upstreams is None:
            assert mine is None
        else:
            np.testing.assert_array_equal(mine, upstreams)
