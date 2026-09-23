"""The read-depth stage's GMM initializer, choosing distinct states (#348).

`cnaster.hmm_initialize.gmm_init` fits `2K` Gaussian components to BAF-mirrored
data and, with `only_minor=False` (the read-depth stage), keeps the `K` with
the most posterior mass (`hmm_initialize.py:444`). Mass is where the data is:
on a genome that is mostly normal, the most populated components are slices
of the normal cluster and their mirror images, which at `p = 0.5` are the
same point. Measured on `tests.fixtures.calicost_instance`: six of the eight
initial states at `p` 0.497 to 0.503 and `log mu` -0.25 to 0.07, two for
eight planted events, and the fit kept three planted states in one fitted
state (copy-state ARI 0.896 against CalicoST's 0.999, which fits eight states
per clone).

`run_core_inference` calls it with `only_minor=False` in both stages, BAF
and read-depth (`hmrf.py:522`), so both are affected.

`gmm_init` here is upstream's with one change, made where the selection reads
its weights: components within one standard deviation of a heavier one
(Mahalanobis, under their mean covariance) are merged into it, mass and all,
before the top `K` are taken. So the selection is still by mass, among
components that differ. With `only_minor=True`, which groups the `2K` into
`K` by k-means on the folded means, it is upstream's unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from cnaster.hmm_initialize import gmm_init as UPSTREAM
from sklearn.mixture import GaussianMixture

__all__ = ["UPSTREAM", "distinct_init", "distinct_weights", "gmm_init", "installed"]

_INSTALLED: list[bool] = [False]


def installed() -> bool:
    """Whether :func:`distinct_init` is active."""
    return _INSTALLED[0]


@contextmanager
def distinct_init() -> Iterator[None]:
    """Hand :func:`gmm_init` to `run_core_inference` for the block.

    `cnaster.hmrf.run_core_inference` binds its initializer as a default
    argument (`hmm_initializer=gmm_init`, `hmrf.py:425`), which rebinding the
    module name does not reach; `port.patch.hmrf.run_core_inference` passes
    it explicitly while this is active.
    """
    previous = _INSTALLED[0]
    _INSTALLED[0] = True

    try:
        yield
    finally:
        _INSTALLED[0] = previous


RADIUS = 1.0
"""Mahalanobis distance under which two components are one state."""


def distinct_weights(
    means: np.ndarray, covariances: np.ndarray, posteriors: np.ndarray
) -> np.ndarray:
    """`posteriors` with each component's mass moved onto a heavier twin.

    Greedy by total mass: a component within :data:`RADIUS` of one already
    kept gives its column to it and keeps zeros. The row sums are unchanged.
    """
    merged = np.array(posteriors, dtype=np.float64, copy=True)
    order = np.argsort(-merged.sum(axis=0), kind="stable")
    kept: list[int] = []

    for component in order:
        for head in kept:
            difference = means[component] - means[head]
            pooled = 0.5 * (covariances[component] + covariances[head])
            if difference @ np.linalg.solve(pooled, difference) < RADIUS**2:
                merged[:, head] += merged[:, component]
                merged[:, component] = 0.0
                break
        else:
            kept.append(int(component))

    return merged


class _Distinct(GaussianMixture):  # type: ignore[misc]
    """`GaussianMixture` whose `predict_proba` merges indistinguishable components."""

    def predict_proba(self, X: Any) -> np.ndarray:
        posteriors = super().predict_proba(X)
        covariances = np.asarray(self.covariances_)

        if self.covariance_type == "diag":
            covariances = np.stack([np.diag(c) for c in covariances])

        return distinct_weights(np.asarray(self.means_), covariances, posteriors)


@contextmanager
def _distinct_mixture() -> Iterator[None]:
    import cnaster.hmm_initialize as module

    previous = module.GaussianMixture
    module.GaussianMixture = _Distinct

    try:
        yield
    finally:
        module.GaussianMixture = previous


def gmm_init(*args: Any, **kwargs: Any) -> Any:
    """Upstream's, choosing among distinct components when `only_minor=False`."""
    only_minor = kwargs.get("only_minor", args[10] if len(args) > 10 else True)

    if only_minor:
        return UPSTREAM(*args, **kwargs)

    with _distinct_mixture():
        return UPSTREAM(*args, **kwargs)
