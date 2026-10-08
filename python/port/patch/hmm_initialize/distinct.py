"""Replaces `cnaster.hmm_initialize.gmm_init`, choosing distinct states (#348).

With `only_minor=False`, upstream keeps the `K` heaviest of `2K` components,
which on a mostly normal genome are near-duplicates of the normal cluster.
Here components within :data:`RADIUS` (Mahalanobis) of a heavier one are merged
into it first. With `only_minor=True` it is upstream's unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from cnaster.hmm_initialize import gmm_init as UPSTREAM
from sklearn.mixture import GaussianMixture

from port.patch._signature import as_upstream

__all__ = ["UPSTREAM", "distinct_weights", "gmm_init"]


RADIUS = 1.0
"""Mahalanobis distance under which two components are one state."""


def distinct_weights(
    means: np.ndarray, covariances: np.ndarray, posteriors: np.ndarray
) -> np.ndarray:
    """`posteriors` with each component's mass moved onto a heavier twin; row sums unchanged."""
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


@as_upstream(UPSTREAM)
def gmm_init(arguments: dict[str, Any]) -> Any:
    """Upstream's, choosing among distinct components when `only_minor=False`.

    `cnaster` binds `gmm_init` as a default argument (`hmrf.py:425`), so
    `port.patch.hmrf.run_core_inference` passes this explicitly (`distinct_init`).
    """
    if arguments.get("only_minor", True):
        return UPSTREAM(**arguments)

    with _distinct_mixture():
        return UPSTREAM(**arguments)
