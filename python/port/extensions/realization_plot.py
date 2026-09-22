r"""Fitted `(mu, p)` over realizations of one planted genome (#291).

**What a stated error is for, drawn beside what it claims.**
`port.extensions.parameter_errors` gives one fit's covariance. That covariance
is a prediction: refit the same genome from fresh counts and the estimates
should scatter inside it. This draws the prediction and the test of it on one
axis per copy state:

- one realization, with 1-sigma error bars and its 1- and 2-sigma contours;
- every other realization, as a point without error bars;
- the planted truth.

`mu` is the rate relative to normal coverage, compared unshifted: the planted
`mu` realizes UMIs against the normal baseline, and the fit's `exp(log_mu)`
estimates the same thing. `logmu_shift` would take either to `mubar`, a
different quantity, and is applied to neither.

A fit whose contours hold the other realizations and the truth is calibrated
and unbiased. One whose other realizations sit inside its contours but whose
truth does not is **precise and biased**, and a panel is the quickest way to
tell the two apart.

**The contours are Mahalanobis radii, not credible levels.** Radius `r` is the
set `(x - m)' S^-1 (x - m) = r^2`. In two dimensions that holds 39.3 and 86.5
per cent of a Gaussian at `r = 1, 2`, not the 68 and 95 a reader carries over
from one; the legend says radius so neither is implied.

One panel per state, because the states sit a decade apart in `mu` and the
contours are a few thousandths wide: a shared axis draws every ellipse as a
point.

An extension, per #274: `cnaster` draws nothing of the kind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["contour", "plot_realizations"]

RADII = (1.0, 2.0)
"""The Mahalanobis radii drawn around the single realization."""


def contour(
    mean: Sequence[float] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
    radius: float,
    n_points: int = 200,
) -> np.ndarray:
    """`(n_points, 2)` on the ellipse `(x - m)' S^-1 (x - m) = radius^2`.

    `m + radius * L u` for `u` on the unit circle and `L` the Cholesky factor
    of `S`: `(L u)' S^-1 (L u) = u' u = 1`, so every point is exactly on the
    contour. Cholesky rather than an eigendecomposition because it refuses a
    covariance that is not positive definite, which is the one a contour
    cannot be drawn for.
    """
    centre = np.asarray(mean, dtype=np.float64).reshape(2)
    factor = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    angles = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(angles), np.sin(angles)])

    points: np.ndarray = centre[:, None] + radius * factor @ circle

    return points.T


def plot_realizations(
    *,
    planted: tuple[np.ndarray, np.ndarray],
    single: tuple[np.ndarray, np.ndarray, np.ndarray],
    others: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str] | None = None,
) -> Any:
    """One panel per copy state, in `(mu, p)`.

    Parameters
    ----------
    planted
        `(mu, p)` per state, the truth.
    single
        `(mu, p, covariance)` for the realization with errors;
        `covariance` is `(n_states, 2, 2)` in `(mu, p)`.
    others
        `(mu, p)` per state for each remaining realization.
    labels
        A title per state. Defaults to the state index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    truth_mu, truth_p = (np.asarray(values, dtype=np.float64) for values in planted)
    mu, p, covariance = (np.asarray(values, dtype=np.float64) for values in single)
    n_states = truth_mu.size

    figure, axes = plt.subplots(
        1, n_states, figsize=(3.2 * n_states, 3.2), squeeze=False, layout="constrained"
    )

    for state, axis in enumerate(axes[0]):
        centre = (mu[state], p[state])
        sigma = np.sqrt(np.diag(covariance[state]))

        for radius, style in zip(RADII, ("-", "--"), strict=True):
            ring = contour(centre, covariance[state], radius)
            axis.plot(
                ring[:, 0],
                ring[:, 1],
                style,
                color="C0",
                linewidth=1.0,
                label=f"radius {radius:g}" if state == 0 else None,
            )

        axis.errorbar(
            *centre,
            xerr=sigma[0],
            yerr=sigma[1],
            fmt="o",
            color="C0",
            markersize=4,
            capsize=2,
            label="one realization" if state == 0 else None,
        )

        if others:
            axis.plot(
                [other[0][state] for other in others],
                [other[1][state] for other in others],
                "o",
                markerfacecolor="none",
                color="C1",
                markersize=4,
                label="other realizations" if state == 0 else None,
            )

        axis.plot(
            truth_mu[state],
            truth_p[state],
            "*",
            color="k",
            markersize=10,
            label="truth" if state == 0 else None,
        )

        axis.set_title(labels[state] if labels is not None else f"state {state}")
        axis.set_xlabel(r"$\mu$")
        axis.set_ylabel("p (minor)")
        axis.ticklabel_format(useOffset=False)

    figure.legend(loc="outside lower center", ncols=5, frameon=False)

    return figure
