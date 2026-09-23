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


def _errors(
    axis: Any,
    centre: tuple[float, float],
    covariance: np.ndarray,
    color: str,
    once: Any,
) -> bool:
    """Contours and 1-sigma bars around `centre`; whether its `mu` is pinned.

    A pinned state's `mu` is exact, so its covariance is singular and has no
    ellipse: its error is the bar on `p` alone.
    """
    pinned = bool(covariance[0, 0] <= 0.0)

    if not pinned:
        for radius, style in zip(RADII, ("-", "--"), strict=True):
            ring = contour(centre, covariance, radius)
            axis.plot(
                ring[:, 0],
                ring[:, 1],
                style,
                color=color,
                linewidth=1.0,
                label=once(f"radius {radius:g}"),
            )

    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    axis.errorbar(
        *centre,
        xerr=sigma[0],
        yerr=sigma[1],
        fmt="none",
        ecolor=color,
        capsize=2,
    )

    return pinned


def plot_realizations(
    *,
    planted: tuple[np.ndarray, np.ndarray],
    single: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    others: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str] | None = None,
    planted_covariance: np.ndarray | None = None,
) -> Any:
    """One panel per copy state, in `(mu, p)`.

    Parameters
    ----------
    planted
        `(mu, p)` per state, the truth.
    single
        `(mu, p, covariance)` for one realization; `covariance` is
        `(n_states, 2, 2)` in `(mu, p)`, or `None` to draw it as a point.
    others
        `(mu, p)` per state for each remaining realization.
    labels
        A title per state. Defaults to the state index.
    planted_covariance
        `(n_states, 2, 2)` to draw the errors on the truth instead: the
        likelihood's covariance evaluated at the planted parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    truth_mu, truth_p = (np.asarray(values, dtype=np.float64) for values in planted)
    mu = np.asarray(single[0], dtype=np.float64)
    p = np.asarray(single[1], dtype=np.float64)
    covariance = None if single[2] is None else np.asarray(single[2], dtype=np.float64)
    n_states = truth_mu.size

    figure, axes = plt.subplots(
        1, n_states, figsize=(3.2 * n_states, 3.2), squeeze=False, layout="constrained"
    )

    shown: set[str] = set()

    def once(name: str) -> str | None:
        """The legend label the first time a series is drawn, else `None`."""
        if name in shown:
            return None

        shown.add(name)

        return name

    for state, axis in enumerate(axes[0]):
        pinned = False

        if covariance is not None:
            pinned = _errors(axis, (mu[state], p[state]), covariance[state], "C0", once)

        axis.plot(
            mu[state],
            p[state],
            "o",
            color="C0",
            markersize=4,
            label=once("one realization"),
        )

        if others:
            axis.plot(
                [other[0][state] for other in others],
                [other[1][state] for other in others],
                "o",
                markerfacecolor="none",
                color="C1",
                markersize=4,
                label=once("other realizations"),
            )

        if planted_covariance is not None:
            pinned = _errors(
                axis,
                (truth_mu[state], truth_p[state]),
                np.asarray(planted_covariance[state], dtype=np.float64),
                "k",
                once,
            )

        axis.plot(
            truth_mu[state],
            truth_p[state],
            "*",
            color="k",
            markersize=10,
            label=once("truth"),
        )

        title = labels[state] if labels is not None else f"state {state}"
        axis.set_title(title + (" (pinned)" if pinned else ""))
        axis.set_xlabel(r"$\mu$")
        axis.set_ylabel("p (minor)")
        axis.ticklabel_format(useOffset=False)

    figure.legend(loc="outside lower center", ncols=5, frameon=False)

    return figure
