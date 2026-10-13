"""`cnaster`'s spot-clone field against the paper's posterior-weighted `H_nm` (#58, #206).

`compute_loglike_spot_assignment` scores the decoded state, and weights RDR per spot.
"""

from typing import Any

import numpy as np
import pytest
from cnaster.hmrf import compute_loglike_spot_assignment

AGREEMENT_TOLERANCE = 1.0e-12
"""Field gap allowed at a point-mass posterior: reassociation (realized 4.3e-14)."""

SHARP = 1.0e6
"""A logit scale at which the softmax is a point mass to machine precision."""


def _emissions(
    n_states: int = 4, n_obs: int = 30, n_spots: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state log scores for both channels, at `cnaster`'s layout."""
    generator = np.random.default_rng(5)

    return (
        generator.normal(-2.0, 1.0, (n_states, n_obs, n_spots)),
        generator.normal(-1.5, 1.0, (n_states, n_obs, n_spots)),
    )


def _posterior(
    scale: float, n_obs: int = 30, n_clones: int = 3, n_states: int = 4
) -> np.ndarray:
    """`Q(k | g, m)`, sharpened by `scale`. Shape `(n_obs, n_clones, n_states)`."""
    generator = np.random.default_rng(23)
    logits = generator.normal(size=(n_obs, n_clones, n_states)) * scale
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))

    return np.asarray(weights / weights.sum(axis=-1, keepdims=True))


def _entropy(posterior: np.ndarray) -> float:
    """Mean posterior entropy in nats -- how far `Q` is from a point mass."""
    safe = np.clip(posterior, 1.0e-300, None)

    return float(-(posterior * np.log(safe)).sum(axis=-1).mean())


def _paper_field(posterior: np.ndarray, rdr: np.ndarray, baf: np.ndarray) -> np.ndarray:
    """Return `-H_nm`, the posterior-weighted expectation, written as defined."""
    n_obs, n_clones, _ = posterior.shape
    n_spots = rdr.shape[2]
    field = np.zeros((n_spots, n_clones))

    for spot in range(n_spots):
        for clone in range(n_clones):
            field[spot, clone] = sum(
                float(
                    (
                        posterior[position, clone]
                        * (rdr[:, position, spot] + baf[:, position, spot])
                    ).sum()
                )
                for position in range(n_obs)
            )

    return field


def _cnaster_field(
    posterior: np.ndarray,
    rdr: np.ndarray,
    baf: np.ndarray,
    **overrides: Any,
) -> np.ndarray:
    """`cnaster`'s field at the decoded state, with the channel weight off."""

    n_obs, n_clones, _ = posterior.shape
    n_spots = rdr.shape[2]

    arguments: dict[str, Any] = {
        "non_zero_weight": False,
        "num_valid_nb_spotwise": np.ones(n_spots),
        "num_valid_bb_spotwise": np.ones(n_spots),
    }
    arguments.update(overrides)

    scored: np.ndarray = compute_loglike_spot_assignment(
        n_spots,
        arguments.pop("num_valid_nb_spotwise"),
        arguments.pop("num_valid_bb_spotwise"),
        # NB an array, not `None`: `numba` cannot type `None` on the unreached branch
        np.zeros(n_spots),
        False,
        rdr,
        baf,
        posterior.argmax(axis=-1),
        n_obs,
        n_clones,
        **arguments,
    )
    return scored


@pytest.mark.analytic
def test_the_two_fields_agree_where_the_posterior_is_a_point_mass() -> None:
    """At a point-mass posterior the fields agree within `AGREEMENT_TOLERANCE`."""
    rdr, baf = _emissions()
    posterior = _posterior(SHARP)

    assert _entropy(posterior) < 1.0e-9, "the fixture's posterior is not a point mass"

    realized = float(
        np.abs(
            _cnaster_field(posterior, rdr, baf) - _paper_field(posterior, rdr, baf)
        ).max()
    )
    assert realized < AGREEMENT_TOLERANCE, (
        f"fields differ by {realized:.3g} at a point mass"
    )


@pytest.mark.bug
@pytest.mark.parametrize(
    ("scale", "floor"),
    [(4.0, 0.02), (1.0, 0.08), (0.25, 0.06)],
)
def test_the_hard_state_field_departs_from_the_paper_off_a_point_mass(
    scale: float, floor: float
) -> None:
    """Off a point mass `cnaster`'s field departs from the paper's by more than `floor`."""
    rdr, baf = _emissions()
    posterior = _posterior(scale)

    paper = _paper_field(posterior, rdr, baf)
    gap = float(
        np.abs(_cnaster_field(posterior, rdr, baf) - paper).max() / np.abs(paper).max()
    )

    assert gap > floor, (
        f"at entropy {_entropy(posterior):.4f} nats the gap is {gap:.4f}, under {floor}"
    )


@pytest.mark.warning
def test_the_channel_weight_scales_the_read_depth_alone() -> None:
    """`rel_valid_emision_weight` scales only RDR: 2 BAF vs 4 RDR segments give 0.5, to 1e-9."""
    rdr, baf = _emissions()
    posterior = _posterior(SHARP)
    n_spots = rdr.shape[2]

    unweighted = _cnaster_field(posterior, rdr, baf)
    weighted = _cnaster_field(
        posterior,
        rdr,
        baf,
        non_zero_weight=True,
        num_valid_nb_spotwise=np.full(n_spots, 4.0),
        num_valid_bb_spotwise=np.full(n_spots, 2.0),
        smooth_indices=np.arange(n_spots),
        smooth_indptr=np.arange(n_spots + 1),
    )

    decoded = posterior.argmax(axis=-1)
    read_depth = np.zeros_like(unweighted)
    for spot in range(n_spots):
        for clone in range(decoded.shape[1]):
            read_depth[spot, clone] = sum(
                rdr[decoded[position, clone], position, spot]
                for position in range(decoded.shape[0])
            )

    np.testing.assert_allclose(weighted, unweighted - 0.5 * read_depth, atol=1.0e-9)

    shift = float(np.abs(weighted - unweighted).max() / np.abs(unweighted).max())
    assert shift > 0.1, f"the weight moved the field by only {shift:.4f}"
