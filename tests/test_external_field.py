"""The field the HMM hands the labelling, against the one the paper defines (#58).

**Step 1 of #206's plan: pin the seam before moving it.** `likelihood.tex`
defines the external field as a posterior-weighted expectation over copy
states,

    H_nm = - sum_{g, k} Q(k_gm | theta', l) [ ln P(u_gn | k) + ln P(b_gn | k) ]

and `compute_loglike_spot_assignment` evaluates it at the single decoded
state instead, then multiplies the read-depth channel by a per-spot factor
the paper does not define. Two tests here measure the first difference as a
number and one pins the second.

Neither is fixed here: `cnaster` is read only, and a refactor of this seam
that does not know what it is preserving will carry both into the rewrite --
which is the whole reason this lands before #206's step 2.
"""

from typing import Any

import numpy as np
import pytest

AGREEMENT_TOLERANCE = 1.0e-12
"""How far the two fields may sit apart where the posterior is a point mass.

Realized 4.3e-14 over twelve spots, three clones, thirty positions and four
states. The two sum the same terms in a different order there, so this is
reassociation; a departure at 1e-12 is a defect.
"""

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
    """`-H_nm`, the posterior-weighted expectation, at `cnaster`'s sign.

    Written as the definition reads rather than as an einsum, because what
    is being checked is that the definition was followed.
    """
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
    from cnaster.hmrf import compute_loglike_spot_assignment

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
        # NB an array rather than `None`: the weighting branch reads it, and
        #    `numba` cannot type `None` there even where `is_tumor_mixed` is
        #    False and the read is unreachable.
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
    """The limit the discrepancy is measured from.

    `sum_k Q(k) ln P(x | k)` is `ln P(x | k*)` exactly when `Q` is a point
    mass at `k*`, so agreement here is the definition and not a property of
    `cnaster`. It is asserted because it is what makes the next test's number
    mean something: a gap at a sharp posterior would be a second, different
    defect, and this separates them.
    """
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
    """`cnaster` evaluates at the decoded state; the paper sums over `Q`.

    Realized relative gaps, as a fraction of the field's own scale, against
    the mean posterior entropy that produced them:

    | entropy (nats) | gap |
    | ---: | ---: |
    | 0.42 | 3.9% |
    | 1.10 | 12.5% |
    | 1.36 | 10.2% |

    **Not monotone in entropy, and the floors say so rather than claiming
    otherwise**: as `Q` flattens the expectation tends to the mean over
    states, which is not always further from the mode's score than a
    half-sharp posterior is. What the numbers establish is that the two are
    different objects wherever the posterior is not degenerate -- which is
    everywhere the variational formulation is doing any work.

    `bug`, and written to fail when `cnaster` adopts the variational form:
    the assertion is that the fields *differ*.
    """
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
    """`rel_valid_emision_weight` multiplies one channel by a ratio of counts.

    `pooled_num_valid_bb / pooled_num_valid_nb` over the spot's smoothed
    neighbourhood, applied to the RDR term and not the BAF term. The paper's
    `H_nm` has no such factor, and `integer_copy_numbers.tex` criticises
    exactly this -- down-weighting the read-depth ratio by an ad-hoc factor
    irrespective of the inferred dispersions -- in CalicoST.

    Pinned rather than argued: with two valid BAF segments per spot against
    four valid RDR segments the weight is exactly one half, and the field is
    then `0.5 * rdr + baf` position by position. `warning` rather than `bug`
    because the comment on `non_zero_weight` states a real problem -- the
    RDR would otherwise out-weigh a BAF channel full of zero-depth segments
    -- and it is the fix that is ad-hoc, not the concern.
    """
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
