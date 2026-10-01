"""`tests.sim_audit`'s copy-state ARI per planted class (#511).

`class_ari` over `planted_classes` is the copy ARI on one class's clone-bins.
The referee is sklearn's `adjusted_rand_score` on bins selected by hand, by
index, not by `planted_classes`; a perfect decode scores 1 in every class
with two or more planted pairs, and a class planted as one pair is NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tests.sim_audit import class_ari, phase_free, planted_classes


def _code(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([a * 1_000 + b for a, b in pairs], dtype=np.int64)


# NB index: 0-3 LOH, 4-7 balanced gain, 8-11 unbalanced gain, 12-15 neutral
TRUTH = _code(
    [(0, 1), (0, 1), (1, 0), (0, 2),
     (2, 2), (2, 2), (3, 3), (3, 3),
     (2, 1), (2, 1), (1, 3), (1, 3),
     (1, 1), (1, 1), (1, 1), (1, 1)]
)  # fmt: skip
DECODE = _code(
    [(0, 1), (1, 0), (1, 0), (0, 2),
     (2, 2), (1, 1), (3, 3), (3, 3),
     (1, 2), (1, 2), (1, 3), (2, 1),
     (1, 1), (2, 1), (1, 1), (1, 1)]
)  # fmt: skip
BY_HAND = {
    "loh": [0, 1, 2, 3],
    "balanced_gain": [4, 5, 6, 7],
    "unbalanced_gain": [8, 9, 10, 11],
    "neutral": [12, 13, 14, 15],
}


@pytest.mark.oracle
def test_each_class_ari_is_sklearns_on_that_class_bins_alone() -> None:
    classes = planted_classes(TRUTH)
    for name in ("loh", "balanced_gain", "unbalanced_gain"):
        index = BY_HAND[name]
        assert np.flatnonzero(classes[name]).tolist() == index
        expected = adjusted_rand_score(TRUTH[index], DECODE[index])
        assert class_ari(TRUTH, DECODE, classes[name]) == pytest.approx(
            expected, abs=5e-5
        )
    # NB the subsets differ from the whole: each decode error above is
    #    priced against its class, not against the neutral bins
    whole = adjusted_rand_score(TRUTH, DECODE)
    assert all(
        abs(class_ari(TRUTH, DECODE, classes[n]) - whole) > 0.01
        for n in ("loh", "balanced_gain", "unbalanced_gain")
    )


@pytest.mark.analytic
def test_a_perfect_decode_scores_one_in_every_class_with_two_planted_pairs() -> None:
    classes = planted_classes(TRUTH)
    for name in ("loh", "balanced_gain", "unbalanced_gain"):
        assert class_ari(TRUTH, TRUTH, classes[name]) == 1.0


@pytest.mark.infra
def test_a_class_planted_as_one_pair_or_not_at_all_is_nan() -> None:
    classes = planted_classes(TRUTH)
    # NB neutral is one pair by definition: sklearn would read 1 for a
    #    constant decode and 0 otherwise, neither a measurement
    assert adjusted_rand_score(TRUTH[12:], DECODE[12:]) == 0.0
    assert np.isnan(class_ari(TRUTH, DECODE, classes["neutral"]))

    single = _code([(2, 2), (2, 2), (1, 1)])
    decode = _code([(2, 2), (3, 3), (1, 1)])
    assert np.isnan(class_ari(single, decode, planted_classes(single)["balanced_gain"]))
    assert np.isnan(class_ari(single, decode, planted_classes(single)["loh"]))


@pytest.mark.oracle
def test_each_phase_free_class_ari_is_sklearns_on_hand_sorted_pairs() -> None:
    """Phase-free: the pairs sorted by hand to `(min, max)`, then sklearn per class.

    The unbalanced gain's swaps `(2, 1) -> (1, 2)` and `(1, 3) -> (2, 1)` are
    one consistent and one not; phase-free, the first is a hit.
    """

    def by_hand(codes: np.ndarray) -> np.ndarray:
        return np.array(
            [
                min(c // 1_000, c % 1_000) * 1_000 + max(c // 1_000, c % 1_000)
                for c in codes
            ]
        )

    truth, decode = by_hand(TRUTH), by_hand(DECODE)
    np.testing.assert_array_equal(phase_free(TRUTH), truth)
    np.testing.assert_array_equal(phase_free(DECODE), decode)
    classes = planted_classes(TRUTH)

    for name in ("loh", "unbalanced_gain"):
        index = BY_HAND[name]
        expected = adjusted_rand_score(truth[index], decode[index])
        assert class_ari(truth, decode, classes[name]) == pytest.approx(
            expected, abs=5e-5
        )
