"""A whole `run_cnamaste`, from the files it reads to the tables it writes,
judged against the instance those files were drawn from.

Realized on `DEV` at 5 outer and 10 inner iterations, with the per-clone
shift on as `port` runs it (#392 stage 3): clone ARI 1.000 on both label
files; per-clone copy-state ARI 1.000, 1.000 and 1.000; per-bin BAF within
0.200 of the planted `min(p, 1 - p)`. Before the shift: 0.814 for the
deleted clone's states and BAF within 0.054, which the same run with the
shift turned off reproduces exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sim.outputs import Run
from sklearn.metrics import adjusted_rand_score

pytestmark = pytest.mark.end2end


@pytest.mark.parametrize("name", ["clone_labels.tsv", "baf_clone_labels.tsv"])
def test_every_spot_is_assigned_its_planted_clone(run: Run, name: str) -> None:
    fitted = run.clone_labels(name)
    assert np.all(fitted >= 0), "a planted spot is missing from the table"
    assert adjusted_rand_score(run.truth.labels, fitted) >= 0.99


def test_fitted_clones_match_planted_clones_one_to_one(run: Run) -> None:
    matching = run.matching()
    assert sorted(matching.values()) == list(range(run.truth.n_clones))


@pytest.mark.xfail(
    strict=True,
    reason="with the per-clone shift on (#392 stage 3) the fit shares states "
    "across clones that the planted model does not: one fitted state, p = 0.30, "
    "holds both a clone's balanced bins and another's deletion at p = 0.88, "
    "and the BAF error is 0.200 against 0.054 with the shift off",
)
def test_each_bins_allele_fraction_is_the_planted_one(run: Run) -> None:
    """`min(p, 1 - p)`: the run reports `p` on whichever haplotype it phased to."""
    seg = run.seglevel()
    assert len(seg) == run.truth.n_obs

    for fitted, planted in run.matching().items():
        truth = run.truth.p_binom[run.truth.states[planted]]
        estimate = seg[f"clone{fitted} p"].to_numpy()
        error = np.abs(
            np.minimum(truth, 1 - truth) - np.minimum(estimate, 1 - estimate)
        )
        assert error.max() <= 0.1, f"clone {planted}: {error.max():.3f}"


def test_each_tumour_clones_copy_state_path_is_the_planted_partition(run: Run) -> None:
    """ARI of the fitted states against the planted, per clone: labels are the run's own."""
    seg = run.seglevel()

    for fitted, planted in run.matching().items():
        if planted == 0:
            continue
        states = seg[f"clone{fitted} Z"].to_numpy()
        ari = adjusted_rand_score(run.truth.states[planted], states)
        assert ari >= 0.95, f"clone {planted}: ARI {ari:.3f}"


def test_the_normal_clone_is_neutral_and_diploid_everywhere(run: Run) -> None:
    seg = run.seglevel()
    (fitted,) = [f for f, p in run.matching().items() if p == 0]
    assert seg[f"clone{fitted} Z"].nunique() == 1
    np.testing.assert_array_equal(seg[f"clone{fitted} A"], 1)
    np.testing.assert_array_equal(seg[f"clone{fitted} B"], 1)


@pytest.mark.xfail(
    strict=True,
    reason="a clone's rates are relative to its own library size, so a deletion "
    "raises its neutral bins' rate off the normal clone's (#276). The per-clone "
    "shift folded at #392 stage 3 was expected to correct it and does not on "
    "DEV: the normal clone decodes to one state and the others' neutral bins "
    "to two more",
)
def test_planted_neutral_bins_share_the_normal_clones_state(run: Run) -> None:
    """Every clone's planted-neutral bins are fitted to the normal clone's state.

    Realized before the shift: planted clone 1, a one-copy deletion over 26 of
    60 bins, has its 34 neutral bins fitted to a state of their own, at log
    rate 0.26 above the normal clone's.
    """
    seg = run.seglevel()
    matching = run.matching()
    (normal,) = [f for f, p in matching.items() if p == 0]
    neutral = seg[f"clone{normal} Z"].mode()[0]

    for fitted, planted in matching.items():
        states = seg[f"clone{fitted} Z"].to_numpy()
        planted_neutral = run.truth.states[planted] == 0
        np.testing.assert_array_equal(
            states[planted_neutral], neutral, err_msg=f"clone {planted}"
        )


def test_the_normal_clone_is_pinned_at_unit_rate(run: Run) -> None:
    """The shift leaves the rates without a scale; the pin fixes the normal
    clone's dominant state at `mu = 1`, so its `logmu` is 0 on every bin."""
    seg = run.seglevel()
    (fitted,) = [f for f, p in run.matching().items() if p == 0]
    np.testing.assert_array_equal(seg[f"clone{fitted} logmu"], 0.0)
