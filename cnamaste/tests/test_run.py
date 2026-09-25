"""A whole `run_cnamaste`, from the files it reads to the tables it writes,
judged against the instance those files were drawn from.

Realized on `DEV` at 5 outer and 10 inner iterations: clone ARI 1.000 on
both label files; per-bin BAF within 0.054 of the planted `min(p, 1 - p)`;
per-clone copy-state ARI 0.814, 1.000 and 1.000. The bounds below leave that
margin and no more.
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
        assert ari >= 0.75, f"clone {planted}: ARI {ari:.3f}"


def test_the_normal_clone_is_neutral_and_diploid_everywhere(run: Run) -> None:
    seg = run.seglevel()
    (fitted,) = [f for f, p in run.matching().items() if p == 0]
    assert seg[f"clone{fitted} Z"].nunique() == 1
    np.testing.assert_array_equal(seg[f"clone{fitted} A"], 1)
    np.testing.assert_array_equal(seg[f"clone{fitted} B"], 1)


@pytest.mark.xfail(
    strict=True,
    reason="a clone's rates are relative to its own library size, so a deletion "
    "raises its neutral bins' rate off the normal clone's (#276); the per-clone "
    "shift folded at #392 stage 3 corrects it",
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
