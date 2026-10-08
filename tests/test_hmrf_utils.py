"""Clone bookkeeping helpers of the spatial layer, checked against identities (#8, #13)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.mark.smoke
@pytest.mark.parametrize("n_clones", [1, 2, 4])
def test_indices_and_assignment_invert_each_other(n_clones: int) -> None:
    """Grouping spots by clone and regrouping returns the assignment."""
    from cnaster.hmrf_utils import get_clone_assignment, get_clone_indices

    rng = np.random.default_rng(3)
    n_spots = 40
    assignment = rng.integers(0, n_clones, size=n_spots)

    indices = get_clone_indices(assignment, range(n_clones))
    recovered = get_clone_assignment(np.zeros((n_spots, 2)), indices)

    np.testing.assert_array_equal(recovered, assignment)


@pytest.mark.smoke
def test_indices_partition_the_spots() -> None:
    """Every spot lands in exactly one clone."""
    from cnaster.hmrf_utils import get_clone_indices

    assignment = np.array([0, 1, 1, 2, 0, 2, 2])
    indices = get_clone_indices(assignment, range(3))

    gathered = np.concatenate(indices)
    np.testing.assert_array_equal(np.sort(gathered), np.arange(assignment.size))


@pytest.mark.smoke
def test_contiguous_clone_ids_are_accepted_and_gaps_are_not() -> None:
    """Ids `0..n-1` are accepted and a gap is refused."""
    from cnaster.hmrf_utils import validate_clone_ids

    assert validate_clone_ids(np.array([0, 1, 2, 1, 0]))

    with pytest.raises(RuntimeError):
        validate_clone_ids(np.array([0, 2, 3]))


@pytest.mark.smoke
@pytest.mark.parametrize("n_clones", [1, 3])
def test_stacking_lays_clones_end_to_end(n_clones: int) -> None:
    """Each clone's observations form a contiguous block, in clone order."""
    from cnaster.hmrf_utils import clone_stack_obs

    rng = np.random.default_rng(11)
    n_obs = 7
    single_X = rng.random((n_obs, 2, n_clones))
    base = rng.random((n_obs, n_clones))
    total = rng.random((n_obs, n_clones))
    lengths = np.array([n_obs])
    sitewise = np.zeros(n_obs)

    stacked_X, stacked_base, stacked_total, stacked_lengths, stacked_sitewise, _ = (
        clone_stack_obs(single_X, base, total, lengths, sitewise, None)
    )

    assert stacked_X.shape == (n_obs * n_clones, 2, 1)
    for clone in range(n_clones):
        block = slice(clone * n_obs, (clone + 1) * n_obs)
        np.testing.assert_array_equal(stacked_X[block, :, 0], single_X[:, :, clone])
        np.testing.assert_array_equal(stacked_base[block, 0], base[:, clone])
        np.testing.assert_array_equal(stacked_total[block, 0], total[:, clone])

    np.testing.assert_array_equal(stacked_lengths, np.tile(lengths, n_clones))
    assert stacked_sitewise.size == n_obs * n_clones


@pytest.mark.smoke
def test_casting_a_sparse_matrix_keeps_its_non_zeros() -> None:
    """The row-wise form carries exactly the stored entries."""
    from cnaster.hmrf_utils import cast_csr

    dense = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    rows = cast_csr(csr_matrix(dense))

    assert len(rows) == dense.shape[0]
    for index, row in enumerate(rows):
        rebuilt = np.zeros(dense.shape[1])
        for column, value in row:
            rebuilt[column] = value
        np.testing.assert_array_equal(rebuilt, dense[index])
