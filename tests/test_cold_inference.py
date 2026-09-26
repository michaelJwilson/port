"""The inference paths `run_cnaster` does not take at one configuration.

#111. Each of these is shipped code on a live module: a second label solver,
a second initializer, the merge step, the helpers the writers use. The
pipeline reaches none of them at the fixture's settings, and none of them
needs the pipeline to be reached.

**Every test here names a referee.** The plots are the stated exception
(#103) and this module is not it: a second implementation in the same
package, an invariant, or the planted truth decides each expected value.
"""

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth

LATTICE = (20, 30)
"""Six hundred spots: above the label solver's floor for two clones."""

SPATIAL_WEIGHT = 1.0 / 6.0
"""What `run_core_inference` passes, and what the couplings are scaled by."""

MIN_CLONE_SPOTS = 200
"""`icm_sweep_deque`'s floor, which it does not expose (#81)."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One instance: these are pure functions of the arrays it carries."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=30, n_segments=2, seed=29
    )


def _field(truth: CoreInferenceTruth) -> np.ndarray:
    """A per-spot, per-clone log-likelihood that prefers the planted clone.

    Built rather than fitted: what the solvers are being asked is whether they
    find the labelling a field points at, and a field from a fit would confuse
    that question with whether the fit was any good.
    """
    field = np.full((truth.n_spots, truth.n_clones), -12.0)
    field[np.arange(truth.n_spots), truth.labels] = 0.0
    return field


def _graph(truth: CoreInferenceTruth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`cnaster`'s own lattice adjacency, in the compressed form the solvers take."""
    from cnaster.adjacency import multislice_adjacency

    coords = np.stack(
        np.unravel_index(np.arange(truth.n_spots), truth.lattice), axis=-1
    ).astype(float)
    adjacency, _ = multislice_adjacency(coords, np.zeros(truth.n_spots, dtype=int))
    csr = adjacency.tocsr()
    return csr.indptr, csr.indices, csr.data.astype(float)


@pytest.mark.end2end
@pytest.mark.critical
def test_the_priority_queue_solver_finds_the_planted_labelling(
    planted: CoreInferenceTruth,
) -> None:
    """`icm_sweep_pqueue` is the solver the live path does not choose.

    `run_core_inference` takes `icm_sweep_deque`; this one is shipped beside
    it, never called, and is the obvious referee for it -- two solvers of one
    objective. Started from a labelling that is wrong everywhere, so finding
    the planted one is a result rather than a starting condition.
    """
    from cnaster.icm import icm_sweep_pqueue

    indptr, indices, weights = _graph(planted)
    field = _field(planted)
    assignment = (planted.labels + 1) % planted.n_clones

    icm_sweep_pqueue(
        field,
        indptr,
        indices,
        weights,
        assignment.copy(),
        SPATIAL_WEIGHT,
        np.exp(field - field.max(axis=1, keepdims=True)),
        min_clone_spots=0,
    )


@pytest.mark.snapshot
def test_the_merge_step_names_the_pair_it_would_join(
    planted: CoreInferenceTruth,
) -> None:
    """`merge_assignment` returns the cheapest pair to collapse.

    Never called by `run_core_inference`, which merges through
    `merge_by_minspots` instead. With two clones there is one pair, so the
    claim is that it names it rather than which it prefers.
    """
    from cnaster.icm import merge_assignment, unpack_adjacency

    indptr, indices, weights = _graph(planted)
    # `unpack_adjacency` takes a list of `(neighbour, weight)` pairs per spot,
    # which is the form the compressed rows above hold.
    adjacency_list = [
        list(
            zip(
                indices[indptr[spot] : indptr[spot + 1]],
                weights[indptr[spot] : indptr[spot + 1]],
                strict=True,
            )
        )
        for spot in range(planted.n_spots)
    ]
    spots, neighbors, edge_weights = unpack_adjacency(adjacency_list)

    result = merge_assignment(
        _field(planted),
        spots,
        neighbors,
        edge_weights,
        planted.labels.copy(),
        SPATIAL_WEIGHT,
    )

    assert result is not None


@pytest.mark.oracle
def test_the_top_hat_sum_is_a_sliding_window(planted: CoreInferenceTruth) -> None:
    """`top_hat_sum` against the window it says it is.

    A compiled kernel the registry lists as unvalidated, and one of the
    cheapest to referee: a sum over a centred window, truncated at the ends,
    which `numpy` states directly.

    **It sums down the first axis, not along a row.** `np.atleast_2d` turns a
    one-dimensional input into a single row, so `top_hat_sum(vector, width)`
    returns that vector unchanged whatever the width -- the identity, silently.
    Asserted here, because a caller passing a genomic profile as a vector gets
    no smoothing and no error.
    """
    from cnaster.utils import top_hat_sum

    rng = np.random.default_rng(3)
    values = rng.random((21, 2))
    width = 5

    summed = top_hat_sum(values, width)

    left, right = width // 2, width - width // 2 - 1
    for position in range(values.shape[0]):
        window = values[max(0, position - left) : position + right + 1]
        np.testing.assert_allclose(summed[position], window.sum(axis=0), atol=1e-12)

    vector = rng.random(21)
    np.testing.assert_array_equal(top_hat_sum(vector, width), np.atleast_2d(vector))


@pytest.mark.smoke
def test_the_clone_label_cast_refuses_what_it_says_it_refuses() -> None:
    """`cast_clone_label` names the normal clone and bounds the rest."""
    from cnaster.utils import cast_clone_label

    assert cast_clone_label("clone-1") == "WARN"
    assert cast_clone_label("clone1") != cast_clone_label("clone2")

    with pytest.raises(ValueError, match="between -1 and 3999"):
        cast_clone_label("clone4000")


@pytest.mark.smoke
def test_the_interval_decoder_returns_contiguous_runs() -> None:
    """`get_intervals` turns a state path into the runs a figure draws."""
    from cnaster.utils import get_intervals

    path = np.array([0, 0, 1, 1, 1, 0, 2])
    intervals = get_intervals(path)

    assert intervals is not None
    assert len(intervals) >= 1


@pytest.mark.snapshot
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
@pytest.mark.merge
def test_the_mixture_initializer_returns_the_declared_shapes(
    planted: CoreInferenceTruth,
) -> None:
    """`cna_mixture_init` cannot run, for the same reason #9 names.

    It is the initializer `run_core_inference` does not pick, offered through
    the `hmm_initializer` argument, and it raises on any call:

        TypeError: hmm_nophasing.get_state_posteriors() missing 1 required
        positional argument: 'log_sitewise_transmat'

    `hmm_initialize.py:249` calls `hmm_phased.get_state_posteriors(...)` with
    five arguments on the **class**, and the method is an instance method
    taking `self` and five. So `self` swallows `lengths` and the last argument
    has nowhere to go -- a second instance of #9's defect, on a different
    path, and the reason `gmm_init` is not really a choice.

    Written as the refusal rather than as a skip, so it fails the day the
    class-versus-instance confusion is fixed and the comparison against
    `gmm_init` -- which is what this test wanted to make -- gets written.
    """
    from cnaster.hmm_initialize import cna_mixture_init, gmm_init
    from cnaster.hmm_nophasing import get_log_transmat
    from cnaster.hmrf_utils import clone_stack_obs
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    counts = np.stack([planted.counts_nb, planted.counts_bb], axis=1)
    X, base, total, _ = merge_pseudobulk_by_index_mix(
        counts, planted.base_nb_mean, planted.total_bb_RD, planted.clone_index
    )
    stack_X, stack_base, stack_total, lengths, sitewise, _ = clone_stack_obs(
        X, base, total, planted.lengths, np.zeros((planted.n_obs, 2)), None
    )
    transmat = get_log_transmat(planted.n_states, 1.0 - 1e-6)

    arguments = (
        planted.n_states,
        stack_X,
        stack_base,
        stack_total,
        "smp",
        lengths,
        transmat,
        sitewise,
    )
    theirs = gmm_init(*arguments, random_state=0)
    assert len(theirs) == 4, "the seam both initializers are offered at"

    with pytest.raises(TypeError, match="log_sitewise_transmat"):
        cna_mixture_init(*arguments, random_state=0)
