"""Inference paths `run_cnaster` does not take, each against a referee (#111).

Referees: a second solver in the package, an invariant, or the planted truth.
"""

import numpy as np
import pytest
from cnaster.adjacency import multislice_adjacency
from cnaster.hmm_initialize import cna_mixture_init, gmm_init
from cnaster.hmm_nophasing import get_log_transmat
from cnaster.hmrf_utils import clone_stack_obs
from cnaster.icm import icm_sweep_pqueue, merge_assignment, unpack_adjacency
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.utils import cast_clone_label, top_hat_sum
from port.sim.truth import CoreInferenceTruth, core_inference_truth

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
    """Return a per-spot, per-clone field preferring the planted clone, built not fitted."""
    field = np.full((truth.n_spots, truth.n_clones), -12.0)
    field[np.arange(truth.n_spots), truth.labels] = 0.0
    return field


def _graph(truth: CoreInferenceTruth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `cnaster`'s lattice adjacency as CSR `(indptr, indices, weights)`."""

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
    """`icm_sweep_pqueue` finds the planted labelling from one wrong everywhere."""

    indptr, indices, weights = _graph(planted)
    field = _field(planted)
    assignment = (planted.labels + 1) % planted.n_clones

    found = assignment.copy()
    icm_sweep_pqueue(
        field,
        indptr,
        indices,
        weights,
        found,
        SPATIAL_WEIGHT,
        np.exp(field - field.max(axis=1, keepdims=True)),
        min_clone_spots=0,
    )
    # NB wrong at every spot before, planted after (#749 WP2)
    assert np.array_equal(assignment == planted.labels, np.zeros(planted.n_spots, bool))
    assert np.array_equal(found, planted.labels)


@pytest.mark.snapshot
def test_the_merge_step_names_the_pair_it_would_join(
    planted: CoreInferenceTruth,
) -> None:
    """`merge_assignment` names the only pair of two clones."""

    indptr, indices, weights = _graph(planted)
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

    # NB `(cost, best merge cost, pair)`: two clones, one pair (#749 WP2)
    assert result[2] == (0, 1)


@pytest.mark.oracle
def test_the_top_hat_sum_is_a_sliding_window(planted: CoreInferenceTruth) -> None:
    """`top_hat_sum` equals a centred, truncated `numpy` window sum, to 1e-12; a 1-D input is returned unchanged."""

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


@pytest.mark.warning
def test_the_clone_label_cast_refuses_what_it_says_it_refuses() -> None:
    """`cast_clone_label` names the normal clone and bounds the rest."""

    assert cast_clone_label("clone-1") == "WARN"
    assert cast_clone_label("clone1") != cast_clone_label("clone2")

    with pytest.raises(ValueError, match="between -1 and 3999"):
        cast_clone_label("clone4000")


@pytest.mark.snapshot
@pytest.mark.usefixtures("cnaster_config", "cnaster_perf_sink")
@pytest.mark.merge
def test_the_mixture_initializer_returns_the_declared_shapes(
    planted: CoreInferenceTruth,
) -> None:
    """`cna_mixture_init` raises `TypeError` (class-vs-instance call, as #9); `gmm_init` runs."""

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
