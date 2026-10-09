"""`port.patch.icm.interface`'s eight-parameter solver call, bitwise against cnaster's
fifteen-argument `icm_sweep_deque` (#59 item 5).

Each comparison seeds the legacy global RNG identically, since the sweep's queue order
draws from it.
"""

import inspect

import numpy as np
import pytest
from cnaster import icm
from port.patch.icm.interface import CsrGraph, fold_unary, icm_sweep
from scipy.sparse import csr_matrix

from tests.adapters import cnaster_sweep
from tests.builders import random_graph

N_SPOTS = 400
N_CLONES = 4
N_SAMPLES = 3
BETA = 0.75
SEED = 6_803


def _problem(
    seed: int = SEED,
) -> tuple[np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
    """A drawn field, a spatial graph, a labelling and each spot's sample."""
    rng = np.random.default_rng(seed)

    field = rng.normal(-50.0, 5.0, (N_SPOTS, N_CLONES))

    graph = random_graph(rng, N_SPOTS, (2, 8))
    assignment = rng.integers(0, N_CLONES, N_SPOTS)
    sample_ids = rng.integers(0, N_SAMPLES, N_SPOTS)

    return field, graph, assignment, sample_ids


_cnaster_sweep = cnaster_sweep


def _patched_sweep(
    field: np.ndarray,
    graph: csr_matrix,
    assignment: np.ndarray,
    beta: float,
    *,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    """The eight-parameter call, on an already-folded field."""
    # NPY002: cnaster's sweep shuffles with the legacy global RNG.
    np.random.seed(seed)  # noqa: NPY002
    labels = assignment.copy()

    result = icm_sweep(
        field, CsrGraph.from_matrix(graph), labels, beta, min_clone_spots=0
    )

    return labels, result.niter, result.cost


def _assert_same(
    expected: tuple[np.ndarray, int, float], actual: tuple[np.ndarray, int, float]
) -> None:
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1] == actual[1]
    assert expected[2] == actual[2], "the cost must be bitwise, not close"


@pytest.mark.patch
@pytest.mark.parametrize("seed", [11, 23])
def test_the_reduced_call_is_bitwise_cnasters(seed: int) -> None:
    """Same labelling, iteration count and cost on the plain problem, bitwise."""
    field, graph, assignment, _ = _problem()

    expected = _cnaster_sweep(field, graph, assignment, BETA, seed=seed)
    actual = _patched_sweep(field, graph, assignment, BETA, seed=seed)

    _assert_same(expected, actual)
    assert not np.array_equal(actual[0], assignment), "the sweep must do something"


@pytest.mark.patch
def test_the_per_sample_weights_fold_bitwise() -> None:
    """Folding `log_persample_weights[c, sample_ids[i]]` once matches cnaster's per- visit add, bitwise."""
    field, graph, assignment, sample_ids = _problem()

    rng = np.random.default_rng(SEED)
    weights = rng.normal(0.0, 2.0, (N_CLONES, N_SAMPLES))

    expected = _cnaster_sweep(
        field,
        graph,
        assignment,
        BETA,
        seed=SEED,
        log_persample_weights=weights,
        sample_ids=sample_ids,
    )
    actual = _patched_sweep(
        fold_unary(field, weights, sample_ids), graph, assignment, BETA, seed=SEED
    )

    _assert_same(expected, actual)
    assert not np.array_equal(
        expected[0], _cnaster_sweep(field, graph, assignment, BETA, seed=SEED)[0]
    ), "the weights must change the answer, or this tests nothing"


@pytest.mark.patch
def test_the_allowed_clone_mask_folds_bitwise() -> None:
    """`onehot_allowed_clones` folded as `-inf` matches cnaster's post-edge-term overwrite, bitwise."""
    field, graph, assignment, _ = _problem()

    rng = np.random.default_rng(SEED)
    allowed = rng.random((N_SPOTS, N_CLONES)) > 0.25
    allowed[np.arange(N_SPOTS), assignment] = True

    assert not allowed.all(), "the mask must forbid something"

    expected = _cnaster_sweep(
        field, graph, assignment, BETA, seed=SEED, onehot_allowed_clones=allowed
    )
    actual = _patched_sweep(
        fold_unary(field, onehot_allowed_clones=allowed),
        graph,
        assignment,
        BETA,
        seed=SEED,
    )

    _assert_same(expected, actual)
    assert allowed[np.arange(N_SPOTS), actual[0]].all(), "a forbidden clone was chosen"


@pytest.mark.patch
@pytest.mark.parametrize("temp", [0.5, 2.0])
def test_the_temperature_folds_into_the_coupling(temp: float) -> None:
    """Passing `spatial_weight / temp` matches cnaster's own quotient, bitwise."""
    field, graph, assignment, _ = _problem()

    expected = _cnaster_sweep(field, graph, assignment, BETA, seed=SEED, temp=temp)
    actual = _patched_sweep(field, graph, assignment, BETA / temp, seed=SEED)

    _assert_same(expected, actual)


@pytest.mark.bug
def test_the_posterior_argument_is_never_read_or_written() -> None:
    """`posterior` is unread: a finite array comes back untouched and the labelling equals `None`'s."""
    field, graph, assignment, _ = _problem()

    posterior = np.full((N_SPOTS, N_CLONES), 0.25)
    witness = posterior.copy()

    with_array = _cnaster_sweep(
        field, graph, assignment, BETA, seed=SEED, posterior=posterior
    )
    with_none = _cnaster_sweep(field, graph, assignment, BETA, seed=SEED)

    np.testing.assert_array_equal(witness, posterior)
    _assert_same(with_none, with_array)


@pytest.mark.patch
def test_the_live_sweep_is_the_csr_one() -> None:
    """`cnaster.icm` exports the last of four `icm_sweep_deque` definitions, the CSR one the patch wraps."""

    source = inspect.getsource(icm)
    assert source.count("\ndef icm_sweep_deque(") == 4, "the shadowing changed"

    parameters = list(inspect.signature(icm.icm_sweep_deque).parameters)

    assert parameters[1:4] == ["adj_indptr", "adj_indices", "adj_weights"]
    assert "adj_spots" not in parameters, "the COO definition is live again"
    assert len(parameters) == 15


@pytest.mark.bug
def test_the_sweep_is_not_reproducible_without_seeding_a_global() -> None:
    """Two unseeded runs of cnaster's sweep differ: it shuffles with the legacy global RNG."""
    field, graph, assignment, _ = _problem()

    # NPY002: cnaster's sweep shuffles with the legacy global RNG.
    np.random.seed(1)  # noqa: NPY002
    first = _patched_sweep(field, graph, assignment, BETA, seed=1)

    # NPY002: cnaster's sweep shuffles with the legacy global RNG.
    np.random.seed(2)  # noqa: NPY002
    second = _patched_sweep(field, graph, assignment, BETA, seed=2)

    assert not np.array_equal(first[0], second[0])


@pytest.mark.analytic
def test_the_mask_is_exact_under_the_edge_term() -> None:
    """`-inf + finite == -inf`, which makes the fold exact."""
    edge = np.array([0.0, 1e300, -1e300, np.finfo(np.float64).max])

    assert np.all(-np.inf + edge == -np.inf)


@pytest.mark.smoke
def test_the_fold_leaves_the_field_it_was_given() -> None:
    """Folding returns a copy; cnaster reads the unfolded field after the sweep."""
    field, _, _, sample_ids = _problem()
    witness = field.copy()

    rng = np.random.default_rng(SEED)
    fold_unary(field, rng.normal(0.0, 2.0, (N_CLONES, N_SAMPLES)), sample_ids)

    np.testing.assert_array_equal(witness, field)


@pytest.mark.warning
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"log_persample_weights": np.zeros((N_CLONES, N_SAMPLES))}, "requires"),
        (
            {
                "log_persample_weights": np.zeros((N_CLONES + 1, N_SAMPLES)),
                "sample_ids": np.zeros(N_SPOTS, dtype=int),
            },
            "clones",
        ),
        ({"onehot_allowed_clones": np.ones((N_SPOTS, N_CLONES + 1), dtype=bool)}, "is"),
    ],
)
def test_the_fold_refuses_a_shape_it_cannot_index(
    kwargs: dict[str, np.ndarray], match: str
) -> None:
    """Mis-shaped weights or mask raise rather than broadcast."""
    field, _, _, _ = _problem()

    with pytest.raises(ValueError, match=match):
        fold_unary(field, **kwargs)


@pytest.mark.smoke
def test_the_graph_is_the_matrixs_own_arrays() -> None:
    """`CsrGraph.from_matrix` references the matrix's arrays without copying."""
    _, graph, _, _ = _problem()
    wrapped = CsrGraph.from_matrix(graph)

    assert wrapped.indptr is graph.indptr
    assert wrapped.indices is graph.indices
    assert wrapped.weights is graph.data
    assert wrapped.n_spots == N_SPOTS


@pytest.mark.smoke
def test_the_sweep_reports_convergence_with_its_sweep_count() -> None:
    """`icm_sweep_deque` has no cap, so every return is its criterion met (T- #617)."""
    field, graph, assignment, _ = _problem()
    np.random.seed(SEED)  # noqa: NPY002

    result = icm_sweep(
        field, CsrGraph.from_matrix(graph), assignment.copy(), BETA, min_clone_spots=0
    )

    assert result.termination.converged
    assert result.termination.iterations == result.niter
