"""The reduced label-solver interface, pinned bitwise against `cnaster`.

Issue #59 item 5. `port.patch.icm_interface` presents
`cnaster.icm.icm_sweep_deque`'s fifteen parameters as eight, by folding the
terms the solver adds inside its own inner loop into the field it is given
and by passing one graph as one argument.

**The claim is equivalence, and nothing else.** `CLAUDE.md`: a patch that
makes the existing code plainer is worth landing on its evidence of
equivalence alone. So every test here is a bitwise comparison against the
fifteen-argument call on the same problem, and the dropped parameters each
carry a test showing what happens to them -- folded, or never read.

The sweep draws from the legacy global `numpy` RNG for its queue order, so
each comparison seeds it identically before both calls. That is a property
of `cnaster`'s solver, recorded in `test_the_sweep_is_not_reproducible`
rather than worked around silently.
"""

import numpy as np
import pytest
from port.patch.icm_interface import CsrGraph, fold_unary, icm_sweep
from scipy.sparse import csr_matrix

N_SPOTS = 400
N_CLONES = 4
N_SAMPLES = 3
BETA = 0.75
SEED = 6_803


def _problem(
    seed: int = SEED,
) -> tuple[np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
    """A field, a spatial graph, a labelling and the sample each spot is in.

    The field is drawn rather than scored: what the solver does with it
    depends on the differences between clones, not on their being
    log-densities, and issue #59's items 1 and 2 are where the field's
    provenance is the subject.
    """
    rng = np.random.default_rng(seed)

    field = rng.normal(-50.0, 5.0, (N_SPOTS, N_CLONES))

    rows, cols, data = [], [], []
    for spot in range(N_SPOTS):
        for neighbour in rng.choice(
            N_SPOTS, size=int(rng.integers(2, 8)), replace=False
        ):
            rows.append(spot)
            cols.append(int(neighbour))
            data.append(float(rng.uniform(0.5, 2.0)))

    graph = csr_matrix((data, (rows, cols)), shape=(N_SPOTS, N_SPOTS))
    assignment = rng.integers(0, N_CLONES, N_SPOTS)
    sample_ids = rng.integers(0, N_SAMPLES, N_SPOTS)

    return field, graph, assignment, sample_ids


def _cnaster_sweep(
    field: np.ndarray,
    graph: csr_matrix,
    assignment: np.ndarray,
    spatial_weight: float,
    *,
    seed: int,
    posterior: np.ndarray | None = None,
    **kwargs: object,
) -> tuple[np.ndarray, int, float]:
    """The fifteen-argument call, as `hmrf.py:307` makes it."""
    from cnaster.icm import icm_sweep_deque

    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
    np.random.seed(seed)  # noqa: NPY002
    labels = assignment.copy()

    niter, cost = icm_sweep_deque(
        single_llf=field,
        adj_indptr=graph.indptr,
        adj_indices=graph.indices,
        adj_weights=graph.data,
        new_assignment=labels,
        spatial_weight=spatial_weight,
        posterior=posterior,
        min_clone_spots=0,
        **kwargs,
    )

    return labels, int(niter), float(cost)


def _patched_sweep(
    field: np.ndarray,
    graph: csr_matrix,
    assignment: np.ndarray,
    beta: float,
    *,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    """The eight-parameter call, on an already-folded field."""
    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
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


@pytest.mark.cnaster
@pytest.mark.parametrize("seed", [11, 23])
def test_the_reduced_call_is_bitwise_cnasters(seed: int) -> None:
    """Same labelling, same iteration count, same cost, on the plain problem.

    Bitwise because the patch reorders no arithmetic: it passes `cnaster`'s
    own solver the same floats through fewer arguments. A tolerance here
    would be admitting that something moved.
    """
    field, graph, assignment, _ = _problem()

    expected = _cnaster_sweep(field, graph, assignment, BETA, seed=seed)
    actual = _patched_sweep(field, graph, assignment, BETA, seed=seed)

    _assert_same(expected, actual)
    assert not np.array_equal(actual[0], assignment), "the sweep must do something"


@pytest.mark.cnaster
def test_the_per_sample_weights_fold_bitwise() -> None:
    """`log_persample_weights[c, sample_ids[i]]`, added once instead of per visit.

    `cnaster` adds it to `single_llf[i, c]` before the edge term, left to
    right; `fold_unary` performs that same first addition once. The floats
    and their order are identical, so the labelling is too -- and if the fold
    had transposed the indexing, the weights would still be the right
    numbers in the wrong places and this would fail.
    """
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


@pytest.mark.cnaster
def test_the_allowed_clone_mask_folds_bitwise() -> None:
    """`onehot_allowed_clones`, folded as `-inf` rather than branched on.

    `cnaster` overwrites the cost with `-inf` **after** adding the edge term;
    the fold writes `-inf` before it. They agree because `-inf` plus a finite
    number is `-inf`, which is asserted separately in
    `test_the_mask_is_exact_under_the_edge_term`.
    """
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


@pytest.mark.cnaster
@pytest.mark.parametrize("temp", [0.5, 2.0])
def test_the_temperature_folds_into_the_coupling(temp: float) -> None:
    """`spatial_weight / temp` is the only use `temp` has.

    So the reduced interface takes the quotient. Bitwise because `cnaster`
    forms exactly that quotient once, before the sweep, and the patch forms
    it in the caller instead.
    """
    field, graph, assignment, _ = _problem()

    expected = _cnaster_sweep(field, graph, assignment, BETA, seed=SEED, temp=temp)
    actual = _patched_sweep(field, graph, assignment, BETA / temp, seed=SEED)

    _assert_same(expected, actual)


@pytest.mark.cnaster
def test_the_posterior_argument_is_never_read_or_written() -> None:
    """The dropped parameter, shown dead rather than asserted to be.

    `posterior` is a **required positional** whose only uses in the live body
    are commented out, and `hmrf.py:314` passes `None` to it. Dropping it
    from the interface is therefore free -- but only if that is true, so it
    is driven: a finite array is passed in, and it comes back untouched while
    the sweep produces the same labelling as passing `None`.
    """
    field, graph, assignment, _ = _problem()

    posterior = np.full((N_SPOTS, N_CLONES), 0.25)
    witness = posterior.copy()

    with_array = _cnaster_sweep(
        field, graph, assignment, BETA, seed=SEED, posterior=posterior
    )
    with_none = _cnaster_sweep(field, graph, assignment, BETA, seed=SEED)

    np.testing.assert_array_equal(witness, posterior)
    _assert_same(with_none, with_array)


@pytest.mark.cnaster
def test_the_live_sweep_is_the_csr_one() -> None:
    """Which of the four `icm_sweep_deque` definitions `cnaster.icm` exports.

    `icm.py` binds that name at lines 363, 513, 658 and 807. Only the last
    survives the module body, and it takes CSR where the first takes the COO
    triple `hmrf.py:284-285` still builds. The patch wraps the live one, so a
    reordering of `icm.py` that changed the winner would change the solver
    silently -- this is what would have to be wrong for this to fail.
    """
    import inspect

    from cnaster import icm

    source = inspect.getsource(icm)
    assert source.count("\ndef icm_sweep_deque(") == 4, "the shadowing changed"

    parameters = list(inspect.signature(icm.icm_sweep_deque).parameters)

    assert parameters[1:4] == ["adj_indptr", "adj_indices", "adj_weights"]
    assert "adj_spots" not in parameters, "the COO definition is live again"
    assert len(parameters) == 15


@pytest.mark.cnaster
def test_the_sweep_is_not_reproducible_without_seeding_a_global() -> None:
    """Why every comparison above seeds `np.random`.

    `cnaster`'s sweep shuffles its queue with the legacy global RNG, which is
    neither seeded nor threaded through a `Generator`. Two runs of one
    problem give two answers. Recorded, not fixed: seeding it inside the
    patch would be a behaviour change hiding inside an interface change.
    """
    field, graph, assignment, _ = _problem()

    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
    np.random.seed(1)  # noqa: NPY002
    first = _patched_sweep(field, graph, assignment, BETA, seed=1)

    # NPY002 is the finding, not the violation: `cnaster`'s sweep shuffles
    # its queue with the legacy global RNG, so a `Generator` cannot reach it.
    np.random.seed(2)  # noqa: NPY002
    second = _patched_sweep(field, graph, assignment, BETA, seed=2)

    assert not np.array_equal(first[0], second[0])


@pytest.mark.analytic
def test_the_mask_is_exact_under_the_edge_term() -> None:
    """`-inf + finite == -inf`, which is what makes the fold exact.

    The fold moves the mask from after the edge term to before it. That is
    only equivalent while the edge term is finite -- it is a sum of edge
    weights -- so the property is pinned rather than assumed.
    """
    edge = np.array([0.0, 1e300, -1e300, np.finfo(np.float64).max])

    assert np.all(-np.inf + edge == -np.inf)


@pytest.mark.analytic
def test_the_fold_leaves_the_field_it_was_given() -> None:
    """A copy, not a write-through.

    The unfolded field is what the next outer iteration recomputes against,
    and `cnaster` reads `single_llf` after the sweep for `merge_assignment`.
    """
    field, _, _, sample_ids = _problem()
    witness = field.copy()

    rng = np.random.default_rng(SEED)
    fold_unary(field, rng.normal(0.0, 2.0, (N_CLONES, N_SAMPLES)), sample_ids)

    np.testing.assert_array_equal(witness, field)


@pytest.mark.analytic
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
    """Broadcasting here would silently score the wrong clone.

    A `(n_clones + 1, n_samples)` weight and a wider mask both have shapes
    `numpy` is willing to do something with, and what it does is not what the
    solver was asked for.
    """
    field, _, _, _ = _problem()

    with pytest.raises(ValueError, match=match):
        fold_unary(field, **kwargs)


@pytest.mark.analytic
def test_the_graph_is_the_matrixs_own_arrays() -> None:
    """`CsrGraph.from_matrix` names the call site's expression, it does not copy.

    `hmrf.py:309-311` passes these three attributes. A copy here would add an
    allocation per outer iteration to an interface change that is supposed to
    cost nothing.
    """
    _, graph, _, _ = _problem()
    wrapped = CsrGraph.from_matrix(graph)

    assert wrapped.indptr is graph.indptr
    assert wrapped.indices is graph.indices
    assert wrapped.weights is graph.data
    assert wrapped.n_spots == N_SPOTS
