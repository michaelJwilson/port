"""`pipeline_clone_assignment` against `cnaster.hmrf`'s, called, bitwise, at gate size (#281)."""

from typing import Any

import numpy as np
import pytest

from tests.adapters import clone_assignment_arguments
from tests.fixtures import spot_clone_field


def _both(arguments: dict[str, Any]) -> tuple[Any, Any]:
    """Return upstream's and the replacement's results, each on its own `prev_assignment`."""
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    def call(function: Any) -> Any:
        return function(
            arguments["single_X"],
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            hmmclass=hmm_nophasing,
        )

    return call(UPSTREAM), call(pipeline_clone_assignment)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
@pytest.mark.parametrize(("n_states", "n_clones"), [(5, 3), (3, 2)])
def test_the_replacement_assigns_what_upstream_assigns(
    n_states: int, n_clones: int
) -> None:
    """Assignment, field and likelihood equal upstream's, bitwise."""
    fixture = spot_clone_field(
        n_states=n_states, n_obs=60, n_spots=36, n_clones=n_clones
    )

    theirs, ours = _both(clone_assignment_arguments(fixture, width=6))

    their_assignment, their_field, their_likelihood = theirs
    our_assignment, our_field, our_likelihood = ours

    np.testing.assert_array_equal(our_field, their_field)
    np.testing.assert_array_equal(our_assignment, their_assignment)

    assert our_likelihood == their_likelihood, (
        f"likelihood {our_likelihood!r} against upstream's {their_likelihood!r}"
    )


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_tumour_mixed_call_goes_to_cnaster_unchanged() -> None:
    """With `single_tumor_prop` the call is delegated: results equal upstream's (#135)."""
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    fixture = spot_clone_field(n_states=3, n_obs=40, n_spots=16, n_clones=2)
    arguments = clone_assignment_arguments(fixture, width=4)
    proportion = np.full(16, 0.7)

    def call(function: Any) -> Any:
        return function(
            arguments["single_X"],
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            single_tumor_prop=proportion,
            hmmclass=hmm_nophasing,
        )

    their_assignment, their_field, their_likelihood = call(UPSTREAM)
    our_assignment, our_field, our_likelihood = call(pipeline_clone_assignment)

    np.testing.assert_array_equal(our_field, their_field)
    np.testing.assert_array_equal(our_assignment, their_assignment)

    assert our_likelihood == their_likelihood


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_merge_loop_merges_what_upstream_merges() -> None:
    """With `merge=True` all three returns equal upstream's, bitwise (#59)."""
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    fixture = spot_clone_field(n_states=4, n_obs=40, n_spots=16, n_clones=4)
    arguments = clone_assignment_arguments(fixture, width=4)

    def call(function: Any) -> Any:
        return function(
            arguments["single_X"],
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            hmmclass=hmm_nophasing,
            merge=True,
        )

    their_assignment, their_field, their_likelihood = call(UPSTREAM)
    our_assignment, our_field, our_likelihood = call(pipeline_clone_assignment)

    np.testing.assert_array_equal(our_field, their_field)
    np.testing.assert_array_equal(our_assignment, their_assignment)

    assert our_likelihood == their_likelihood


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_self_only_pooling_assigns_what_upstream_assigns() -> None:
    """Self-only `smooth_mat` gives upstream's three returns, bitwise (#488, #513)."""
    import cnaster.hmrf
    import scipy.sparse as sp
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import (
        UPSTREAM,
        _self_only,
        pipeline_clone_assignment,
    )

    fixture = spot_clone_field(n_states=4, n_obs=60, n_spots=36, n_clones=3)
    arguments = clone_assignment_arguments(fixture, width=6)
    identity = sp.identity(36, format="csr")

    assert _self_only(identity)
    assert _self_only(sp.identity(36, dtype=np.int8, format="csr"))
    assert not _self_only(sp.csr_matrix(np.ones((36, 36))))
    assert not _self_only(2 * identity)

    theirs_pooled = cnaster.hmrf.pool_spatio_genomic_counts(
        arguments["single_X"],
        arguments["single_base_nb_mean"],
        arguments["single_total_bb_RD"],
        identity.indices,
        identity.indptr,
        None,
        False,
    )

    for name, theirs in zip(
        ("single_X", "single_base_nb_mean", "single_total_bb_RD"),
        theirs_pooled[:3],
        strict=True,
    ):
        np.testing.assert_array_equal(arguments[name], theirs)

    def call(function: Any) -> Any:
        return function(
            arguments["single_X"],
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            smooth_mat=identity,
            hmmclass=hmm_nophasing,
        )

    theirs, ours = call(UPSTREAM), call(pipeline_clone_assignment)

    np.testing.assert_array_equal(ours[1], theirs[1])
    np.testing.assert_array_equal(ours[0], theirs[0])
    assert ours[2] == theirs[2]


@pytest.mark.infra
@pytest.mark.usefixtures("cnaster_config")
def test_a_smooth_matrix_that_pools_neighbours_is_refused() -> None:
    """A `smooth_mat` pooling a spot with a neighbour raises; port does not pool (#513)."""
    import scipy.sparse as sp
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import (
        PooledSmoothing,
        pipeline_clone_assignment,
    )

    fixture = spot_clone_field(n_states=4, n_obs=60, n_spots=36, n_clones=3)
    arguments = clone_assignment_arguments(fixture, width=6)
    pooling = sp.identity(36, format="csr") + sp.eye(36, k=1, format="csr")

    with pytest.raises(PooledSmoothing, match="#513"):
        pipeline_clone_assignment(
            arguments["single_X"],
            arguments["single_base_nb_mean"],
            arguments["single_total_bb_RD"],
            arguments["res"],
            arguments["pred"],
            arguments["adjacency_mat"],
            arguments["prev_assignment"].copy(),
            arguments["sample_ids"],
            arguments["spatial_weight"],
            smooth_mat=pooling,
            hmmclass=hmm_nophasing,
        )
