"""`pipeline_clone_assignment` against `cnaster`'s, at a gate size (#281).

**The largest drop-in in the repository was compared only on the release
tier.** `tests/test_patched_entry_point.py` puts two whole `run_cnaster`
runs side by side, which is the stronger claim and the slower one; it
carries `release`, so the per-pull-request tier never ran the function's
body at all. 56 of its 104 statements were unreached, which is how #281's
guard found it.

This is the same claim at a size the gate can afford: one call, one
fixture, the three returned values against upstream's. It does not replace
the whole-run comparison -- that one says twelve replacements compose --
it says this one replacement is right before the composition is asked
about.

**Referee: `cnaster.hmrf.pipeline_clone_assignment`, called.** Not
transcribed: the field is a sum over bins in a fixed order on the same
doubles, so `np.array_equal` is the bar and a tolerance would hide a
reordering.
"""

from typing import Any

import numpy as np
import pytest

from tests.fixtures import SpotCloneField, spot_clone_field


def _lattice_adjacency(n_spots: int, width: int) -> Any:
    """A four-neighbour grid, as `construct_multislice_lattice_adjacency` builds.

    Built here rather than drawn, because the claim is about the solver
    reading one graph and the graph should be one a reader can check by
    inspection: spot `i` neighbours `i - 1`, `i + 1`, `i - width` and
    `i + width` where those exist.
    """
    from scipy.sparse import coo_matrix

    rows: list[int] = []
    columns: list[int] = []

    for spot in range(n_spots):
        row, column = divmod(spot, width)

        for neighbour_row, neighbour_column in (
            (row, column + 1),
            (row + 1, column),
        ):
            neighbour = neighbour_row * width + neighbour_column

            if neighbour_column < width and neighbour < n_spots:
                rows.extend((spot, neighbour))
                columns.extend((neighbour, spot))

    data = np.ones(len(rows))

    return coo_matrix((data, (rows, columns)), shape=(n_spots, n_spots)).tocsr()


def _arguments(fixture: SpotCloneField, width: int) -> dict[str, Any]:
    """Everything both arms are handed, built once so neither can differ.

    `pred` is the concatenated path the fit returns and `res` the four
    parameters beside it, at `(n_states, 1)` -- the shape `cnaster` reads
    and the only one a fit produces (#278).
    """
    n_obs, n_spots = fixture.counts_nb.shape

    single_X = np.zeros((n_obs, 2, n_spots))
    single_X[:, 0, :] = fixture.counts_nb
    single_X[:, 1, :] = fixture.counts_bb

    generator = np.random.default_rng(fixture.seed)

    return {
        "single_X": single_X,
        "single_base_nb_mean": fixture.base_nb_mean,
        "single_total_bb_RD": fixture.total_bb_RD,
        "res": {
            "new_log_mu": fixture.log_mu.reshape(-1, 1),
            "new_alphas": fixture.alphas.reshape(-1, 1),
            "new_p_binom": fixture.p_binom.reshape(-1, 1),
            "new_taus": fixture.taus.reshape(-1, 1),
        },
        "pred": fixture.pred.T.reshape(-1),
        "adjacency_mat": _lattice_adjacency(n_spots, width),
        "prev_assignment": generator.integers(0, fixture.n_clones, size=n_spots).astype(
            np.int64
        ),
        "sample_ids": np.zeros(n_spots, dtype=np.int64),
        "spatial_weight": 1.5,
    }


def _both(arguments: dict[str, Any]) -> tuple[Any, Any]:
    """Upstream's and the replacement's, on inputs neither may mutate.

    Each arm gets its own copy of `prev_assignment`: the function writes
    into it, so sharing one would let whichever ran first decide what the
    second was asked.
    """
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
    """The assignment, the field and the likelihood, all three.

    All three because each can be right while another is wrong: the field
    is what the fused kernel computes, the assignment is what the solver
    makes of it, and the likelihood reads both back through the graph. A
    test on the assignment alone passes a field that is wrong by a constant.
    """
    fixture = spot_clone_field(
        n_states=n_states, n_obs=60, n_spots=36, n_clones=n_clones
    )

    theirs, ours = _both(_arguments(fixture, width=6))

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
    """The one branch the fused field does not cover, delegated rather than guessed.

    `single_tumor_prop is not None` is a different quantity -- #135 finds the
    E step accepts the proportion and never reads it -- so the replacement
    hands the call over. Identity of the result is the claim: a delegation
    that recomputed anything would be a second implementation of the branch
    it is avoiding.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    fixture = spot_clone_field(n_states=3, n_obs=40, n_spots=16, n_clones=2)
    arguments = _arguments(fixture, width=4)
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
    """`merge=True`, which is a quarter of the function and a separate claim.

    The merge loop calls `cnaster.hmrf.merge_assignment` on a COO triple the
    replacement builds with three array expressions where upstream runs two
    pure-Python passes over every non-zero (#59 item 3). Same consumer, same
    decision, so the assignment it converges on is the referee -- and the
    loop is where a wrong triple shows up, because on `merge=False` the
    round trip is computed and discarded.

    Four clones on sixteen spots, so there is a merge worth making: with two
    the loop can only take one step before the graph is one clone.
    """
    from cnaster.hmm_nophasing import hmm_nophasing
    from port.patch.hmrf.clone_assignment import UPSTREAM, pipeline_clone_assignment

    fixture = spot_clone_field(n_states=4, n_obs=40, n_spots=16, n_clones=4)
    arguments = _arguments(fixture, width=4)

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
