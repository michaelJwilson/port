"""The stages `run_cnaster` calls, one at a time.

`tests/test_run_cnaster_round_trip.py` establishes that the pipeline reaches
the end; it asserts nothing about what any stage computed. These do, on the
same fixture, so a failure names a stage rather than the run.

**The bins are renumbered downstream of `normal_baf_bin_filter`** (#105), so a
stage after it cannot be refereed against the planted bin index. Every test
here sits upstream of that filter or is indifferent to it, and the ones that
cannot be are #105's to unblock.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures import CoreInferenceTruth, core_inference_truth
from tests.run_config import write_run_cnaster_config
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

pytestmark = pytest.mark.preprocessing

LATTICE = (25, 40)
"""Rows and columns. A thousand spots, which is `icm_sweep_deque`'s floor times five."""

FLIP_EVERY = 3
"""Every third block is stored on the other haplotype, for the phasing test."""

BALANCED_STATE = 0
"""The planted diploid balanced state, which casts no phase vote (#106)."""

TRIVIAL_AGREEMENT = 1.0 - 1.0 / FLIP_EVERY
"""What an unphased answer scores, which is why a threshold is not a test.

Phase is defined up to a global complement, so a comparison against the
planted haplotype takes the better of the two directions. Against
`flip_every=3` an identically-zero indicator therefore scores
`max(2/3, 1/3)` = **0.667** having flipped nothing, and any pin below that
measures the fixture's flip rate rather than the phasing.

The earlier pins here -- 0.65, realized 0.661 and 0.667 -- were exactly that
(#122). The figures #108 recorded, 0.857 and 0.925, are above it and were
measuring something; they were measured on the Markov-chain genome #120
retired.
"""

STRONG_MARGIN = 0.1
"""How far from balance a planted state has to sit to count as strong."""

SHIPPED_T_PHASEING = 0.99999
"""`zenodo_sim_config.yaml`'s `hmm.t_phaseing`, which `run_cnaster` passes.

Restated so the sweep below is against what ships rather than against a
number this file chose. `hmm.t` is stickier still, 0.9999999.
"""

RESOLVING_SELF_TRANSITIONS = (0.7, 0.6, 0.5)
"""Swept `t` at which the decode resolves three states on every platform (#142).

**Where the threshold sits is not reproducible, so it is not asserted.** This
machine resolves three states from 0.85 down; a GitHub runner collapses at
0.85 and reaches only two at 0.8. Both agree from 0.7 down, and both collapse
at every value at or above 0.9. Those two bands are what the tests claim, and
the boundary between them is reported rather than pinned.

The fit is a non-convex optimization over a multimodal likelihood and `t`
moves the basin, so which optimum it reaches turns on floating-point
arithmetic -- the same platform sensitivity #147 found in `opt/fit`, here
changing a decoded state count rather than a convergence flag.

The implied segment length is `1 / (1 - t)` bins: 10 at the collapsing end,
3.3 at the resolving one, against the 10-to-25-bin events this fixture plants.
The prior has to be weaker than the truth before the truth is recoverable,
which is the finding rather than any particular number.
"""

PHASING_SELF_TRANSITION = 1.0 - 1e-6
"""What `phased` passes, between the two shipped values and representative."""

PHASING_EPS_BAF = 0.1
"""`phasing.py:67`'s deadband, inside which a block casts no phase vote.

A local in `initial_phase_given_partition`, so it is restated rather than
imported; `test_no_block_casts_a_vote_because_every_one_decodes_balanced`
fails if the value drifts, since the count it pins depends on it.
"""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One instance for the module: the stages below are pure functions of it."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=40, n_segments=3, seed=11
    )


@pytest.fixture(scope="module")
def loaded(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Any]:
    """`load_input_data`'s return, from files written once for the module.

    The configuration stays installed for the body of every test: these stages
    read the global rather than taking it as an argument, and a fixture that
    restored it on the way out would leave them reading `None`.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    root: Path = tmp_path_factory.mktemp("stages")
    written = write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)
    config_path = write_run_cnaster_config(written, planted)

    # The pipeline's own configuration rather than the loader's subset: these
    # stages read sections the loader never touches -- `preprocessing`,
    # `hmrf`, `int_copy_num` -- so a partial global makes them raise
    # `AttributeError` rather than run.
    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield load_input_data(get_global_config())
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.end2end
@pytest.mark.planted
def test_the_sample_list_is_the_one_slice_the_fixture_wrote(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """One slice in, one slice out, and every spot assigned to it.

    `get_sample_list` derives the slices from `adata.obs["sample"]` by
    removing adjacent duplicates, so a fixture written in spot order returns
    one name however many spots carry it.
    """
    from cnaster.io import get_sample_list

    sample_list, sample_ids = get_sample_list(loaded.adata)

    assert sample_list == ["S1"]
    assert sample_ids.shape == (planted.n_spots,)
    assert set(np.unique(sample_ids)) == {0}


@pytest.mark.infra
@pytest.mark.analytic
def test_no_tumour_proportion_file_gives_no_proportion(loaded: Any) -> None:
    """`preprocessing.tumorprop_file: None` returns `None`, not zeros.

    The distinction is load-bearing downstream: `run_core_inference` branches
    on `single_tumor_prop is None` and an array of zeros would take the mixed
    path with every spot called normal.
    """
    from cnaster.io import read_tumor_prop

    assert read_tumor_prop(loaded.adata) is None


@pytest.mark.end2end
@pytest.mark.planted
@pytest.mark.critical
def test_the_rectangular_partition_recovers_the_planted_bands(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """A partition along the axis the clones are banded on returns them.

    The fixture lays clones in horizontal bands -- `labels` is a function of
    the row alone -- so a one-by-`M` partition of the lattice is the planted
    labelling, and the strongest claim available about the initializer is that
    it reproduces it exactly rather than approximately.

    This is what `run_cnaster` starts its outer loop from, and until now
    nothing in this repository ran it: the solver tests supply their own
    `initial_clone_index` and step over `spatial.py` entirely (#95).
    """
    from cnaster.spatial import initialize_clones

    coordinates = np.asarray(loaded.coords, dtype=float)
    # NB the bands run along `x`: `tissue_positions.csv` writes the lattice
    #    row as `x`, and `labels` is a function of the row alone. So the
    #    partition is `n_clones` by one, and the transpose recovers nothing --
    #    which is the check, since a partition on the wrong axis still
    #    partitions.
    index = initialize_clones(
        coordinates,
        np.zeros(planted.n_spots, dtype=int),
        planted.n_clones,
        1,
        random_state=None,
    )

    recovered = np.empty(planted.n_spots, dtype=np.int64)
    for clone, spots in enumerate(index):
        recovered[spots] = clone

    assert sum(len(spots) for spots in index) == planted.n_spots
    assert len(index) == planted.n_clones
    # NB the partition names its clones by position, and the fixture names
    #    its own by band, so the two agree up to which end counts as first.
    agreement = max(
        float(np.mean(recovered == planted.labels)),
        float(np.mean(recovered == planted.n_clones - 1 - planted.labels)),
    )
    assert agreement == 1.0


@pytest.mark.infra
@pytest.mark.analytic
def test_the_partition_covers_every_spot_exactly_once(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """Whatever the geometry, a partition loses no spot and duplicates none.

    It is **not** balanced, and that is the function working rather than
    failing: `rectangle_partition` cuts the bounding box into equal
    rectangles, so a 25 by 40 lattice split two ways on each axis gives
    [260, 260, 240, 240] -- thirteen rows against twelve. `best_equal_partition`
    is the one that balances, and it chooses among candidates this produces.
    """
    from cnaster.spatial import initialize_clones

    coordinates = np.asarray(loaded.coords, dtype=float)
    index = initialize_clones(
        coordinates, np.zeros(planted.n_spots, dtype=int), 2, 2, random_state=None
    )

    covered = np.concatenate(index)
    np.testing.assert_array_equal(np.sort(covered), np.arange(planted.n_spots))
    assert [len(spots) for spots in index] == [260, 260, 240, 240]


@pytest.mark.end2end
@pytest.mark.planted
def test_the_clone_label_table_carries_every_spot_once(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """`construct_df_clone_label` is the run's output table, one row per spot."""
    from cnaster.io import construct_df_clone_label

    table = construct_df_clone_label(
        np.asarray(loaded.barcodes),
        np.asarray(loaded.coords, dtype=float),
        planted.labels,
    )

    assert len(table) == planted.n_spots
    assert set(table.columns) == {"sample_id", "x", "y", "clone_label"}
    # NB the barcode is the index rather than a column, which is what the
    #    written `clone_labels.tsv` carries in its first field.
    assert table.index.nunique() == planted.n_spots
    np.testing.assert_array_equal(
        np.sort(table.clone_label.to_numpy()), np.sort(planted.labels)
    )


@pytest.fixture(scope="module")
def flipped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """An instance whose files store every third block on the other haplotype.

    One block per bin (`blocks_per_bin=(1, 2)` draws from `[1, 2)`), so a
    flipped block is a flipped bin and the planted phase is a vector over the
    segmentation `cnaster` derives. With several blocks to a bin the planted
    phase would be ambiguous wherever they disagreed, which is a fixture
    question rather than a phasing one.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

    # NB `self_transition` is loosened from the default 0.99: over sixty bins
    #    a 0.99 chain barely leaves the state it starts in, and starting in
    #    the balanced one leaves no block that carries a phase at all.
    # NB a phase-rich genome, deliberately. Phase is defined only where the
    #    allele share is imbalanced, and #120 made the default genome mostly
    #    neutral -- 78 per cent of this instance's bins, which leaves two
    #    blocks carrying a phase out of sixty. That is the right default and
    #    the wrong instance for this test, so the events here cover the
    #    genome instead of decorating it.
    truth = core_inference_truth(
        n_clones=2,
        n_states=3,
        lattice=LATTICE,
        n_obs=60,
        n_segments=2,
        events=(8, 12),
        event_bins=(10, 25),
        seed=5,
    )
    pre_image = unsegment(
        truth, blocks_per_bin=(1, 2), unassigned_genes=0, flip_every=FLIP_EVERY
    )

    root: Path = tmp_path_factory.mktemp("phasing")
    written = write_tmp_inputs(truth, pre_image, root)
    config_path = write_run_cnaster_config(written, truth)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield truth, pre_image, load_input_data(get_global_config()), written
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.fixture(scope="module")
def phased(flipped: Any) -> Any:
    """`initial_phase_given_partition` on the flipped instance, run once.

    The fit inside it is the expensive part, and the tests below read the same
    output from three sides: the phase it returned, the decode that produced
    it, and the segmentation it refined.
    """
    from cnaster.hmm_nophasing import get_log_transmat
    from cnaster.omics import (
        assign_initial_blocks,
        form_gene_snp_table,
        summarize_counts_for_blocks,
    )
    from cnaster.phasing import initial_phase_given_partition

    truth, pre_image, loaded, written = flipped
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)

    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )
    table = assign_initial_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids, initial_min_umi=1
    )
    blocks = summarize_counts_for_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids
    )

    res, recovered, refined = initial_phase_given_partition(
        blocks.X,
        blocks.lengths,
        # NB BAF only: a zero exposure is what `run_cnaster` passes here too,
        #    and it is `(n_obs, n_spots)` rather than a vector.
        np.zeros_like(blocks.total_bb_RD),
        blocks.total_bb_RD,
        None,
        # NB the planted partition, as `run_cnaster` passes `initialize_clones`'
        #    output. One clone over every spot pools clones whose states
        #    differ, and the imbalance the vote reads washes out in the mix.
        truth.clone_index,
        truth.n_states,
        get_log_transmat(truth.n_states, 1.0 - 1e-6),
        np.zeros(blocks.X.shape[0]),
        "sp",
        1.0 - 1e-6,
        0,
        fix_NB_dispersion=False,
        shared_NB_dispersion=True,
        fix_BB_dispersion=False,
        shared_BB_dispersion=True,
        max_iter=100,
        tol=1e-3,
        threshold=0.5,
    )

    return truth, pre_image, blocks, res, recovered, refined


def _recovered_on(
    recovered: np.ndarray, planted: np.ndarray, mask: np.ndarray
) -> float:
    """Agreement over `mask`, taking phase up to a global complement.

    Which is what makes an absolute threshold meaningless below
    `TRIVIAL_AGREEMENT`: the better of the two directions is reported, so an
    answer that flipped nothing still scores the fixture's unflipped fraction.
    """
    return max(
        float(np.mean(recovered[mask] == planted[mask])),
        float(np.mean(recovered[mask] != planted[mask])),
    )


@pytest.mark.warning
@pytest.mark.subject
def test_no_block_casts_a_vote_because_every_one_decodes_balanced(
    phased: Any,
) -> None:
    """**`phase_indicator` comes back zero everywhere: nothing is phased (#122).**

    Not an absence of signal. 114 of this instance's 120 clone-blocks are
    planted in a non-balanced state, at `p` of 0.58 or 0.88. What happens is
    that `phasing.py:121` fits the clone-stacked phased HMM, `:141` decodes it,
    and **every block in both clones decodes to state 0**, whose fitted BAF is
    0.50004. `:156` then reads a block within `EPS_BAF` of balance as
    normal-like and `:162` sets its vote to `-1`, so nothing votes and `:174`
    leaves the zero default in place.

    The fit also drives the other two states to 0.99999 and 0.108, both at the
    boundary of the parameter -- the same collapse seen from the parameters
    rather than from the decode.

    Pinned as `cnaster`'s behaviour, not as the method's: the referee is the
    module's own arithmetic, and what the method claims is the next test.
    """
    truth, _, _, res, recovered, _ = phased

    fitted_p = np.asarray(res["new_p_binom"]).ravel()
    n_clones = len(truth.clone_index)
    decoded = np.argmax(res["log_gamma"], axis=0).reshape(n_clones, recovered.size)
    base_states = decoded % truth.n_states
    phase_mask = decoded < truth.n_states
    model_baf = np.where(phase_mask, fitted_p[base_states], 1.0 - fitted_p[base_states])

    np.testing.assert_allclose(fitted_p[0], 0.5, atol=5e-4)
    assert (base_states == BALANCED_STATE).all(), "some block decoded off balance"

    silent = np.abs(model_baf - 0.5) < PHASING_EPS_BAF
    assert silent.all(), f"{int(silent.size - silent.sum())} clone-blocks still vote"

    per_block = truth.states[:, : recovered.size]
    imbalanced = int((per_block != BALANCED_STATE).sum())
    assert imbalanced > 0.9 * per_block.size, f"{imbalanced} of {per_block.size}"

    np.testing.assert_array_equal(recovered, np.zeros_like(recovered))


@pytest.mark.end2end
@pytest.mark.planted
@pytest.mark.xfail(strict=True, reason="the vote returns no phase at all (#122)")
def test_the_phasing_recovers_the_planted_haplotype(phased: Any) -> None:
    """Every block stored on the other haplotype is the one phasing flips back.

    The claim `phasing.py` exists to support, and the one the round trip
    cannot make: it drives the stage but compares nothing. Written against a
    fixture that plants the answer -- `unsegment(flip_every=3)` stores every
    third block's B count as `total - B`, and nothing in the files says which
    ones.

    **The threshold is `TRIVIAL_AGREEMENT`, rather than a number beside it.**
    An absolute pin is what let #122 sit here unnoticed: 0.661 cleared a
    threshold of 0.65 while the indicator was constant. Recovery has to beat
    what flipping nothing already scores, or it is not recovery.

    Two regimes are still excluded by `cnaster`'s own rule: a block whose state
    is balanced casts no vote (#106), and phase is defined up to a global
    complement.

    `strict`, so that a fix upstream turns this red rather than passing
    quietly.
    """
    truth, pre_image, _, _, recovered, _ = phased

    planted = ~pre_image.phase_indicator
    assert recovered.shape == planted.shape

    # The blocks the rule can speak for: those whose planted state is not the
    # balanced one, since a balanced block casts no vote.
    per_block = truth.states[:, : recovered.size]
    speakable = np.all(per_block != BALANCED_STATE, axis=0)
    assert speakable.sum() > 0.25 * recovered.size, "too few blocks carry a phase"

    agreement = _recovered_on(recovered, planted, speakable)
    assert agreement > TRIVIAL_AGREEMENT, (
        f"phase agreement {agreement:.3f} over {speakable.sum()} blocks, against "
        f"{TRIVIAL_AGREEMENT:.3f} for flipping nothing"
    )

    # And the subset a fit cannot mistake for balance.
    strong = np.any(np.abs(truth.p_binom - 0.5)[per_block] >= STRONG_MARGIN, axis=0)
    strong_agreement = _recovered_on(recovered, planted, strong)
    assert strong_agreement > TRIVIAL_AGREEMENT, (
        f"strong-block agreement {strong_agreement:.3f} over {strong.sum()}"
    )


@pytest.fixture(scope="module")
def phase_inputs(flipped: Any) -> Any:
    """The clone-stacked arrays and the initializer's output, as `phasing.py`
    builds them (`:75`, `:100`, `:105`), so the tests below can take the fit
    apart without re-deriving the pre-image.
    """
    from cnaster.hmm_initialize import gmm_init
    from cnaster.hmm_nophasing import get_log_transmat
    from cnaster.hmrf_utils import clone_stack_obs
    from cnaster.omics import (
        assign_initial_blocks,
        form_gene_snp_table,
        summarize_counts_for_blocks,
    )
    from cnaster.pseudobulk import merge_pseudobulk_by_index_mix

    truth, _, loaded, written = flipped
    alleles = (loaded.cell_snp_Aallele, loaded.cell_snp_Ballele)
    table = form_gene_snp_table(
        loaded.unique_snp_ids, str(written.hgtable), loaded.adata
    )
    table = assign_initial_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids, initial_min_umi=1
    )
    blocks = summarize_counts_for_blocks(
        table, loaded.adata, *alleles, loaded.unique_snp_ids
    )

    zero_exposure = np.zeros_like(blocks.total_bb_RD)
    sitewise = np.zeros(blocks.X.shape[0])
    pooled = merge_pseudobulk_by_index_mix(
        blocks.X,
        zero_exposure,
        blocks.total_bb_RD,
        truth.clone_index,
        None,
        threshold=0.5,
    )
    stacked = clone_stack_obs(*pooled[:3], blocks.lengths, sitewise, pooled[3])

    init_log_mu, init_p_binom, _, _ = gmm_init(
        truth.n_states,
        *stacked[:3],
        "sp",
        stacked[3],
        get_log_transmat(truth.n_states, PHASING_SELF_TRANSITION),
        stacked[4],
        random_state=0,
        in_log_space=False,
        only_minor=True,
    )
    n_clones = pooled[0].shape[2]

    return truth, stacked, init_log_mu, init_p_binom, n_clones


def _decode_occupancy(result: Any, n_states: int, n_clones: int) -> np.ndarray:
    """How many clone-blocks decode to each base state, phase folded away."""
    decoded = np.argmax(result["log_gamma"], axis=0).reshape(n_clones, -1)
    return np.bincount((decoded % n_states).ravel(), minlength=n_states)


def _fit(phase_inputs: Any, *, t: float, max_iter: int, planted: bool) -> Any:
    """`phasing.py:121`'s call, with the starting point and `t` as knobs."""
    from cnaster.hmm_phased import hmm_phased

    truth, stacked, init_log_mu, init_p_binom, _ = phase_inputs
    if planted:
        init_log_mu = np.zeros((truth.n_states, 1))
        init_p_binom = np.sort(np.minimum(truth.p_binom, 1.0 - truth.p_binom)).reshape(
            -1, 1
        )

    return hmm_phased(params="sp", t=t).optimize(
        stacked[0],
        stacked[3],
        truth.n_states,
        stacked[1],
        total_bb_RD=stacked[2],
        log_sitewise_transmat=stacked[4],
        fix_NB_dispersion=False,
        shared_NB_dispersion=True,
        fix_BB_dispersion=False,
        shared_BB_dispersion=True,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        max_iter=max_iter,
        tol=1e-12,
    )


@pytest.mark.warning
@pytest.mark.subject
def test_the_phasing_refuses_a_non_zero_exposure(flipped: Any) -> None:
    """**The BAF-only call is enforced, not chosen (#122).**

    `phasing.py:64` opens with `assert np.all(single_base_nb_mean == 0)`, so
    the zero exposure `run_cnaster` passes is the only exposure the function
    accepts. That eliminates the first of #122's three candidates outright:
    the collapse cannot be attributed to the call site, because no other call
    is reachable.
    """
    from cnaster.phasing import initial_phase_given_partition

    truth, _, _, _ = flipped

    with pytest.raises(AssertionError):
        initial_phase_given_partition(
            np.zeros((4, 2, 1)),
            np.array([4]),
            np.ones((4, 1)),
            np.ones((4, 1)),
            None,
            truth.clone_index,
            truth.n_states,
            np.zeros((truth.n_states, truth.n_states)),
            np.zeros(4),
            "sp",
            PHASING_SELF_TRANSITION,
            0,
            fix_NB_dispersion=False,
            shared_NB_dispersion=True,
            fix_BB_dispersion=False,
            shared_BB_dispersion=True,
            max_iter=1,
            tol=1e-3,
            threshold=0.5,
        )


@pytest.mark.end2end
@pytest.mark.planted
def test_the_initializer_recovers_the_planted_minor_bafs(phase_inputs: Any) -> None:
    """**And the second candidate is eliminated: `gmm_init` is right (#122).**

    Asked for a minor-BAF initialization (`only_minor=True`), it returns
    `[0.1197, 0.4191, 0.5000]` against the planted minor BAFs of
    `[0.12, 0.42, 0.50]` -- every state to within 0.001. Whatever destroys the
    decode, it is not the starting point handed to the fit.
    """
    truth, _, _, init_p_binom, _ = phase_inputs

    planted_minor = np.sort(np.minimum(truth.p_binom, 1.0 - truth.p_binom))
    initialized = np.sort(np.asarray(init_p_binom).ravel())

    np.testing.assert_allclose(initialized, planted_minor, atol=2e-3)


@pytest.mark.bug
@pytest.mark.subject
def test_the_fit_collapses_the_decode_between_its_first_two_iterations(
    phase_inputs: Any,
) -> None:
    """**So it is the fit, and it happens at iteration two (#122).**

    Started from the planted minor BAFs, one Baum-Welch iteration still holds
    them -- `[0.1198, 0.4194, 0.6961]` -- and the decode uses **all three**
    states, 28 / 40 / 52 clone-blocks. Two iterations put every one of the 120
    on a single state, and it never comes back: at 20 iterations the fit has
    settled on `[0.1048, 0.5000, 1.0000]`, one state at the pooled mean and one
    parked at the boundary with nothing assigned to it.

    The truth is representable and decodable, then, and the fit walks away
    from it. This is the attribution #122 asks for, and it holds from the
    planted starting point, so it is not a basin the initializer chose.
    """
    truth, _, _, _, n_clones = phase_inputs

    first = _decode_occupancy(
        _fit(phase_inputs, t=SHIPPED_T_PHASEING, max_iter=1, planted=True),
        truth.n_states,
        n_clones,
    )
    second = _decode_occupancy(
        _fit(phase_inputs, t=SHIPPED_T_PHASEING, max_iter=2, planted=True),
        truth.n_states,
        n_clones,
    )

    assert int((first > 0).sum()) == truth.n_states, f"first iteration {first}"
    assert int((second > 0).sum()) == 1, f"second iteration {second}"


@pytest.mark.warning
@pytest.mark.subject
@pytest.mark.parametrize("t", [SHIPPED_T_PHASEING, PHASING_SELF_TRANSITION, 0.99, 0.9])
def test_the_collapse_holds_across_every_self_transition_that_ships(
    phase_inputs: Any, t: float
) -> None:
    """What drives it: the fixed near-unity self-transition (#122).

    `params="sp"` fits the start probabilities and the BAF states and leaves
    the transition fixed, so `t` is imposed rather than learned. Across the
    whole shipped range -- `zenodo_sim_config.yaml` sets
    `t_phaseing = 0.99999` and `t = 0.9999999` -- the decode is one state.
    `test_..._only_a_transition_no_configuration_ships_decodes_the_truth`
    is the other end of the same sweep.
    """
    truth, _, _, _, n_clones = phase_inputs

    occupancy = _decode_occupancy(
        _fit(phase_inputs, t=t, max_iter=100, planted=True), truth.n_states, n_clones
    )

    assert int((occupancy > 0).sum()) == 1, f"t={t} decoded {occupancy}"


@pytest.mark.cnaster
@pytest.mark.subject
def test_only_a_transition_no_configuration_ships_decodes_the_truth(
    phase_inputs: Any,
) -> None:
    """At `t = 0.5` the decode recovers all three states: 27 / 66 / 27.

    The contrast that makes the sweep above a finding rather than a list of
    failures -- the fit is not incapable of the instance, it is prevented from
    reaching it by a transition prior that forbids switching. `0.5` is four
    orders of magnitude from anything `cnaster` ships, so this is a diagnosis
    and not a proposed setting.
    """
    truth, _, _, _, n_clones = phase_inputs

    occupancy = _decode_occupancy(
        _fit(phase_inputs, t=0.5, max_iter=100, planted=True), truth.n_states, n_clones
    )

    assert int((occupancy > 0).sum()) == truth.n_states, f"decoded {occupancy}"


@pytest.mark.cnaster
@pytest.mark.subject
def test_the_refinement_conserves_every_block(phased: Any) -> None:
    """The segmentation `initial_phase_given_partition` hands on.

    Split from the recovery claim because it holds whatever the vote did: the
    refinement redistributes blocks between contigs and is not allowed to lose
    or invent one. Kept live rather than folded into the `xfail` above, which
    would have taken this pin down with it.
    """
    _, _, blocks, _, _, refined = phased

    assert refined.sum() == blocks.X.shape[0], "the refinement lost a block"
    assert len(refined) >= len(blocks.lengths)


@pytest.mark.warning
@pytest.mark.subject
@pytest.mark.parametrize("t", [0.9999999, SHIPPED_T_PHASEING, 0.999, 0.99, 0.95, 0.9])
def test_the_decode_collapses_at_every_self_transition_down_to_nine_tenths(
    phase_inputs: Any, t: float
) -> None:
    """Six values spanning seven orders of magnitude in `1 - t`, all collapsing.

    #142. The sweep #129 opened was four values and too coarse to say anything
    about where this ends. This is the upper band, and it reproduces: every
    value here puts all 120 clone-blocks on one state on this machine and on a
    GitHub runner alike.

    `0.9` implies a ten-bin segment, already *shorter* than the events the
    fixture plants, and it still collapses. Every value `cnaster` ships is in
    this band.
    """
    truth, _, _, _, n_clones = phase_inputs

    occupancy = _decode_occupancy(
        _fit(phase_inputs, t=t, max_iter=100, planted=True), truth.n_states, n_clones
    )

    assert int((occupancy > 0).sum()) == 1, f"t={t} decoded {occupancy}"


@pytest.mark.cnaster
@pytest.mark.subject
@pytest.mark.parametrize("t", RESOLVING_SELF_TRANSITIONS)
def test_the_decode_resolves_once_the_prior_is_weak_enough(
    phase_inputs: Any, t: float
) -> None:
    """**The instance is decodable, so the collapse is the prior's doing.**

    The lower band, and the half of #142 that matters: three states come back
    once `t` is weak enough, so #122's collapse is the fit's and not the
    data's -- the attribution #129 rested on, measured directly.

    Only values that resolve on both platforms are asserted. The boundary
    itself moves between them, and `RESOLVING_SELF_TRANSITIONS` says why it is
    reported instead of pinned.

    `zenodo_sim_config.yaml` sets `t = 0.9999999` and `t_phaseing = 0.99999`,
    implying segments of ten million and one hundred thousand bins against a
    sixty-bin genome. Those are not priors on segment length; they assert one
    segment.
    """
    truth, _, _, _, n_clones = phase_inputs

    occupancy = _decode_occupancy(
        _fit(phase_inputs, t=t, max_iter=100, planted=True), truth.n_states, n_clones
    )

    assert int((occupancy > 0).sum()) == truth.n_states, f"t={t} decoded {occupancy}"


@pytest.mark.end2end
@pytest.mark.planted
@pytest.mark.parametrize("t", RESOLVING_SELF_TRANSITIONS)
def test_resolving_the_state_count_is_not_recovering_the_parameters(
    phase_inputs: Any, t: float
) -> None:
    """**A weaker prior buys the state count and not the states (#142).**

    Wherever the decode resolves, the fit still does not recover all three
    planted minor BAFs. On this machine at `t = 0.5` the folded fit is
    `[0.129, 0.212, 0.494]` against planted `[0.12, 0.42, 0.50]`: the two
    extremes land, the middle one is out by 0.21.

    So a fix for #122 that stopped at the state count would be measuring the
    wrong thing. Asserted as a count of recovered parameters rather than
    against those numbers, which are as platform-dependent as the threshold
    is -- the claim is that the set is incomplete, not which member is missing.

    The middle state does return at `t = 0.4`, within 0.01, with the occupancy
    changing character to 66/17/37. That is four orders below anything
    `cnaster` ships and well past a defensible prior, so it is recorded here
    rather than swept: it says the parameter is reachable, not that the value
    is usable.
    """
    truth, _, _, _, _ = phase_inputs

    fitted = np.asarray(
        _fit(phase_inputs, t=t, max_iter=100, planted=True)["new_p_binom"]
    ).ravel()
    folded = np.sort(np.minimum(fitted, 1.0 - fitted))
    planted = np.sort(np.minimum(truth.p_binom, 1.0 - truth.p_binom))

    recovered = sum(bool(min(abs(folded - value)) <= 0.02) for value in planted)

    assert recovered < truth.n_states, (
        f"t={t} recovered every planted state after all: {folded} against {planted}"
    )
