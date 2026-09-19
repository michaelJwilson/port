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
from tests.tmp_inputs import WrittenInputs, write_tmp_inputs
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
def written(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> WrittenInputs:
    """The fixture as files, written once for the module.

    Separate from `loaded` because the prep chain needs the gene table's path
    as well as what the loader returned, and a loader that also carried the
    paths would be two things.
    """
    root: Path = tmp_path_factory.mktemp("stages")
    return write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)


@pytest.fixture(scope="module")
def loaded(planted: CoreInferenceTruth, written: WrittenInputs) -> Iterator[Any]:
    """`load_input_data`'s return, from files written once for the module.

    The configuration stays installed for the body of every test: these stages
    read the global rather than taking it as an argument, and a fixture that
    restored it on the way out would leave them reading `None`.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import load_input_data

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


@pytest.mark.smoke
def test_no_tumour_proportion_file_gives_no_proportion(loaded: Any) -> None:
    """`preprocessing.tumorprop_file: None` returns `None`, not zeros.

    The distinction is load-bearing downstream: `run_core_inference` branches
    on `single_tumor_prop is None` and an array of zeros would take the mixed
    path with every spot called normal.
    """
    from cnaster.io import read_tumor_prop

    assert read_tumor_prop(loaded.adata) is None


@pytest.mark.end2end
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


@pytest.mark.snapshot
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
@pytest.mark.parametrize(
    "t",
    [0.9999999, PHASING_SELF_TRANSITION, SHIPPED_T_PHASEING, 0.999, 0.99, 0.95, 0.9],
)
def test_the_decode_collapses_at_every_self_transition_down_to_nine_tenths(
    phase_inputs: Any, t: float
) -> None:
    """Seven values spanning seven orders of magnitude in `1 - t`, all collapsing.

    #142, absorbing #129's four-value sweep: that test asserted the same
    thing over a subset of these values with an identical body, so it is
    gone and its one value #142 lacked -- `PHASING_SELF_TRANSITION` -- is
    here. Every value puts all 120 clone-blocks on one state, on this
    machine and on a GitHub runner alike.

    `0.9` implies a ten-bin segment, already *shorter* than the events the
    fixture plants, and it still collapses. Every value `cnaster` ships is in
    this band.
    """
    truth, _, _, _, n_clones = phase_inputs

    occupancy = _decode_occupancy(
        _fit(phase_inputs, t=t, max_iter=100, planted=True), truth.n_states, n_clones
    )

    assert int((occupancy > 0).sum()) == 1, f"t={t} decoded {occupancy}"


@pytest.mark.snapshot
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


NORMAL_BASELINE_TOLERANCE = 0.15
"""Total variation between the fitted normal baseline and the planted one.

Realized 0.101 on this fixture. The baseline is a per-bin **share** over the
candidate spots, so the comparison is between two distributions over its 40
bins and total variation is what states it: a per-bin relative error is
dominated by the low-count bins (median 12 per cent, 95th percentile 56) and
says more about the draw than about the stage.
"""


@pytest.fixture(scope="module")
def normal_stage(planted: Any, loaded: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """The normal stage run once: its candidate mask and its baseline.

    `determine_normal_candidates` reads a clone assignment and per-clone BAF
    profiles; the fixture knows both, so neither is taken from a fit. That is
    what makes the claims below end-to-end against planted truth rather than
    an agreement between two of `cnaster`'s own stages.

    Module-scoped because the three assertions are three claims about one
    run, not three runs: repeating the stage per test would treble a
    thirty-second call to restate the same arrays.
    """
    from cnaster.config import get_global_config
    from cnaster.normal_spot import (
        determine_normal_baseline,
        determine_normal_candidates,
    )
    from scipy.sparse import eye as sparse_eye

    truth = planted
    config = get_global_config()
    single_X = np.stack([truth.counts_nb, truth.counts_bb], axis=1)
    single_X_rdr = single_X[:, 0, :].astype(float)

    candidate = determine_normal_candidates(
        config,
        {"new_assignment": truth.labels},
        truth.p_binom[truth.states],
        single_X,
        single_X_rdr,
        sparse_eye(truth.n_spots, format="csr"),
        None,
    )
    rdr_normal, _, _ = determine_normal_baseline(single_X_rdr.copy(), candidate, config)

    return truth, np.asarray(candidate), np.asarray(rdr_normal).ravel()


@pytest.mark.end2end
def test_the_normal_candidates_are_the_planted_balanced_clone(
    normal_stage: tuple[Any, np.ndarray, np.ndarray],
) -> None:
    """**The stage selects the clone the fixture planted as balanced (#160).**

    `cnaster` picks by the smallest BAF deviation from 0.5, summed over the
    genome with a 0.05 deadband. The fixture plants by copy state: one clone
    sits at the balanced state in more of its bins than any other. The two
    rules are different, so their agreeing is a claim rather than a tautology,
    and it is the claim the whole normal-baseline path rests on.

    Every candidate must come from that clone. A candidate drawn from a clone
    carrying events would put tumour coverage into the baseline every later
    stage divides by.
    """
    truth, candidate, _ = normal_stage

    planted_balanced = int(
        np.argmax(
            [np.mean(truth.states[clone] == 0) for clone in range(truth.n_clones)]
        )
    )

    assert candidate.sum() > 0, "the stage selected no normal spots at all"
    assert set(np.unique(truth.labels[candidate]).tolist()) == {planted_balanced}, (
        f"candidates came from clones {np.unique(truth.labels[candidate])}, "
        f"and the planted balanced clone is {planted_balanced}"
    )


@pytest.mark.end2end
def test_the_normal_baseline_follows_the_planted_exposure(
    normal_stage: tuple[Any, np.ndarray, np.ndarray],
) -> None:
    """**The baseline is the planted exposure over the spots it selected (#160).**

    `determine_normal_baseline` sums the read-depth channel over the candidate
    spots and normalizes, so it estimates the per-bin share of the library the
    normal population carries. The fixture planted that share as
    `base_nb_mean`, and the normal clone sits at `mu = 1` in most of its bins,
    so the planted expectation is the exposure itself.

    Every later stage divides by this baseline, so an error here is an error
    in every copy ratio the pipeline reports. Nothing checked it before.
    """
    truth, candidate, rdr_normal = normal_stage

    planted_share = np.asarray(truth.base_nb_mean)[:, candidate].sum(axis=1)
    planted_share = planted_share / planted_share.sum()

    total_variation = 0.5 * float(np.abs(rdr_normal - planted_share).sum())

    assert total_variation < NORMAL_BASELINE_TOLERANCE, (
        f"the fitted baseline is {total_variation:.4f} from the planted one in "
        f"total variation, over {rdr_normal.size} bins"
    )


@pytest.mark.analytic
def test_the_normal_baseline_is_a_distribution_over_bins(
    normal_stage: tuple[Any, np.ndarray, np.ndarray],
) -> None:
    """Conservation: the baseline is a share, so it sums to one.

    Holds of any correct implementation -- the stage divides by its own total
    -- and it is what makes the total-variation comparison above a comparison
    between two distributions rather than between two scales.
    """
    _, _, rdr_normal = normal_stage

    assert float(np.sum(rdr_normal)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.min(rdr_normal)) >= 0.0


SHIPPED_NORMAL_CONFIDENCE = (0.01, 0.99)
"""`zenodo_sim_config.yaml`'s `quality.normal_allele_specific_confidence`.

`tests/run_config.py` widens it to `(0.0, 1.0)` so the fixture's bins survive
into the round trip, so the shipped value is restated here and passed
explicitly: what the filter does at what ships is the claim, and a test reading
the widened configuration would be measuring the widening.
"""


def _prep_chain(loaded: Any, written: WrittenInputs) -> tuple[Any, Any, Any]:
    """The five `omics` calls `run_cnaster` makes between loading and binning.

    Returns the gene-SNP table, the binned counts, and the per-bin table the
    normal stage reads. `tests/test_run_cnaster_prep.py` asserts this chain
    recovers the planted segmentation and counts on its own instance; here it
    is the input to the two filters, not the subject.
    """
    from cnaster.omics import (
        assign_initial_blocks,
        binned_gene_snp,
        create_bin_ranges,
        form_gene_snp_table,
        summarize_counts_for_bins,
        summarize_counts_for_blocks,
    )

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
    table = create_bin_ranges(
        table,
        loaded.adata,
        *alleles,
        loaded.unique_snp_ids,
        blocks.X,
        blocks.total_bb_RD,
        blocks.lengths,
        secondary_min_umi=1,
        secondary_min_snp_umi=1,
        secondary_min_normal_umi=0,
    )
    binned = summarize_counts_for_bins(
        table,
        loaded.adata,
        blocks.X,
        blocks.total_bb_RD,
        np.ones(int(table.block_id.dropna().nunique()), dtype=bool),
        nu=1.0,
        logphase_shift=0.0,
        geneticmap_file=None,
    )

    return table, binned, binned_gene_snp(table)


def _balanced_clone(truth: CoreInferenceTruth) -> int:
    """Which clone the fixture planted at the balanced state in most bins."""
    return int(
        np.argmax(
            [np.mean(truth.states[clone] == 0) for clone in range(truth.n_clones)]
        )
    )


@pytest.fixture(scope="module")
def prepared(loaded: Any, written: WrittenInputs) -> tuple[Any, Any, Any]:
    """The prep chain run once, for every test below that needs bins."""
    return _prep_chain(loaded, written)


@pytest.fixture(scope="module")
def baf_filtered(
    planted: CoreInferenceTruth, prepared: tuple[Any, Any, Any]
) -> tuple[Any, Any, np.ndarray]:
    """`normal_baf_bin_filter` at the shipped interval, and what it removed.

    The table is copied in: the function writes `None` into its caller's
    `bin_id` column and then renumbers it, so a caller that kept the reference
    has a different table afterwards (#89). Copying is what lets the removal
    set be read off as the difference between the two.
    """
    from cnaster.normal_spot import normal_baf_bin_filter

    truth, binned, _ = planted, *prepared[1:]
    table = prepared[0]
    index_normal = np.flatnonzero(truth.labels == _balanced_clone(truth))

    filtered, counts = normal_baf_bin_filter(
        table.copy(),
        binned.X.copy(),
        binned.base_nb_mean.copy(),
        binned.total_bb_RD.copy(),
        1.0,
        0.0,
        index_normal,
        None,
        confidence_interval=SHIPPED_NORMAL_CONFIDENCE,
    )

    dropped = filtered.bin_id.isna() & table.bin_id.notna()
    removed = np.unique(table.loc[dropped, "bin_id"].to_numpy().astype(int))

    return filtered, counts, removed


@pytest.mark.end2end
def test_the_baf_filter_removes_the_imbalanced_bins_of_the_normal_clone(
    planted: CoreInferenceTruth, baf_filtered: tuple[Any, Any, np.ndarray]
) -> None:
    """**The removal set is exactly the planted non-balanced bins (#38, #160).**

    The filter pools B-allele counts over the normal spots, fits a
    beta-binomial with `p` forced to 0.5, and drops the bins whose pooled
    count falls outside the interval. The fixture plants which bins those are:
    the balanced clone sits at `p = 0.5` everywhere except inside its own
    events, where the planted `p` is 0.58 or above.

    So the two sets have to agree, and they do **exactly** -- eight bins
    removed, eight planted, no bin either way. That is the claim the whole
    filter exists to support, and nothing checked it before.

    Set equality rather than a rate: a recall figure would let a filter that
    removed the genome score well, and a precision figure alone would let one
    that removed nothing.
    """
    _, _, removed = baf_filtered

    planted_imbalanced = np.flatnonzero(planted.states[_balanced_clone(planted)] != 0)

    np.testing.assert_array_equal(removed, planted_imbalanced)


@pytest.mark.end2end
def test_the_filtered_segmentation_is_the_planted_one_less_the_removals(
    planted: CoreInferenceTruth, baf_filtered: tuple[Any, Any, np.ndarray]
) -> None:
    """The chromosomes shorten by what was removed from each, not by a total.

    `lengths` is rebuilt from the surviving bins per chromosome, so a removal
    charged to the wrong chromosome -- the error an off-by-one in the
    renumbering would make -- moves the boundary without changing the total.
    A scalar count of survivors cannot see it; the vector can.

    Realized `[8, 17, 7]` against the planted `[10, 21, 9]`: two removals in
    the first chromosome, four in the second, two in the third.
    """
    _, counts, removed = baf_filtered

    chromosome_of_bin = np.repeat(
        np.arange(planted.lengths.size), np.asarray(planted.lengths)
    )
    expected = np.asarray(planted.lengths) - np.bincount(
        chromosome_of_bin[removed], minlength=planted.lengths.size
    )

    np.testing.assert_array_equal(np.asarray(counts.lengths), expected)
    assert counts.X.shape[0] == int(expected.sum())


@pytest.mark.analytic
def test_the_surviving_bins_are_renumbered_onto_a_contiguous_range(
    baf_filtered: tuple[Any, Any, np.ndarray],
) -> None:
    """Conservation: the survivors are relabelled `0 .. n-1`, once each.

    Which is why a stage downstream of this filter cannot be refereed against
    a planted bin index (#105): the label a bin carries afterwards is its rank
    among the survivors, not the bin it was. Holds of any correct
    implementation, so it says nothing about `cnaster` being right -- it says
    the renumbering is a bijection, which is what makes the two claims above
    readable.
    """
    filtered, counts, _ = baf_filtered

    surviving = np.sort(filtered.bin_id.dropna().unique().astype(int))

    np.testing.assert_array_equal(surviving, np.arange(counts.X.shape[0]))


@pytest.fixture(scope="module")
def one_gene_per_bin(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CoreInferenceTruth, np.ndarray, np.ndarray]:
    """`filter_normal_diffexp` on an instance whose bins hold one gene each.

    `genes_per_bin=(1, 2)` draws from `[1, 2)`, so every bin is one gene and
    the bin's counts are the gene's. That is the instance on which the
    filter's **selection** is what decides the answer, which the module
    fixture's multi-gene instance is not: see the pinned defect below.

    The configuration is installed for the body of this fixture alone and the
    work is done inside it, because `filter_normal_diffexp` reads none of it
    and the module already holds two instances whose globals would otherwise
    interleave.
    """
    from cnaster.config import YAMLConfig, get_global_config, set_global_config
    from cnaster.io import get_sample_list, load_input_data
    from cnaster.normal_spot import filter_normal_diffexp

    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=LATTICE, n_obs=40, n_segments=3, seed=11
    )
    root: Path = tmp_path_factory.mktemp("one_gene")
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, genes_per_bin=(1, 2)), root
    )
    config_path = write_run_cnaster_config(written, truth)

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        loaded = load_input_data(get_global_config())
        _, binned, df_bin_info = _prep_chain(loaded, written)
        sample_list, sample_ids = get_sample_list(loaded.adata)
        retained = np.asarray(
            filter_normal_diffexp(
                loaded.exp_counts,
                df_bin_info,
                truth.labels == _balanced_clone(truth),
                sample_list=sample_list,
                sample_ids=sample_ids,
            )
        )
    finally:
        set_global_config(None)
        set_global_config(previous)

    return truth, retained, np.asarray(binned.X[:, 0, :])


@pytest.mark.end2end
def test_the_expression_filter_keeps_every_planted_count_it_should(
    one_gene_per_bin: tuple[CoreInferenceTruth, np.ndarray, np.ndarray],
) -> None:
    """**Nothing in this instance is differentially expressed, so nothing goes.**

    The filter drops a gene whose expression differs between the normal
    candidates and the rest by more than `logfcthreshold_t = 4` -- a factor of
    16 -- among genes above the 80th percentile of total UMIs. The fixture's
    widest planted contrast is `mu = 5` against `mu = 1`, a log fold change of
    2.32, and the realized maximum over every gene is **2.47**. So the
    prediction from the planted parameters is that the filter returns its
    input unchanged, and it does, bitwise.

    Asserted against `truth.counts_nb` rather than against the binner's output:
    the planted counts are what the claim is about, and comparing two
    `cnaster` stages to each other would pass equally well if both were wrong.

    What this does **not** establish is that the selection fires correctly when
    something is differentially expressed. It cannot on this fixture: the 13
    genes above the UMI gate are every one of them an `unassigned_*` gene,
    which carries no copy-number signal by construction, and no planted
    contrast reaches 16-fold. A positive case needs a fixture with a wider
    expression separation, and is #160's to place.
    """
    truth, retained, _ = one_gene_per_bin

    np.testing.assert_array_equal(retained, truth.counts_nb.astype(float))


@pytest.mark.bug
def test_the_expression_filter_empties_every_bin_holding_more_than_one_gene(
    planted: CoreInferenceTruth,
    loaded: Any,
    prepared: tuple[Any, Any, Any],
) -> None:
    """**A separator mismatch zeroes 32 of 40 bins, 82 per cent of the UMIs.**

    `binned_gene_snp` writes `INCLUDED_GENES` as `",".join(...)`
    (`omics.py:261`); `filter_normal_diffexp` reads it back with
    `genestr.split(" ")` (`normal_spot.py:903`). So a bin holding more than one
    gene yields a single name -- `"gene_0_0,gene_0_1"` -- that matches nothing
    in `adata.var`, its gene set is empty, and its counts are summed over no
    genes at all.

    The result is not a filter doing too much: **no gene passes either
    threshold on this instance** -- 0 of the 131 that survive
    `sc.pp.filter_genes` -- and the bins still come back at zero. Bins holding
    exactly
    one gene carry no comma and survive bitwise, which is what identifies the
    mechanism rather than merely pinning the symptom.

    Gated behind `config.quality.filter_normal_diffexp`, which this fixture and
    `zenodo_sim_config.yaml` leave off; a run that turns it on loses the read
    depth of every multi-gene bin. Reported upstream; `port` does not land the
    fix.
    """
    from cnaster.normal_spot import filter_normal_diffexp

    _, binned, df_bin_info = prepared
    genes_per_bin = np.array(
        [len(text.split(",")) for text in df_bin_info.INCLUDED_GENES.to_numpy()]
    )
    assert genes_per_bin.max() > 1, "the instance holds no multi-gene bin to lose"

    retained = np.asarray(
        filter_normal_diffexp(
            loaded.exp_counts,
            df_bin_info,
            planted.labels == _balanced_clone(planted),
            sample_list=["S1"],
            sample_ids=np.zeros(planted.n_spots, dtype=int),
        )
    )

    emptied = retained.sum(axis=1) == 0
    np.testing.assert_array_equal(emptied, genes_per_bin > 1)
    np.testing.assert_array_equal(
        retained[~emptied], np.asarray(binned.X[~emptied, 0, :], dtype=float)
    )
