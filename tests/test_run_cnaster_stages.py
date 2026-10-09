"""The stages `run_cnaster` calls, one at a time, against the planted truth of one fixture.

Bins are renumbered after `normal_baf_bin_filter` (#105), so tests sit upstream of it or
are indifferent.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cnaster.config import get_global_config
from cnaster.hmm_initialize import gmm_init
from cnaster.hmm_nophasing import get_log_transmat
from cnaster.hmm_phased import hmm_phased
from cnaster.hmrf_utils import clone_stack_obs
from cnaster.io import (
    construct_df_clone_label,
    get_sample_list,
    load_input_data,
    read_tumor_prop,
)
from cnaster.normal_spot import (
    determine_normal_baseline,
    determine_normal_candidates,
    filter_normal_diffexp,
    normal_baf_bin_filter,
)
from cnaster.omics import binned_gene_snp
from cnaster.phasing import initial_phase_given_partition
from cnaster.pseudobulk import merge_pseudobulk_by_index_mix
from cnaster.spatial import initialize_clones
from port.sim.inputs import (
    WrittenInputs,
    read_to_bins,
    write_tmp_inputs,
    written_config,
)
from port.sim.run_config import (
    FLIP_EVERY,
    SHIPPED_T_PHASEING,
    PlantedInstance,
    write_run_cnaster_config,
)
from port.sim.truth import CoreInferenceTruth, balanced_clone, core_inference_truth
from port.sim.unsegment import unsegment
from scipy.sparse import eye as sparse_eye

from tests.adapters import cnaster_initial_phase
from tests.fixtures import END_TO_END_LATTICE, end_to_end_truth

pytestmark = pytest.mark.preprocessing


BALANCED_STATE = 0
"""The planted diploid balanced state, which casts no phase vote (#106)."""

TRIVIAL_AGREEMENT = 1.0 - 1.0 / FLIP_EVERY
"""Agreement an all-zero indicator scores, phase being up to a complement (#122, #120)."""

STRONG_MARGIN = 0.1
"""How far from balance a planted state has to sit to count as strong."""

RESOLVING_SELF_TRANSITIONS = (0.7, 0.6, 0.5)
"""Swept `t` resolving three states on every platform; the boundary is not (#142, #147)."""

PHASING_SELF_TRANSITION = 1.0 - 1e-6
"""What `phased` passes, between the two shipped values."""

PHASING_EPS_BAF = 0.1
"""`phasing.py:67`'s deadband, restated since it is a local there."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """One planted instance for the module: equal bands, events in both clones (#298)."""
    return end_to_end_truth(normal_clone=False)


@pytest.fixture(scope="module")
def written(
    planted: CoreInferenceTruth, tmp_path_factory: pytest.TempPathFactory
) -> WrittenInputs:
    """The fixture as files, written once for the module."""
    root: Path = tmp_path_factory.mktemp("stages")
    return write_tmp_inputs(planted, unsegment(planted, flip_every=0), root)


@pytest.fixture(scope="module")
def loaded(planted: CoreInferenceTruth, written: WrittenInputs) -> Iterator[Any]:
    """`load_input_data`'s return; the pipeline's full configuration stays installed for each test."""

    with written_config(write_run_cnaster_config(written, planted)) as config:
        yield load_input_data(config)


@pytest.mark.end2end
def test_the_sample_list_is_the_one_slice_the_fixture_wrote(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """One slice in, one slice out, every spot assigned to it."""

    sample_list, sample_ids = get_sample_list(loaded.adata)

    assert sample_list == ["S1"]
    assert sample_ids.shape == (planted.n_spots,)
    assert set(np.unique(sample_ids)) == {0}


@pytest.mark.smoke
def test_no_tumour_proportion_file_gives_no_proportion(loaded: Any) -> None:
    """`preprocessing.tumorprop_file: None` returns `None`, not zeros."""

    assert read_tumor_prop(loaded.adata) is None


@pytest.mark.end2end
@pytest.mark.critical
def test_the_rectangular_partition_recovers_the_planted_bands(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """The rectangular partition reproduces the planted bands exactly (#95)."""

    coordinates = np.asarray(loaded.coords, dtype=float)
    # NB the bands run along `x`, so the partition is `n_clones` by one; the transpose
    # recovers nothing.
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
    # NB the two agree up to which end is first.
    agreement = max(
        float(np.mean(recovered == planted.labels)),
        float(np.mean(recovered == planted.n_clones - 1 - planted.labels)),
    )
    assert agreement == 1.0


@pytest.mark.analytic
def test_the_partition_covers_every_spot_exactly_once(
    planted: CoreInferenceTruth, loaded: Any
) -> None:
    """The partition covers every spot exactly once, though unbalanced."""

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

    table = construct_df_clone_label(
        np.asarray(loaded.barcodes),
        np.asarray(loaded.coords, dtype=float),
        planted.labels,
    )

    assert len(table) == planted.n_spots
    assert set(table.columns) == {"sample_id", "x", "y", "clone_label"}
    # NB the barcode is the index, as `clone_labels.tsv`'s first field.
    assert table.index.nunique() == planted.n_spots
    np.testing.assert_array_equal(
        np.sort(table.clone_label.to_numpy()), np.sort(planted.labels)
    )


@pytest.fixture(scope="module")
def flipped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """An instance whose files store every third block on the other haplotype."""

    # NB `self_transition` loosened from 0.99 so the chain leaves its start state.
    # NB a phase-rich genome with no normal clone (#120, #298), so most blocks carry a
    # phase.
    truth = core_inference_truth(
        n_clones=2,
        n_states=3,
        lattice=END_TO_END_LATTICE,
        n_obs=60,
        n_segments=2,
        events=(8, 12),
        event_bins=(10, 25),
        seed=5,
        normal_clone=False,
    )
    pre_image = unsegment(
        truth, blocks_per_bin=(1, 2), unassigned_genes=0, flip_every=FLIP_EVERY
    )

    root: Path = tmp_path_factory.mktemp("phasing")
    written = write_tmp_inputs(truth, pre_image, root)

    with written_config(write_run_cnaster_config(written, truth)) as config:
        yield truth, pre_image, load_input_data(config), written


@pytest.fixture(scope="module")
def phased(flipped: Any) -> Any:
    """`initial_phase_given_partition` on the flipped instance, run once."""
    truth, pre_image, loaded, written = flipped
    blocks = read_to_bins(written, loaded=loaded, through="blocks").blocks
    res, recovered, refined = cnaster_initial_phase(truth, blocks, 1.0 - 1e-6)

    return truth, pre_image, blocks, res, recovered, refined


def _recovered_on(
    recovered: np.ndarray, planted: np.ndarray, mask: np.ndarray
) -> float:
    """Agreement over `mask`, phase taken up to a global complement."""
    return max(
        float(np.mean(recovered[mask] == planted[mask])),
        float(np.mean(recovered[mask] != planted[mask])),
    )


@pytest.mark.warning
def test_no_block_casts_a_vote_because_every_one_decodes_balanced(
    phased: Any,
) -> None:
    """`phase_indicator` is zero everywhere: every block decodes balanced, so none votes (#122)."""
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
    """Phasing flips back the planted flipped blocks above `TRIVIAL_AGREEMENT` (#122, #106)."""
    truth, pre_image, _, _, recovered, _ = phased

    planted = ~pre_image.phase_indicator
    assert recovered.shape == planted.shape

    # Blocks whose planted state is not balanced, the only ones that vote.
    per_block = truth.states[:, : recovered.size]
    speakable = np.all(per_block != BALANCED_STATE, axis=0)
    assert speakable.sum() > 0.25 * recovered.size, "too few blocks carry a phase"

    agreement = _recovered_on(recovered, planted, speakable)
    assert agreement > TRIVIAL_AGREEMENT, (
        f"phase agreement {agreement:.3f} over {speakable.sum()} blocks, against "
        f"{TRIVIAL_AGREEMENT:.3f} for flipping nothing"
    )

    strong = np.any(np.abs(truth.p_binom - 0.5)[per_block] >= STRONG_MARGIN, axis=0)
    strong_agreement = _recovered_on(recovered, planted, strong)
    assert strong_agreement > TRIVIAL_AGREEMENT, (
        f"strong-block agreement {strong_agreement:.3f} over {strong.sum()}"
    )


@pytest.fixture(scope="module")
def phase_inputs(flipped: Any) -> Any:
    """Clone-stacked arrays and initializer output as `phasing.py:75-105` builds them."""

    truth, _, loaded, written = flipped
    blocks = read_to_bins(written, loaded=loaded, through="blocks").blocks

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
    """`initial_phase_given_partition` refuses a non-zero exposure, ruling out the call site (#122)."""

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
    """`gmm_init` recovers the planted minor BAFs within 0.001 (#122)."""
    truth, _, _, init_p_binom, _ = phase_inputs

    planted_minor = np.sort(np.minimum(truth.p_binom, 1.0 - truth.p_binom))
    initialized = np.sort(np.asarray(init_p_binom).ravel())

    np.testing.assert_allclose(initialized, planted_minor, atol=2e-3)


@pytest.mark.bug
def test_the_fit_collapses_the_decode_between_its_first_two_iterations(
    phase_inputs: Any,
) -> None:
    """From the planted start the decode collapses to one state at iteration two (#122)."""
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
    """The refinement conserves every block across contigs."""
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
    """Every `t` from 0.9 to `PHASING_SELF_TRANSITION` collapses the decode (#142, #129)."""
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
    """Three states resolve at `RESOLVING_SELF_TRANSITIONS`, so the collapse is the prior's (#142)."""
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
    """Where the decode resolves, not all three planted minor BAFs are recovered (#142)."""
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
"""Total-variation bound between fitted and planted normal baseline shares."""


@pytest.fixture(scope="module")
def normal_stage(planted: Any, loaded: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """The normal stage run once from the planted clones and BAF profiles."""

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
    """Normal candidates all come from the planted balanced clone (#160)."""
    truth, candidate, _ = normal_stage

    planted_balanced = balanced_clone(truth)

    assert candidate.sum() > 0, "the stage selected no normal spots at all"
    assert set(np.unique(truth.labels[candidate]).tolist()) == {planted_balanced}, (
        f"candidates came from clones {np.unique(truth.labels[candidate])}, "
        f"and the planted balanced clone is {planted_balanced}"
    )


@pytest.mark.end2end
def test_the_normal_baseline_follows_the_planted_exposure(
    normal_stage: tuple[Any, np.ndarray, np.ndarray],
) -> None:
    """The baseline is the planted exposure within `NORMAL_BASELINE_TOLERANCE` TV (#160)."""
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
    """The baseline sums to one."""
    _, _, rdr_normal = normal_stage

    assert float(np.sum(rdr_normal)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.min(rdr_normal)) >= 0.0


SHIPPED_NORMAL_CONFIDENCE = (0.01, 0.99)
"""The shipped `quality.normal_allele_specific_confidence`, which the run config widens."""


def _prep_chain(loaded: Any, written: WrittenInputs) -> tuple[Any, Any, Any]:
    """The five `omics` calls `run_cnaster` makes between loading and binning."""

    chain = read_to_bins(written, loaded=loaded)

    return chain.table, chain.bins, binned_gene_snp(chain.table)


@pytest.fixture(scope="module")
def prepared(loaded: Any, written: WrittenInputs) -> tuple[Any, Any, Any]:
    """The prep chain run once, for every test below that needs bins."""
    return _prep_chain(loaded, written)


@pytest.fixture(scope="module")
def baf_filtered(
    planted: CoreInferenceTruth, prepared: tuple[Any, Any, Any]
) -> tuple[Any, Any, np.ndarray]:
    """`normal_baf_bin_filter` at the shipped interval on a copy, and what it removed (#89)."""

    truth, binned, _ = planted, *prepared[1:]
    table = prepared[0]
    index_normal = np.flatnonzero(truth.labels == balanced_clone(truth))

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
    """The removed bins are exactly the planted non-balanced bins of the normal clone (#38, #160)."""
    _, _, removed = baf_filtered

    planted_imbalanced = np.flatnonzero(planted.states[balanced_clone(planted)] != 0)

    np.testing.assert_array_equal(removed, planted_imbalanced)


@pytest.mark.end2end
def test_the_filtered_segmentation_is_the_planted_one_less_the_removals(
    planted: CoreInferenceTruth, baf_filtered: tuple[Any, Any, np.ndarray]
) -> None:
    """Each chromosome's length drops by its own removals: `[8, 17, 7]` from `[10, 21, 9]`."""
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
    """Survivors are relabelled `0 .. n-1`, once each (#105)."""
    filtered, counts, _ = baf_filtered

    surviving = np.sort(filtered.bin_id.dropna().unique().astype(int))

    np.testing.assert_array_equal(surviving, np.arange(counts.X.shape[0]))


@pytest.fixture(scope="module")
def one_gene_per_bin(
    planted_instance: PlantedInstance, tmp_path_factory: pytest.TempPathFactory
) -> tuple[CoreInferenceTruth, np.ndarray, np.ndarray]:
    """`filter_normal_diffexp` on an instance with one gene per bin."""

    truth = planted_instance[0]
    root: Path = tmp_path_factory.mktemp("one_gene")
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, genes_per_bin=(1, 2)), root
    )

    with written_config(write_run_cnaster_config(written, truth)) as config:
        loaded = load_input_data(config)
        _, binned, df_bin_info = _prep_chain(loaded, written)
        sample_list, sample_ids = get_sample_list(loaded.adata)
        retained = np.asarray(
            filter_normal_diffexp(
                loaded.exp_counts,
                df_bin_info,
                truth.labels == balanced_clone(truth),
                sample_list=sample_list,
                sample_ids=sample_ids,
            )
        )

    return truth, retained, np.asarray(binned.X[:, 0, :])


@pytest.mark.end2end
def test_the_expression_filter_keeps_every_planted_count_it_should(
    one_gene_per_bin: tuple[CoreInferenceTruth, np.ndarray, np.ndarray],
) -> None:
    """No planted contrast reaches 16-fold, so the filter returns `truth.counts_nb` bitwise (#160)."""
    truth, retained, _ = one_gene_per_bin

    np.testing.assert_array_equal(retained, truth.counts_nb.astype(float))


@pytest.mark.bug
def test_the_expression_filter_empties_every_bin_holding_more_than_one_gene(
    planted: CoreInferenceTruth,
    loaded: Any,
    prepared: tuple[Any, Any, Any],
) -> None:
    """`INCLUDED_GENES` joined by `,` but split on ` ` zeroes every multi-gene bin (`bug`)."""

    _, binned, df_bin_info = prepared
    genes_per_bin = np.array(
        [len(text.split(",")) for text in df_bin_info.INCLUDED_GENES.to_numpy()]
    )
    assert genes_per_bin.max() > 1, "the instance holds no multi-gene bin to lose"

    retained = np.asarray(
        filter_normal_diffexp(
            loaded.exp_counts,
            df_bin_info,
            planted.labels == balanced_clone(planted),
            sample_list=["S1"],
            sample_ids=np.zeros(planted.n_spots, dtype=int),
        )
    )

    emptied = retained.sum(axis=1) == 0
    np.testing.assert_array_equal(emptied, genes_per_bin > 1)
    np.testing.assert_array_equal(
        retained[~emptied], np.asarray(binned.X[~emptied, 0, :], dtype=float)
    )
