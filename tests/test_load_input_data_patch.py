"""`port.patch.io.load_input_data` returns what `cnaster`'s does.

#167. The patch removes passes, not rows: every spot, gene and SNP that
survives `cnaster`'s loader survives this one, and every count is the same
integer. So the referee is `cnaster` itself, field for field, and the
comparison is bitwise rather than to a tolerance -- there is no arithmetic
here for a tolerance to absorb.

What is **not** claimed: that either loader is right. Both read the same files
and both could mis-read them together. That claim belongs to the stage tests
in `tests/test_run_cnaster_stages.py`, which judge what comes out of the
loader against the truth the fixture planted.
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

GATE_LATTICE = (25, 40)
GATE_OBS = 40
"""The dev instance: a thousand spots over forty bins.

What a merge is gated on. The size at which a **speedup** is established is
the release-tier one below, per the Measurement rule: a ratio read here
decides nothing.
"""

STRESS_LATTICE = (50, 50)
STRESS_OBS = 400
"""2,500 spots over 400 bins -- 782 SNPs and 1,187 genes once binned.

Chosen as the largest instance whose fixture builds in under four seconds, so
the measurement is repeatable inside a test run rather than an offline note.
A Visium slide is 5,000 spots against 500,000 SNPs, where the arrays this
patch does not allocate are gigabytes rather than megabytes; the direction is
established here and the magnitude there is arithmetic, not measurement.
"""


def _instance(
    root: Path, lattice: tuple[int, int], n_obs: int
) -> tuple[CoreInferenceTruth, Any, WrittenInputs, Path]:
    """Plant an instance and write it, returning its configuration path.

    The pre-image comes back too: it carries the planted per-gene counts and
    their names, which is what the loader is judged against below. The binned
    truth cannot serve, because the loader drops genes and a bin total would
    then disagree for a reason that is the filter working.
    """
    truth = core_inference_truth(
        n_clones=2, n_states=3, lattice=lattice, n_obs=n_obs, n_segments=3, seed=11
    )
    pre_image = unsegment(truth, flip_every=0)
    written = write_tmp_inputs(truth, pre_image, root)

    return truth, pre_image, written, write_run_cnaster_config(written, truth)


@pytest.fixture(scope="module")
def planted_instance(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CoreInferenceTruth, Any, WrittenInputs, Path]:
    """The dev instance, planted and written once for the module."""
    root: Path = tmp_path_factory.mktemp("patch_gate")

    return _instance(root, GATE_LATTICE, GATE_OBS)


@pytest.fixture(scope="module")
def gate_config(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
) -> Iterator[Any]:
    """The dev instance's configuration, installed for the module."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    config_path = planted_instance[3]

    previous = get_global_config()
    set_global_config(None)
    set_global_config(YAMLConfig.from_file(config_path))
    try:
        yield get_global_config()
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.fixture(scope="module")
def both_loaders(gate_config: Any) -> tuple[Any, Any]:
    """Both loaders run once on the same instance."""
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data as patched

    return upstream(gate_config), patched(gate_config)


@pytest.mark.patch
def test_the_patch_returns_the_same_counts_as_cnaster(
    both_loaders: tuple[Any, Any],
) -> None:
    """Every count matrix, bitwise, including the two allele matrices dense.

    The allele matrices are where the patch does most of its work -- it never
    builds the dense `(A + B)` upstream builds to take a row sum -- so their
    agreeing is the claim, not a formality.
    """
    reference, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.adata.layers["count"]),
        np.asarray(reference.adata.layers["count"]),
    )
    np.testing.assert_array_equal(patched.cell_snp_Aallele, reference.cell_snp_Aallele)
    np.testing.assert_array_equal(patched.cell_snp_Ballele, reference.cell_snp_Ballele)
    np.testing.assert_array_equal(
        patched.exp_counts.sparse.to_dense().to_numpy(),
        reference.exp_counts.sparse.to_dense().to_numpy(),
    )


@pytest.mark.patch
def test_the_patch_keeps_the_same_spots_genes_and_snps(
    both_loaders: tuple[Any, Any],
) -> None:
    """The filters select the same rows and columns, in the same order.

    Order matters and is easy to lose: the loader sorts spots into the SNP
    barcodes' order, and a patch that filtered before sorting would return the
    same *set* of spots against a different allele matrix. Comparing the
    barcode sequence rather than the count catches that; comparing the shapes
    would not.
    """
    reference, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.barcodes), np.asarray(reference.barcodes)
    )
    np.testing.assert_array_equal(
        np.asarray(patched.adata.var.index), np.asarray(reference.adata.var.index)
    )
    np.testing.assert_array_equal(patched.unique_snp_ids, reference.unique_snp_ids)
    np.testing.assert_array_equal(
        np.asarray(patched.coords, dtype=float),
        np.asarray(reference.coords, dtype=float),
    )
    assert (patched.across_slice_adjacency_mat is None) == (
        reference.across_slice_adjacency_mat is None
    )


def _planted_alignment(
    pre_image: Any, written: WrittenInputs, loaded: Any
) -> np.ndarray:
    """The planted count matrix, permuted into the loader's order.

    The loader sorts spots into the SNP barcodes' order and drops genes, so
    every comparison against planted counts needs both alignments. One helper
    rather than one per test: the alignment is bookkeeping, and repeating it
    would make three tests look like three claims about it.
    """
    planted = np.asarray(pre_image.adata.layers["count"])
    planted_genes = {
        str(name): column
        for column, name in enumerate(np.asarray(pre_image.adata.var.index))
    }
    spot_of = {str(barcode): spot for spot, barcode in enumerate(written.barcodes)}

    rows = np.array([spot_of[str(barcode)] for barcode in loaded.barcodes])
    columns = np.array(
        [planted_genes[str(name)] for name in np.asarray(loaded.adata.var.index)]
    )

    return np.asarray(planted[np.ix_(rows, columns)])


MIN_PERCENT_EXPRESSED_SPOTS = 5.0e-3
"""`load_input_data`'s default gene floor, as a fraction of the spots.

Restated rather than imported: it is a default argument on a 480-line function,
so a test that read it back off the signature would pass whatever it became.
`# BUG actually a fraction` is `cnaster`'s own note beside it -- the name says
percent and the arithmetic says fraction, which at 5e-3 over a thousand spots
is a floor of five spots rather than of five.
"""


@pytest.mark.end2end
def test_the_patched_loader_returns_the_planted_counts_for_every_gene_it_keeps(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    both_loaders: tuple[Any, Any],
) -> None:
    """**Every retained count is the planted count, bitwise (#167).**

    The judgement the equivalence tests cannot make: they establish that the
    two loaders agree, not that either read the files correctly. This compares
    the patched loader's matrix against the counts the fixture planted, gene by
    gene and spot by spot, after aligning both by name -- the loader sorts its
    spots into the SNP barcodes' order, so a comparison in array order would be
    comparing two different permutations and would fail for the wrong reason.

    The genes the filter drops are excluded here and are the next test's
    subject, because "the retained counts are right" and "the right genes were
    retained" are two claims and a single assertion could not tell which broke.
    """
    _, pre_image, written, _ = planted_instance
    _, patched = both_loaders

    np.testing.assert_array_equal(
        np.asarray(patched.adata.layers["count"]),
        _planted_alignment(pre_image, written, patched),
    )


@pytest.mark.end2end
def test_the_loader_drops_exactly_the_genes_too_few_spots_express(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    both_loaders: tuple[Any, Any],
) -> None:
    """**Which genes survive is decided by the planted counts, and it holds.**

    `load_input_data` keeps a gene expressed in at least
    `min_percent_expressed_spots * n_spots` spots. That floor is a statement
    about the data, so the planted counts settle it without running anything:
    the set the loader kept has to be the set the planted matrix says is above
    the floor.

    Realized on the dev instance: 138 genes written, 132 kept, and the six
    dropped are the six the planted counts put under five spots -- one
    expressed nowhere at all, the rest in one to four. The
    assertion is set equality, so a filter that dropped one gene too many or
    kept one too few fails whichever way it erred.
    """
    _, pre_image, _, _ = planted_instance
    _, patched = both_loaders

    planted = np.asarray(pre_image.adata.layers["count"])
    names = np.asarray(pre_image.adata.var.index)

    floor = MIN_PERCENT_EXPRESSED_SPOTS * planted.shape[0]
    expected = {str(name) for name in names[(planted > 0).sum(axis=0) >= floor]}

    assert {str(name) for name in patched.adata.var.index} == expected


@pytest.mark.backend
def test_the_sparse_return_carries_the_same_matrix(gate_config: Any) -> None:
    """`sparse_counts=True` is the same numbers in a different container.

    The flag exists to be measured rather than adopted: the dense return is
    `cnaster`'s type contract and every caller indexes it as an array. What it
    changes is bytes, not values, and that is what this pins.
    """
    import scipy.sparse as sp
    from port.patch.io import load_input_data as patched

    dense = patched(gate_config)
    sparse = patched(gate_config, sparse_counts=True)

    assert sp.issparse(sparse.cell_snp_Aallele)
    np.testing.assert_array_equal(
        sparse.cell_snp_Aallele.toarray(), dense.cell_snp_Aallele
    )
    np.testing.assert_array_equal(
        sparse.cell_snp_Ballele.toarray(), dense.cell_snp_Ballele
    )


def synthetic_ranges(
    n_snps: int, n_ranges: int, seed: int = 11
) -> tuple[np.ndarray, Any]:
    """SNP ids and filter ranges over a genome, both sorted as the loader needs.

    Shared with the benchmark module. The ids are `cnaster`'s own text form --
    `{chromosome}_{position}_{ref}_{alt}` -- because the filter parses them with
    `split("_")`, so a test handing it tuples would not exercise the parse.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    chromosomes = rng.integers(1, 23, n_snps)
    positions = rng.integers(0, 250_000_000, n_snps)
    order = np.lexsort((positions, chromosomes))
    snp_ids = np.array(
        [f"{chromosomes[k]}_{positions[k]}_A_T" for k in order], dtype=object
    )

    range_chromosomes = rng.integers(1, 23, n_ranges)
    range_starts = rng.integers(0, 250_000_000, n_ranges)
    range_order = np.lexsort((range_starts, range_chromosomes))

    ranges = pd.DataFrame(
        {
            "Chr": range_chromosomes[range_order],
            "Start": range_starts[range_order],
            "End": range_starts[range_order] + 2_000_000,
        }
    )

    return snp_ids, ranges


def range_filter_loop(unique_snp_ids: np.ndarray, ranges: Any) -> np.ndarray:
    """`cnaster.io.load_input_data`'s range filter, transcribed verbatim.

    Lines 740-772 of `io.py`, as the call the patch replaces. Transcribed
    rather than imported because it is inline in a 480-line function behind a
    configuration key, so there is no way to call it on its own -- which is
    also why nothing had ever run it.
    """
    num_ranges = ranges.shape[0]
    indicator_filter = np.array([True] * len(unique_snp_ids))
    j = 0

    for i in range(len(unique_snp_ids)):
        this_chr = int(unique_snp_ids[i].split("_")[0])
        this_pos = int(unique_snp_ids[i].split("_")[1])

        while j < num_ranges and (
            (ranges.Chr.to_numpy()[j] < this_chr)
            or (
                (ranges.Chr.to_numpy()[j] == this_chr)
                and (ranges.End.to_numpy()[j] <= this_pos)
            )
        ):
            j += 1

        if (
            j < num_ranges
            and (ranges.Chr.to_numpy()[j] == this_chr)
            and (ranges.Start.to_numpy()[j] <= this_pos)
            and (ranges.End.to_numpy()[j] > this_pos)
        ):
            indicator_filter[i] = False

    return indicator_filter


@pytest.mark.patch
@pytest.mark.parametrize(
    ("n_snps", "n_ranges"), [(1_000, 50), (10_000, 200), (2_000, 1)]
)
def test_the_vectorized_range_filter_drops_the_snps_the_loop_drops(
    n_snps: int, n_ranges: int
) -> None:
    """The same mask, element for element, at three shapes.

    Three because the filter's answer depends on the *arrangement* rather than
    the count: many ranges over few SNPs and one range over many are the two
    ends of the forward pointer's behaviour, and a single range is where an
    off-by-one at the array's end shows.

    Exactness is available here and a tolerance would be meaningless: the
    output is a boolean vector, so the two implementations either select the
    same SNPs or they do not.
    """
    from port.patch.io import _range_mask

    snp_ids, ranges = synthetic_ranges(n_snps, n_ranges)

    expected = range_filter_loop(snp_ids, ranges)
    realized = _range_mask(snp_ids, ranges)

    assert 0 < int((~expected).sum()) < n_snps, (
        "the instance removes everything or nothing, so agreement says little"
    )
    np.testing.assert_array_equal(realized, expected)


def _loaders() -> list[tuple[str, Any]]:
    """Both implementations, for the claims that hold of either.

    The three filter-file branches below are `cnaster`'s as much as the
    patch's, and neither had ever been executed. Parametrizing is what stops
    the patch's arrival from being the only reason they are covered: the same
    claim is put to both subjects, and a divergence names which one moved.
    """
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data as patched

    return [("cnaster", upstream), ("patch", patched)]


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_gene_file_removes_the_genes_it_names_and_no_others(
    name: str,
    load_input_data: Any,
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    gate_config: Any,
) -> None:
    """**`filter_gene_file`, which no run in this repository has ever taken.**

    `zenodo_sim_config.yaml` leaves it `None` and so does `tests/run_config.py`,
    so the branch ships unexercised. The claim is exact and comes from the
    fixture rather than from the loader: name ten planted genes, and the ten
    are gone and everything else is where it was.

    Asserted as the whole retained set rather than as a membership test on the
    ten, because a filter that removed them along with half the genome would
    pass the membership test.
    """
    _, pre_image, written, _ = planted_instance

    baseline = load_input_data(gate_config)
    doomed = sorted(str(name) for name in baseline.adata.var.index)[:10]

    gene_file = written.root / "filter_genes.txt"
    gene_file.write_text("\n".join(doomed) + "\n")

    # NB `str`, not the `Path`: `cnaster` logs `len(filter_gene_file)` and a
    #    `Path` has no length, so the loader raises `TypeError` before it
    #    filters anything. Pinned by the `bug` test below.
    filtered = load_input_data(gate_config, filter_gene_file=str(gene_file))

    assert {str(name) for name in filtered.adata.var.index} == {
        str(name) for name in baseline.adata.var.index
    } - set(doomed)
    np.testing.assert_array_equal(
        np.asarray(filtered.adata.layers["count"]),
        np.asarray(baseline.adata.layers["count"])[
            :,
            [
                column
                for column, name in enumerate(baseline.adata.var.index)
                if str(name) not in set(doomed)
            ],
        ],
    )


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_range_file_removes_the_snps_inside_the_ranges_it_names(
    name: str,
    load_input_data: Any,
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    gate_config: Any,
) -> None:
    """**`filter_range_file`, likewise never taken, and against planted SNPs.**

    The ranges are written around the positions the fixture planted, so which
    SNPs should go is known before the loader runs: every SNP whose coordinate
    falls inside a range, and only those. That is the claim, as set equality on
    the surviving ids, and it is what the vectorized mask has to reproduce in
    situ rather than on synthetic ranges.

    The allele matrices have to lose the same columns, which a mask applied to
    one array and not the other would not.
    """
    baseline = load_input_data(gate_config)

    chromosome = np.array(
        [int(str(snp).split("_")[0]) for snp in baseline.unique_snp_ids]
    )
    position = np.array(
        [int(str(snp).split("_")[1]) for snp in baseline.unique_snp_ids]
    )

    # NB a window around every third planted SNP, wide enough to be a range and
    #    narrow enough to leave its neighbours alone.
    centres = np.flatnonzero(np.arange(position.size) % 3 == 0)
    # NB `chr` prefixed: `get_filter_ranges` tests `"chr" in ranges.Chr.iloc[0]`
    #    and raises `TypeError: argument of type 'numpy.int64' is not iterable`
    #    on a file whose chromosomes are bare integers, so the prefixed form is
    #    the only one it reads.
    ranges = "\n".join(
        f"chr{chromosome[k]}\t{position[k] - 10}\t{position[k] + 10}" for k in centres
    )

    range_file = planted_instance[2].root / "filter_ranges.tsv"
    range_file.write_text(ranges + "\n")

    filtered = load_input_data(gate_config, filter_range_file=range_file)

    inside = np.zeros(position.size, dtype=bool)
    for k in centres:
        inside |= (chromosome == chromosome[k]) & (np.abs(position - position[k]) < 10)

    np.testing.assert_array_equal(
        filtered.unique_snp_ids, baseline.unique_snp_ids[~inside]
    )
    np.testing.assert_array_equal(
        filtered.cell_snp_Aallele, baseline.cell_snp_Aallele[:, ~inside]
    )
    np.testing.assert_array_equal(
        filtered.cell_snp_Ballele, baseline.cell_snp_Ballele[:, ~inside]
    )


@pytest.mark.end2end
@pytest.mark.parametrize(("name", "load_input_data"), _loaders())
def test_the_normal_index_file_annotates_the_spots_it_names(
    name: str,
    load_input_data: Any,
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    gate_config: Any,
) -> None:
    """**`normal_idx_file`, the third branch nothing runs.**

    The annotation is what `run_cnaster` would take as given rather than infer,
    so the claim is that the spots named are the spots marked: the planted
    balanced clone's barcodes in, the same barcodes out as `normal`, and every
    other spot `tumor`.
    """
    truth, _, written, _ = planted_instance

    balanced = int(
        np.argmax(
            [np.mean(truth.states[clone] == 0) for clone in range(truth.n_clones)]
        )
    )
    normal = [
        str(written.barcodes[spot]) for spot in np.flatnonzero(truth.labels == balanced)
    ]

    normal_file = written.root / "normal_idx.txt"
    normal_file.write_text("\n".join(normal) + "\n")

    annotated = load_input_data(gate_config, normal_idx_file=normal_file)

    marked = {
        str(barcode)
        for barcode, label in zip(
            annotated.adata.obs.index,
            annotated.adata.obs["tumor_annotation"],
            strict=True,
        )
        if label == "normal"
    }

    assert marked == set(normal)


@pytest.fixture(scope="module")
def outlier_configs(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
) -> dict[str, Path]:
    """The same instance's configuration with each outlier branch turned on.

    `tests/run_config.py` leaves both off, as `zenodo_sim_config.yaml` does, so
    these are written here rather than added there: turning one on for the
    whole suite would change what every other stage test is measuring.
    """
    import yaml

    from tests.run_config import run_cnaster_config

    truth, _, written, _ = planted_instance
    paths = {}

    for key in ("local_outlier_filter", "normalize_gene_outliers"):
        config = run_cnaster_config(written, truth)
        config["quality"][key] = True
        path = written.root / f"config_{key}.yaml"
        path.write_text(yaml.safe_dump(config))
        paths[key] = path

    return paths


@pytest.fixture
def installed(outlier_configs: dict[str, Path]) -> Any:
    """Install one of those configurations for the body of a test."""
    from cnaster.config import YAMLConfig, get_global_config, set_global_config

    def install(key: str) -> Any:
        set_global_config(None)
        set_global_config(YAMLConfig.from_file(outlier_configs[key]))

        return get_global_config()

    previous = get_global_config()
    try:
        yield install
    finally:
        set_global_config(None)
        set_global_config(previous)


@pytest.mark.end2end
def test_the_outlier_filter_leaves_every_planted_count_alone(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    installed: Any,
) -> None:
    """**`quality.local_outlier_filter`, a branch nothing had run, is a no-op here.**

    It zeroes the genes `LocalOutlierFactor` flags on per-gene UMI totals. On
    this instance it flags **none**, so the prediction is that the counts come
    back exactly as planted, and they do.

    That is not a null result about the fixture, which carries a 34-fold spread
    between the `unassigned_*` genes at 26,016 UMIs and the median bin gene at
    772. It is a statement about the filter: `n_neighbors` is hard-coded at 200
    against 138 genes, `scikit-learn` clamps it to `n_samples - 1`, and a local
    outlier factor taken over a nearly complete neighbourhood is one for every
    point. So the branch cannot flag anything on a dataset with fewer genes
    than its own neighbour count -- which it does not report, since it logs
    "Removed 0 outlier genes" as though it had looked.
    """
    from port.patch.io import load_input_data

    _, pre_image, written, _ = planted_instance

    filtered = load_input_data(installed("local_outlier_filter"))

    np.testing.assert_array_equal(
        np.asarray(filtered.adata.layers["count"]),
        _planted_alignment(pre_image, written, filtered),
    )


@pytest.mark.warning
def test_the_downsampler_misses_its_own_threshold_by_two_per_cent(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    installed: Any,
) -> None:
    """**`quality.normalize_gene_outliers` does nothing, by a 2.5 per cent margin.**

    It scales a gene at or above the 95th percentile of UMI totals down to
    `target = 0.05 * (total - top_total)`, and only where the gene is **above**
    that target. On this instance the seven top genes carry 26,016 UMIs each
    against a target of 26,680: the branch runs, compares, and scales nothing.

    Pinned as a warning rather than left as a passing identity, because the
    margin is 2.5 per cent of one number. A fixture redraw moves it, and then
    the arithmetic this test does not reach starts running with nothing
    checking it. What a positive case needs is a planted gene an order of
    magnitude above the rest, which `unsegment` cannot express today.
    """
    from port.patch.io import load_input_data

    _, pre_image, written, _ = planted_instance

    scaled = load_input_data(installed("normalize_gene_outliers"))
    expected = _planted_alignment(pre_image, written, scaled).astype(float)

    totals = expected.sum(axis=0)
    top = totals >= np.percentile(totals, 95)
    target = 0.05 * (totals.sum() - totals[top].sum())

    assert totals[top].max() < target, (
        f"the top gene at {totals[top].max():_.0f} now exceeds the target at "
        f"{target:_.0f}, so the downsampler fires and this test is the wrong one"
    )
    np.testing.assert_array_equal(
        np.asarray(scaled.adata.layers["count"], dtype=float), expected
    )


@pytest.mark.bug
def test_the_gene_filter_counts_the_path_rather_than_the_genes(
    planted_instance: tuple[CoreInferenceTruth, Any, WrittenInputs, Path],
    gate_config: Any,
) -> None:
    """**`len(filter_gene_file)` is the length of the filename (`io.py:725`).**

    ```python
    logger.info(f"Removing {len(filter_gene_file)} genes based on input file={filter_gene_file}.")
    ```

    The argument is the path, not the gene list, so what is logged as the
    number of genes removed is the number of **characters in the path** -- and
    a `pathlib.Path`, which has no length, raises `TypeError` before the filter
    runs at all. Either way the branch is unusable as it stands: a caller
    passing a `Path` loses the run, and one passing a `str` is told a number
    that has nothing to do with its data.

    The filtering itself is correct, which the `end2end` test above establishes
    on the same file. This pins the reporting around it, and the crash, and
    both are `cnaster`'s to fix.
    """
    from cnaster.io import load_input_data as upstream

    _, _, written, _ = planted_instance

    gene_file = written.root / "filter_genes_bug.txt"
    gene_file.write_text("gene_0_0\n")

    with pytest.raises(TypeError, match="has no len"):
        upstream(gate_config, filter_gene_file=gene_file)

    # NB the `str` form survives, and reports the path's character count as
    #    the number of genes: 10 characters here against the one gene named.
    filtered = upstream(gate_config, filter_gene_file=str(gene_file))

    assert "gene_0_0" not in set(map(str, filtered.adata.var.index))
