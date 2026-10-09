"""T- #817: a `--sal` run's `cnamaste.h5` against the run's own outputs; `run_plots` redraws them byte for byte."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from port.extensions import cnamaste as c
from port.qa import stage
from port.qa.audit import drawn_config
from port.scripts.run_cnaster import main as run_cnaster_main
from port.scripts.run_plots import main

from tests import ROOT

MANIFEST = Path("sim/manifests/dev_tree_1s_hard.toml")


@pytest.fixture(scope="module")
def output(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Run `run_cnaster_port --sal` on dev_tree_1s_hard r0 (`9ec90dc2`) at `SOURCE_DATE_EPOCH=0`."""

    root = tmp_path_factory.mktemp("run_plots")
    here, epoch = Path.cwd(), os.environ.get("SOURCE_DATE_EPOCH")
    os.chdir(ROOT)
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        member = next(stage.members(MANIFEST, root / "sim", n=1))
        assert member.hash == "9ec90dc2"
        run_cnaster_main(["--sal", str(drawn_config(member.sample, root / "run", {}))])
        yield root / "run" / "output"
    finally:
        os.chdir(here)
        if epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH")
        else:
            os.environ["SOURCE_DATE_EPOCH"] = epoch


@pytest.mark.merge
@pytest.mark.backend
def test_run_plots_draws_every_page_the_run_wrote_byte_for_byte(
    output: Path, tmp_path: Path
) -> None:
    """`run_plots` redraws all 19 PDFs of the run from the file alone, byte for byte."""

    assert main([str(output / "cnamaste.h5"), "--out", str(tmp_path)]) == 0

    wrote = sorted(p.relative_to(output) for p in output.rglob("*.pdf"))
    drawn = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.pdf"))
    assert len(wrote) == 19
    assert drawn == wrote
    assert [
        p for p in wrote if (output / p).read_bytes() != (tmp_path / p).read_bytes()
    ] == []


@pytest.mark.merge
@pytest.mark.smoke
def test_the_stages_are_what_the_run_wrote(output: Path) -> None:
    """Each stage group equals the file the run wrote for it, in run order."""

    h5 = output / c.FILE
    run = next(output.glob("clone*"))
    found = c.stages(h5)
    assert found[0] == "inputs"
    order = [found.index(g) for g in ("inputs", "adjacency", "baf", "rdrbaf", "clone_assignment", "integer_copy", "integer_clones")]  # fmt: skip
    assert order == sorted(order)
    assert list(c.levels(h5)) == [
        "phasing_min_snp_umis",
        "secondary_min_umi",
        "normal_baf_filter",
        "min_segment_normal_umi",
        "normal_candidates",
    ]
    assert sum(g.startswith("figures/") for g in found) == 19
    # NB pages hold references, not data; each level's counts are written once
    assert all(not c.read(h5, g)[0] for g in found if g.startswith("figures/"))
    assert sorted(g for g in found if g.startswith("counts/")) == [
        "counts/normal_candidates",
        "counts/phasing_min_snp_umis",
        "counts/secondary_min_umi",
    ]

    spots, _ = c.read(h5, "inputs")
    labels = pd.read_csv(run / "clone_labels.tsv", sep="\t", comment="#")
    at = pd.Index(spots["barcodes"]).get_indexer(labels["barcode"].astype(str))
    final, final_attrs = c.read(h5, "clone_assignment")
    column = "cnaster_clone_label" if "cnaster_clone_label" in labels else "clone_label"
    np.testing.assert_array_equal(final["assignment"][at], labels[column].to_numpy())

    seglevel = pd.read_csv(run / "cnv_seglevel.tsv", sep="\t", comment="#")
    copies, _ = c.read(h5, "integer_copy")
    for allele in ("A", "B"):
        expected = seglevel[[f"clone{k} {allele}" for k in copies["clones"]]].to_numpy()
        np.testing.assert_array_equal(copies[allele], expected)

    fit, fit_attrs = c.read(h5, "rdrbaf")
    with np.load(
        next(run.glob("rdrbaf_final_nstates*_smp.npz")), allow_pickle=True
    ) as written:
        for ours, theirs in (("log_mu", "new_log_mu"), ("p_binom", "new_p_binom")):
            np.testing.assert_array_equal(fit[ours], np.ravel(written[theirs]))
        # NB the stage's clones precede `reindex_clones`: one permutation apart
        pairs = {
            (int(b), int(a))
            for b, a in zip(fit["assignment"], final["assignment"], strict=True)
        }
        assert len({b for b, _ in pairs}) == len({a for _, a in pairs}) == len(pairs)
        reindexed = np.empty_like(fit["pred_cnv"])
        for b, a in pairs:
            reindexed[:, a] = fit["pred_cnv"][:, b]
        np.testing.assert_array_equal(reindexed, np.asarray(written["pred_cnv"]))
    assert fit_attrs["level"] == final_attrs["level"] == "normal_candidates"
    assert fit["pred_cnv"].shape[0] == len(seglevel)

    merged = pd.read_csv(run / "clone_labels_integer.tsv", sep="\t", comment="#")
    integer, _ = c.read(h5, "integer_clones")
    at = pd.Index(spots["barcodes"]).get_indexer(merged["barcode"].astype(str))
    np.testing.assert_array_equal(integer["integer_ids"][integer["assignment"]][at], merged["integer_clone_label"].to_numpy())  # fmt: skip
