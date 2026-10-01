"""`dev_tree` r0 through `run_cnaster_port --sal`, one stage at a time (#467).

Each test judges one stage against r0's truth, so a failure names the stage.
The runs come from `tests.sim_stages`: one run per configuration, cached, so
the stages after the first cost a read.

- **Input:** the drawn counts carry each event's read-depth ratio. The
  referee for every later stage: a stage cannot recover what the data does
  not hold.
- **Clones:** `--sal` recovers the planted clones.
- **HMM:** with the planted clones held (`--oracle-start`), each event's
  fitted `log mu` is its planted depth. It is not: single-copy losses are
  fitted at copy-neutral LOH's depth (#471), pinned below.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

MIN_GENES = 100
"""Events with fewer genes are left out: their ratio's noise is the bound."""


def _events(truth: Any) -> list[tuple[int, int, int, str, int, int]]:
    """Every planted `(chr, start, end, clone, A, B)` that is not `(1, 1)`."""
    table = pd.read_csv(truth / "truth_acn_profile.tsv", sep="\t")
    names = [c[: -len("_A_copy")] for c in table.columns if c.endswith("_A_copy")]
    return [
        (
            int(r.chr),
            int(r.start),
            int(r.end),
            n,
            int(r[f"{n}_A_copy"]),
            int(r[f"{n}_B_copy"]),
        )
        for _, r in table.iterrows()
        for n in names
        if (int(r[f"{n}_A_copy"]), int(r[f"{n}_B_copy"])) != (1, 1)
    ]


@pytest.fixture(scope="module")
def r0() -> Any:
    from tests.sim_stages import r0

    return r0()


@pytest.fixture(scope="module")
def depth(r0: Any) -> dict[tuple[int, int, int, str], tuple[int, float]]:
    """Per planted event and clone: its genes, and its log read-depth ratio to normal."""
    import anndata as ad
    import scipy.sparse as sp

    from tests.sim_fixtures import references

    resources = references()

    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")

    genes = pd.read_csv(resources / "hgTables_hg38_gencode.txt", sep="\t")
    genes = genes.drop_duplicates("name2").set_index("name2")
    labels = pd.read_csv(r0 / "truth_clone_labels.tsv", sep="\t").set_index("barcode")

    matrices, barcodes = [], []

    for path in sorted(r0.glob("*/filtered_feature_bc_matrix.h5ad")):
        slice_ = ad.read_h5ad(path)
        matrices.append(sp.csr_matrix(slice_.X))
        barcodes.extend(slice_.obs_names)

    counts = sp.vstack(matrices).tocsr()
    label = labels["labels"].reindex(barcodes).to_numpy()
    position = genes.reindex(slice_.var_names)
    totals = {
        c: np.asarray(counts[label == c].sum(axis=0)).ravel() for c in np.unique(label)
    }

    ratios = {}

    for chrom, start, end, clone, _, _ in _events(r0):
        inside = (
            (position["chrom"] == f"chr{chrom}")
            & (position["cdsStart"] >= start)
            & (position["cdsStart"] < end)
        ).to_numpy()
        share = totals[clone][inside].sum() / totals[clone].sum()
        normal = totals["normal"][inside].sum() / totals["normal"].sum()
        ratios[(chrom, start, end, clone)] = (
            int(inside.sum()),
            float(np.log(share / normal)),
        )

    return ratios


@pytest.mark.merge
@pytest.mark.end2end
def test_the_r0_counts_carry_each_event_s_depth(
    r0: Any, depth: dict[tuple[int, int, int, str], tuple[int, float]]
) -> None:
    """Each event's log read-depth ratio is `log((A + B) / 2)`, to 0.2.

    Measured on r0 (`3381575a`, 60 x 50 per slice): losses -0.784 to -0.630,
    LOH -0.026 to +0.048, the chr11 gain +0.684. The worst is clone_2's chr7
    loss, 193 genes, 0.091 beyond `log(1/2)` (at 42 x 42 it was chr18's
    loss, 0.135). 0.2 is under
    a third of the 0.693 separating a loss from LOH, which is what the later
    stages need the data to carry. chr15's `(3, 1)` holds no gene (the
    acrocentric arm) and chr16's 31, so `MIN_GENES` leaves both out.
    """
    judged = 0

    for chrom, start, end, clone, a, b in _events(r0):
        genes, ratio = depth[(chrom, start, end, clone)]

        if genes < MIN_GENES:
            continue

        judged += 1
        assert abs(ratio - np.log((a + b) / 2)) <= 0.2, (chrom, clone, (a, b), ratio)

    assert judged >= 10, judged


@pytest.mark.release
@pytest.mark.end2end
def test_the_clone_stage_recovers_r0() -> None:
    """`--sal` on main recovers r0's four clones: ARI 0.9997 at e5ee447 (#470)."""
    from tests.sim_stages import stages

    run = stages("r0", ("--sal",))

    assert run.recovery["n_clones"] == 4
    assert run.recovery["ari"] >= 0.99, run.recovery


def _fitted(run: Any, truth: Any) -> dict[tuple[int, int, int, str], float]:
    """Per planted event and clone: the mean fitted `log mu` over its segments."""
    # NB the run's clones, merged where `port` merged them (#518, #613).
    spots = run.outputs.get("spot_labels.tsv")
    fitted = (
        run.outputs["clone_labels.tsv"][["barcode", "clone_label"]]
        if spots is None
        else spots[["barcode", "clone_label_decode"]].rename(
            columns={"clone_label_decode": "clone_label"}
        )
    )
    planted = pd.read_csv(truth / "truth_clone_labels.tsv", sep="\t")
    both = planted.merge(fitted, on="barcode")
    index = pd.crosstab(both["labels"], both["clone_label"]).idxmax(axis=1)
    segments = run.outputs["cnv_seglevel.tsv"]

    means = {}

    for chrom, start, end, clone, _, _ in _events(truth):
        rows = segments[
            (segments["CHR"] == chrom)
            & (segments["END"] > start)
            & (segments["START"] < end)
        ]
        means[(chrom, start, end, clone)] = float(
            rows[f"clone{int(index[clone])} logmu"].mean()
        )

    return means


@pytest.mark.release
@pytest.mark.bug
def test_the_hmm_fits_r0_s_losses_at_the_depth_of_its_loh(
    r0: Any, depth: dict[tuple[int, int, int, str], tuple[int, float]]
) -> None:
    """**#471:** with the planted clones held, a one-copy loss and a
    copy-neutral LOH share one fitted state.

    The counts separate them by 0.7 in `log` depth (above). The fit puts both
    within 0.1 of each other, near -0.14, at every state count and transition
    tried (7 and 5 states; `t` 0.9999999 and 0.9999), so the decode calls
    every loss `(0, 2)`. Fails when the fit separates them.
    """
    from tests.sim_stages import stages

    run = stages("r0", ("--sal",), oracle=True)
    fitted = _fitted(run, r0)
    losses, loh = [], []

    for chrom, start, end, clone, a, b in _events(r0):
        genes, _ = depth[(chrom, start, end, clone)]

        if genes < MIN_GENES:
            continue

        if a + b == 1:
            losses.append(fitted[(chrom, start, end, clone)])
        elif (a, b) in {(0, 2), (2, 0)}:
            loh.append(fitted[(chrom, start, end, clone)])

    assert losses
    assert loh
    assert abs(np.mean(losses) - np.mean(loh)) < 0.1, (losses, loh)
