"""The phase-switch kernel from the segment lineage, against `cnaster`'s (#438).

Three referees. Where `cnaster` reads the map correctly -- contigs whose
string order is their numeric order -- the drop-in is `cnaster`'s kernel
bitwise away from contig boundaries (`patch`). On a 22-chromosome map it is
a per-pair computation written from the definition (`oracle`). And
`cnaster`'s reading of chr2-9 is pinned as the defect it is (`bug`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NU = 1.0
SHIFT = 0.0


def _map(path: Path, contigs: range, rate: float = 1.0) -> Path:
    """A map at `rate` cM/Mb with a jitter, markers every 500 kb to 60 Mb."""
    rng = np.random.default_rng(11)
    rows = []

    for contig in contigs:
        positions = np.arange(0, 60_000_001, 500_000)
        steps = rng.uniform(0.5, 1.5, positions.size - 1) * rate * 0.5
        cm = np.concatenate(([0.0], np.cumsum(steps)))
        rows += [
            {"chrom": f"chr{contig}", "pos": int(p), "pos_cm": float(c)}
            for p, c in zip(positions, cm, strict=True)
        ]

    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _blocks(contigs: range) -> pd.DataFrame:
    """Twenty blocks per contig, 50-150 kb wide, every 1-3 Mb.

    Each block is a gene, a SNP inside it, and a second gene, so its first
    and last rows are genes: the extent `cnaster` reads from its rows and the
    one read here from its genes are the same, and the comparison is of the
    map alone.
    """
    rng = np.random.default_rng(13)
    rows = []
    block = 0

    for contig in contigs:
        starts = np.cumsum(rng.integers(1_000_000, 3_000_000, 20))
        for start in starts:
            width = int(rng.integers(50_000, 150_000))
            s = int(start)
            rows += [
                {
                    "CHR": contig,
                    "START": s,
                    "END": s + 10,
                    "is_interval": True,
                    "block_id": block,
                },
                {
                    "CHR": contig,
                    "START": s + 5,
                    "END": s + 6,
                    "is_interval": False,
                    "block_id": block,
                },
                {
                    "CHR": contig,
                    "START": s + width - 10,
                    "END": s + width,
                    "is_interval": True,
                    "block_id": block,
                },
            ]
            block += 1

    return pd.DataFrame(rows)


@pytest.mark.cnaster
@pytest.mark.patch
@pytest.mark.usefixtures("cnaster_config")
def test_the_kernel_is_cnasters_where_cnaster_reads_the_map(tmp_path: Path) -> None:
    """Contigs 1 and 2 sort alike as strings and integers: every within-contig entry equal.

    The contig's last entry differs by design: `log 1/2` here, `cnaster`'s
    `min_prob` there.
    """
    from cnaster.recomb import get_sitewise_transmat as upstream
    from port.patch.recomb import get_sitewise_transmat

    contigs = range(1, 3)
    path = _map(tmp_path / "map.tsv", contigs)
    table = _blocks(contigs)

    theirs = upstream("block_id", table.copy(), path, NU, SHIFT)
    ours = get_sitewise_transmat("block_id", table.copy(), path, NU, SHIFT)

    last = np.r_[table.groupby("block_id").CHR.first().diff().to_numpy()[1:] != 0, True]

    np.testing.assert_array_equal(ours[~last], theirs[~last])
    np.testing.assert_array_equal(ours[last], np.log(0.5))


@pytest.mark.oracle
@pytest.mark.usefixtures("cnaster_config")
def test_the_kernel_is_haldane_over_each_contigs_own_map(tmp_path: Path) -> None:
    """On 22 chromosomes, every within-contig entry is `log((1 - exp(-2 nu d)) / 2)` to 1e-12.

    The referee interpolates each position with `np.interp` on its own
    contig's rows, and walks the blocks pair by pair.
    """
    from cnaster.config import get_global_config
    from port.patch.recomb import get_sitewise_transmat

    contigs = range(1, 23)
    path = _map(tmp_path / "map.tsv", contigs)
    table = _blocks(contigs)
    frame = pd.read_csv(path, sep="\t")
    floor = get_global_config().phasing.min_prob

    ours = get_sitewise_transmat("block_id", table, path, NU, SHIFT)

    edges = table.groupby("block_id").agg(
        CHR=("CHR", "first"), START=("START", "first"), END=("END", "last")
    )
    expected = []

    for k in range(len(edges)):
        if k + 1 == len(edges) or edges.CHR.iloc[k + 1] != edges.CHR.iloc[k]:
            expected.append(np.log(0.5))
            continue

        rows = frame[frame.chrom == f"chr{edges.CHR.iloc[k]}"]
        cm_end = np.interp(edges.END.iloc[k], rows.pos, rows.pos_cm)
        cm_start = np.interp(edges.START.iloc[k + 1], rows.pos, rows.pos_cm)
        p = max((1.0 - np.exp(-2.0 * NU * (cm_start - cm_end))) / 2.0, floor)
        expected.append(min(np.log(0.5), np.log(p) - SHIFT))

    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=0.0)


@pytest.mark.bug
@pytest.mark.usefixtures("cnaster_config")
def test_cnaster_reads_chr2_to_9_as_chr1s_last_centimorgan(tmp_path: Path) -> None:
    """`get_reference_recomb_rates` sorts `chrom` as strings; the cursor assumes integers.

    Every within-contig entry on chr2-9 is `cnaster`'s floor, and on no other
    contig; the drop-in's are Haldane's there as everywhere.
    """
    from cnaster.config import get_global_config
    from cnaster.recomb import get_sitewise_transmat as upstream
    from port.patch.recomb import get_sitewise_transmat

    contigs = range(1, 23)
    path = _map(tmp_path / "map.tsv", contigs)
    table = _blocks(contigs)
    floor = np.log(get_global_config().phasing.min_prob) - SHIFT

    chrom = table.groupby("block_id").CHR.first().to_numpy()
    within = np.r_[chrom[1:] == chrom[:-1], False]

    theirs = upstream("block_id", table.copy(), path, NU, SHIFT)
    ours = get_sitewise_transmat("block_id", table.copy(), path, NU, SHIFT)

    stuck = sorted({int(c) for c in chrom[within & (theirs == floor)]})
    assert stuck == list(range(2, 10))
    assert not np.any(ours[within] == floor)


def _cnaster_log_switch(distance: np.ndarray, shift: float, floor: float) -> np.ndarray:
    """`cnaster`'s per-bin law (`recomb.py:56-66, :158`): Haldane over cM, floor, times `e^-shift`."""
    p = np.maximum((1.0 - np.exp(-2.0 * NU * distance)) / 2.0, floor)
    return np.minimum(np.log(0.5), np.log(p) - shift)


def _composed(log_ab: np.ndarray, log_bc: np.ndarray) -> np.ndarray:
    """`p_ac` from `p_ab` and `p_bc` on a two-state chain: `1 - 2 p = (1 - 2 p_ab)(1 - 2 p_bc)`."""
    return np.log(
        (1.0 - (1.0 - 2.0 * np.exp(log_ab)) * (1.0 - 2.0 * np.exp(log_bc))) / 2.0
    )


@pytest.mark.analytic
def test_the_composable_law_composes_over_any_binning() -> None:
    """Splitting a distance into two bins changes nothing (#449), to 1e-12 relative."""
    from port.extensions.segments import composable_log_switch

    rng = np.random.default_rng(449)
    first, second = rng.uniform(1e-4, 5.0, 200), rng.uniform(1e-4, 5.0, 200)

    whole = composable_log_switch(first + second, NU, -2.0)
    split = _composed(
        composable_log_switch(first, NU, -2.0), composable_log_switch(second, NU, -2.0)
    )

    np.testing.assert_allclose(split, whole, rtol=1e-12)


@pytest.mark.bug
@pytest.mark.parametrize(
    ("distance", "one_bin", "two_bins"),
    [(0.05, 0.3516, 0.2954), (0.002, 0.0739, 0.1369)],
)
def test_cnasters_law_depends_on_the_binning(
    distance: float, one_bin: float, two_bins: float
) -> None:
    """With `cnaster`'s shift of -2 and floor of 0.01, two bins imply another switch rate (#449).

    Fails when the law composes. The `e^2` factor lowers the composed
    probability (0.05 cM: 0.352 in one bin, 0.295 in two); the floor raises
    it (0.002 cM: 0.074 in one bin, 0.137 in two).
    """
    whole = _cnaster_log_switch(np.array([distance]), -2.0, 0.01)
    half = _cnaster_log_switch(np.array([distance / 2]), -2.0, 0.01)
    split = _composed(half, half)

    assert float(np.exp(whole[0])) == pytest.approx(one_bin, abs=1e-4)
    assert float(np.exp(split[0])) == pytest.approx(two_bins, abs=1e-4)


@pytest.mark.analytic
@pytest.mark.usefixtures("cnaster_config")
def test_the_kernel_takes_the_composable_law_only_when_installed(
    tmp_path: Path,
) -> None:
    """`composable_switch()` gives `composable_log_switch` within contigs; outside, `cnaster`'s law."""
    from port.extensions.segments import composable_log_switch
    from port.patch.recomb import composable_switch, get_sitewise_transmat

    contigs = range(1, 3)
    path = _map(tmp_path / "map.tsv", contigs)
    table = _blocks(contigs)
    frame = pd.read_csv(path, sep="\t")
    edges = table.groupby("block_id").agg(
        CHR=("CHR", "first"), START=("START", "first"), END=("END", "last")
    )

    default = get_sitewise_transmat("block_id", table.copy(), path, NU, -2.0)
    with composable_switch():
        composable = get_sitewise_transmat("block_id", table.copy(), path, NU, -2.0)

    for k in range(len(edges) - 1):
        if edges.CHR.iloc[k + 1] != edges.CHR.iloc[k]:
            assert composable[k] == default[k] == np.log(0.5)
            continue

        rows = frame[frame.chrom == f"chr{edges.CHR.iloc[k]}"]
        d = np.interp(edges.START.iloc[k + 1], rows.pos, rows.pos_cm) - np.interp(
            edges.END.iloc[k], rows.pos, rows.pos_cm
        )
        np.testing.assert_allclose(
            composable[k], composable_log_switch(np.array([d]), NU, -2.0)[0], rtol=1e-12
        )

    assert not np.array_equal(composable, default)
