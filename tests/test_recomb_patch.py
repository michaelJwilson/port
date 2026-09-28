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
