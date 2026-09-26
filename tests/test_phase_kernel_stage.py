"""`get_sitewise_transmat` on a derived genome, and what it says there (#160).

`tests/test_recomb.py` refereed the two functions underneath this one against
the closed form; `tests/test_sitewise_kernel.py` refereed the block the kernel
becomes. Neither runs the wrapper `run_cnaster` actually calls, which takes the
gene-SNP table the pipeline derived, the genetic map on disk, and the two
configured constants, and returns the per-bin log switch probability the
phasing HMM is written over.

Running it on the fixture's own genome is what shows that **every entry the
phasing recursion reads is one half** -- the two claims below, in that order:
first that the arithmetic is the closed form, then what the closed form
evaluates to at the shipped constants.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import CoreInferenceTruth
from tests.run_config import PlantedInstance
from tests.tmp_inputs import WrittenInputs, read_to_bins, written_config

pytestmark = pytest.mark.preprocessing

NU = 1.0
LOGPHASE_SHIFT = -2.0
MIN_PROB = 1.0e-2
"""`zenodo_sim_config.yaml`'s `phasing.nu`, `logphase_shift` and `min_prob`.

Restated rather than read back off the configuration: the second test's claim
is about what these values do, so a test that took them from the file would
pass whatever they became and stop saying it.
"""

SATURATED = np.log(0.5)
"""What a switch probability of one half is in log space.

`get_sitewise_transmat` clips at it: `min(log 0.5, log p - logphase_shift)`.
A site at this value says the phase after it is independent of the phase
before, which is the same as having no kernel at all.
"""


@pytest.fixture(scope="module")
def planted(planted_instance: PlantedInstance) -> CoreInferenceTruth:
    """The session's gate instance."""
    return planted_instance[0]


@pytest.fixture(scope="module")
def binned_genome(
    planted_instance: PlantedInstance,
) -> Iterator[tuple[WrittenInputs, Any, np.ndarray]]:
    """The prep chain, then `get_sitewise_transmat` over the bins it derived.

    The table is the one `run_cnaster` passes -- `form_gene_snp_table` through
    `create_bin_ranges` -- so the coordinates the genetic map is interpolated
    at are the pipeline's rather than ones this module chose.
    """
    from cnaster.recomb import get_sitewise_transmat

    _, _, written, config_path = planted_instance

    with written_config(config_path):
        table = read_to_bins(written, through="ranges").table
        kernel = np.asarray(
            get_sitewise_transmat(
                segment_key="bin_id",
                df_gene_snp=table,
                geneticmap_file=str(written.genetic_map),
                nu=NU,
                logphase_shift=LOGPHASE_SHIFT,
            )
        )

        yield written, table, kernel


def _closed_form(written: WrittenInputs, table: Any) -> np.ndarray:
    """The kernel, from the map and the bins, without `cnaster`.

    Haldane's mapping function on the interpolated centimorgan positions,
    floored at `min_prob`, taken to `min(log 0.5, log p - logphase_shift)`.
    Written out rather than imported so the comparison has two sides:
    `cnaster` walks its positions in a Python loop over interleaved
    `(start, end)` pairs, and this is the same model in array form.
    """
    genetic_map = pd.read_csv(written.genetic_map, sep="\t")

    grouped = table.dropna(subset=["bin_id"]).groupby("bin_id")
    first = grouped.agg({"CHR": "first", "START": "first"})
    last = grouped.agg({"CHR": "last", "END": "last"})

    def centimorgans(chromosome: Any, position: Any) -> float:
        contig = genetic_map[genetic_map.chrom == f"chr{int(chromosome)}"]

        return float(
            np.interp(position, contig.pos.to_numpy(), contig.pos_cm.to_numpy())
        )

    interleaved = np.empty(2 * len(first))
    interleaved[0::2] = [
        centimorgans(c, p) for c, p in zip(first.CHR, first.START, strict=True)
    ]
    interleaved[1::2] = [
        centimorgans(c, p) for c, p in zip(last.CHR, last.END, strict=True)
    ]

    chromosomes = np.repeat(first.CHR.to_numpy(), 2)
    distance = np.diff(interleaved, append=interleaved[-1])
    same_contig = np.append(chromosomes[1:] == chromosomes[:-1], False)

    switch = np.where(
        same_contig,
        np.maximum((1.0 - np.exp(-2.0 * NU * np.abs(distance))) / 2.0, MIN_PROB),
        MIN_PROB,
    )

    return np.asarray(np.minimum(np.log(0.5), np.log(switch) - LOGPHASE_SHIFT)[1::2])


@pytest.mark.oracle
def test_the_sitewise_kernel_is_the_mapping_function_on_the_derived_bins(
    planted: CoreInferenceTruth,
    binned_genome: tuple[WrittenInputs, Any, np.ndarray],
) -> None:
    """**The wrapper is Haldane on the pipeline's own coordinates, exactly.**

    The referee is the closed form evaluated independently -- the genetic map
    interpolated at each bin's first and last coordinate, Haldane's mapping
    function on the gap, floored, shifted and clipped -- so the comparison is
    between two implementations of one model rather than between a run and a
    recorded number.

    Exact, to `0.0`: the two differ in arrangement (a Python loop over
    interleaved pairs against an array) and not in arithmetic, so there is no
    error for a tolerance to absorb. One entry per bin, and the fixture's 40
    bins come back as 40.
    """
    written, table, kernel = binned_genome

    assert kernel.shape == (planted.n_obs,)
    np.testing.assert_allclose(
        kernel, _closed_form(written, table), rtol=0.0, atol=1e-12
    )


@pytest.mark.warning
def test_every_entry_the_recursion_reads_carries_no_phase_information(
    planted: CoreInferenceTruth,
    binned_genome: tuple[WrittenInputs, Any, np.ndarray],
) -> None:
    """**All 37 entries the forward pass consumes are `log 0.5`.**

    At the shipped constants the kernel says the phase after a bin is
    independent of the phase before it, everywhere it is used, which is the
    same as having no kernel: the bins are 200 kb apart, the fixture's map puts
    that at 0.174 cM, Haldane gives `p = 0.147`, `logphase_shift = -2`
    multiplies it by `e^2 = 7.39` to 1.09, and the clip at one half takes it to
    `p = 0.5`.

    **The three entries that are not saturated are the three the recursion
    never reads.** `forward_lattice` runs `for t in range(1, le)` over each
    chromosome with `idx = cumlen + t - 1`, so the last index of every
    chromosome is skipped -- and that is exactly where
    `compute_numbat_phase_switch_prob` leaves `min_prob`, since it fills the
    transition from each site to the next and a chromosome's last site has no
    next within its own chain. Realized: the unsaturated indices are
    `[9, 30, 39]` against lengths `[10, 21, 9]`, and the two sets are equal.

    So the boundary value is padding rather than a claim about contiguity, and
    the finding is the saturation alone. The threshold is where
    `log p - logphase_shift` meets `log 0.5`, at `p = 1/(2e^2) = 0.0677`, which
    is `0.035 cM` -- about **80 kb** at this map's 0.87 cM/Mb. Bins wider than
    that carry nothing, and `zenodo_sim_config` ships 5 Mb as the bin cap.

    This is the kernel `initial_phase_given_partition` and the phasing HMM are
    written over, and it is a candidate root cause for #122, where the phasing
    recovers no haplotype at all. Pinned rather than asserted away: what the
    constants should be is `cnaster`'s question and the paper's.
    """
    _, _, kernel = binned_genome

    saturated = np.isclose(kernel, SATURATED)

    read_by_recursion: list[int] = []
    start = 0
    for length in np.asarray(planted.lengths):
        read_by_recursion += list(range(start, start + int(length) - 1))
        start += int(length)

    assert saturated[read_by_recursion].all(), (
        "an entry the recursion reads now carries phase information"
    )
    np.testing.assert_array_equal(
        np.flatnonzero(~saturated),
        np.setdiff1d(np.arange(planted.n_obs), read_by_recursion),
    )
    np.testing.assert_allclose(
        kernel[~saturated], np.log(MIN_PROB) - LOGPHASE_SHIFT, rtol=0.0, atol=1e-12
    )
