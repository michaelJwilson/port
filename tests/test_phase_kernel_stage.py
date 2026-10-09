"""`get_sitewise_transmat` on the pipeline's derived genome, against the closed form
(#160).

At the shipped constants every entry the phasing recursion reads is `log 0.5`.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from port.sim.inputs import WrittenInputs, read_to_bins, written_config
from port.sim.run_config import PlantedInstance
from port.sim.truth import CoreInferenceTruth

pytestmark = pytest.mark.preprocessing

NU = 1.0
LOGPHASE_SHIFT = -2.0
MIN_PROB = 1.0e-2
"""`zenodo_sim_config.yaml`'s `phasing.nu`, `logphase_shift` and `min_prob`, restated.
"""

SATURATED = np.log(0.5)
"""A switch probability of one half in log space: `get_sitewise_transmat`'s clip."""


@pytest.fixture(scope="module")
def binned_genome(
    planted_instance: PlantedInstance,
) -> Iterator[tuple[WrittenInputs, Any, np.ndarray]]:
    """The prep chain, then `get_sitewise_transmat` over the bins it derived."""
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
    """Haldane on the interpolated cM positions, floored, shifted and clipped, without
    cnaster.
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
    """The wrapper equals Haldane on the pipeline's own coordinates, exactly; 40 bins."""
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
    """All 37 entries the forward pass reads are `log 0.5`; the 3 unsaturated are
    chromosome ends.

    Saturation sets in above ~80 kb at this map's 0.87 cM/Mb; candidate root cause for
    #122.
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
