"""`run_calicost`: CalicoST on `run_cnaster_port`'s configuration (#347).

What is pinned here:

- the translated configuration carries the `run_cnaster` value of every key
  the two programs share, and CalicoST's own reader reads it back (`infra`);
- the scorer maps CalicoST's bins back to the planted ones, and merges clones
  by their integer profile (`analytic`);
- a whole CalicoST run on the dev instance recovers the planted clones
  (`end2end`, `release`: it needs the `calicost` extra, which CI does not
  install, and runs for minutes).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

SHARED = {
    "n_clones": ("hmrf", "n_clones"),
    "n_clones_rdr": ("hmrf", "n_clones_rdr"),
    "min_spots_per_clone": ("hmrf", "min_spots_per_clone"),
    "min_avgumi_per_clone": ("hmrf", "min_avgumi_per_clone"),
    "tumorprop_threshold": ("hmrf", "tumorprop_threshold"),
    "max_iter_outer": ("hmrf", "max_iter_outer"),
    "spatial_weight": ("hmrf", "spatial_weight"),
    "n_states": ("hmm", "n_states"),
    "t": ("hmm", "t"),
    "t_phaseing": ("hmm", "t_phaseing"),
    "fix_NB_dispersion": ("hmm", "fix_NB_dispersion"),
    "shared_NB_dispersion": ("hmm", "shared_NB_dispersion"),
    "fix_BB_dispersion": ("hmm", "fix_BB_dispersion"),
    "shared_BB_dispersion": ("hmm", "shared_BB_dispersion"),
    "max_iter": ("hmm", "max_iter"),
    "tol": ("hmm", "tol"),
    "gmm_random_state": ("hmm", "gmm_random_state"),
    "nu": ("phasing", "nu"),
    "logphase_shift": ("phasing", "logphase_shift"),
    "npart_phasing": ("phasing", "npart_phasing"),
    "min_percent_expressed_spots": ("quality", "min_percent_expressed_spots"),
    "min_snpumi_perspot": ("quality", "spot_min_snp_umis"),
    "secondary_min_umi": ("quality", "secondary_min_snp_umi"),
    "geneticmap_file": ("references", "geneticmap_file"),
    "hgtable_file": ("references", "hgtable_file"),
    "nonbalance_bafdist": ("int_copy_num", "nonbalance_bafdist"),
    "nondiploid_rdrdist": ("int_copy_num", "nondiploid_rdrdist"),
}
"""CalicoST key -> the `run_cnaster` section and key it takes its value from."""


def _document(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    from tests.fixtures import core_inference_truth
    from tests.run_config import write_run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

    truth = core_inference_truth(n_obs=40, lattice=(6, 5), seed=3)
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )
    config = write_run_cnaster_config(written, truth)
    return yaml.safe_load(config.read_text()), config


@pytest.mark.infra
def test_every_shared_key_carries_the_run_cnaster_value(tmp_path: Path) -> None:
    """27 keys equal, the inputs are the sample's, and one initialization runs."""
    from port.scripts.run_calicost import calicost_config

    document, _ = _document(tmp_path)
    config = calicost_config(document)
    sheet = pd.read_csv(document["paths"]["sample_sheet"], sep=r"\s+").iloc[0]

    for key, (section, name) in SHARED.items():
        assert config[key] == document[section][name], key

    assert config["spaceranger_dir"] == sheet["spaceranger_dir"]
    assert config["snp_dir"] == sheet["snp_dir"]
    assert config["output_dir"] == document["paths"]["output_dir"] + "_calicost"
    assert config["num_hmrf_initialization_end"] == (
        config["num_hmrf_initialization_start"] + 1
    )
    assert config["maxspots_pooling"] == 1
    assert config["np_threshold"] == -math.inf


@pytest.mark.infra
def test_calicost_reads_back_the_written_configuration(tmp_path: Path) -> None:
    """CalicoST's own parser returns every translated value, typed."""
    pytest.importorskip("calicost")
    from port.scripts.run_calicost import (
        calicost_config,
        compatible,
        write_calicost_config,
    )

    document, _ = _document(tmp_path)
    config = calicost_config(document)
    path = write_calicost_config(config, tmp_path / "calicost.txt")

    with compatible():
        from calicost.arg_parse import read_configuration_file

        read = read_configuration_file(str(path))

    for key, value in config.items():
        assert read[key] == value, key


@pytest.mark.infra
def test_a_colon_in_a_value_is_refused(tmp_path: Path) -> None:
    """CalicoST splits every line on `:`, so such a value would be read cut short."""
    from port.scripts.run_calicost import write_calicost_config

    with pytest.raises(ValueError, match="cannot read a ':'"):
        write_calicost_config({"output_dir": "C:/run"}, tmp_path / "c.txt")


@pytest.mark.infra
def test_the_shims_are_put_back(tmp_path: Path) -> None:
    """`compatible()` leaves numpy, scipy and pandas as it found them."""
    import scipy.sparse
    from port.scripts.run_calicost import compatible

    before = (
        hasattr(np, "NAN"),
        hasattr(scipy.sparse.spmatrix, "A"),
        hasattr(pd.Series, "nonzero"),
    )

    with compatible():
        assert np.isnan(getattr(np, "NAN"))  # noqa: B009
        assert pd.Series([False, True]).nonzero()[0].tolist() == [1]

    assert before == (
        hasattr(np, "NAN"),
        hasattr(scipy.sparse.spmatrix, "A"),
        hasattr(pd.Series, "nonzero"),
    )


@pytest.mark.analytic
def test_a_merged_bin_maps_back_to_each_planted_bin_it_covers() -> None:
    """Rows spanning two planted genes cover both; a gene no row spans is -1."""
    from tests.fixtures import core_inference_truth
    from tests.recovery_audit import planted_rows
    from tests.tmp_inputs import GENE_LENGTH, GENE_SPACING

    truth = core_inference_truth(n_obs=40, lattice=(6, 5), seed=3)
    first = int(truth.lengths[0])
    seglevel = pd.DataFrame(
        {
            "CHR": [1, 1, 2],
            "START": [0, 2 * GENE_SPACING, 0],
            "END": [GENE_SPACING + GENE_LENGTH, 3 * GENE_SPACING + GENE_LENGTH, 0],
        }
    )
    rows = planted_rows(seglevel, truth)
    expected = np.full(int(np.sum(truth.lengths)), -1)
    expected[[0, 1]] = 0
    expected[[2, 3]] = 1
    expected[first] = 2

    np.testing.assert_array_equal(rows, expected)


@pytest.mark.analytic
def test_clones_of_one_integer_profile_merge_to_the_smallest() -> None:
    """Equal `(A, B)` at every bin is one clone; one differing B keeps two."""
    from tests.recovery_audit import integer_clones

    a = np.array([[1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 2]])
    b = np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]])

    np.testing.assert_array_equal(integer_clones(a, b), [0, 1, 0, 3])


@pytest.mark.end2end
@pytest.mark.release
def test_calicost_recovers_the_planted_clones_of_the_dev_instance() -> None:
    """CalicoST, aligned, on the dev instance at the figures' configuration.

    Tolerances are set from the measured run in #347's pull request.
    """
    pytest.importorskip("calicost")
    import matplotlib as mpl

    from tests.fixtures import dev_instance
    from tests.recovery_audit import run_arm

    mpl.use("Agg")
    recovery, _ = run_arm(dev_instance(), ["--no-figures"], calicost=True)

    assert recovery.ari >= 0.5, recovery


def _l_shaped() -> np.ndarray:
    """A 20 x 20 grid less its top-right quadrant: 300 spots, one block empty."""
    xs, ys = np.meshgrid(np.arange(20), np.arange(20))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    return coords[~((coords[:, 0] >= 10) & (coords[:, 1] >= 10))]


@pytest.mark.bug
@pytest.mark.release
def test_calicosts_initializer_does_not_terminate_on_an_l_shaped_clone() -> None:
    """`rectangle_initialize_initial_clone` loops forever on an L of 300 spots.

    Four clones get four blocks, so every redraw is a permutation and the
    empty block is always some clone's (`utils_hmrf.py:216`). Run in a child
    with a 20 s limit, because the defect is that it never returns; written
    to fail the day it does. `cnaster` #248 is the same defect.
    """
    import subprocess
    import sys

    pytest.importorskip("calicost")
    script = (
        "import numpy as np, sys, types\n"
        "t = types.ModuleType('turtle'); t.reset = lambda: None\n"
        "sys.modules['turtle'] = t\n"
        "from calicost.utils_hmrf import rectangle_initialize_initial_clone\n"
        "xs, ys = np.meshgrid(np.arange(20), np.arange(20))\n"
        "c = np.column_stack([xs.ravel(), ys.ravel()])\n"
        "c = c[~((c[:, 0] >= 10) & (c[:, 1] >= 10))]\n"
        "rectangle_initialize_initial_clone(c, 4, random_state=0)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", script], timeout=20, check=True)


@pytest.mark.infra
def test_the_guard_refuses_the_l_and_passes_a_square_through() -> None:
    """The refusal names the block and the floor; a square grid is CalicoST's."""
    pytest.importorskip("calicost")
    from port.scripts.run_calicost import (
        UnterminatedInitialization,
        compatible,
        terminating,
    )

    xs, ys = np.meshgrid(np.arange(20), np.arange(20))
    square = np.column_stack([xs.ravel(), ys.ravel()])

    with compatible():
        from calicost import calicost_main

        original = calicost_main.rectangle_initialize_initial_clone(
            square, 4, random_state=0
        )

        with terminating():
            with pytest.raises(UnterminatedInitialization, match="0 spots"):
                calicost_main.rectangle_initialize_initial_clone(
                    _l_shaped(), 4, random_state=0
                )
            checked = calicost_main.rectangle_initialize_initial_clone(
                square, 4, random_state=0
            )

    for ours, theirs in zip(checked, original, strict=True):
        np.testing.assert_array_equal(ours, theirs)


@pytest.mark.infra
def test_the_aligned_palette_colours_every_pair_up_to_the_cap() -> None:
    """CalicoST's own colours kept; `(5, 2)`, which failed a run, now has one."""
    pytest.importorskip("calicost")
    from port.scripts.run_calicost import _palette, compatible

    with compatible():
        from calicost.utils_plotting import get_full_palette

        theirs, _ = get_full_palette()
        ours, ordered = _palette(12)

    assert (5, 2) not in theirs
    assert all(ours[pair] == colour for pair, colour in theirs.items())
    assert {
        (major, total - major)
        for total in range(1, 13)
        for major in range(total, (total - 1) // 2, -1)
    } <= set(ours)
    assert len(ordered) == len(set(ordered)) == len(ours)


@pytest.mark.infra
def test_the_aligned_palette_is_callable_while_installed(tmp_path: Path) -> None:
    """Called through CalicoST's module inside `aligned`, it returns, not recurses."""
    pytest.importorskip("calicost")
    from port.scripts.run_calicost import aligned, compatible

    document, _ = _document(tmp_path)
    document["int_copy_num"]["max_total_copy"] = 12

    with compatible():
        from calicost import utils_plotting

        with aligned(document):
            palette, _ = utils_plotting.get_full_palette()

    assert (5, 2) in palette
