"""CalicoST easy (`23989aa4`): 223 outlier genes, from the loader to the bins (#574).

`quality.local_outlier_filter` flags 223 genes carrying 49.4% of easy's UMIs.
Each test follows them one stage further, against `cnaster`'s own call:

- **Loader:** port's dense and sparse reads zero exactly those genes, and
  leave every other gene's counts equal to the unfiltered read.
- **Binning:** both `create_bin_ranges` calls of a `--sal` run see the same
  zeroed counts and cut the same bins as `cnaster`'s own function, 1,847 then
  1,690. Restoring the 223 genes' counts moves neither, so the bin count is
  decided upstream of binning, by the read depth the filter leaves.
- **Outputs:** the filter off reproduces #487's head exactly, 1,716 bins and
  phase-free exact altered 0.750; on, `main` reads 1,690 and 0.630. The 0.750
  was an unfiltered run, not a binning departure.
"""

from __future__ import annotations

import copy
import importlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.preprocessing

OUTLIERS = 223
"""Genes the filter flags on easy, logged alike by all three loaders (#574)."""

BINNING = "create_bin_ranges"
"""The driver's binning call, captured around whatever it is bound to."""


def _dense(x: Any) -> np.ndarray:
    import scipy.sparse as sp

    return np.asarray(x.toarray() if sp.issparse(x) else x)


def _config(sample: Any, root: Path, on: bool) -> tuple[Any, dict[str, Any]]:
    """`sample`'s config with the outlier filter `on`, set global, and the loader's arguments."""
    import yaml
    from cnaster.config import YAMLConfig, set_global_config

    from tests.sim_fixtures import write_sim_inputs

    path = write_sim_inputs(sample, root, {"quality.local_outlier_filter": on})
    config = YAMLConfig(yaml.safe_load(path.read_text()))
    set_global_config(config)
    return config, {
        "filter_gene_file": config.references.filtergenelist_file,
        "filter_range_file": config.references.filterregion_file,
        "min_snp_umis": config.quality.spot_min_snp_umis,
        "min_percent_expressed_spots": config.quality.min_percent_expressed_spots,
    }


def _raw(sample: Any) -> Any:
    """Easy's AnnData from `cnaster`'s loader with the outlier filter off."""
    from cnaster.io import load_input_data

    config, arguments = _config(sample, Path(tempfile.mkdtemp()), False)
    return load_input_data(config, **arguments).adata


def _zeroed(counts: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Genes with no counts in `counts` and some in `raw`."""
    return np.flatnonzero((counts.sum(0) == 0) & (raw.sum(0) > 0))


@pytest.fixture(scope="module")
def easy() -> Any:
    from tests.sim_audit import SAMPLES
    from tests.sim_fixtures import load_simulated

    return load_simulated(SAMPLES["easy"])


@pytest.mark.merge
@pytest.mark.patch
def test_the_loaders_zero_the_outlier_genes_and_no_other(
    easy: Any, tmp_path: Path
) -> None:
    """Dense and sparse reads equal `cnaster`'s bitwise: 223 genes zeroed, the rest raw."""
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data

    raw = _raw(easy)
    config, arguments = _config(easy, tmp_path, True)
    theirs = upstream(config, **arguments).adata
    counts = _dense(theirs.layers["count"])
    reference = _dense(
        raw[list(theirs.obs.index), list(theirs.var.index)].layers["count"]
    )
    zeroed = _zeroed(counts, reference)

    assert zeroed.size == OUTLIERS
    np.testing.assert_array_equal(
        np.delete(counts, zeroed, 1), np.delete(reference, zeroed, 1)
    )

    for sparse in (False, True):
        ours = load_input_data(config, **arguments, sparse_counts=sparse).adata
        assert list(ours.var.index) == list(theirs.var.index)
        assert list(ours.obs.index) == list(theirs.obs.index)
        np.testing.assert_array_equal(_dense(ours.layers["count"]), counts)


@pytest.fixture(scope="module")
def binned(easy: Any) -> tuple[Any, list[tuple[Any, Any, Any]]]:
    """One `--sal` run on easy, and each `create_bin_ranges` call's inputs and output."""
    from tests.sim_audit import run_arm
    from tests.sim_stages import DRIVER

    driver = importlib.import_module(DRIVER)
    current = getattr(driver, BINNING)
    calls: list[tuple[Any, Any, Any]] = []

    # NB the inputs are copied before the call: `create_bin_ranges` writes
    #    `bin_id` into the table it is given.
    def capture(*args: Any, **kwargs: Any) -> Any:
        inputs = (copy.deepcopy(args), copy.deepcopy(kwargs))
        result = current(*args, **kwargs)
        calls.append((*inputs, result.copy()))
        return result

    setattr(driver, BINNING, capture)

    try:
        recovery, _ = run_arm(easy, ["--sal", "--no-plots"])
    finally:
        setattr(driver, BINNING, current)

    return recovery, calls


@pytest.mark.release
@pytest.mark.patch
def test_both_binning_calls_cut_cnaster_s_bins_on_the_zeroed_counts(
    easy: Any, binned: tuple[Any, list[tuple[Any, Any, Any]]]
) -> None:
    """Each call: 223 genes zeroed, `cnaster`'s bins, and the same bins with them restored.

    1,847 bins at the phased segmentation, 1,690 after the normal-UMI merge.
    """
    from cnaster.omics import create_bin_ranges as upstream

    recovery, calls = binned
    raw = _raw(easy)

    assert len(calls) == 2
    assert recovery.bins == 1690

    for (args, kwargs, result), bins in zip(calls, (1847, 1690), strict=True):
        key = kwargs.get("key", "bin_id")
        adata = args[1]
        restored = raw[list(adata.obs.index), list(adata.var.index)]
        counts = _dense(adata.layers["count"])
        zeroed = _zeroed(counts, _dense(restored.layers["count"]))

        assert zeroed.size == OUTLIERS
        assert result[key].nunique() == bins

        replay = upstream(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        assert replay[key].equals(result[key])

        unfiltered = list(copy.deepcopy(args))
        unfiltered[1] = adata.copy()
        unfiltered[1].layers["count"] = restored.layers["count"].copy()
        again = upstream(*unfiltered, **copy.deepcopy(kwargs))
        assert again[key].nunique() == bins


@pytest.mark.release
@pytest.mark.snapshot
def test_the_outlier_filter_moves_easy_s_recovery_by_its_stated_amounts(
    easy: Any, binned: tuple[Any, list[tuple[Any, Any, Any]]]
) -> None:
    """CalicoST easy (`23989aa4`): off reproduces #487's head; on is `main`.

    Clones are recovered either way.

    | filter | bins | copy ARI | exact altered | phase-free |
    | --- | --- | --- | --- | --- |
    | off | 1,716 | 0.884 | 0.369 | 0.750 |
    | on  | 1,690 | 0.898 | 0.252 | 0.630 |
    """
    from tests.sim_audit import run_arm

    on, _ = binned
    off, _ = run_arm(
        easy, ["--sal", "--no-plots"], {"quality.local_outlier_filter": False}
    )

    for run, bins, copy_ari, altered, minor in (
        (on, 1690, 0.8984, 0.252, 0.6299),
        (off, 1716, 0.8844, 0.3692, 0.75),
    ):
        assert run.ari >= 0.98, run
        assert run.bins == bins, run
        assert run.copy_ari == pytest.approx(copy_ari, abs=1e-3), run
        assert run.exact_altered == pytest.approx(altered, abs=1e-3), run
        assert run.exact_altered_minor == pytest.approx(minor, abs=1e-3), run
