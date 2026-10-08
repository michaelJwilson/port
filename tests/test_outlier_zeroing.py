"""CalicoST easy (`2d4ce9a9`): 223 outlier genes followed from the loader to the bins (T-
#574).

Referee: `cnaster`'s own loader and `create_bin_ranges`; off reproduces PR- #487's head
(T- #593).
"""

from __future__ import annotations

import copy
import importlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _config(sample: Any, root: Path, on: bool) -> Iterator[tuple[Any, dict[str, Any]]]:
    """`sample`'s config with the outlier filter `on`, global while open, and the loader's
    arguments.
    """
    import yaml
    from port.sim.fixtures import write_sim_inputs
    from port.sim.inputs import written_config

    path = write_sim_inputs(sample, root, {"quality.local_outlier_filter": on})
    with written_config(yaml.safe_load(path.read_text())) as config:
        yield (
            config,
            {
                "filter_gene_file": config.references.filtergenelist_file,
                "filter_range_file": config.references.filterregion_file,
                "min_snp_umis": config.quality.spot_min_snp_umis,
                "min_percent_expressed_spots": config.quality.min_percent_expressed_spots,
            },
        )


def _raw(sample: Any) -> Any:
    """Easy's AnnData from `cnaster`'s loader with the outlier filter off."""
    from cnaster.io import load_input_data

    with _config(sample, Path(tempfile.mkdtemp()), False) as (config, arguments):
        return load_input_data(config, **arguments).adata


def _zeroed(counts: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Genes with no counts in `counts` and some in `raw`."""
    return np.flatnonzero((counts.sum(0) == 0) & (raw.sum(0) > 0))


@pytest.fixture(scope="module")
def easy() -> Any:
    from port.sim.fixtures import SAMPLES, load_simulated

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
    with _config(easy, tmp_path, True) as (config, arguments):
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


@pytest.mark.merge
@pytest.mark.patch
def test_with_the_flag_off_the_loaders_keep_every_outlier_gene(
    easy: Any, tmp_path: Path
) -> None:
    """With the filter off, dense and sparse reads equal `cnaster`'s with the 223 genes
    counted.
    """
    from cnaster.io import load_input_data as upstream
    from port.patch.io import load_input_data

    with _config(easy, tmp_path / "on", True) as (on, on_arguments):
        flagged = upstream(on, **on_arguments).adata
    with _config(easy, tmp_path / "off", False) as (config, arguments):
        theirs = upstream(config, **arguments).adata
        counts = _dense(theirs.layers["count"])
        zeroed = _zeroed(
            _dense(flagged.layers["count"]),
            _dense(
                theirs[list(flagged.obs.index), list(flagged.var.index)].layers["count"]
            ),
        )

        assert zeroed.size == OUTLIERS
        assert (counts[:, zeroed].sum(axis=0) > 0).all()

        for sparse in (False, True):
            ours = load_input_data(config, **arguments, sparse_counts=sparse).adata
            assert list(ours.var.index) == list(theirs.var.index)
            assert list(ours.obs.index) == list(theirs.obs.index)
            np.testing.assert_array_equal(_dense(ours.layers["count"]), counts)


@pytest.fixture(scope="module")
def binned(easy: Any) -> tuple[Any, list[tuple[Any, Any, Any]]]:
    """One `--sal` run on easy, and each `create_bin_ranges` call's inputs and output."""
    from port.qa.audit import audit_sample

    from tests.sim_stages import DRIVER

    driver = importlib.import_module(DRIVER)
    current = getattr(driver, BINNING)
    calls: list[tuple[Any, Any, Any]] = []

    # NB copied first: `create_bin_ranges` writes `bin_id` into its input.
    def capture(*args: Any, **kwargs: Any) -> Any:
        inputs = (copy.deepcopy(args), copy.deepcopy(kwargs))
        result = current(*args, **kwargs)
        calls.append((*inputs, result.copy()))
        return result

    setattr(driver, BINNING, capture)

    try:
        recovery, _ = audit_sample(
            easy, ["--sal", "--no-plots"], {"quality.local_outlier_filter": True}
        )
    finally:
        setattr(driver, BINNING, current)

    return recovery, calls


@pytest.mark.release
@pytest.mark.patch
def test_both_binning_calls_cut_cnaster_s_bins_on_the_zeroed_counts(
    easy: Any, binned: tuple[Any, list[tuple[Any, Any, Any]]]
) -> None:
    """Each binning call sees 223 genes zeroed and cuts `cnaster`'s bins (1,847, then
    1,690), unchanged if restored.
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
    """Filter off reproduces PR- #487's head (1,716 bins, phase-free 0.750); on gives 1,690
    and 0.630.
    """
    from port.qa.audit import audit_sample

    on, _ = binned
    off, _ = audit_sample(
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
