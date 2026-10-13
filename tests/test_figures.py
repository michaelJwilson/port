"""Two `cnaster` plotting entry points the pipeline never calls, against written-out referees.

`plot_loh_density`'s figure is compared in `tests/test_plot_loh_density.py` (#355, #103).
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
import yaml
from cnaster.plot_loh_density import nan_gaussian_filter1d

mpl.use("Agg")


SAMPLES = 4
"""Validation rows to synthesize: two groups of two, so a group mean exists."""


@pytest.mark.smoke
# NB one figure's form (#403): reruns where this module or the lock changes
@pytest.mark.deprecate
def test_the_smoother_fills_where_there_is_no_coverage() -> None:
    """`nan_gaussian_filter1d` fills a NaN with the valid Gaussian-weighted mean, to 1e-12."""

    data = np.array([[0.4, np.nan, 0.6, 0.5]])
    smoothed = nan_gaussian_filter1d(data, sigma=1.0, fill_value=0.5)

    # NB scipy's defaults: radius 4 sigma, row reflected at its ends
    radius = int(4.0 + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * offsets**2)
    row = data[0]
    padded = np.concatenate([row[::-1], row, row[::-1]])
    window = padded[row.size + 1 + offsets]
    valid = ~np.isnan(window)
    expected = np.sum(kernel[valid] * window[valid]) / np.sum(kernel[valid])

    assert smoothed[0, 1] == pytest.approx(expected, rel=1e-12)
    assert 0.4 < smoothed[0, 1] < 0.6


@pytest.mark.smoke
# NB one figure's form (#403): reruns where this module or the lock changes
@pytest.mark.deprecate
def test_the_validation_metrics_load_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_validation_stats` reads synthesized YAMLs; `plot_metrics` writes to the cwd."""
    monkeypatch.chdir(tmp_path)
    from cnaster.plot_validation_stats import load_validation_stats, plot_metrics

    rng = np.random.default_rng(2)
    for index in range(SAMPLES):
        sample = f"numcnas{1 + index % 2}.2_cnasize1e7_ploidy2_random{index}"
        (tmp_path / f"validation_stats_{sample}.yaml").write_text(
            yaml.safe_dump(
                {
                    "sample_id": sample,
                    "initialization": "rectangle",
                    "loglike": float(-1e4 * rng.random()),
                    "num_clones": 2,
                    "normal_rate": float(rng.random()),
                    "match_rate": float(rng.random()),
                    "correct_rate": float(rng.random()),
                    "ari": float(rng.random()),
                    "clone_mapping_success_rate": float(rng.random()),
                    "normal_recovery_rate": float(rng.random()),
                    "cna_recovery_rate": float(rng.random()),
                    "cna_false_positive_rate": float(rng.random()),
                }
            )
        )

    frame = load_validation_stats(tmp_path)

    assert len(frame) == SAMPLES
    assert set(frame.sample_id) == {
        f"numcnas{1 + index % 2}.2_cnasize1e7_ploidy2_random{index}"
        for index in range(SAMPLES)
    }

    plot_metrics(frame, "rectangle", output_dir=str(tmp_path))
