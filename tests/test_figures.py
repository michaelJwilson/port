"""The two plotting entry points the pipeline never calls.

`tests/test_run_cnaster_round_trip.py` draws nineteen figures and takes
`plot_copy_number_profile` to 96.73 per cent and `plot_genomic` to 69.51.
Two modules stay at zero because `run_cnaster` has no call to them:
`plot_loh_density`, which renders a density cloud of loss of heterozygosity,
and `plot_validation_stats`, which reads a directory of validation YAMLs that
some harness outside this repository writes.

**These assert that the figure renders, and nothing about what is in it.**
That is a deliberate exception to the rule against a test whose only claim is
the absence of an exception, taken on the maintainer's call: a plotting entry
point that raises is a broken pipeline, and the alternative -- byte
reproduction, the standard `CLAUDE.md` names for a figure -- needs the
matplotlib timestamp pinned first. #103 carries what these leave undone, and
it is most of what a figure test would be.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from tests.fixtures import CoreInferenceTruth, core_inference_truth

SAMPLES = 4
"""Validation rows to synthesize. Two groups of two, so a group mean exists."""


@pytest.fixture(scope="module")
def planted() -> CoreInferenceTruth:
    """A small instance: these render what they are given without fitting."""
    return core_inference_truth(
        n_clones=2, n_states=3, lattice=(8, 10), n_obs=30, n_segments=2, seed=3
    )


def _result(truth: CoreInferenceTruth) -> dict[str, Any]:
    """The `res_combine` the plots read, built from the planted truth.

    Three keys, and the plot is indifferent to whether they came from a fit:
    the assignment, the per-clone copy state path, and the allele shares.
    """
    return {
        "new_assignment": truth.labels,
        "pred_cnv": truth.states.T,
        "new_p_binom": truth.p_binom.reshape(-1, 1),
    }


# NB two tests stood here and are removed rather than repaired (#259).
#    They rendered `cnaster.plot_loh_density`, which `cnaster@port`
#    deleted outright -- the module is gone, not moved, so there is no
#    import to update and nothing left to render. Its superseded copy
#    had been parked in a string literal beside it (#270); the whole
#    file went with the cleanup.


@pytest.mark.smoke
def test_the_validation_metrics_load_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_validation_stats` reads a directory of YAMLs, and `plot_metrics` draws it.

    The YAMLs are synthesized here because nothing in this repository writes
    them: they come from a validation harness `cnaster` ships no caller for.
    The names follow the pattern `plot_metrics` parses --
    `numcnas{n}_cnasize{s}_ploidy{p}_random{r}` -- since a name it cannot
    parse leaves every extracted column NaN and the grouping empty.

    **`output_dir` is accepted and never read.** `plot_validation_stats.py:207`
    writes `Path(".") / f"{method}_validation.pdf"`, so the figure lands in the
    caller's working directory whatever is passed -- which is how this test
    first rewrote a tracked file in the repository root. The directory is
    changed rather than the argument trusted.
    """
    monkeypatch.chdir(tmp_path)
    import yaml
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
