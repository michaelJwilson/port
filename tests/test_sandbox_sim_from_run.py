"""#460: a version-3 manifest written from a run's outputs recovers what planted them.

`port.sandbox.sim_from_run` reads a run's `clone_labels.tsv` and
`cnv_segments.tsv`. Here those are written from a draw's own truth, so the
referee is the manifest the draw was made from: its clone count, its shared
and unique events, its states, its array and which clones each slice holds.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest
from port.sandbox.sim_from_run import read_run, to_toml
from port.sim.draw import Drawn, draw, extended, from_document
from port.sim.fixtures import SIM_ROOT, references

from tests.test_sim_draw import _manifest

MANIFESTS = SIM_ROOT / "manifests"


def _as_run(drawn: Drawn, into: Path) -> Path:
    """The draw's truth in a run's `clone_labels.tsv` and `cnv_segments.tsv`."""
    truth = pd.read_csv(drawn.path / "truth_clone_labels.tsv", sep="\t")
    profile = pd.read_csv(drawn.path / "truth_acn_profile.tsv", sep="\t")
    number = {clone: k for k, clone in enumerate(drawn.clones)}

    into.mkdir(parents=True)
    pd.DataFrame(
        {
            "barcode": truth["barcode"],
            "sample_id": truth["sample_id"],
            "x": truth["x"],
            "y": truth["y"],
            "clone_label": truth["labels"].map(number),
        }
    ).to_csv(into / "clone_labels.tsv", sep="\t", index=False)
    pd.concat(
        pd.DataFrame(
            {
                "clone": number[clone],
                "CHR": profile["chr"],
                "START": profile["start"],
                "END": profile["end"],
                "A": profile[f"{clone}_A_copy"],
                "B": profile[f"{clone}_B_copy"],
            }
        )
        for clone in drawn.clones
    ).to_csv(into / "cnv_segments.tsv", sep="\t", index=False)
    return into


@pytest.mark.end2end
def test_a_manifest_from_a_run_recovers_the_one_that_planted_it(
    tmp_path: Path,
) -> None:
    """dev_shared_unique on a 20 x 20 array: 3 clones, 1 shared and 2 unique events.

    With its offsets stated, the written manifest is one `port.sim.draw`
    reads, and every recovered state is one the source can plant.
    """
    resources = references()
    if resources is None:
        pytest.skip("CalicoST's GRCh38_resources not found; set $PORT_GRCH38")
    source = _manifest("dev_shared_unique")
    drawn = draw(source, tmp_path / "drawn", resources=resources)

    text = to_toml(read_run(_as_run(drawn, tmp_path / "run")))
    stated = text.replace("# offset = [?, ?]", "offset = [0.0, 0.0]").replace(
        'extends = "calicost_grch38.toml"',
        f'extends = "{MANIFESTS / "calicost_grch38.toml"}"',
    )
    (tmp_path / "from_run.toml").write_text(stated)
    written = tomllib.loads(text)
    parsed = from_document(extended(tmp_path / "from_run.toml"), MANIFESTS)

    assert written["cna"]["n_clones"] == source.cna["n_clones"] == 3
    assert written["cna"]["shared"] == source.cna["shared"] == 1
    assert written["cna"]["unique"] == source.cna["unique"] == 2
    assert {tuple(s) for s in written["cna"]["states"]} <= {
        tuple(s) for s in source.cna["states"]
    }
    assert (written["array"]["rows"], written["array"]["columns"]) == (20, 20)
    assert [set(s.clones) for s in parsed.slices] == [set(parsed.tumour)]
    assert "offset" not in "".join(
        line for line in text.splitlines() if not line.startswith("#")
    )
