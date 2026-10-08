"""`port.extensions.config_audit` against `cnaster`'s shipped config and ours (#324).

Pins the AST scan, the findings #324 tabulates for `zenodo_sim_config.yaml` (`4adad4d`),
`run_config.py`'s stated findings, and `--audit-config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests import TESTS

SHIPPED = TESTS / "data" / "zenodo_sim_config.yaml"

UNUSED = {
    ("run.bafonly", "unread"),
    ("hmm.params", "unread"),
    ("betabinom.run_default", "unread"),
    ("int_copy_num.rdr_weight", "unread"),
    ("hmm.em_xtol", "solver"),
    ("hmm.em_xrtol", "solver"),
    ("hmrf.min_spots_per_clone", "floor"),
}
"""What both configurations share: keys kept for the mirror, and one floor."""


def _kinds(document: dict[str, Any], **keywords: Any) -> set[tuple[str, str]]:
    from port.extensions.config_audit import audit

    return {(f.key, f.kind) for f in audit(document, **keywords)}


@pytest.mark.infra
def test_the_scan_counts_code_and_not_comments_or_strings() -> None:
    """The AST scan counts `hmm.n_states`, not reads in a comment or string (`rdr_weight`,
    `hmrf.np_merge`).
    """
    from port.extensions.config_audit import cnaster_reads

    reads = cnaster_reads(("hmm", "hmrf", "int_copy_num", "quality"))

    assert ("hmm", "n_states") in reads
    assert ("quality", "min_normal_count_perbin") in reads
    assert ("int_copy_num", "ploidy") in reads
    assert ("int_copy_num", "rdr_weight") not in reads
    assert ("hmrf", "np_merge") not in reads


@pytest.mark.warning
def test_the_shipped_config_carries_what_324_tabulates() -> None:
    """The shipped config: four unread keys, two unused tolerances, a floor, two off
    (#448).
    """
    shipped = yaml.safe_load(SHIPPED.read_text())

    assert _kinds(shipped, check_paths=False) == UNUSED | {
        ("int_copy_num.nonbalance_bafdist", "disabled"),
        ("int_copy_num.nondiploid_rdrdist", "disabled"),
        # NB port's, stated for #518; `run_cnaster` ignores it.
        ("int_copy_num.merge_agreement", "port"),
    }


@pytest.mark.warning
@pytest.mark.merge
def test_the_test_config_carries_only_what_it_states(tmp_path: Path) -> None:
    """The test config: the mirror's unread keys and the floor; `max_total_copy` reported
    as `port` (#313).
    """
    from port.sim.inputs import write_tmp_inputs
    from port.sim.run_config import run_cnaster_config
    from port.sim.truth import dev_instance
    from port.sim.unsegment import unsegment

    truth = dev_instance()
    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), tmp_path
    )

    assert _kinds(run_cnaster_config(written, truth), check_paths=False) == UNUSED | {
        ("int_copy_num.max_total_copy", "port")
    }


@pytest.mark.infra
def test_a_start_params_length_that_disagrees_with_n_states_is_reported() -> None:
    document = {"hmm": {"n_states": 5}, "betabinom": {"start_params": "0.5,0.5"}}

    assert ("betabinom.start_params", "length") in _kinds(document, check_paths=False)


@pytest.mark.infra
def test_audit_config_lists_the_findings_and_runs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from port.scripts.run_cnaster import main

    config = tmp_path / "config.yaml"
    config.write_text(SHIPPED.read_text())

    assert main([str(config), "--audit-config"]) == 0

    printed = capsys.readouterr().out

    assert "unread   hmm.params" in printed
    assert "path     references.geneticmap_file" in printed


@pytest.mark.infra
def test_a_key_only_port_reads_is_reported_as_ports_not_as_unread() -> None:
    """`max_total_copy` is read by `--copy-cap` and by no live `cnaster` code."""
    from port.extensions.config_audit import PORT_READS, cnaster_reads

    document = {"int_copy_num": {"max_total_copy": 12}}

    assert ("int_copy_num", "max_total_copy") in PORT_READS
    assert ("int_copy_num", "max_total_copy") not in cnaster_reads(("int_copy_num",))
    assert _kinds(document, check_paths=False) == {
        ("int_copy_num.max_total_copy", "port")
    }
