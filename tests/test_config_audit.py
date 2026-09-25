"""`port.extensions.config_audit` against `cnaster`'s shipped config and ours (#324).

- the AST scan finds reads the live code makes and not reads that sit in a
  comment or a string literal (`infra`);
- the shipped `zenodo_sim_config.yaml`, vendored at `cnaster` `4adad4d`,
  carries exactly the findings #324 tabulates (`warning`: shipped values
  that assert more than the code does with them);
- `tests/run_config.py` carries exactly the findings its docstring states,
  so a key `cnaster` starts or stops reading fails here (`warning`);
- `run_cnaster_port --audit-config` lists them and runs nothing (`infra`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

SHIPPED = Path(__file__).parent / "data" / "zenodo_sim_config.yaml"

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
    """`hmm.n_states` is read; `rdr_weight` and `hmrf.np_merge` only look read.

    `int_copy_num.rdr_weight`'s one read is a comment (`integer_copy.py:144`)
    and `hmrf.np_merge`'s sit inside a string literal in `run_cnaster`, so a
    grep counts both and the AST counts neither.
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
    """Four unread keys, two unused tolerances, three strings, a floor, two off."""
    shipped = yaml.safe_load(SHIPPED.read_text())

    assert _kinds(shipped, check_paths=False) == UNUSED | {
        ("hmm.em_xtol", "string"),
        ("hmm.em_ftol", "string"),
        ("hmm.em_xrtol", "string"),
        ("int_copy_num.nonbalance_bafdist", "disabled"),
        ("int_copy_num.nondiploid_rdrdist", "disabled"),
    }


@pytest.mark.warning
def test_the_test_config_carries_only_what_it_states(tmp_path: Path) -> None:
    """The mirror's unread keys and the floor; nothing disabled, nothing a string.

    `int_copy_num.max_total_copy` (#313) is `port`'s: `cnaster` never reads it,
    so a plain `run_cnaster` ignores the 12 the test config states, while
    `run_cnaster_port --copy-cap` applies it. Reported as `port`, not `unread`.
    """
    from tests.fixtures import dev_instance
    from tests.run_config import run_cnaster_config
    from tests.tmp_inputs import write_tmp_inputs
    from tests.unsegment import unsegment

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
