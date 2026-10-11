"""The inventory: every function `run_cnamaste` reaches has per-function rows, or a stated reason it has none.

`audit.callgraph.reached()` is the static graph from `run_cnamaste` (live
definitions, plots cut). `NOT_EXECUTED` lists the reached functions the easy
run does not execute (measured once with a profiler over a full run, plots
stubbed), each with the reason. Every other reached function has at least one
row in a `test_fn_*.py` table. `python -m test_fn_inventory` (from
`cnamaste/tests`) prints the inventory as a Markdown table.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import pytest

import test_defects
import test_fn_hmm
import test_fn_hmrf
import test_fn_integer
import test_fn_io
import test_fn_normal
import test_fn_omics
import test_fn_outputs
import test_fn_phasing
import test_fn_spatial
from audit.callgraph import Graph, dead_copies, reached
from audit.digest import Pins
from audit.fn import Row, snapshot

MODULES = (test_fn_io, test_fn_omics, test_fn_phasing, test_fn_spatial, test_fn_normal, test_fn_hmm, test_fn_hmrf, test_fn_integer, test_fn_outputs)

NOT_EXECUTED: dict[str, str] = {
    "annotation:load_clone_labels": "annotation.clone_label is None on easy",
    "annotation:load_clone_ranges": "annotation.clone_ranges is None on easy",
    "annotation:assign_clone_ranges": "annotation.clone_ranges is None on easy",
    "cna_hmrf_result:CnaHMRFResult.items": "reached only by name (`.items()` on a dict elsewhere); no caller",
    "cna_hmrf_result:CnaHMRFResult.values": "reached only by name (`.values` on a frame elsewhere); no caller",
    "he:join_tables_xy": "the staged sample has no H&E image; get_he_image returns before the join",
    "hmm_nophasing:hmm_nophasing.unpack_param_errors": "propagate_hmm_param_errors is False in both fits",
    "integer_copy:hill_climbing_integer_copynumber_oneclone": "only for a stated ploidy; run_cnamaste breaks after the first (None) ploidy",
    "spatial:banded": "admits_assignment holds for every RDR split on easy",
    "spatial:best_equal_partition": "initial_clone_index_baf is never None (initialize_clones sets it)",
    "spatio_genomic_counts:LockableMixin.lock": "reached only by name (`res_combine.lock()` is CnaHMRFResult's)",
    "spatio_genomic_counts:SpatioGenomicCounts.__getitem__": "the pipeline unpacks the container; nothing subscripts it",
    "spatio_genomic_counts:SpatioGenomicCounts.__setitem__": "nothing assigns into the container",
    "spatio_genomic_counts:SpatioGenomicCounts.copy": "reached only by name (`.copy()` on arrays and frames)",
    "spatio_genomic_counts:SpatioGenomicCounts.items": "reached only by name",
}
"""Reached in the static graph, not executed on easy: why. A row may still test one (e.g. unpack_param_errors)."""

ISSUES = (20, 30, 45, 46, 58, 80, 81, 84, 105, 113, 115, 122, 135, 136, 143, 146, 165, 172, 177, 180, 181, 182, 189, 191, 236, 240, 241, 244,
          267, 269, 293, 311, 320, 417, 440, 446, 449, 466, 468, 479, 483, 560, 561, 788, 799, 860, 869, 871)
"""Issues the per-function work places: each has a row here or in test_defects.py."""

DEDUP = ("dead copies of reached functions remain in the source (shadowed definitions and definitions inside string literals): "
         "the deduplication change comments out the differing unreachable copies and deletes the exact ones")


def rows() -> list[Row]:
    return [r for m in MODULES for kind in ("ORACLE", "INVARIANT", "CAPTURED") for r in getattr(m, kind, [])]


@pytest.fixture(scope="module")
def graph() -> Graph:
    return reached()


def test_every_reached_function_has_rows_or_a_reason(graph: Graph) -> None:
    """Static graph = tested + not executed on easy; no row names a function outside the graph."""
    keys, tested = graph.keys(), {r.function for r in rows()}
    assert not tested - keys, f"rows for functions run_cnamaste does not reach: {sorted(tested - keys)}"
    assert not set(NOT_EXECUTED) - keys, f"NOT_EXECUTED names unreached functions: {sorted(set(NOT_EXECUTED) - keys)}"
    missing = keys - tested - set(NOT_EXECUTED)
    assert not missing, f"reached and executed, without a row: {sorted(missing)}"


def test_every_row_kind_is_known_and_paired(graph: Graph) -> None:
    """Each row is one of the three kinds and states what it checks and where its input comes from."""
    for r in rows():
        assert r.kind in ("oracle", "invariant", "captured") and r.what and r.source, r.id


def test_live_definitions_are_the_reached_ones(graph: Graph) -> None:
    """Where a name is defined more than once, the graph entered the live definition, never a dead copy."""
    assert graph.shadowed, "the source has dead copies of reached functions"
    for key, dead in graph.shadowed.items():
        live = graph.find(key)
        assert live.line not in dead, f"{key}: the reached definition is a dead copy"
        assert sorted(dead + [live.line]) == dead_copies(live.module)[live.qualname], f"{key}: every copy is either live or dead"


@pytest.mark.xfail(strict=True, reason=DEDUP)
def test_no_dead_copies_remain(graph: Graph) -> None:
    """No reached function has a shadowed or string-literal copy."""
    assert not graph.shadowed, {k: v for k, v in graph.shadowed.items()}


def test_every_placed_issue_has_a_row() -> None:
    """Each issue in ISSUES is cited by a per-function row (`Ticket#N:`) or has its test_defects.py row."""
    cited = " ".join(r.defect or "" for r in rows()) + " " + " ".join(r.what for r in rows())
    defects = " ".join(r.issue for r in test_defects.ROWS)
    missing = [n for n in ISSUES if not re.search(rf"#{n}(?!\d)", cited + " " + defects)]
    assert not missing, f"issues with no row: {missing}"


ROWS = rows()


@pytest.mark.snapshot
@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_snapshot_returns(row: Row, ctx: Any, request: pytest.FixtureRequest) -> None:
    """Every row's function returns what its pin records, on every input the row gives it (`audit.fn.snapshot`).

    A snapshot is not validation: the pin is what the function returns today, defects included, with no
    judgement attached. A change means "state why", not "wrong": the PR that moves a pin says why the output
    moved, and re-pins with `--update-digests`. A strict xfail's fix shows as its pin moving and the xfail flipping.
    """
    snapshot(row, ctx, request)


@pytest.mark.snapshot
def test_snapshot_pins_are_the_rows(sim_hash: str, request: pytest.FixtureRequest) -> None:
    """The pins are exactly the rows': none missing, none for a row that no longer exists."""
    keys = {r.id for r in ROWS}
    assert len(keys) == len(ROWS), "row keys repeat"
    if request.config.getoption("--update-digests"):
        request.node.user_properties.append(("digest-keys", [sim_hash, sorted(keys)]))
        return
    pins = set(Pins.of(sim_hash))
    assert not keys - pins, f"rows without a pin (--update-digests): {sorted(keys - pins)}"
    assert not pins - keys, f"stale pins: {sorted(pins - keys)}"


def inventory(graph: Graph) -> list[tuple[str, ...]]:
    """function | module:line | reached from | input source | assertion kinds | known issue."""
    by_function: dict[str, list[Row]] = {}
    for r in rows():
        by_function.setdefault(r.function, []).append(r)
    out = []
    for fn in sorted(graph.nodes, key=lambda f: (f.module, f.line)):
        parent = graph.parents[fn]
        found = by_function.get(fn.key, [])
        sources = "; ".join(sorted({r.source for r in found})) or f"not executed: {NOT_EXECUTED.get(fn.key, '')}"
        kinds = ", ".join(sorted({r.kind for r in found}))
        leaf = fn.qualname.split(".")[-1]
        known = sorted({r.defect.split(":")[0] for r in found if r.defect} | {r.issue for r in test_defects.ROWS if r.function.endswith(leaf) or f".{leaf} " in r.function})
        via = (parent.key + (" (by name)" if fn in graph.by_name else "")) if parent else "(entry point)"
        out.append((fn.key, fn.where, via, sources, kinds, ", ".join(known)))
    return out


if __name__ == "__main__":
    g = reached()
    print("| function | module:line | reached from | input source | assertion kinds | known issue |")
    print("|---|---|---|---|---|---|")
    for line in inventory(g):
        print("| " + " | ".join(x.replace("|", "/") for x in line) + " |")
    sys.exit(0)
