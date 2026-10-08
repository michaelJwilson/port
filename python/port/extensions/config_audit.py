"""What a `run_cnaster` configuration states against what `cnaster` does with it (#324).

`audit` reports, per key, against the installed `cnaster`: **port** (read only
by a `port` patch), **unread**, **solver** (an `em_*` the solver ignores),
**string** (a number YAML loads as a string), **length**, **floor** (below the
ICM's hard-coded floor, #81), **disabled** (an uncrossable `int_copy_num`
threshold), **path** (missing file). Read only.
"""

from __future__ import annotations

import ast
import inspect
import re
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

from port.extensions.integer_copy import DEFAULT_MAX_TOTAL_COPY

__all__ = ["INDIRECT", "PORT_READS", "Finding", "audit", "cnaster_reads"]

INDIRECT: frozenset[tuple[str, str]] = frozenset(
    {
        # `hmm_initialize` is handed `config.hmm` and reads it one level down.
        ("hmm", "gmm_min_binom_prob"),
        ("hmm", "gmm_max_binom_prob"),
        # `em_*` are read by `getattr`; `audit` asks the solver which.
    }
)
"""Reads the AST walk cannot see, because the section is bound to a name first."""

PORT_READS: dict[tuple[str, str], str] = {
    ("int_copy_num", "max_total_copy"): "--copy-cap (#313)",
    ("int_copy_num", "merge_agreement"): "clone_labels_integer.tsv (#518)",
    ("quality", "min_segment_mb"): "the read-depth segment floor (#551)",
    ("quality", "min_segment_normal_umi"): "the read-depth segment floor (#551)",
}
"""Keys `cnaster` never reads that a `port` patch does, and the flag that reads them."""

NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

DEFAULT_TOTAL_COPY = DEFAULT_MAX_TOTAL_COPY
"""`cnaster.integer_copy`'s `max_total_copy`, which no key changes."""


class Finding(NamedTuple):
    """One key, what is wrong with it, and why."""

    key: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind:<8} {self.key}: {self.detail}"


def _live_modules() -> list[Path]:
    import cnaster

    # NB a namespace package: no `__file__`, one entry on `__path__`.
    root = Path(next(iter(cnaster.__path__)))
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if not {"deprecated", "sandbox"} & set(path.relative_to(root).parts)
    ]


@cache
def cnaster_reads(sections: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """Every `<name>.<section>.<key>` attribute read in live `cnaster` code."""
    found: set[tuple[str, str]] = set(INDIRECT)

    for path in _live_modules():
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in sections
            ):
                found.add((node.value.attr, node.attr))

    return frozenset(found)


def _em_keys_used(document: dict[str, Any]) -> set[str]:
    """The `em_*` keys `get_em_solver_params` reads for the configured solver."""
    from cnaster import config as cnaster_config
    from cnaster.hmm_utils import get_em_solver_params

    hmm = document.get("hmm") or {}
    probe = {key: 0 for key in hmm if str(key).startswith("em_")}
    previous = cnaster_config._global_config

    try:
        cnaster_config.set_global_config(
            cnaster_config.YAMLConfig({"hmm": {"solver": hmm.get("solver"), **probe}})
        )
        taken = get_em_solver_params()
    except (ValueError, AttributeError):
        return set(probe)
    finally:
        cnaster_config.set_global_config(previous)

    return {f"em_{key}" for key in taken}


def _icm_floor() -> int:
    from cnaster.icm import icm_sweep_deque

    return int(inspect.signature(icm_sweep_deque).parameters["min_clone_spots"].default)


def audit(document: dict[str, Any], *, check_paths: bool = True) -> list[Finding]:
    """Every finding for a loaded configuration, in file order."""
    sections = tuple(k for k, v in document.items() if isinstance(v, dict))
    reads = cnaster_reads(sections)
    em_used = _em_keys_used(document)
    findings: list[Finding] = []

    for section in sections:
        for key, value in document[section].items():
            name = f"{section}.{key}"

            if str(key).startswith("em_") and section == "hmm":
                if key not in em_used:
                    findings.append(
                        Finding(
                            name,
                            "solver",
                            f"unread by solver {document['hmm'].get('solver')!r}",
                        )
                    )
            elif (section, key) in PORT_READS and (section, key) not in reads:
                findings.append(
                    Finding(
                        name,
                        "port",
                        f"cnaster ignores it; run_cnaster_port reads it under "
                        f"{PORT_READS[section, key]}",
                    )
                )
            elif (section, key) not in reads:
                findings.append(
                    Finding(name, "unread", "no live cnaster code reads it")
                )

            if isinstance(value, str) and NUMBER.match(value.strip()):
                findings.append(
                    Finding(
                        name, "string", f"{value!r} loads as a string, not a number"
                    )
                )

    hmm = document.get("hmm") or {}
    betabinom = document.get("betabinom") or {}
    starts = betabinom.get("start_params")

    if starts is not None and "n_states" in hmm:
        count = len(str(starts).split(","))
        if count != int(hmm["n_states"]):
            findings.append(
                Finding(
                    "betabinom.start_params",
                    "length",
                    f"{count} values for hmm.n_states = {hmm['n_states']}",
                )
            )

    floor = _icm_floor()
    minimum = (document.get("hmrf") or {}).get("min_spots_per_clone")

    if minimum is not None and int(minimum) < floor:
        findings.append(
            Finding(
                "hmrf.min_spots_per_clone",
                "floor",
                f"{minimum} does not govern: the ICM merges clones under "
                f"{floor} spots first, a default no key changes (#81)",
            )
        )

    from port.patch.integer_copy import stated_total

    copies = document.get("int_copy_num") or {}

    # NB parsed as `--copy-cap`: `"none"` is no cap, `0` is refused (#466).
    try:
        total = stated_total(copies.get("max_total_copy")) or DEFAULT_TOTAL_COPY
    except ValueError as error:
        findings.append(Finding("int_copy_num.max_total_copy", "invalid", str(error)))
        total = DEFAULT_TOTAL_COPY

    bafdist = copies.get("nonbalance_bafdist")

    if bafdist is not None and float(bafdist) >= 0.5:
        findings.append(
            Finding(
                "int_copy_num.nonbalance_bafdist",
                "disabled",
                f"{bafdist} >= 0.5: |p - 0.5| can never exceed it",
            )
        )

    # NB `mu = (A + B) / 2`, so under the cap `|mu - 1| <= total / 2 - 1`.
    reach = total / 2 - 1
    rdrdist = copies.get("nondiploid_rdrdist")

    if rdrdist is not None and float(rdrdist) >= reach:
        findings.append(
            Finding(
                "int_copy_num.nondiploid_rdrdist",
                "disabled",
                f"{rdrdist} >= {reach}: no decodable state (A + B <= {total}) has "
                f"|mu - 1| above it",
            )
        )

    if check_paths:
        for section in ("paths", "references"):
            for key, value in (document.get(section) or {}).items():
                # NB written by the run, not read by it.
                if key in {"output_dir", "perf_path"} or not isinstance(value, str):
                    continue
                if value.lower() != "none" and not Path(value).exists():
                    findings.append(
                        Finding(f"{section}.{key}", "path", f"{value} does not exist")
                    )

    return findings
