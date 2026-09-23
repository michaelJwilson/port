"""What a `run_cnaster` configuration states against what `cnaster` does with it (#324).

`cnaster`'s YAML carries no schema and no check, so a key can be read by
nothing, or govern something other than its name says, and the run proceeds
the same either way. `audit` reports those, per key, against the installed
`cnaster`:

- **unread**: no live module reads it. Reads are found by walking the AST of
  every installed module outside `deprecated/` and `sandbox/`, so a read in a
  comment or inside a string literal does not count, plus the indirect reads
  in :data:`INDIRECT`;
- **solver**: an `em_*` tolerance the configured solver does not take, asked of
  `cnaster.hmm_utils.get_em_solver_params` itself rather than restated here;
- **string**: a number YAML 1.1 loads as a string (`1e-4` is one);
- **length**: `betabinom.start_params` against `hmm.n_states`;
- **floor**: `hmrf.min_spots_per_clone` below the ICM's own hard-coded floor,
  which merges first (#81);
- **disabled**: `int_copy_num` thresholds no fitted state can cross;
- **path**: a file the configuration names that does not exist.

The audit reads; it changes nothing. `run_cnaster_port --audit-config` prints
it, and every run prints its count.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

__all__ = ["INDIRECT", "Finding", "audit", "cnaster_reads"]

INDIRECT: frozenset[tuple[str, str]] = frozenset(
    {
        # `hmm_initialize` is handed `config.hmm` and reads it one level down.
        ("hmm", "gmm_min_binom_prob"),
        ("hmm", "gmm_max_binom_prob"),
        # `get_em_solver_params` reads `em_*` by `getattr`; which ones depends
        # on the solver, and `audit` asks it rather than listing them here.
    }
)
"""Reads the AST walk cannot see, because the section is bound to a name first."""

NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

DEFAULT_TOTAL_COPY = 6
"""`cnaster.integer_copy`'s `max_total_copy`, which no key changes."""


@dataclass(frozen=True)
class Finding:
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

    copies = document.get("int_copy_num") or {}
    total = int(copies.get("max_total_copy") or DEFAULT_TOTAL_COPY)

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
