"""A run's integer-clone rule and the configuration keys it reads (#331, #518, T- #817).

Port writes no table beside `cnaster`'s: what `cnv_states.tsv`,
`cnv_segments.tsv`, `cnv_binlevel.tsv`, `clone_labels_integer.tsv`,
`gene_segments.tsv` and `manifest.json` held is in `cnamaste.h5`
(`docs/cnamaste-h5.md`), and `clone_labels.tsv` is left as `cnaster` writes
it. A reader derives what those tables tabulated from the file: a bin's rate
in its clone is `exp(log_mu[Z] - logmu_shift)` of `/rdrbaf` (#613), the
integer clones are `/clone_assignment_int`.

`integer_clones` is the one rule merging clones by their decoded `(A, B)`: at
`int_copy_num.merge_agreement` of bins, 0.99 unless configured, for the run's
`/clone_assignment_int` (#518); at 1.0 for a run with no merge of its own
(`port.qa.audit.merged_clones`, #344).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "MERGE_AGREEMENT",
    "config_keys",
    "installed_keys",
    "integer_clones",
    "merge_agreement",
    "run_directories",
]


def run_directories(output_dir: Path, since: float | None = None) -> Iterator[Path]:
    """Each directory under `output_dir` holding a finished run's tables.

    Given `since`, a `time.time()`, only those whose `cnv_seglevel.tsv` was
    written at or after it: `output_dir` is shared by every configuration
    that names it, and a directory an earlier run left is not this run's
    (T- #617).
    """
    for table in sorted(Path(output_dir).rglob("cnv_seglevel.tsv")):
        if since is not None and table.stat().st_mtime < since:
            continue
        if any(table.parent.glob("rdrbaf_final_nstates*_smp.npz")):
            yield table.parent


MERGE_AGREEMENT = 0.99
"""The share of bins at which two integer profiles must agree to be one clone, unset.

0.99 because the split pair it exists to join agrees at 0.9993 on `dev_tree`
and every distinct pair on CalicoST easy, hard and `dev_tree` at 0.9863 or
less (#518); 1.0 joins only identical profiles (#344).
"""


def integer_clones(
    frame: pd.DataFrame, agreement: float = MERGE_AGREEMENT
) -> dict[str, str]:
    """Each clone id -> the smallest id whose integer copy profile it matches.

    `frame` is `cnv_seglevel.tsv`, or any table with `clone{c} A` and
    `clone{c} B` per bin. In id order, each clone joins the first earlier
    named clone whose `(A, B)` agree with its own at no less than
    `agreement` of the bins, and names itself otherwise; the smallest id
    names a group, so the normal clone keeps `0`. At 1.0, every bin (#344).

    Below 1.0 it is #518's merge: on `dev_tree` 60 x 50 without the
    Neyman-Pearson merge, one planted clone split by slice decodes alike at
    0.9993 of 2,895 bins, while every distinct pair on CalicoST easy, hard
    and `dev_tree` agrees at 0.9863 or less.
    """
    if not 0.0 < agreement <= 1.0:
        msg = f"merge agreement must be in (0, 1], got {agreement!r}"
        raise ValueError(msg)

    ids = [c.split()[0][len("clone") :] for c in frame.columns if c.endswith(" A")]
    ordered = sorted(
        ids, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c)
    )
    named: list[tuple[str, np.ndarray]] = []
    names: dict[str, str] = {}

    for clone in ordered:
        profile = frame[[f"clone{clone} A", f"clone{clone} B"]].to_numpy(dtype=int)
        match = next(
            (
                name
                for name, other in named
                if float(np.mean(np.all(profile == other, axis=1))) >= agreement
            ),
            None,
        )

        if match is None:
            named.append((clone, profile))
            match = clone

        names[clone] = match

    return names


def config_keys(config: Path | None) -> dict[str, Any]:
    """The configuration's copy caps, ploidy and state count, where set."""
    if config is None:
        return {}

    import yaml

    return _wanted(yaml.safe_load(Path(config).read_text()))


def installed_keys() -> dict[str, Any]:
    """`config_keys` of the configuration the run installed (`cnaster.config`), `{}` where none is."""
    from cnaster.config import YAMLConfig, get_global_config

    def plain(node: Any) -> Any:
        return (
            {k: plain(v) for k, v in vars(node).items()}
            if isinstance(node, YAMLConfig)
            else node
        )

    installed = get_global_config()
    return {} if installed is None else _wanted(plain(installed))


def _wanted(document: Any) -> dict[str, Any]:
    """The keys `config_keys` reads, wherever they sit in `document`."""
    wanted = {"n_states", "max_total_copy", "merge_agreement", "ploidy", "output_dir"}
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in wanted and not isinstance(value, dict):
                    found[key] = value
                walk(value)

    walk(document)
    return found


def merge_agreement(keys: dict[str, Any]) -> float:
    """`int_copy_num.merge_agreement` from `config_keys`, `MERGE_AGREEMENT` where unset (#518)."""
    stated = keys.get("merge_agreement")
    return MERGE_AGREEMENT if stated is None else float(stated)
