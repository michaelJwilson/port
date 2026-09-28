"""Draw new samples from a version-3 TOML manifest (#445)."""

from __future__ import annotations

import os
from pathlib import Path

RESOURCE_FILES = ("hgTables_hg38_gencode.txt", "genetic_map_GRCh38_merged.tab.gz")
"""What `draw` reads from CalicoST's `GRCh38_resources`."""


def references() -> Path:
    """CalicoST's `GRCh38_resources`: `$PORT_GRCH38`, else the uv git checkout."""
    candidates = [os.environ.get("PORT_GRCH38", "")]
    candidates += sorted(
        str(p)
        for p in (Path.home() / ".cache/uv/git-v0/checkouts").glob(
            "*/*/GRCh38_resources"
        )
    )

    for candidate in candidates:
        if candidate and all((Path(candidate) / f).exists() for f in RESOURCE_FILES):
            return Path(candidate)

    msg = "CalicoST's GRCh38_resources not found; set $PORT_GRCH38"
    raise FileNotFoundError(msg)
