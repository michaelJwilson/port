"""`port.sim.fixtures.realization_hash` reads decoded content, not storage (#595).

Referee: SHA-256 over stripped names and plain bytes, computed here.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest
from port.sim.fixtures import realization_hash

FILES = {
    "barcodes.txt": b"AAAC-1\nAAAG-1\n",
    "spatial/tissue_positions_list.csv": b"AAAC-1,1,0,0,0,0\nAAAG-1,1,0,1,0,1\n",
    "truth_clone_labels.tsv": b"barcode\tlabels\nAAAC-1\tnormal\nAAAG-1\tclone_1\n",
}
"""A fixture of three text inputs, one under a subdirectory."""


def _write(root: Path, *, plain: bool, packed: bool) -> Path:
    for name, data in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if plain:
            path.write_bytes(data)
        if packed:
            path.with_name(path.name + ".gz").write_bytes(gzip.compress(data, mtime=0))
    return root


@pytest.mark.snapshot
def test_plain_gzipped_and_both_read_one_hash(tmp_path: Path) -> None:
    """Plain, `.gz` and both layouts hash alike, to SHA-256 over stripped names and plain bytes computed here."""
    layouts = {
        "plain": (True, False),
        "packed": (False, True),
        "both": (True, True),
    }
    found = {
        layout: realization_hash(_write(tmp_path / layout, plain=p, packed=g))
        for layout, (p, g) in layouts.items()
    }

    digest = hashlib.sha256()
    # NB `os.walk` hashes a directory's files before its subdirectories'
    for name in ("barcodes.txt", "truth_clone_labels.tsv"):
        digest.update(name.encode())
        digest.update(FILES[name])
    digest.update(b"spatial/tissue_positions_list.csv")
    digest.update(FILES["spatial/tissue_positions_list.csv"])

    assert found == dict.fromkeys(layouts, digest.hexdigest()[:8])


@pytest.mark.warning
def test_plain_and_gzipped_copies_that_differ_are_refused(tmp_path: Path) -> None:
    """A `.gz` decoding to other bytes than its plain copy names the file."""
    root = _write(tmp_path, plain=True, packed=True)
    (root / "barcodes.txt.gz").write_bytes(gzip.compress(b"AAAC-1\n", mtime=0))

    with pytest.raises(ValueError, match=r"barcodes\.txt.* decode differently"):
        realization_hash(root)
