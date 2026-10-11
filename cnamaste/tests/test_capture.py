"""The capture's file handling: hard-link deduplication and the staging of a drawn sample (Ticket#836).

`audit.capture.deduplicated` rewrites a staged file with each repeated dataset a hard link to its first copy; a
reader must not tell the two apart. `audit.capture.stage_inputs` stages a drawn sample (one directory per slice
beside a shared `snp/`) one sample sheet row per slice, each input linked unchanged.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from audit.capture import SLICE_INPUTS, SNP_INPUTS, TRUTH, deduplicated, input_files, stage_inputs, truth_files

PROJECT = Path(__file__).resolve().parents[1]

EASY = PROJECT / "tests" / "data" / "sim_2d4ce9a9.hdf5"


def _contents(path: Path) -> dict[str, Any]:
    """Every group's and dataset's attributes, and every dataset's dtype, shape and bytes, by name.

    Walked by name: `visititems` visits an object once, so a hard-linked dataset's second name would go unread."""
    found: dict[str, Any] = {}

    def walk(group: Any, prefix: str) -> None:
        found[prefix or "/"] = {k: repr(v) for k, v in group.attrs.items()}
        for name in group:
            obj, path = group[name], f"{prefix}/{name}"
            if isinstance(obj, h5py.Group):
                walk(obj, path)
            else:
                attrs = {k: repr(v) for k, v in obj.attrs.items()}
                found[path] = (attrs, obj.dtype.str, obj.shape, np.ascontiguousarray(obj[()]).tobytes())

    with h5py.File(path, "r") as h5:
        walk(h5, "")
    return found


def test_a_deduplicated_file_reads_as_its_source(tmp_path: Path) -> None:
    """Easy's staged file, deduplicated: the same names, attributes, dtypes, shapes and bytes, in no more bytes. The
    capture writes it deduplicated already, so a second pass is idempotent in size.

    Fails where a link points at a dataset that differs in bytes or attributes, or a dataset is dropped."""
    target = tmp_path / "linked.hdf5"
    links = deduplicated(EASY, target)
    assert links > 0
    assert _contents(target) == _contents(EASY)
    print(f"{links} links; {EASY.stat().st_size:_} -> {target.stat().st_size:_} bytes")
    assert target.stat().st_size <= EASY.stat().st_size


def _drawn_sample(root: Path) -> Path:
    """Two slices, `a` and `b`, beside a shared `snp/`, each input file holding its own name; the truth gzipped."""
    sample = root / "r0"
    for name in [f"{s}/{n}" for s in ("a", "b") for n in SLICE_INPUTS] + [f"snp/{n}" for n in SNP_INPUTS]:
        (sample / name).parent.mkdir(parents=True, exist_ok=True)
        (sample / name).write_text(name)
    for name in TRUTH:
        (sample / f"{name}.gz").write_bytes(gzip.compress(name.encode()))
    pd.DataFrame({"sample_id": ["a", "b"]}).to_csv(sample / "sample_sheet.tsv", sep="\t", index=False)
    return sample


def test_a_drawn_sample_stages_one_row_per_slice(tmp_path: Path) -> None:
    """Each slice's row names its own directory and the shared `snp/`; every input reads back as written.

    Fails where a slice is dropped or its row names another slice's directory, or a staged input is not its source."""
    sample = _drawn_sample(tmp_path)
    root = tmp_path / "staged"
    stage_inputs(sample, root, {"references": {}}, tmp_path / "resources")
    target = root / "inputs" / "r0"
    sheet = pd.read_csv(root / "sample_sheet.tsv", sep="\t")
    assert sheet["sample_id"].tolist() == ["a", "b"]
    assert sheet["spaceranger_dir"].tolist() == [str(target / "a"), str(target / "b")]
    assert sheet["snp_dir"].tolist() == [str(target / "snp")] * 2
    names = sorted(input_files(sample))
    assert names == sorted([f"{s}/{n}" for s in ("a", "b") for n in SLICE_INPUTS] + [f"snp/{n}" for n in SNP_INPUTS])
    assert all((target / n).read_text() == n for n in names)
    assert truth_files(sample) == {name: name.encode() for name in TRUTH}
