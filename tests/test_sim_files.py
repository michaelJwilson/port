"""#460: committed compressed sample files against their pre-compression SHA-256, and `compress`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from port.sim.files import compress, located, read_bytes
from port.sim.fixtures import REPOSITORY

PLAIN = {
    "sim/normal_baseline.txt.gz": "1ec533480ec0435919a475a326fcc71781e3f30b90249920819f411297c2188e",
    "sim/numcnas1.2_cnasize5e7_ploidy2_random0/barcodes.txt.gz": "085e7ccaca66b2c4558cb8f23b6c84637709c210a68a8c4d5791e92637af8c1c",
    "sim/numcnas1.2_cnasize5e7_ploidy2_random0/spatial/tissue_positions_list.csv.gz": "3b8d5ccd050afe509d9bbb68180c27225c7c22631e0780a90396e966551c4d67",
    "sim/numcnas1.2_cnasize5e7_ploidy2_random0/truth_acn_profile.tsv.gz": "95104a9d8c4d09e249f8693f97be1565195d766396bc0416d92b3b91a316ed01",
    "sim/numcnas1.2_cnasize5e7_ploidy2_random0/truth_clone_labels.tsv.gz": "62c53e39388631890c7c3e142fc5b33d5882a0a5d65bd78bc170763acdbce466",
    "sim/numcnas1.2_cnasize5e7_ploidy2_random0/unique_snp_ids.npy.gz": "0d3f570e3209216d5f4fae6ced5d29aff0d6a1ef8aa72a9480279efe0ad64bdf",
    "sim/numcnas6.3_cnasize1e7_ploidy2_random0/barcodes.txt.gz": "085e7ccaca66b2c4558cb8f23b6c84637709c210a68a8c4d5791e92637af8c1c",
    "sim/numcnas6.3_cnasize1e7_ploidy2_random0/spatial/tissue_positions_list.csv.gz": "3b8d5ccd050afe509d9bbb68180c27225c7c22631e0780a90396e966551c4d67",
    "sim/numcnas6.3_cnasize1e7_ploidy2_random0/truth_acn_profile.tsv.gz": "46f6223f04d4398b56f315555dc6471ce07230a0907925d0254193843912b177",
    "sim/numcnas6.3_cnasize1e7_ploidy2_random0/truth_clone_labels.tsv.gz": "62c53e39388631890c7c3e142fc5b33d5882a0a5d65bd78bc170763acdbce466",
    "sim/numcnas6.3_cnasize1e7_ploidy2_random0/unique_snp_ids.npy.gz": "0d3f570e3209216d5f4fae6ced5d29aff0d6a1ef8aa72a9480279efe0ad64bdf",
}
"""SHA-256 of each committed `.gz`'s content, as committed plain before #460."""


@pytest.mark.snapshot
@pytest.mark.parametrize("name", sorted(PLAIN))
def test_each_compressed_file_holds_the_bytes_it_replaced(name: str) -> None:
    """Decompressed, a committed `.gz` is the file it replaced, byte for byte."""
    path = REPOSITORY / name

    assert hashlib.sha256(read_bytes(path.with_suffix(""))).hexdigest() == PLAIN[name]


@pytest.mark.analytic
def test_compression_round_trips_and_writes_the_same_bytes_twice(
    tmp_path: Path,
) -> None:
    """`compress` is deterministic and round-trips; `located` prefers the plain file."""
    source = tmp_path / "table.tsv"
    source.write_bytes(b"gene\tlambda\n" * 1000)
    first = compress(source, tmp_path / "first.tsv.gz").read_bytes()
    second = compress(source, tmp_path / "second.tsv.gz").read_bytes()

    assert first == second
    assert read_bytes(tmp_path / "first.tsv") == source.read_bytes()
    assert located(source) == source
