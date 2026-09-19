"""Type stub for the compiled `port.oxiport` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs. `mypy --strict` catches only the
direction where the stub is missing something a caller uses.

The array arguments are the CSR triple as `scipy` stores it, passed as plain
arrays rather than as a matrix so the kernel borrows them without a conversion
on either side.
"""

import numpy as np
import numpy.typing as npt

def double(x: int) -> int: ...
def csr_pair_row_sums(
    a_data: npt.NDArray[np.float64],
    a_indptr: npt.NDArray[np.int64],
    b_data: npt.NDArray[np.float64],
    b_indptr: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]: ...
def csr_positive_per_column(
    data: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
    n_cols: int,
) -> npt.NDArray[np.int64]: ...
