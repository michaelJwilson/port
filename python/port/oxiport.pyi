"""Type stub for the compiled `port.oxiport` Rust extension.

Hand-written, so it can drift: keep the signatures here matching the
`#[pyfunction]` definitions in src/lib.rs. `mypy --strict` catches only the
direction where the stub is missing something a caller uses.
"""

import numpy as np
from numpy.typing import NDArray

def double(x: int) -> int: ...
def forward_lattice(
    lengths: NDArray[np.int64],
    log_transmat: NDArray[np.float64],
    log_startprob: NDArray[np.float64],
    log_emission: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def backward_lattice(
    lengths: NDArray[np.int64],
    log_transmat: NDArray[np.float64],
    log_emission: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def forward_lattice_phased(
    lengths: NDArray[np.int64],
    log_transmat: NDArray[np.float64],
    log_startprob: NDArray[np.float64],
    log_emission: NDArray[np.float64],
    log_sitewise_transmat: NDArray[np.float64],
    penalize_phase_only_on_same_cnv: bool,
) -> NDArray[np.float64]: ...
def backward_lattice_phased(
    lengths: NDArray[np.int64],
    log_transmat: NDArray[np.float64],
    log_emission: NDArray[np.float64],
    log_sitewise_transmat: NDArray[np.float64],
    penalize_phase_only_on_same_cnv: bool,
) -> NDArray[np.float64]: ...
