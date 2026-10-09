//! `oxicnamaste`, cnamaste's own compiled kernels (T- #836 K0).
//!
//! Empty of kernels: K7 copies `oxiport`'s forward-backward lattices here
//! and K5a sal's alpha expansion, each naming its source. The crate depends
//! on no `port` crate, so cnamaste builds and ships on its own.

use pyo3::prelude::*;

/// The crate's version, `Cargo.toml`'s, which the wheel carries too.
#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn oxicnamaste(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
