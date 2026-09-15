use pyo3::prelude::*;

/// Doubles an integer.
///
/// Placeholder binding: it demonstrates the Rust-to-Python pattern real
/// numerical kernels follow and implements nothing itself. `port.__init__`
/// re-exports it and `tests/test_oxiport_bindings.py` asserts it exists, so
/// the compiled extension is covered from the first commit rather than from
/// whenever the first real kernel lands.
#[pyfunction]
pub fn double(x: i64) -> i64 {
    x * 2
}

/// The compiled extension module. `python/port/__init__.py` re-exports it as
/// `port.oxiport` (see `module-name` in `pyproject.toml`).
#[pymodule]
fn oxiport(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    Ok(())
}
