use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod lattice;

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

fn contiguous<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn checked_lengths(lengths: &[i64], n_obs: usize) -> PyResult<()> {
    if lengths.iter().any(|&le| le < 1) || lengths.iter().sum::<i64>() as usize != n_obs {
        return Err(PyValueError::new_err(
            "lengths must be positive and sum to the emission's second axis",
        ));
    }
    Ok(())
}

fn checked_len(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} has {actual} entries where {expected} are needed"
        )));
    }
    Ok(())
}

fn into_array(py: Python<'_>, out: Vec<f64>, n_obs: usize) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let rows = out.len() / n_obs.max(1);
    let array = Array2::from_shape_vec((rows, n_obs), out)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// `cnaster`'s unphased forward lattice; `log_alpha`, `(n_states, n_obs)`.
#[pyfunction]
fn forward_lattice<'py>(
    py: Python<'py>,
    lengths: PyReadonlyArray1<'py, i64>,
    log_transmat: PyReadonlyArray2<'py, f64>,
    log_startprob: PyReadonlyArray1<'py, f64>,
    log_emission: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let [n_states, n_obs, n_spots] = *log_emission.shape() else {
        unreachable!()
    };
    let lengths = contiguous(&lengths, "lengths")?;
    checked_lengths(lengths, n_obs)?;
    let transmat = contiguous(&log_transmat, "log_transmat")?;
    let startprob = contiguous(&log_startprob, "log_startprob")?;
    let emission = contiguous(&log_emission, "log_emission")?;
    checked_len("log_transmat", transmat.len(), n_states * n_states)?;
    checked_len("log_startprob", startprob.len(), n_states)?;
    let mut out = vec![0.0; n_states * n_obs];
    py.detach(|| {
        lattice::forward(
            lengths, transmat, startprob, emission, n_states, n_obs, n_spots, &mut out,
        )
    });
    into_array(py, out, n_obs)
}

/// `cnaster`'s unphased backward lattice; `log_beta`, `(n_states, n_obs)`.
#[pyfunction]
fn backward_lattice<'py>(
    py: Python<'py>,
    lengths: PyReadonlyArray1<'py, i64>,
    log_transmat: PyReadonlyArray2<'py, f64>,
    log_emission: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let [n_states, n_obs, n_spots] = *log_emission.shape() else {
        unreachable!()
    };
    let lengths = contiguous(&lengths, "lengths")?;
    checked_lengths(lengths, n_obs)?;
    let transmat = contiguous(&log_transmat, "log_transmat")?;
    let emission = contiguous(&log_emission, "log_emission")?;
    checked_len("log_transmat", transmat.len(), n_states * n_states)?;
    let mut out = vec![0.0; n_states * n_obs];
    py.detach(|| {
        lattice::backward(
            lengths, transmat, emission, n_states, n_obs, n_spots, &mut out,
        )
    });
    into_array(py, out, n_obs)
}

/// `cnaster`'s phased forward lattice over `2K` paired states.
#[pyfunction]
fn forward_lattice_phased<'py>(
    py: Python<'py>,
    lengths: PyReadonlyArray1<'py, i64>,
    log_transmat: PyReadonlyArray2<'py, f64>,
    log_startprob: PyReadonlyArray1<'py, f64>,
    log_emission: PyReadonlyArray3<'py, f64>,
    log_sitewise_transmat: PyReadonlyArray1<'py, f64>,
    penalize_phase_only_on_same_cnv: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let [n_paired, n_obs, n_spots] = *log_emission.shape() else {
        unreachable!()
    };
    let lengths = contiguous(&lengths, "lengths")?;
    checked_lengths(lengths, n_obs)?;
    let transmat = contiguous(&log_transmat, "log_transmat")?;
    let startprob = contiguous(&log_startprob, "log_startprob")?;
    let emission = contiguous(&log_emission, "log_emission")?;
    let sitewise = contiguous(&log_sitewise_transmat, "log_sitewise_transmat")?;
    let n_states = n_paired.div_ceil(2);
    checked_len("log_transmat", transmat.len(), n_states * n_states)?;
    checked_len("log_startprob", startprob.len(), n_states)?;
    checked_len("log_sitewise_transmat", sitewise.len(), n_obs)?;
    let mut out = vec![0.0; n_paired * n_obs];
    py.detach(|| {
        lattice::forward_phased(
            lengths,
            transmat,
            startprob,
            emission,
            sitewise,
            penalize_phase_only_on_same_cnv,
            n_paired,
            n_obs,
            n_spots,
            &mut out,
        )
    });
    into_array(py, out, n_obs)
}

/// `cnaster`'s phased backward lattice over `2K` paired states.
#[pyfunction]
fn backward_lattice_phased<'py>(
    py: Python<'py>,
    lengths: PyReadonlyArray1<'py, i64>,
    log_transmat: PyReadonlyArray2<'py, f64>,
    log_emission: PyReadonlyArray3<'py, f64>,
    log_sitewise_transmat: PyReadonlyArray1<'py, f64>,
    penalize_phase_only_on_same_cnv: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let [n_paired, n_obs, n_spots] = *log_emission.shape() else {
        unreachable!()
    };
    let lengths = contiguous(&lengths, "lengths")?;
    checked_lengths(lengths, n_obs)?;
    let transmat = contiguous(&log_transmat, "log_transmat")?;
    let emission = contiguous(&log_emission, "log_emission")?;
    let sitewise = contiguous(&log_sitewise_transmat, "log_sitewise_transmat")?;
    let n_states = n_paired.div_ceil(2);
    checked_len("log_transmat", transmat.len(), n_states * n_states)?;
    checked_len("log_sitewise_transmat", sitewise.len(), n_obs)?;
    let mut out = vec![0.0; n_paired * n_obs];
    py.detach(|| {
        lattice::backward_phased(
            lengths,
            transmat,
            emission,
            sitewise,
            penalize_phase_only_on_same_cnv,
            n_paired,
            n_obs,
            n_spots,
            &mut out,
        )
    });
    into_array(py, out, n_obs)
}

/// The compiled extension module. `python/port/__init__.py` re-exports it as
/// `port.oxiport` (see `module-name` in `pyproject.toml`).
#[pymodule]
fn oxiport(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(forward_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(backward_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(forward_lattice_phased, m)?)?;
    m.add_function(wrap_pyfunction!(backward_lattice_phased, m)?)?;
    Ok(())
}
