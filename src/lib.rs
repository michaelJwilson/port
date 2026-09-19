//! `oxiport`: two CSR reductions, measured against the loader and **rejected**.
//!
//! **1.8x and 0.97x against the forms `port.patch.input_data` runs, where the
//! bar is 2x (#184).** The row sums read 2.21x under `pytest-benchmark` and
//! 1.75x under a `perf_counter` loop over the same two calls, so the claim is
//! the width of its own measurement; the gene counts are the slower of the
//! pair. They are written out in full because the negative result
//! is worth more than the diff: what `cnaster` paid on this path was an
//! intermediate matrix, #167 had already stopped paying it, and what was left
//! is a `scipy` matvec and a `bincount` — one C pass each, which is what these
//! are. Rust bought the loop it was already getting, `rayon` included.
//!
//! `tests/test_oxiport_loader_bench.py` carries the table and
//! `tests/test_oxiport_loader.py` the equivalence, so a later kernel proposed
//! for this path has a reference that is measured rather than assumed.
//!
//! Both read their inputs as borrowed slices and allocate one output vector of
//! the answer's own length, so what they cost in memory is the answer. A third
//! kernel, a one-pass CSR-to-dense `int64` cast, was written and measured for
//! `get_spaceranger_counts`' triple copy and is **not** here: it is 1.0x at the
//! gate size and 1.47x at a slide's, and landing it would mean reimplementing
//! that reader rather than orchestrating it. #167 owns that.
//!
//! **Shape: validate, then release the GIL, then fold.** Each kernel checks its
//! indices once, up front, so the hot loop is infallible and `rayon` has
//! nothing to carry back across a thread boundary; and each holds only borrowed
//! slices across `detach`, so no Python object is touched while the
//! interpreter is free. The validation pass is linear in the index arrays and
//! is what turns a malformed matrix into a message rather than a segfault.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Work below which the parallel path is not worth its scheduling.
///
/// Which path runs is a cost decision and not a numerical one: both fold each
/// row's values in the same order, so they agree bitwise and the threshold is
/// invisible to a caller. It is stated as a constant rather than left to
/// `rayon`'s splitting because the gate-sized instances in this repository sit
/// below it, and a benchmark that silently changed path between sizes would be
/// measuring the threshold.
const PARALLEL_THRESHOLD: usize = 1_000;

/// Doubles an integer.
///
/// Placeholder binding from the crate's first commit, kept because
/// `tests/test_oxiport_bindings.py` pins it and because it is the smallest
/// possible check that the extension built and imported at all.
#[pyfunction]
pub fn double(x: i64) -> i64 {
    x * 2
}

/// Check that every row's `indptr` span lies inside `data`, once.
///
/// Returns the row count. Doing this before the compute loop is what makes
/// that loop infallible, which is what lets it run under `rayon` and outside
/// the GIL: an error raised inside a parallel fold would have to be carried
/// back across both boundaries.
fn validated_rows(data: &[f64], indptr: &[i64]) -> PyResult<usize> {
    if indptr.is_empty() {
        return Err(PyValueError::new_err(
            "indptr must carry at least one entry",
        ));
    }

    for row in 0..indptr.len() - 1 {
        let (start, end) = (indptr[row], indptr[row + 1]);

        if start < 0 || end < start || (end as usize) > data.len() {
            return Err(PyValueError::new_err(format!(
                "row {row} spans [{start}, {end}) which is not inside its data"
            )));
        }
    }

    Ok(indptr.len() - 1)
}

/// One row's stored values, summed. Bounds are the caller's to have checked.
#[inline]
fn row_total(data: &[f64], indptr: &[i64], row: usize) -> f64 {
    data[indptr[row] as usize..indptr[row + 1] as usize]
        .iter()
        .sum()
}

/// Row sums of `A + B`, for two CSR matrices, without forming `A + B`.
///
/// `(A + B).sum(axis=1)` in `scipy` builds the sum matrix first — a third CSR
/// with up to `nnz(A) + nnz(B)` entries — and then reduces it away. The sums
/// need neither: each row's total is the sum of its own stored values in each
/// matrix, so one pass over both `data` arrays gives the answer with one
/// allocation of `n_rows`.
///
/// Only the `indptr` arrays are read besides the data; the column indices are
/// irrelevant to a row sum, which is what makes the intermediate avoidable and
/// the rows independent enough to fold in parallel.
///
/// # Errors
///
/// If the two `indptr` arrays disagree on the row count, either is empty, or a
/// stated row range runs past its `data`.
#[pyfunction]
#[pyo3(name = "csr_pair_row_sums")]
pub fn csr_pair_row_sums<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray1<'py, f64>,
    a_indptr: PyReadonlyArray1<'py, i64>,
    b_data: PyReadonlyArray1<'py, f64>,
    b_indptr: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (a_data, a_indptr) = (a_data.as_slice()?, a_indptr.as_slice()?);
    let (b_data, b_indptr) = (b_data.as_slice()?, b_indptr.as_slice()?);

    if a_indptr.len() != b_indptr.len() {
        return Err(PyValueError::new_err(
            "the two matrices disagree on the row count",
        ));
    }

    let n_rows = validated_rows(a_data, a_indptr)?;
    validated_rows(b_data, b_indptr)?;

    let totals = py.detach(|| {
        let of_row =
            |row: usize| row_total(a_data, a_indptr, row) + row_total(b_data, b_indptr, row);

        if n_rows >= PARALLEL_THRESHOLD {
            (0..n_rows).into_par_iter().map(of_row).collect()
        } else {
            (0..n_rows).map(of_row).collect::<Vec<f64>>()
        }
    });

    Ok(PyArray1::from_vec(py, totals))
}

/// Per-column counts of **positive** values, for a CSR matrix.
///
/// `np.sum(X > 0, axis=0)` builds a second matrix of the full shape to count
/// its entries; `getnnz(axis=0)` counts what is stored, which differs wherever
/// a stored entry is zero. This counts the positive ones, so it is the first
/// expression's answer at the second's cost — and the loader's gene filter is
/// the first expression.
///
/// The parallel path folds a private vector per worker and reduces them, since
/// the scatter is by column while the split is by entry. That costs `n_cols`
/// per worker, which is the one place in this crate where the parallel form
/// allocates more than the sequential one; it is bounded by the answer's own
/// size times the thread count, not by the matrix.
///
/// # Errors
///
/// If `data` and `indices` disagree in length, or a column index falls outside
/// `n_cols`.
#[pyfunction]
#[pyo3(name = "csr_positive_per_column")]
pub fn csr_positive_per_column<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (data, indices) = (data.as_slice()?, indices.as_slice()?);

    if data.len() != indices.len() {
        return Err(PyValueError::new_err(format!(
            "{} values against {} column indices",
            data.len(),
            indices.len()
        )));
    }

    if let Some(&column) = indices.iter().find(|&&c| c < 0 || (c as usize) >= n_cols) {
        return Err(PyValueError::new_err(format!(
            "column index {column} is outside a matrix {n_cols} wide"
        )));
    }

    let counts = py.detach(|| {
        if data.len() >= PARALLEL_THRESHOLD {
            data.par_iter()
                .zip(indices.par_iter())
                .fold(
                    || vec![0_i64; n_cols],
                    |mut acc, (value, &column)| {
                        if *value > 0.0 {
                            acc[column as usize] += 1;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0_i64; n_cols],
                    |mut left, right| {
                        for (l, r) in left.iter_mut().zip(right.iter()) {
                            *l += r;
                        }
                        left
                    },
                )
        } else {
            let mut acc = vec![0_i64; n_cols];

            for (value, &column) in data.iter().zip(indices.iter()) {
                if *value > 0.0 {
                    acc[column as usize] += 1;
                }
            }

            acc
        }
    });

    Ok(PyArray1::from_vec(py, counts))
}

/// The compiled extension module. `python/port/__init__.py` re-exports it as
/// `port.oxiport` (see `module-name` in `pyproject.toml`).
#[pymodule]
fn oxiport(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    m.add_function(wrap_pyfunction!(csr_pair_row_sums, m)?)?;
    m.add_function(wrap_pyfunction!(csr_positive_per_column, m)?)?;
    Ok(())
}
