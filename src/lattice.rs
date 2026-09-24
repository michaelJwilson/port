//! `cnaster`'s forward and backward lattices, compiled once at build time.
//!
//! `cnaster.hmm_nophasing.hmm_nophasing.{forward,backward}_lattice` are
//! `@njit` without `cache=True`, so every process compiles them on first
//! call: 9.75 s of a 40 s dev run. These are the same recursions, in the
//! same floating-point order -- the same sequential sums, the same
//! `max + ln(sum(exp(a - max)))` -- so the results are bitwise `cnaster`'s,
//! with the compile moved to `maturin build`.
//!
//! One change of work, not of arithmetic: the per-site sum of the emission
//! over its last axis is computed once per `(state, site)` instead of once
//! per `(state, state, site)` in the backward pass. Each sum is the same
//! sequence of additions, so the value is identical; it is `K` times fewer.
//!
//! Arrays arrive as flat, C-contiguous slices with their shape; `lib.rs`
//! takes them from NumPy through the `numpy` crate.

/// `cnaster`'s `numba_logsumexp`: `-inf` or `+inf` maxima returned as they are.
#[inline]
fn logsumexp(a: &[f64]) -> f64 {
    // NB `np.max`: a NaN anywhere is the maximum.
    let mut a_max = f64::NEG_INFINITY;
    for &x in a {
        if x.is_nan() {
            return f64::NAN;
        }
        if x > a_max {
            a_max = x;
        }
    }
    if a_max.is_infinite() {
        return a_max;
    }
    let mut total = 0.0;
    for &x in a {
        total += (x - a_max).exp();
    }
    a_max + total.ln()
}

/// `(n_states, n_obs)` sums of `emission` over its last axis, sequentially.
fn emission_sums(emission: &[f64], n_states: usize, n_obs: usize, n_spots: usize) -> Vec<f64> {
    let mut sums = vec![0.0; n_states * n_obs];
    for j in 0..n_states {
        for t in 0..n_obs {
            let base = (j * n_obs + t) * n_spots;
            let mut total = 0.0;
            for s in 0..n_spots {
                total += emission[base + s];
            }
            sums[j * n_obs + t] = total;
        }
    }
    sums
}

/// Work below which the segments run on the calling thread: spawning costs
/// more than a small lattice does.
const PARALLEL_WORK: usize = 1 << 18;

/// Run `segment(start, length, local)` over every contig in `lengths`, and
/// write each `(n_rows, length)` local block into `out`, `(n_rows, n_obs)`.
///
/// Contigs are independent recursions, so they run on separate threads when
/// there is enough work, each into its own buffer. The arithmetic of a
/// contig does not depend on which thread runs it, so the output is the
/// same bitwise either way.
fn over_segments<F>(
    lengths: &[i64],
    n_rows: usize,
    n_obs: usize,
    work: usize,
    out: &mut [f64],
    segment: F,
) where
    F: Fn(usize, usize, &mut [f64]) + Sync,
{
    let mut spans = Vec::with_capacity(lengths.len());
    let mut start = 0usize;
    for &le in lengths {
        spans.push((start, le as usize));
        start += le as usize;
    }

    let run = |span: &(usize, usize)| {
        let mut local = vec![0.0; n_rows * span.1];
        segment(span.0, span.1, &mut local);
        local
    };

    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let blocks: Vec<Vec<f64>> = if threads > 1 && spans.len() > 1 && work >= PARALLEL_WORK {
        let per = spans.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = spans
                .chunks(per)
                .map(|group| scope.spawn(|| group.iter().map(run).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("lattice thread"))
                .collect()
        })
    } else {
        spans.iter().map(run).collect()
    };

    for ((start, le), local) in spans.iter().zip(blocks) {
        for r in 0..n_rows {
            out[r * n_obs + start..r * n_obs + start + le]
                .copy_from_slice(&local[r * le..(r + 1) * le]);
        }
    }
}

/// Unphased forward pass: `log_alpha`, `(n_states, n_obs)`, into `out`.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    lengths: &[i64],
    log_transmat: &[f64],
    log_startprob: &[f64],
    emission: &[f64],
    n_states: usize,
    n_obs: usize,
    n_spots: usize,
    out: &mut [f64],
) {
    let sums = emission_sums(emission, n_states, n_obs, n_spots);
    let work = n_states * n_states * n_obs;

    over_segments(lengths, n_states, n_obs, work, out, |cumlen, le, local| {
        let mut buf = vec![0.0; n_states];
        for j in 0..n_states {
            local[j * le] = log_startprob[j] + sums[j * n_obs + cumlen];
        }
        for t in 1..le {
            for j in 0..n_states {
                for i in 0..n_states {
                    buf[i] = local[i * le + t - 1] + log_transmat[i * n_states + j];
                }
                local[j * le + t] = logsumexp(&buf) + sums[j * n_obs + cumlen + t];
            }
        }
    });
}

/// Unphased backward pass: `log_beta`, `(n_states, n_obs)`, into `out`.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    lengths: &[i64],
    log_transmat: &[f64],
    emission: &[f64],
    n_states: usize,
    n_obs: usize,
    n_spots: usize,
    out: &mut [f64],
) {
    let sums = emission_sums(emission, n_states, n_obs, n_spots);
    let work = n_states * n_states * n_obs;

    over_segments(lengths, n_states, n_obs, work, out, |cumlen, le, local| {
        let mut buf = vec![0.0; n_states];
        for i in 0..n_states {
            local[i * le + le - 1] = 0.0;
        }
        for t in (0..le.saturating_sub(1)).rev() {
            for i in 0..n_states {
                for j in 0..n_states {
                    buf[j] = local[j * le + t + 1]
                        + log_transmat[i * n_states + j]
                        + sums[j * n_obs + cumlen + t + 1];
                }
                local[i * le + t] = logsumexp(&buf);
            }
        }
    });
}

/// `cnaster.hmm_phased.update_combined_transmat`, into `combined`.
fn combined_transmat(
    combined: &mut [f64],
    n_states: usize,
    log_transmat: &[f64],
    self_trans: f64,
    switch_trans: f64,
    penalize_same_cnv: bool,
    log_half: f64,
) {
    let width = 2 * n_states;
    for a in 0..n_states {
        for b in 0..n_states {
            let base = log_transmat[a * n_states + b];
            let (keep, switch) = if penalize_same_cnv {
                (log_half + base, log_half + base)
            } else {
                (self_trans + base, switch_trans + base)
            };
            combined[a * width + b] = keep;
            combined[a * width + n_states + b] = switch;
            combined[(n_states + a) * width + b] = switch;
            combined[(n_states + a) * width + n_states + b] = keep;
        }
    }
    if penalize_same_cnv {
        for i in 0..n_states {
            let base = log_transmat[i * n_states + i];
            combined[i * width + i] = self_trans + base;
            combined[i * width + i + n_states] = switch_trans + base;
            combined[(i + n_states) * width + i] = switch_trans + base;
            combined[(i + n_states) * width + i + n_states] = self_trans + base;
        }
    }
}

/// Phased forward pass over `n_paired = 2 * n_states` states.
#[allow(clippy::too_many_arguments)]
pub fn forward_phased(
    lengths: &[i64],
    log_transmat: &[f64],
    log_startprob: &[f64],
    emission: &[f64],
    log_sitewise: &[f64],
    penalize_same_cnv: bool,
    n_paired: usize,
    n_obs: usize,
    n_spots: usize,
    out: &mut [f64],
) {
    let n_states = n_paired.div_ceil(2);
    let sums = emission_sums(emission, n_paired, n_obs, n_spots);
    let log_half = 0.5f64.ln();
    let work = n_paired * n_paired * n_obs;

    over_segments(lengths, n_paired, n_obs, work, out, |cumlen, le, local| {
        let mut combined = vec![0.0; n_paired * n_paired];
        let mut buf = vec![0.0; n_paired];
        for j in 0..n_paired {
            let start = log_half + log_startprob[j % n_states];
            local[j * le] = start + sums[j * n_obs + cumlen];
        }
        for t in 1..le {
            let idx = cumlen + t - 1;
            let switch = log_sitewise[idx];
            let keep = (1.0 - switch.exp()).ln();
            combined_transmat(
                &mut combined,
                n_states,
                log_transmat,
                keep,
                switch,
                penalize_same_cnv,
                log_half,
            );
            for j in 0..n_paired {
                for i in 0..n_paired {
                    buf[i] = local[i * le + t - 1] + combined[i * n_paired + j];
                }
                local[j * le + t] = logsumexp(&buf) + sums[j * n_obs + cumlen + t];
            }
        }
    });
}

/// Phased backward pass over `n_paired = 2 * n_states` states.
#[allow(clippy::too_many_arguments)]
pub fn backward_phased(
    lengths: &[i64],
    log_transmat: &[f64],
    emission: &[f64],
    log_sitewise: &[f64],
    penalize_same_cnv: bool,
    n_paired: usize,
    n_obs: usize,
    n_spots: usize,
    out: &mut [f64],
) {
    let n_states = n_paired.div_ceil(2);
    let sums = emission_sums(emission, n_paired, n_obs, n_spots);
    let log_half = 0.5f64.ln();
    let work = n_paired * n_paired * n_obs;

    over_segments(lengths, n_paired, n_obs, work, out, |cumlen, le, local| {
        let mut combined = vec![0.0; n_paired * n_paired];
        let mut buf = vec![0.0; n_paired];
        for i in 0..n_paired {
            local[i * le + le - 1] = 0.0;
        }
        for t in (0..le.saturating_sub(1)).rev() {
            let idx = cumlen + t;
            let switch = log_sitewise[idx];
            let keep = (1.0 - switch.exp()).ln();
            combined_transmat(
                &mut combined,
                n_states,
                log_transmat,
                keep,
                switch,
                penalize_same_cnv,
                log_half,
            );
            for i in 0..n_paired {
                for j in 0..n_paired {
                    buf[j] = local[j * le + t + 1]
                        + combined[i * n_paired + j]
                        + sums[j * n_obs + cumlen + t + 1];
                }
                local[i * le + t] = logsumexp(&buf);
            }
        }
    });
}
