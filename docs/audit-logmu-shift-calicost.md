# Audit: `logmu_shift` across the paper, CalicoST and `cnaster`

Issue #131, and the evidence for #5.

Paper at `0478c74`; `cnaster` at `4adad4d` (`finish_annotation_mushift3`);
CalicoST at `c1abcae`.

**Three references, three different quantities.** The paper normalizes the
read-depth *mean*. CalicoST computes a shift and spends it on the *allele*
channel. `cnaster` computes the same shift and spends it nowhere: its one call
site is commented out and replaced by a warning. So the de-biasing the paper
defines is implemented by neither, and the thing CalicoST does implement is
not what the paper means by `\bar{\mu}`.

---

## What each reference says

### The paper: a constraint on the rate

`emission.tex:16` states the mean and the constraint that de-biases it:

> As a result, `<u_gn>_k = λ_g T_n · μ_k / (Σ_g λ_g μ_k)` and we define
> de-biased equivalents, `μ̄_k`, that satisfy the further constraint
> `Σ_g λ_g μ̄_k = 1`.

The `\todo` beside it gives the reason: a large amplification at one locus
takes a larger share of the library, which biases every other read-depth
parameter low. The normalization lives **in the negative binomial's mean**,
and it is over the genome (`g`), per copy state (`k`).

`emission.tex:18` then reads the normalized parameter as the copy number:
`μ̄_k = (a_k + b_k)/2`. So the integer-copy decoding of
`docs/audit-integer-copy-calicost.md` consumes `μ̄`, not `μ` — which makes
this not a cosmetic normalization but the thing that gives the RDR axis its
units.

### CalicoST: a shift, applied to the allele channel only

`hmm_NB_BB_phaseswitch.py:526-531` builds it per clone:

```python
logmu_shift.append(
    scipy.special.logsumexp(new_log_mu[this_pred_cnv, :] + np.log(kwargs["lambd"]).reshape(-1, 1), axis=0)
)
logmu_shift = np.vstack(logmu_shift)          # (n_clones, n_spots)
```

`logsumexp(log_mu[pred_cnv] + log λ)` is exactly `log Σ_g λ_g μ_{k(g)}` — the
paper's denominator, evaluated **along the decoded path** rather than over the
states. It is then passed to `compute_emission_probability_nb_betabinom_mix`,
where `hmm_NB_BB_nophasing_v2.py:141-147` uses it:

```python
this_weighted_tp.append(
    tumor_prop[range_s:range_t, s] * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c, s])
    / (tumor_prop[range_s:range_t, s] * np.exp(log_mu[i, s] - kwargs["logmu_shift"][c, s]) + 1 - tumor_prop[range_s:range_t, s])
)
```

and the result multiplies only `p_binom` in the beta-binomial mixture. **The
negative binomial mean in the same function is untouched:**

```python
nb_mean = base_nb_mean[idx_nonzero_rdr, s] * (tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[idx_nonzero_rdr, s])
```

So CalicoST's shift answers a different question from the paper's constraint:
it converts a tumour *cell* fraction into a tumour *read* fraction, because a
clone carrying amplifications contributes more reads per cell. That is a real
quantity and a defensible one. It is not `Σ_g λ_g μ̄_k = 1`, and CalicoST's
RDR mean is not divided by anything.

### `cnaster`: computed, and never applied

`hmm_nophasing.py:133` defines `compute_logmu_shifts`, `@njit`, with the
vectorized original preserved in its own docstring:

```python
return scipy.special.logsumexp(log_mu[clone_copy_states, :] + normal_log_lambda.reshape(-1, 1), axis=0)
```

The hand-unrolled body computes the same `logsumexp` per clone and broadcasts
each clone's scalar across that clone's segments, returning a flat
`(n_segments,)` array.

Its only call site, `hmm_nophasing.py:275-283`:

```python
if normal_log_lambda is not None:
    # logmu_shifts = compute_logmu_shifts(log_mu, copy_states, normal_log_lambda, clone_lengths)
    logger.warning("logmu_shifts are not currently supported.")

# TODO fold in logmu_shifts; assumed concatenated (repeated) along the genomic axis.
```

Three consequences, none of them stated in the code:

1. **The emission is the unnormalized one.** `_nb_logpmf_1d` receives
   `exp(log_mu[i, s])` directly, so `cnaster` fits `μ`, not `μ̄`, and the
   integer-copy decoder downstream reads a `μ` the paper's `(a+b)/2` identity
   does not apply to.
2. **The warning fires inside the per-state loop**, so a fit at `K` states
   over `S` samples emits `K x S` identical warnings per emission evaluation.
   At the dev instance's `K = 10` that is ten per call, and the emission is
   called once per solver objective evaluation.
3. **`cna_hmrf_result.py:48` declares `new_log_mu_shift: np.ndarray | None =
   None`** and nothing ever populates it. `hmm.py:159` and `hmm.py:213` carry
   the commented-out computation and the commented-out assignment.

---

## The divergence, as a table

| | the paper | CalicoST | `cnaster` |
| --- | --- | --- | --- |
| quantity | `Σ_g λ_g μ̄_k = 1` | `log Σ_g λ_g μ_{k(g)}` per clone | same as CalicoST |
| computed over | states `k` | the decoded path | the decoded path |
| shape | per state | `(n_clones, n_spots)` | `(n_segments,)`, clone-broadcast |
| applied to RDR mean | **yes** | no | no |
| applied to BAF mixing | not stated | **yes** | no |
| applied at all | — | yes | **no** |

The row that matters is the last two: `cnaster` has CalicoST's quantity and
neither implementation's application.

---

## Roadmap: the data structures a fix needs

The user's framing in #131 -- "returning a logmu array, together with an
`n_state x n_clone` shifts array from the fitting, propagating through emission
calls for coloring" -- is the right shape, and it is **not** the shape either
existing implementation has.

1. **The shift is per (state, clone), not per clone.** CalicoST's is
   `(n_clones, n_spots)` because it is evaluated on one decoded path, so every
   state shares a clone's shift. The paper's constraint is per state: `μ̄_k`
   normalizes state `k`'s own rate. An `(n_states, n_clones)` array is what
   lets a state be de-biased without first fixing the decode, and it removes
   the circularity that forces CalicoST to recompute the shift after every
   Baum-Welch iteration.
2. **`compute_logmu_shifts` returns the wrong rank twice over.** It is
   `(n_segments,)` where the caller needs `(n_states, n_clones)`, and it takes
   `log_mus` indexed as `log_mus[state]` -- one sample only, where
   `log_mu` is `(n_states, n_spots)` everywhere else in the module. Both
   follow from it having been written for the clone-concatenated pseudobulk,
   where a clone *is* a sample.
3. **`clone_lengths` is the wrong carrier.** It encodes "clones concatenate
   along the genomic axis" as an arithmetic convention the function reimplements
   with `start_idx` bookkeeping. `snakes_and_ladders`' `Ragged` (`lengths` plus
   `values`) is the structure this is an open-coded instance of, and
   `hmrf_utils.clone_stack_lengths` already builds the tiling. Reaching for
   `Ragged` here is `CLAUDE.md`'s "reach for what `snakes_and_ladders` already
   carries", and it is the difference between the fix being a `port` patch and
   an upstream one.
4. **Two results containers need the field populated, not declared.**
   `cna_hmrf_result.new_log_mu_shift` exists; the M step returns `new_log_mu`
   and would need to return the pair. Downstream, anything that colours by
   copy state reads `log_mu` and would have to read `log_mu - shift` instead,
   which is the propagation #131 names.
5. **Decide which application is wanted before writing it.** The paper's is
   in the RDR mean; CalicoST's is in the BAF tumour weighting. They are not
   alternatives -- a complete implementation has both, and they use the shift
   for different purposes. A patch that adds one and calls #5 closed would
   leave the other unstated, which is the failure `CLAUDE.md`'s stated-difference
   rule exists to prevent.

---

## What `snakes_and_ladders` would have to carry

`CLAUDE.md` requires an audit to name its upstream correspondence.

The normalization is **not** application specific: it is a constraint on a
count emission's rate given a per-observation exposure, which is
`NegativeBinomialEmission` plus a covariate. Upstream already has the
covariate (#57, #77), and #65 is the ticket for a varying kernel that reaches
a fit. A `normalized=True` mode on the family -- or, more honestly, a
`log_partition` hook the family exposes so the M step can subtract it -- would
make the paper's `μ̄` expressible upstream and make `cnaster`'s version a
consumer rather than an implementation.

Where the correspondence stops: CalicoST's tumour-read-fraction reweighting is
application specific. It mixes two populations at a per-site proportion, and
upstream has no notion of a mixture over an observation-indexed weight. That
half belongs in `cnaster` whatever upstream does.

---

## What this does not settle

No measurement. This audit reads three sources; it does not run CalicoST
(#131 says not to) and it does not measure what the missing normalization
costs in recovered parameters. That measurement is #5's, and the fixture for
it exists: `core_inference_truth` plants `log_mu` directly, so the bias is
the difference between the planted value and the fitted one under a genome
whose `Σ_g λ_g μ_k` is deliberately far from 1.
