# Copy-state starts at oracle clones (#540)

**TL;DR.** On dev_tree 60 × 50 r0 at the planted clones: 42 starts, 1,015
trials, each polished by `sal`'s EM for up to 60 s and scored on the whole
call.

- **The lattice start is the best on BAF only and the best on BAF + RDR on
  raw rows** (`copy_starts.lattice_start`). It places every integer
  `(A, B)` up to the rows' read-depth ceiling, as the integer decode does,
  and scores each row under the mixture's own IID emission (`sal`'s
  `CountPairEmission`). It then assigns rows by likelihood plus log weight,
  a classification EM, fits the NB size, BB concentration and a BAF error
  rate by that likelihood, and keeps the `n_states` states of highest
  weight.
  - BAF only: it reaches the best fit any trial reached, in 4.9 s. Port's
    `distinct`, today's BAF-stage start, ends 40.4 nats below (up to 114.8).
    With the BB concentration fitted by EM over soft posteriors
    (`lattice-em`) it ends 35.0 below. It holds the best fit on 1% and 5%
    corrupted BAF calls.
  - BAF + RDR: it ends 8.8 nats below the best (in 21 s with 4 workers on 4
    cores), beside 10 Mb-smoothed `kmeans++` at 8.9 and `--sal`'s
    `kmeans++x5+em` at 39.6.
  - Before its polish it places all four planted states, the one-copy loss
    (log mu -0.69) apart from CN-LOH (0.00). The polish merges them, as
    every fit does.
  - It is not robust to read-depth outliers on BAF + RDR: with 5% of rows
    at x8 or /8 it ends 369-412 nats below, because the outliers form a
    populated state of their own; `kmeans++` holds at 11.
- **BAF + RDR: seeding `kmeans++` on rows smoothed over 10 Mb ties the lattice.**
  `kmeans++` there ends 8.9 nats below the best fit reached (at most 9.3 over
  3 seeds) in 1.7 s. `--sal`'s `kmeans++x5+em` on the raw rows ends 39.6
  nats below (39.9 at most) in 13.9 s: 30 nats worse at 8x the cost. Every
  other start on raw rows ends 23-49 nats below, `hmcx5` and `hmcx5+em`
  152-158.
- **BAF + RDR: no polished fit separates the one-copy loss from
  copy-neutral LOH.** Across 382 polished fits the LOH states (folded
  p < 0.1) sit at a median log mu of -0.12, between the loss (-0.65, 137
  rows) and CN-LOH (0.01, 456 rows), and none is within 0.1 of the loss,
  though the lattice start places both. The mixture objective merges them,
  whatever the start: #471's defect.
- **BAF only: read-depth seeding needs a baseline the BAF stage lacks.**
  Seeding from the BAF + RDR call's read-depth quantiles ends 25.3 nats
  below the best. In the pipeline the BAF stage runs before any normal
  baseline exists; against the stand-in it has, each bin's share of every
  clone's reads (`pooled_exposure`), the same seeding ends 118.7 nats
  below. `distinct` smoothed along the genome holds at 39.8-41.9. BAF alone
  separates two levels, LOH and balanced, and every start finds both, so on
  this stage the gap measures fit quality, not state recovery. `sal`'s starts
  do not reach this stage in the pipeline yet (`sal_mixture` skips it).
- **Masks** on the rows a start seeds from, or fits on, do not beat
  smoothing on either stage.
- **Outliers**: with 5% of rows at read depth x8 or /8, `kmeans++` and
  `kmeans++x5+em` end 11 nats below the corrupted call's best, the lattice
  369, `distinct` 140. Masking the top 5% of |log RDR| from the seeding
  takes `kmeans++` to 0.1 but `kmeans++x5+em` to 375: the polish still
  reads the masked rows.
- **Refused**:
  - `cnaster`'s `cna_mixture_init` calls `get_state_posteriors` without
    `log_sitewise_transmat`;
  - `gaussian-em` collapses a variance on BAF + RDR, as in #489;
  - `sal`'s `prior` seeds the pair `(total, fraction x total)`, the joint
    form's, and on this independent-form instance its rates exceed 1
    ("every beta must be positive"), #547;
  - `sal`'s starts refuse counts without the covariate, so covariate-free
    starts exist only as `cnaster`'s and port's GMMs;
  - on smoothed rows, `sal`'s EM refuses 25 of 72 best-of-five trials ("M
    step did not settle");
  - `emission++` draws a negative probability on 12 BAF + RDR trials.

**Seeding.** `sal_mixture.instance_of`, the instance `--sal`'s start seeds
from, seeds every BAF near 0.01 and no read-depth state below neutral
(#547). Its fit is the same model, but a start could not place a loss.
`copy_starts.instance` corrects both: exposure over 100, and the B column
over the common trial count, as `sal`'s `rate_space` writes it. An earlier
run on the uncorrected instance is discarded. These numbers are the
corrected instance's; #547 corrects `--sal`'s, and installs the lattice
under it with a 300 normal-UMI segment floor. The other starts are in
`port.sandbox.extensions.copy_starts`.

**Conditions.**
- The calls come from one `--sal --oracle-start` run of dev_tree r0, rebuilt
  at the planted clones for both stages (`tests.studies.copy_starts capture`).
- Every trial is a start, then `sal`'s EM on the whole call, within 60 s.
  Seeds 0-2; deterministic starts once. 4 forked workers on the 4-core host,
  one thread each, with nothing else running. Seconds include the start.
- Gap: below the best log-likelihood any trial reached on the same call.
- Found: a fitted state within 0.1 of a planted state's pooled log mu and
  0.05 of its folded p (BAF only: p alone).
- The capture is `data/copy_state_starts_capture_r0.npz`, the trials
  `data/copy_state_starts_r0.json`. Regenerate with
  `python -m tests.studies.copy_starts run docs/nb/data/copy_state_starts_capture_r0.npz OUT.pkl`,
  then `python -m tests.studies.copy_start_notebook OUT.pkl`.
