# Copy-state starts at oracle clones (#540)

**TL;DR.** On dev_tree 60 × 50 r0 at the planted clones: 39 starts, 919
trials, each polished by `sal`'s EM for up to 60 s and scored on the whole
call.

- **BAF + RDR: seeding on rows smoothed over 10 Mb is the best start.**
  `kmeans++` there ends 8.9 nats below the best fit reached (at most 9.3 over
  3 seeds) in 1.7 s. `--sal`'s `kmeans++x5+em` on the raw rows ends 39.6
  nats below (39.9 at most) in 13.9 s: 30 nats worse at 8x the cost. Every
  start on raw rows ends 23-49 nats below, `hmcx5` and `hmcx5+em` 152-158.
- **BAF + RDR: no start or arm separates the one-copy loss from
  copy-neutral LOH.** Across 343 fits, the LOH states (folded p < 0.1) sit
  at a median log mu of -0.12, between the loss (-0.65, 137 rows) and CN-LOH
  (0.01, 456 rows). None is within 0.1 of the loss; 15 are within 0.1 of
  CN-LOH, found in 4% of fits. The mixture objective merges the two, whatever
  the start: #471's defect.
- **BAF only: seeding from the BAF + RDR call's read-depth quantiles ends
  21.4 nats below the best, in 4.9 s.** Port's `distinct`, today's BAF-stage
  start, ends 36.5 nats below at its median but up to 110.9 over seeds.
  Smoothing its seeding rows (any window) holds it at 36.4-38.0. One
  `emission++x5+em` trial on 10 Mb smoothing reached the best fit; its other
  two seeds were refused. BAF alone separates two levels, LOH and balanced,
  and every start finds both, so the gap measures fit quality, not state
  recovery. `sal`'s starts do not reach this stage in the pipeline yet
  (`sal_mixture` skips it).
- **Masks** on the rows a start seeds from, or fits on, do not beat
  smoothing on either stage.
- **Outliers**: 5% of rows at read depth x8 or /8 leave the shortlist
  140-181 nats below the corrupted call's best, masked or not. A mask on the
  seeding rows does not reach the polish, which still reads them.
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
corrected instance's; `--sal` itself is unchanged until #547.

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
