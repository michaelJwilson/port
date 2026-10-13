# Study: the read-depth + BAF HMM start (#489)

**TL;DR:** `sal`'s covariate-aware mixture starts put the HMM at a better
start than port's `distinct` GMM (#348) on every sample. End to end, any of
the three polished best-of-five starts lifts CalicoST hard from clone ARI
0.8652 (5 clones) to **0.9829 (4)**, copy ARI 0.8652 → 0.9055, and phase-free
exact altered 0.497 → 0.627. `dev_tree` 60 × 50 and easy are unchanged.
`--sal` installed `kmeans++x5+em`, the most likely of the three per call
(`--hmm-start`), until #547 replaced it with the lattice start
(`docs/nb/copy_state_starts.md`).

## Method

`run_study --hmm-starts capture | per-call`, retired by #749 WP6 once the
streams superseded it; the module is in git history.

1. **Capture.** One `--sal --hmm-start none` run per sample, pickling every
   initializer call. The read-depth + BAF call is the one studied:
   - 60 × 50: 28,850 bins × clones;
   - easy: 12,552;
   - hard: 12,504.
2. **Per call.** The call becomes `sal`'s `MixtureInstance`, **conditioned
   on each bin's exposure and trials** (sal #933/#1083) and seeded in rate
   space. Each start is a `sal.search.mixture_starts.TimedStart`: the start,
   then `sal`'s EM polish under a 160 s budget, 3 seeds on 4 workers, as
   `sal`'s `docs/nb/emission_mixture_starts.ipynb` runs them. Port's
   `distinct` is registered as one more start.
3. **Reference.** A pseudobulk of inferred clones has no generating truth, so
   the reference is the highest log-likelihood any trial reached. Every gap
   is ≥ 0.

## Per call: gap below the best fit reached [nats], mean ± sd over seeds / cell seconds per seed

| start | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `kmeans++x5+em` | 9.2 ± 8.6 / 23 s | 4.5 ± 1.2 / 5 s | 12.9 ± 2.5 / 6 s |
| `datax5+em` | 25.7 ± 6.1 / 21 s | 7.3 ± 1.2 / 5 s | 10.0 ± 7.6 / 7 s |
| `emission++x5+em` | 37.6 ± 13.1 / 27 s | 8.2 ± 1.3 / 6 s | 18.9 ± 2.0 / 6 s |
| `prior` | 40.4 ± 20.2 / 6 s | 4.5 ± 3.2 / 2 s | 11.4 ± 7.9 / 2 s |
| `kmeans++` | 32.4 ± 21.5 / 6 s | 42.0 ± 25.8 / 2 s | 23.6 ± 0.8 / 3 s |
| `emission++` | 53.8 ± 28.8 / 5 s | 25.2 ± 28.7 / 2 s | 25.0 ± 13.0 / 3 s |
| `perturbed` | 76.9 / 7 s | 67.2 / 5 s | 27.8 / 5 s |
| `objective`, `restart`, `quantile`, `anneal`, `tempering`, `annealx5+em` | 78.8 | 66.5 | 28.3 |
| `distinct` (port, #348) | 93.1 / 13 s | 26.9 / 8 s | 14.8 / 5 s |
| `hmc` | 97.1 ± 2.6 / 22 s | 65.2 ± 0.3 / 10 s | 21.3 ± 3.9 / 11 s |
| `gibbs-anneal` | 99.4 ± 6.5 / 3 s | 68.4 ± 0.6 / 2 s | 54.4 ± 1.8 / 2 s |
| `data` | 103.0 ± 22.4 / 5 s | 46.2 ± 28.8 / 2 s | 18.5 ± 5.3 / 2 s |
| `burn-in` | 104.9 ± 4.5 / 2 s | 70.6 ± 4.1 / 2 s | 50.7 ± 3.1 / 2 s |
| `gaussian-em` | refused | refused | refused |

The figure this table drew, `plots/studies/hmm_starts.png`, is retired: #540's
`docs/nb/copy_state_starts.ipynb` draws runtime against gap for every start
on both the BAF-only and the BAF + RDR stage.

## Failings, and what was done about them

- **`gaussian-em` refuses on every sample.** A component's variance collapses
  to 1.7e-20 / 1.8e-10 / 6.7e-206, and `sal` refuses rather than floor it. The
  rate rows of zero-count bins are exact duplicates at 0. Not competitive as
  it stands; the informative-seeding arm below removes those rows.
- **The surrogate starts hand over one fit between them:** `objective`,
  `restart`, `quantile`, `anneal`, `tempering`, and `perturbed` within a nat.
  The Gaussian surrogate reads rate rows whose noise is the inverse of their
  counts, so noisy low-count bins set its optimum. #502 carries weighting and
  trimming by signal-to-noise.
- **Every fit spends several states on near-duplicates of neutral:** 4 of 7
  within 0.03 of neutral log μ on 60 × 50. A covariate-aware
  merge-and-reseed polish was tried as the fix for the competitive start: two
  components within 0.1 log μ and 0.05 p, the later one re-placed by D²
  sampling and re-polished, kept only if the likelihood rises. On easy it kept
  no move: the duplicates are what the mixture likelihood prefers, so a
  likelihood-driven move cannot remove them. Not adopted.
  `sal.opt.split_merge` refuses a covariate, which is the upstream change
  this would otherwise use.

**Seeding from informative bins only** (counts > 0 and trials ≥ median; the
fit still reads every bin) was measured on every start. Gap before → after, in
nats:

| start | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `gaussian-em` | refused → 49.5 | refused → 129.3 | refused → 45.3 |
| `objective` (the surrogate group) | 86.7 → 86.5 | 66.5 → 66.9 | 28.7 → 30.9 |
| `kmeans++x5+em` | 17.2 → 18.4 | 4.5 → 26.2 | 13.2 → 8.2 |
| `emission++x5+em` | 45.6 → 62.1 | 8.2 → 9.6 | 19.3 → 20.5 |
| `datax5+em` | 33.7 → 34.1 | 7.3 → 7.9 | 10.4 → 25.6 |

The gaps are against the best of both runs, so they differ slightly from the
table above.
- Trimming makes `gaussian-em` run, but it is not competitive.
- It does not move the surrogate group.
- It is mixed for the leading start.

Not adopted. What remains for the surrogate is weighting rows rather than
trimming them (#502).

## End to end

`--sal` on #500 (+ #496's loader), `--hmm-start` as named. Clone ARI (clones)
/ copy ARI / phase-free exact altered:

| start | 60 × 50 | easy | hard |
| --- | --- | --- | --- |
| `distinct` (before) | 1.0 (4) / 0.9828 / 0.9348 | 0.9861 (4) / 0.8984 / 0.6299 | 0.8652 (5) / 0.8652 / 0.4970 |
| `kmeans++x5+em` | 1.0 (4) / 0.9828 / 0.9348 | 0.9861 (4) / 0.8984 / 0.6299 | **0.9829 (4) / 0.9055 / 0.6272** |
| `emission++x5+em` | same | same | same |
| `datax5+em` | same | same | same |

The three hand the HMM different states, and it converges to the same fit
from each. What they share, and `distinct` lacks, is a start that includes the
deletion/LOH state (p ≈ 0.08, log μ ≈ −0.67) and a gain (log μ ≈ 0.2).

Through `--sal` itself on this branch, where `--sal` installs `kmeans++x5+em`
(`--time-stages` on):

| sample | clone ARI (clones) | copy ARI | exact altered (phase-free) | wall | peak |
| --- | --- | --- | --- | --- | --- |
| 60 × 50 | 1.0 (4) | 0.9828 | 0.9197 (0.9348) | 224 s | 7.67 GB |
| easy | 0.9861 (4) | 0.8984 | 0.252 (0.6299) | 110 s | 4.16 GB |
| hard | 0.9829 (4) | 0.9055 | 0.0355 (0.6272) | 78 s | 4.16 GB |

## Upstream correspondence

Every start here is `sal`'s as shipped. Two things would need a `sal` change:
a covariate in `split_and_merge`, and weights in the Gaussian surrogate and D²
seeding (#502).
