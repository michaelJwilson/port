# Audit: recovered clones, states and integer copies against the planted truth

Issue #313. `cnaster` at `4adad4d`; `snakes_and_ladders` at `186bc59`; `port`
at `af2d87e` plus this branch. Harness: `python -m tests.recovery_audit`.

**The largest cause is one `cnaster` defect, #320: the normal candidates
admit whole tumor clones, 380 of 583 spots at the figure configuration and 440
of 879 at convergence.** The RDR baseline is their sum. So the normal clone
reads mu 0.34–0.55 at the admitted clones' gains, 2–5 fitted states go to
normal bins, and every amplification is fitted low. With the planted normal
spots handed in, exact integer copies on altered bins rise from 0.484 to 0.749,
and the truth beats the fit by 1,553 nats instead of losing by 5,809. The
oracle collapses two other configurations to one clone, so no fix is landed.
The rest of the audit's errors are configuration (5 fitted states against 8
used), the label solver (`--sal`, #312), and a fixture that could not referee
integer copies.

## Conditions

- 4 cores. Each arm is its own process, run one at a time on an otherwise idle
  host.
- Walls include `numba`'s per-process compile of the uncached lattices (#318
  removes it) and are secondary here: every claim below is against the truth.
- **Instances:** `dev` is 1,600 spots in 4 bands (480/400/360/360) and 1,000
  bins, with 10 planted states of which 8 are used. `lattice` is the same
  genome with `COPY_LATTICE` states, `2 mu = A + B` (9 planted, 8 used).
- **Configurations:** F is the figures', 5 states, `max_iter_outer=1`,
  `max_iter=3`. C is converged, planted `n_states`, 3 × 30.
- **Metrics:**
  - `state` is the share of clone-bins in the Hungarian-matched state;
  - `mu err` is the mean `|mu_fit - mu_planted|` per clone-bin;
  - `copies` is `A + B` equal to `2 mu` planted, over **altered** bins alone,
    because an all-normal decode scores 0.944 over all bins;
  - `cand` is normal candidates / of which tumor;
  - `fit - truth` is `-log P(x)` at the planted `(mu, p)` minus at the fit
    (`--likelihood`). Positive means the model prefers the fit.

## Results

| instance | config | arm | ARI | clones | state | mu err | copies (altered) | cand | fit − truth (nats) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | F | default | 0.919 | 4 | 0.573 | 0.186 | 0.000 | 583 / 380 | — |
| dev | F | `--sal` | 1.000 | 4 | 0.947 | 0.076 | 0.135 | — | — |
| dev | F | oracle normal | 0.581 | 7 | 0.622 | 0.096 | 0.170 | 480 / 0 | — |
| dev | C | default | 1.000 | 4 | 0.708 | 0.145 | 0.206 | 879 / 440 | +6,345 |
| dev | C | `--sal` | 1.000 | 4 | 0.573 | 0.050 | 0.377 | — | +2,829 |
| dev | C | oracle normal | 0.000 | 1 | — | diverged | — | 480 / 0 | — |
| lattice | F | default | 0.606 | 7 | 0.786 | 0.072 | 0.265 | — | — |
| lattice | F | `--sal` | 1.000 | 4 | 0.764 | 0.176 | 0.170 | — | — |
| lattice | F | oracle normal | 0.000 | 1 | 0.817 | 0.050 | — | 480 / 0 | — |
| lattice | C | default = `--sal` | 1.000 | 4 | 0.733 | 0.089 | 0.484 | 879 / 399 | +5,809 |
| lattice | C | oracle normal | 1.000 | 4 | 0.684 | **0.026** | **0.749** | 480 / 0 | **−1,553** |
| lattice | C | oracle normal, `--sal` | 1.000 | 4 | 0.681 | 0.028 | 0.691 | 480 / 0 | −1,671 |

At lattice C, `--sal` and the default return identical fits. The earlier
dev sweep at 5 states adds two rows:
- 1 × 30: default ARI 0.791 (6 clones), `--sal` 1.000;
- 3 × 30: default 0.587 (3 clones), `--sal` 1.000.

## Causes

| finding (#313's baseline) | cause | class | evidence |
| --- | --- | --- | --- |
| ARI 0.919, 48 spots of clone 3 labelled normal | the ICM label solve | label solver | `--sal` 1.000 at every configuration; the default falls to 0.587–0.791 at 5 states converged (#312) |
| five amplifications in one state | 5 fitted states against 8 used | configuration | at 8–10 states they separate under `--sal` (chr7 below) |
| normal state split three ways; fitted `mu` below 1 | tumor spots in the normal candidates | model input, `cnaster` defect (#320) | the normal clone's off-neutral bins: 55/55 (lattice) and 64/65 (dev) are bins amplified in an admitted clone; pseudobulk RDR 0.55 there against 0.52 predicted; fit beats truth by 5,809 nats and loses by 1,553 once removed |
| amplified `mu` fitted low | the same | #320 | lattice C: 1.30, 1.56, 1.88 against 1.5, 2.5–3, 2; oracle 1.47, 2.20–2.77, 1.97 |
| integer copy 3 for `2 mu` = 6.5 and 10 | the dev grid is not integer, and `cnaster` caps `A + B` at 6 (`max_total_copy`, not configurable) | fixture, and a stated `cnaster` limit | `COPY_LATTICE` fixture added, `copy_lattice=True` |
| a fitted deletion used by no bin | an extra state with nothing to hold | configuration | absent at C |

**#314 candidates, tested and not causes here:**
- #30, the M step stopping early: setting its hard-coded `ftol`/`gtol` to 1e-10
  reached 7 solves per run and changed no fitted value to four decimals, at dev
  F, dev C and lattice C. `hmm.em_ftol` is not read by the HMM's M step at all.
- #146, the constant transition, is what lets `fit − truth` be scored without
  state matching. It was not varied.

## chr7, clone II (bins 670–722)

Planted: 34 bins at `mu` 3.25, `p` 0.73, then 19 at 5.00, 0.88.

| config | arm | fitted states | integer copy |
| --- | --- | --- | --- |
| F | default | one: 1.507, 0.204 | 3, 3 |
| F | `--sal` | one: 2.49, 0.21 | 5, 5 |
| 8 states, 3 × 30 | default | one: 2.581, 0.167 | 5, 5 |
| 8 states, 3 × 30 | `--sal` | **two: 3.001, 0.281 and 4.127, 0.133** | 6, 6 |

The two events separate only with enough states **and** `--sal`.
- **The remaining `mu` gap is #320:** 0.92x and 0.83x of planted. Clone II
  itself is never a candidate; its deflation arrives through states shared
  with clones 1 and 3, which are (199 and 181 spots at F, 399 and 41 at C).
- **The copies are the decoder's cap:** 6.5 is no integer and 10 exceeds
  `max_total_copy = 6`, so no decoder output could be right. The lattice
  fixture's analog, (1,4) and (3,3), is what the copy column above referees.

## A realistic selector (sandbox)

`port.sandbox.normal_candidates.two_pass` takes the candidates from a first
run's fitted normal clone (#320). Scored with `--two-pass-normal`:

| instance | config | tumor candidates | ARI | mu err | copies (altered) |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | F | 49 | 0.859 | 0.035 | 0.471 |
| dev | C | 0 | 0.000 (1 clone) | diverged | — |
| lattice | F | 110 of 110 | 0.000 (1 clone) | — | — |
| lattice | C | 0 | 1.000 | 0.026 | 0.749 |

It equals the oracle wherever the first pass labels the clones exactly, and
inherits the oracle's collapse. It stays in the sandbox.

## Not established

- **Why the oracle collapses two runs to one clone.** The RDR-stage refinement
  merges 16 initial clones into one when the baseline is clean, and at dev C
  the fitted `mu` diverge to about 1e16. A correct baseline exposes this second
  defect, so #320 lands no candidate selector until it is understood.
- **Stage-by-stage truth injection** (planted clones into the HMM, planted
  states into the HMRF) was not run. The oracle-normal arm is the one stage
  injected.
