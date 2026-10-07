# Study: recovery across a population of simulations (#544)

**TL;DR:** `run_cnaster_port --sal` on 679 runs, each drawn from its own seed of `population.toml` (seed 0 hashes to `1ef7403e`):
- **Clones.** A clone is detected at half its rate at **10^6.08 UMIs** (95% interval [6.01, 6.16], about 1.2 × 10⁶). The coupling J = 1.0 and 1.4 give the same value. At J = 0.8 and 2.8 the threshold is 0.05–0.21 dex higher, and the paired intervals exclude zero.
- **CNAs at J = 1.**
  - LOH is recovered at half its rate at **12.6 Mb** (L50 10^7.10).
  - Balanced gains are recovered at half their rate at **93 Mb** (10^7.97).
  - Imbalanced gains level off at about 0.4–0.5 recovery from 40 Mb upward. The cause: 57% of (1,2) bins are decoded as (1,1).
- **False positives.** A true-(1,1) segment is called something else at a rate of **1.08 × 10⁻³** ([0.90, 1.30] × 10⁻³). The rate is flat across its SNP UMIs.
- **Sufficiency.** The rule stated in advance passes except for imbalanced gain, whose L50 interval is 0.36 dex (limit 0.3). That interval stopped narrowing after 219 more members, because the crossing lies on the plateau.

**Rerun of the J = 1 arm at f4a0cc0 (sal b61dfba, #633):** the 445 members that ran at J = 1, one
thread per worker as before (#638: the result depends on the thread count). J = 0.8, 1.4 and 2.8 were not
rerun, so panel (a) of the rerun's figure is J = 1 alone, over the 193 base members, not the 76 paired.
- **Runs that raised: 0 of 445**, against 12 of 445 at J = 1 (13 of 679 over every J): none of the 11 `trials must be >= 2` or
  the 1 `M step did not settle` recurs. sal #1136 rewrote that EM; the cause is not isolated further.
- **Clones.** UMI50 10^6.08 [6.03, 6.13], against 10^6.12 [6.06, 6.17] (191 members; 2 raised).
- **CNAs.** LOH L50 10^7.10 [7.01, 7.19], unchanged. Balanced gains 10^7.81 [7.75, 7.87] (65 Mb), against
  10^7.97 [7.90, 8.06]: the intervals do not overlap. Imbalanced gains stay on the plateau, 10^8.08 [7.92, 8.28].
- **False positives.** 1.13 × 10⁻³ [0.95, 1.34] (1,536 of 1,358,202 segments), against 1.08 × 10⁻³ [0.90, 1.30].
- **Per member** on the 433 that ran in both: 173 have identical clone scores. Spot-weighted clone completeness
  has median 0.744 → 0.759, and 697 clones are detected against 682 of 1,299. 235 members move by more than
  0.001, 101 up and 134 down. Fitted clones exceed planted by 2 in 23 members, against 5.
- Key figure: `docs/plots/paper/key_studies/546_population.png` (`data 6daefbd0 · code f4a0cc0`). The
  numbers below are the original's.

The figure, sensitivity and false positive rate, is committed as
`plots/studies/population_recovery.png`; `run_study --population
report --out DIR` redraws it into `DIR/figures/`.

(a) Clone sensitivity against log10 clone UMIs, per J. (b) CNA sensitivity against length at J = 1, per copy-state class. (c) The false positive rate of true-(1,1) segments against the SNP-covering UMIs they hold, at J = 1.
- A clone counts as detected when at least 90% of its spots are in its matched fitted clone.
- A CNA counts as recovered when at least 90% of its bins decode to the planted pair, up to phase.
- Curves are weighted logistic fits, with 95% bands from a cluster bootstrap over members (2,000 resamples). Points are 0.1 dex bins, with the same resampled intervals.

The page is set as `combined.pdf`'s spatial row: 4.80 in wide, 7 pt text.

## Method

`run_study --population run | rescore | report` (#546).

- **Members.** One member is one seed of `sim/manifests/population.toml`: a 60 × 50 slice with 3 tumour clones.
  - Each clone's size is drawn log-uniform over 100–1,000 spots (`[layout.size]`).
  - The tree has 1 trunk event, 4 per leaf and 1 per internal node, with class-balanced copy states.
  - CNA lengths are exponential with mean 20 Mb and a floor of 1 Mb.
  - Counts come from the Pólya urn (#549).
  - `population_long.toml` is the same with a mean length of 60 Mb.
  - A member's hash is `port.sim.fixtures.realization_hash` of its drawn r0; the table names the first seed's of each manifest.
- **Runs.** Each member runs `run_cnaster_port --sal --no-plots` once per `hmrf.spatial_weight` J. J_c = ln 2 is the critical coupling of the q = 4 Potts model on the triangular lattice; J runs from 1.15 to 4 × J_c.
- **Scoring.**
  - Clones are matched by overlap on `clone_labels.tsv`.
  - CNAs are scored on the matched clone's integer copies in `cnv_seglevel.tsv`.
  - True-(1,1) segments are scored as called (1,1) or not, with the SNP A + B reads they hold in the planted clone's spots.
- **Sufficiency (`sufficiency`, stated before any number was read).** Every rule bin holds ≥ 20 items, and each crossing's 95% interval is ≤ 0.3 dex wide.

| Stage | Seeds | Manifest | J | Records |
| --- | --- | --- | --- | --- |
| 1 | 0–79 | population (seed 0 `1ef7403e`) | 0.8, 1.0, 1.4, 2.8 | 312 |
| 2 | 80–199 | population | 1.0 | 115 |
| 3 | 1000–1059 | population_long (seed 1000 `e68161e9`) | 1.0 | 58 |
| 4 | 1060–1259 | population_long | 1.0 | 194 |

- **Panel (a)** reads the 76 members that ran at every J. The comparison is paired: the same members and the same resamples at each J.
- **Panel (b)** reads all 433 members at J = 1; **panel (c)** reads the 364 of them in which at least one clone was detected, since a segment is scored only in a detected clone.
- **Runtime.** Median wall time per run is 90 s (90th percentile 138 s), on one thread per worker and 4 workers.

## Results

### (a) Clone detection against UMIs

| J | members | clones | UMI50, log10 [95%] | change against J = 1, dex [95%] |
| --- | --- | --- | --- | --- |
| 0.8 | 76 | 228 | 6.20 [6.12, 6.32] | [+0.05, +0.21] |
| 1.0 | 76 | 228 | 6.08 [6.01, 6.16] | — |
| 1.4 | 76 | 228 | 6.08 [6.02, 6.15] | [−0.05, +0.04], unresolved |
| 2.8 | 76 | 228 | 6.20 [6.13, 6.29] | [+0.05, +0.20] |

The detection threshold is lowest in the middle of the J range. At J = 0.8 the prior orders too little. At J = 2.8 it smooths small clones into their neighbours.

### (b) CNA recovery against length, J = 1

| class | events | L50, log10 bp [95%] |
| --- | --- | --- |
| LOH | 1,395 | 7.10 [6.98, 7.21] |
| balanced gain | 1,096 | 7.97 [7.89, 8.06] |
| imbalanced gain | 1,261 | 8.02 [7.87, 8.23], on the plateau |
| all | 3,752 | 7.75 [7.67, 7.83] |

Imbalanced-gain recovery is 0.37–0.48 from 10^7.4 to 10^8.4 bp. Decoded states, over 80 long-arm members:

| planted | bins | decoded as |
| --- | --- | --- |
| (1, 2) | 5,496 | (1, 1) 57.3%, (1, 2) 37.0%, (0, 2) 3.5%, (2, 2) 2.2% |
| (1, 3) | 5,021 | (1, 3) 62.7%, (1, 2) 19.7%, (1, 1) 8.5%, (2, 2) 8.1% |

A one-copy gain is called neutral more often than not, whatever its length. A logistic that rises to 1 therefore places L50 on the plateau, and more members do not narrow it:

| stage 4 records added | 0 | 51 | 102 | 153 | 194 |
| --- | --- | --- | --- | --- | --- |
| imbalanced-gain L50 interval, dex | 0.48 | 0.35 | 0.35 | 0.37 | 0.36 |

Whether the (1,1) calls come from the HMM's states or from the integer-copy decoding is not established here; the kept `cnv_seglevel.tsv` of every run allows the check without a rerun.

### (c) False positive rate of (1, 1) segments, J = 1

| SNP UMIs in segment, log10 | false positive rate [95%] | segments |
| --- | --- | --- |
| 0.75 | 1.85e-3 [0.97, 2.93] | 12,461 |
| 1.25 | 1.34e-3 [0.85, 1.94] | 269,916 |
| 1.75 | 0.99e-3 [0.84, 1.17] | 720,030 |
| 2.25 | 1.01e-3 [0.85, 1.19] | 354,901 |
| 2.75 | 1.29e-3 [0.79, 1.91] | 23,263 |
| all | 1.08e-3 [0.90, 1.30] | 1,382,099 (1,491 false) |

## Differences, failures and defects

- **Draws refused.** 15 of the 460 seeds could not be drawn: `no placement of clone_k in 1000 clears the others`. Three large clones do not always fit on 3,000 spots without overlap. A refused seed has no record, so the population is conditioned on its clones fitting, which slightly under-represents three near-1,000-spot clones. Panel (a) reads rates within a UMI bin, so its curves are conditional on UMIs; the conditioning bears on how the members fall across the bins, not on each bin's rate.
- **Runs that raised: 13 of 679.** The report counts each one.
  - 11 raised `ValueError('trials must be >= 2 and weight positive')` in sal's mixture-start EM (`sal/emissions/mstep.py`, snakes_and_ladders 3ad4b04): 2 of 193 base runs and 9 of 252 long-arm runs at J = 1. Recorded here for snakes_and_ladders, which this repository does not write to.
  - 1 raised `a component's M step did not settle at EM iteration 1` in sal (`sal/opt/emission_mixture.py:210`, seed 64, J = 1). Recorded here for snakes_and_ladders.
  - 1 raised `every dispersion must be positive` at seed 65, J = 2.8. This is a defect in port's own M-step gradient path, filed as #555.
- **cnaster reads the sample sheet without a string dtype**, so a numeric-looking sample id breaks `_create_clone_gridspec`. The generator redraws such ids (`test_no_sample_id_reads_as_a_number`). Recorded here for cnaster.
- **References.** The paper states no population-level sensitivity, so this study has no paper figure to agree or disagree with. CalicoST was not run on the population.

## Data and reproduction

- `studies/population_records.jsonl.gz`: every record, one JSON per run.
- `studies/population_summary.json` and `population_tables.md`; the figure as `plots/studies/population_recovery.png` (the `.pdf` is no longer committed).

```
run_study --population run --seeds 0:80 --J 0.8,1.0,1.4,2.8 --workers 4 --out DIR
run_study --population run --seeds 80:200 --J 1.0 --workers 4 --out DIR
run_study --population run --seeds 1000:1260 --J 1.0 --workers 4 \
    --manifest sim/manifests/population_long.toml --out DIR
run_study --population report --out DIR --study2-J 1.0
```

cnaster 4adad4d, snakes_and_ladders 3ad4b04, port c2fb3cf plus the figure commits on #546.
