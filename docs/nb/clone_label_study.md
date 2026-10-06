# Clone-label starts × Potts solvers at dev_tree (#541)

**Rerun at 7d1ba8b (sal b61dfba, #633):** the numbers below are the rerun's. The
deterministic starts and every graph cut reproduce the original. sal b61dfba moves the stochastic paths:
its Potts samplers draw different chains, and its EM (#1136) now fits spectral clustering's states
(0/31 rows fall back to the lattice, against 29/29) and most posterior draws (33/100 fall back, against
92/94). It also adds two solvers, `sal:wolff-heat-bath` and `sal:swendsen-wang-heat-bath` (#1142). The
end-to-end runs ran serially at default threads: at 1 thread, 6 of 16 differ (#638).

**TL;DR.** On dev_tree 60 × 50 r0 (`3381575a`; 6,000 spots, 2 slices), with copy states from
#540's `kmeans++x5+em` and fields built from each start's labels:

- **The solver does not decide the outcome; the start does.** Every graph-cut
  row reaches TRW-S's certified bound on 27 of 29 problems, and is within
  3e-4 nats per spot of it on the other two. The rows are `sal`'s
  `alpha-expansion` and `alpha-beta-swap`, and port's `alpha`,
  `alpha-rust`, `-merge` and `-fuse-merge`. They end at the same labels, in
  0.03–0.06 s (port's pure-Python `alpha`: 2.7 s).
  - ICM, `sal`'s or `cnaster`'s, stops 0.15–0.18 nats per spot above the
    bound.
  - The samplers stop 0.01 above (Swendsen-Wang heat bath 0.004); Wolff and its heat bath 1.2;
    `bifurcation` 3.4.
  - As #492 found, `--sal`'s `alpha-rust-fuse-merge` is as good as any.
- **`grid2`, the pipeline's start, locks its error in.** From its rectangles
  the solve plus floor ends at clone ARI 0.856 with 5 clones. Alternating
  (solve, floor, refit, rebuild) stalls at 0.8605: main's `--sal` dev_tree
  result before #547, 0.8612 with 5 clones.
- **A start that reads `grid2`'s field reaches the planted clones, and so
  does over-partitioning.** Alternated with `--sal`'s solver, each ends at
  clone ARI 0.9989–1.0 with 4 clones:
  - mean field, 0.03 s past the field;
  - 1-hop smoothed argmax;
  - agglomerative clustering along the graph;
  - `grid3` relabelled by the field;
  - a posterior draw;
  - `grid3` itself: 18 rectangles, which the solver and floor merge to 4
    by round 3.

  Each lets the first refit fit clones rather than `grid2`'s 8 rectangles.
  The globally optimal labelling of `grid2`'s own field is not such a start,
  and ends at 0.856.
- **Data-blind starts with too few patches do not:** uniform, Wolff at
  J ≥ 1 (whole slices), spectral clustering (0.57). `umi_grow` and Wolff at
  J = 0.5 reach 0.90–0.92 with 3 clones. `normal-first` (0.86, 5 clones)
  inherits `grid2`'s rectangles outside the normal clone. The field-free
  `BAF agglomerative` reaches 0.92 (3 clones).
- **Joint annealing or sampling of states and labels adds nothing** over
  alternating from the same start (mean field: 0.8597 and 0.9985 against
  0.9989; one chain each, 0.9985 and 0.9993 in the original), and costs
  1.3–1.6× the wall.
- **The coupling:** ×0.1 costs 0.085–0.20 ARI from `grid2`, mean field and
  `normal-first`, and 0.33 from the posterior draw; ×10 changes those three by
  −0.002 to +0.011 and the posterior draw by +0.14 (one draw, seed 0).

**End to end: no start is adopted, and `--sal` keeps its own.** Clone ARI
(clones) / copy ARI. Each run puts the start in place of the first clone
assignment of each stage, on that fit's field, with `--sal` otherwise
(`port.studies.clone_labels e2e`, #553's `--sal`):

| first assignment | dev_tree r0 (`3381575a`) | dev_shared_unique r0 (`097bb52b`) | CalicoST easy (`2d4ce9a9`) | CalicoST hard (`8797710b`) |
| --- | --- | --- | --- | --- |
| `--sal` (none) | 1.0 (4) / 0.9825 | 0.9983 (4) / 0.9941 | 0.9861 (4) / 0.9035 | 0.9829 (4) / 0.9181 |
| mean field | 1.0 (4) / 0.9825 | 0.9971 (4) / 0.9941 | 0.9861 (4) / 0.9035 | 0.9838 (4) / 0.9135 |
| agglomerative | 0.9587 (4) / 0.9767 | 0.997 (4) / 0.9941 | 0.9861 (4) / 0.9035 | 0.9838 (4) / 0.9135 |
| smoothed argmax 1 | 0.9993 (4) / 0.9825 | 0.926 (5) / 0.9597 | 0.2764 (5) / 0.7313 | 0.8793 (4) / 0.9141 |

- **The stall the study finds is already gone end to end.** The pipeline
  starts the BAF + RDR stage from the BAF stage's clones, not from
  `grid2`, and with #547's seeding those give dev_tree 1.0 (4).
- **Mean field** gains 0.001 clone ARI on hard and costs 0.005 copy ARI there; on
  dev_shared_unique it now costs 0.0012 clone ARI at equal copy ARI (the original:
  +0.001 clone, −0.036 copy).
- **Agglomerative** costs dev_tree 0.041.
- **The smoothed argmax** breaks easy (0.28, 5 clones).

The starts stay in `port.sandbox.clone_starts` with these numbers; the
solver stays `alpha-rust-fuse-merge`.

**Conditions.**
- The capture is one `--sal --oracle-start` run of dev_tree r0 (`3381575a`; spot order
  checked against the truth). Its segmentation is that run's, the planted
  clones' BAF stage. Every field and start is built from labels alone, and
  the planted labels only score.
- The field-reading starts read `grid2`'s field: the profiles `grid2`'s
  labels give.
- `sal`'s EM refused the states of 33 of 100 posterior-draw rows (the
  original: 92 of 94, and all 29 of spectral clustering's). Those rows take
  the lattice's states (`states_by`).
- Stochastic starts: seeds 0–9, solved on 0–2. Stochastic solvers: 3 seeds.
  `field_argmax`, `tempering`, `max-product` and `bifurcation` take no
  start by design.
- 4 spawned workers on the 4-core host; seconds are per job, not isolated.
  End-to-end runs: one at a time, default threads (#638: the result depends on the
  thread count; 3 at once also exceeds the host's 15 GB on dev_tree).
- Regenerate with
  `run_study --clone-labels capture SAMPLE CAPTURE.npz`, then
  `run CAPTURE.npz OUT.pkl`, then
  `run_study --clone-label-notebook OUT.pkl`.
