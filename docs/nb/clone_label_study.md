# Clone-label starts × Potts solvers at dev_tree (#541)

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
  - The samplers stop 0.01 above; Wolff 1.2; `bifurcation` 3.4.
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
  alternating from the same start (mean field: 0.9985 and 0.9993 against
  0.9989), and costs 1.3–1.6× the wall.
- **The coupling:** ×0.1 costs 0.10–0.19 ARI from every start; ×10 changes
  it by at most 0.003 but from the posterior draw.

**End to end: no start is adopted, and `--sal` keeps its own.** Clone ARI
(clones) / copy ARI. Each run puts the start in place of the first clone
assignment of each stage, on that fit's field, with `--sal` otherwise
(`tests.studies.clone_labels e2e`, #553's `--sal`):

| first assignment | dev_tree r0 (`3381575a`) | dev_shared_unique r0 (`097bb52b`) | CalicoST easy (`23989aa4`) | CalicoST hard (`1ae26365`) |
| --- | --- | --- | --- | --- |
| `--sal` (none) | 1.0 (4) / 0.9825 | 0.9971 (4) / 0.9941 | 0.9861 (4) / 0.9035 | 0.9829 (4) / 0.9181 |
| mean field | 1.0 (4) / 0.9825 | 0.9983 (4) / 0.9577 | 0.9861 (4) / 0.9035 | 0.9838 (4) / 0.9135 |
| agglomerative | 0.9587 (4) / 0.9767 | 0.9971 (4) / 0.9941 | 0.9861 (4) / 0.9035 | 0.9838 (4) / 0.9135 |
| smoothed argmax 1 | 0.9993 (4) / 0.9825 | 0.9378 (5) / 0.9556 | 0.2783 (5) / 0.7313 | 0.8724 (4) / 0.9139 |

- **The stall the study finds is already gone end to end.** The pipeline
  starts the BAF + RDR stage from the BAF stage's clones, not from
  `grid2`, and with #547's seeding those give dev_tree 1.0 (4).
- **Mean field** gains 0.001 clone ARI on dev_shared_unique and hard, and
  costs 0.036 and 0.005 copy ARI there. It doubles dev_shared_unique's wall
  (171 against 90 s).
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
- `sal`'s EM refused the states of every posterior draw and of spectral
  clustering ("trials must be >= 2", "M step did not settle"). Those rows
  take the lattice's states (`states_by`), polished where `sal` allowed.
- Stochastic starts: seeds 0–9, solved on 0–2. Stochastic solvers: 3 seeds.
  `field_argmax`, `tempering`, `max-product` and `bifurcation` take no
  start by design.
- 4 spawned workers on the 4-core host; seconds are per job, not isolated.
  End-to-end runs: 3 at a time, one core each.
- Regenerate with
  `python -m tests.studies.clone_labels capture SAMPLE CAPTURE.npz`, then
  `run CAPTURE.npz OUT.pkl`, then
  `python -m tests.studies.clone_label_notebook OUT.pkl`.
