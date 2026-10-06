# Paper figures: dev_tree_1s_easy r0 (7ba9b01f)

**TL;DR:** clone ARI 0.9798, copy ARI (phase-free)
0.9335, exact altered (phase-free)
0.8992 (phased 0.1473), from one
`run_cnaster_port --sal --png-copies` run on `sim/manifests/dev_tree_1s_easy.toml` r0 at
code `8bed5a1`: 127 s wall, 3.42 GB peak.
Ledger `run_id` `8bed5a1-dev_tree_1s_easy_r0_7ba9b01f-2114` (`docs/metrics/`, fixture `dev_tree_1s_easy_r0_7ba9b01f`).

Regenerate from a clean tree, so the stamp carries no `+`; it draws r0 into
`.cache/paper_figures/` where `--draw` is not given, and refuses any r0 not
hashing to `7ba9b01f`:

    run_study --paper-figures --fixture dev_tree_1s_easy --out docs/plots/paper

`--truth-only` writes `truth/` alone, with no run. Every figure is stamped
`dev_tree_1s_easy 7ba9b01f · code <sha>`. `run/spatial.png` and `run/combined.png`
draw panel (a) on a slide mocked from the planted labels (`port.sim.he_slide`):
the fixture has no H&E image, and the run never reads the mock.
`truth/phase.png` is flat: this r0 plants 0 phase switches.
`solver_combined.png` is not drawn from this fixture: `--solvers POTTS.record COPY.record`
draws it from a `port.studies.potts_stream` and a `port.studies.copy_state_stream` record,
and its stamp names both records' data hashes.
The genomic panels of `truth/truth_combined.png`, `truth/clones_genomic.png`,
`truth/clone_profiles.png`, `run/genomic.png`, `run/combined.png` and
`compare/copy_genomic_truth_vs_fit.png` draw every altered bin at 2x its extent
(`port.extensions.genomic_axis`, T- #683), and their stamps end `· axis: altered` and the scale used.

| File | Question | Source |
| --- | --- | --- |
| `truth/truth_combined.png` | What was planted, on one page: tree, (A, B) profile, RDR and BAF per clone? | `port.sim.truth_figure.truth_combined_figure` |
| `truth/simulated_tree.png` | Which events sit on which edge of the simulated clone tree? | `port.sim.truth_figure.simulated_tree_figure`, `truth_combined`'s panel (a) alone |
| `truth/spatial.png` | Which clone was each spot drawn from? | `port.sim.analysis.plot_spatial` |
| `truth/clones_genomic.png` | What RDR and BAF does each planted clone give along the genome? | `port.sim.analysis.plot_clones_genomic_truth` |
| `truth/clone_profiles.png` | What (A, B) does each clone carry along the genome? | `port.sim.analysis.plot_clone_profiles` |
| `truth/coverage.png` | Do spot UMI and SNP reads follow their laws? | `port.sim.analysis.plot_coverage` |
| `truth/phase.png` | Where does the planted phase switch? | `port.sim.analysis.plot_phase` |
| `truth/baseline.png` | What normal expression baseline were counts drawn from? | `port.sim.analysis.plot_baseline` |
| `run/combined.png` | What did the run fit, genome and array on one page? | `port.extensions.combined_figure.combined_figure` |
| `run/genomic.png` | What RDR, BAF and integer (A, B) did the run fit per clone? | `port.extensions.combined_figure.genomic_figure` |
| `run/spatial.png` | Where are the fitted clones on the array? | `port.extensions.combined_figure.spatial_figure` |
| `run/copy_number_profile.png` | What integer (A, B) did the run decode per clone? | the run's `plots/copy_number_profile.png` |
| `run/clones_spatial.png` | Which fitted clone is each spot? | the run's `plots/clones_spatial.png` |
| `run/clones_genomic.png` | What RDR and BAF did the run fit per clone? | the run's `plots/clones_genomic.png` |
| `run/rdr_baf_clones_genomic.png` | What RDR and BAF were the clones fitted to? | the run's `plots/rdr_baf_clones_genomic.png` |
| `compare/clones_truth_vs_fit.png` | Do the fitted clones recover the planted ones, and which matches which? | `labels_figure`: `port.qa.audit.score_sample`'s ARI and matching |
| `compare/copy_confusion.png` | Which (A, B) is each planted pair decoded as? | `confusion_figure`: `port.qa.scoring.copy_confusion` |
| `compare/copy_genomic_truth_vs_fit.png` | Where along the genome is a matched clone's (A, B) decoded wrong, or swapped? | `genomic_compare_figure`: `score`'s clone-bins |
| `compare/exact_by_class.png` | Which planted classes are recovered exactly, with and without phase? | `exact_figure`: `port.qa.scoring.exact_by_class` |
| `solver_combined.png` | How far above the best does each spatial solver and each copy-state start end, and how fast? | `solver_figure`: `port.studies.potts_plot.draw`, `port.studies.copy_state_plot.draw` |

## Key studies

Each figure is redrawn when its study is rerun, and stamped `data <hash> · code <sha>`.

| File | Question | Source | Regenerate |
| --- | --- | --- | --- |
| `key_studies/557_copy-states.png` | Which copy-state start, polished by `--sal` Baum-Welch, recovers the planted states at known clones? | `port.studies.copy_state_plot` (#540, PR #557) | `run_study --copy-state-stream sim/manifests/baseline/dev_tree_1s_hard.toml OUT --problems 10 --seeds 10 --held-out 3 --settings python/port/studies/copy_sampler_settings.json` |
| `key_studies/554_clone-starts.png` | Does the clone-label start or the Potts solver decide the clones, and what does each start reach? | `docs/nb/clone_label_study.ipynb` via `port.studies.clone_label_notebook` (#541, PR #554) | `run_study --clone-labels capture SAMPLE CAPTURE.npz`, `... run CAPTURE.npz OUT.pkl`, `run_study --clone-label-notebook OUT.pkl` |
| `key_studies/546_population.png` | At J = 1, how many UMIs does a clone need to be detected, how long must a CNA be to be recovered, and how often is a true-(1,1) segment called altered? | `port.studies.population_report.figures` (#544, PR #546) | `run_study --population run --seeds 0:200 --J 1 --out DIR`, then `... run --seeds 1000:1260 --J 1 --manifest sim/manifests/population_long.toml --out DIR`, then `... report --out DIR --study2-J 1` |
