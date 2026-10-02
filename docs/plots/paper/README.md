# Paper figures: dev_tree_1s_easy r0 (22a5eb85)

**TL;DR:** clone ARI 0.9780, copy ARI (phase-free)
0.9784, exact altered (phase-free)
0.9646 (phased 0.5310), from one
`run_cnaster_port --sal --png-copies` run on `sim/manifests/dev_tree_1s_easy.toml` r0 at
code `0eaac23`: 226 s wall, 3.54 GB peak.
Ledger `run_id` `0eaac23-dev_tree_1s_easy_ln_r0-1935` (`docs/metrics/`, fixture `dev_tree_1s_easy_ln_r0`).

Regenerate from a clean tree, so the stamp carries no `+`; it draws r0 into
`.cache/paper_figures/` where `--draw` is not given, and refuses any r0 not
hashing to `22a5eb85`:

    python -m tests.studies.paper_figures --fixture dev_tree_1s_easy --out docs/plots/paper

`--truth-only` writes `truth/` alone, with no run. Every figure is stamped
`dev_tree_1s_easy 22a5eb85 · code <sha>`. `run/spatial.png` and `run/combined.png`
draw panel (a) on a slide mocked from the planted labels (`tests.he_slide`):
the fixture has no H&E image, and the run never reads the mock.
`truth/phase.png` is flat: this r0 plants 0 phase switches.

| File | Question | Source |
| --- | --- | --- |
| `truth/truth_combined.png` | What was planted, on one page: tree, (A, B) profile, RDR and BAF per clone, spatial clones? | `port.sim.truth_figure.truth_combined_figure` |
| `truth/mutation_tree.png` | Which events sit on which edge of the clone tree? | `port.sim.analysis.plot_tree` |
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
| `compare/clones_truth_vs_fit.png` | Do the fitted clones recover the planted ones, and which matches which? | `labels_figure`: `tests.sim_audit.score`'s ARI and matching |
| `compare/copy_confusion.png` | Which (A, B) is each planted pair decoded as? | `confusion_figure`: `tests.sim_audit.copy_confusion` |
| `compare/copy_genomic_truth_vs_fit.png` | Where along the genome is a matched clone's (A, B) decoded wrong, or swapped? | `genomic_compare_figure`: `score`'s clone-bins |
| `compare/exact_by_class.png` | Which planted classes are recovered exactly, with and without phase? | `exact_figure`: `tests.sim_audit.planted_classes` |

## Key studies

Each figure is redrawn when its study is rerun, and stamped `data <hash> · code <sha>`.

| File | Question | Source | Regenerate |
| --- | --- | --- | --- |
| `key_studies/546_population.png` | At J = 1, how many UMIs does a clone need to be detected, how long must a CNA be to be recovered, and how often is a true-(1,1) segment called altered? | `tests.studies.population_report.figures` (#544, PR #546) | `python -m tests.studies.population run --seeds 0:200 --J 1 --out DIR`, then `... run --seeds 1000:1260 --J 1 --manifest sim/manifests/population_long.toml --out DIR`, then `... report --out DIR --study2-J 1` |
