# Paper figures: dev_tree_1s_easy r0 (7ba9b01f)

**TL;DR:** clone ARI 0.9798, copy ARI (phase-free)
0.9730, exact altered (phase-free)
0.9729 (phased 0.2209), from one
`run_cnaster_port --sal --png-copies` run on `sim/manifests/dev_tree_1s_easy.toml` r0 at
code `b816552`: 260 s wall, 3.42 GB peak.
Ledger `run_id` `b816552-dev_tree_1s_easy_r0-1508` (`docs/metrics/`, fixture `dev_tree_1s_easy_r0` `7ba9b01f`).

Regenerate from a clean tree, so the commit above carries no `+`; it draws r0 into
`.cache/paper_figures/` where `--draw` is not given, and refuses any r0 not
hashing to `7ba9b01f`:

    run_study --paper-figures --fixture dev_tree_1s_easy --out docs/plots/paper

`--truth-only` draws the truth alone, with no run. This directory commits the four
headline figures alone (#745); every other page the run draws (the truth's panels,
the run's own pages, cnaster's copies and the truth-against-fit comparisons) is
written under `.cache/plots/paper/` and regenerated on demand (`curate`). The figures
carry no stamp (#743): `truth_combined.png` and `combined.png` are `dev_tree_1s_easy` r0
`7ba9b01f` at code `b816552`, as above. `combined.png` draws panel (a) on a slide
mocked from the planted labels (`port.sim.he_slide`): the fixture has no H&E image,
and the run never reads the mock. `solver_combined.png` is not drawn from this fixture:
`--solvers POTTS.record COPY.record` draws it from a `port.studies.potts_stream` and a
`port.studies.copy_state_stream` record, and `solver_combined.md` beside it names both
records, their data hashes and their settings. `pop_combined.png` is the population
study's, its data in `docs/studies/`. The genomic panels of `truth_combined.png` and
`combined.png` draw every altered bin at 2x its extent (`port.extensions.genomic_axis`,
T- #683), each on its own segmentation, so their contig widths differ.

| File | Question | Source |
| --- | --- | --- |
| `truth_combined.png` | What was planted, on one page: tree, (A, B) profile, RDR and BAF per tumour clone, and the phase switches? | `port.sim.truth_figure.truth_combined_figure` |
| `truth_combined_multisample.png` | The same, for the fixture's multi-sample counterpart: the same clones and laws on two overlapping slices, phase switch errors on? | `multisample_pages`: `truth_combined_figure` on `MULTISAMPLE[fixture]`'s r0 (`sim/manifests/dev_tree_easy.toml`) |
| `spatial_multisample.png` | Where is each true clone on each of the counterpart's slices? | `multisample_pages`: `port.sim.analysis.plot_spatial` |
| `he_multisample.png` | What H&E slide do the counterpart's planted clones stain, slice by slice? | `he_slices_figure`: `port.sim.he_slide.mock_he` per slice |
| `combined.png` | What did the run fit, genome and array on one page? | `port.extensions.combined_figure.combined_figure` |
| `pop_combined.png` | How many UMIs does a clone need, how long must a CNA be, at each stay probability 1 - t, and how often is a true-(1, 1) segment called altered? | `port.studies.population_report.combined` on `docs/studies/population_summary_limited.json` (#745's limited rerun; #746 the full) |
| `solver_combined.md` | What was `solver_combined.png` drawn from? | `solver_note`: both records' data hashes, their settings and the code |
| `solver_combined.png` | How far above the best does each spatial solver and each copy-state start end, and how fast? | `solver_figure`: `port.studies.potts_plot.draw`, `port.studies.copy_state_plot.draw` |
| `solver_combined_forward.md` | What was `solver_combined_forward.png` drawn from, and what did each polish spend? | `polish_note`: both records' data hashes, median seconds, passes and forward-backward passes |
| `solver_combined_forward.png` | Does the copy-state starts' ranking survive fitting by L-BFGS on the forward log-likelihood in place of EM? | `polish_figure`: `port.studies.copy_state_plot.draw` on a `--polish em` and a `--polish forward` record (#748) |

## Key studies

None committed: `554_clone-starts.png` and `557_copy-states.png` are superseded by
`solver_combined.png` (a) and (b) (#745).
