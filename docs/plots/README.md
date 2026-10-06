# Figures

**No PNG is committed here outside the two exceptions below; each figure
is regenerated on demand by the command below that draws it.** Every
generator writes to `.cache/plots/` by default (`port.qa.provenance.PLOTS`),
untracked; the paths below are relative to it. `lattice/`, `sim/` and
`sim_qa/` exist only there now.
`tests/test_ci_entry.py` guards that `docs/` tracks no PNG outside
`docs/plots/paper/`, T- #624's paper set, and
`studies/population_recovery.png`, whose runs survive only as
`studies/population_records.jsonl.gz`. The figures committed before are in
history: `git show ba34716:docs/plots/<path>.png`.

| figures | command |
| --- | --- |
| the dev instance's, below, and `lattice/` | `run_figures [--out DIR]` (`--cnaster` for plain `cnaster`), or `python -m tests.ci --figures` |
| `realizations.png`, `realizations_truth.png`, `realizations.npz` | `run_audit --errors [--output PATH]` |
| `realizations_copies.png` | `run_audit --copy [--output PATH]` |
| `metrics_history.png`, `metrics_history_classes.png` | `run_study --metrics-history [OUT.png [OUT_CLASSES.png]]` |
| `sim_qa/` | `python -m port.sim.analysis plot sim/generated/dev_tree/r0` (writes `<r>/qa/`) |
| `sim/cna_lengths.png` | `run_study --cna-lengths [OUT.png]` |
| `studies/potts_*.png` | `run_study --potts-plot STREAM.pkl` (`docs/study-field-strength.md`) |
| `studies/copy_states_*.png` | `run_study --copy-state-plot STREAM.pkl` (`docs/study-copy-states.md`) |
| `studies/copy_state_starts.png`, `studies/clone_label_study.png` | `run_study --copy-start-notebook RESULTS.pkl`, `run_study --clone-label-notebook RESULTS.pkl` |
| `studies/population_recovery.png` | `run_study --population report --out DIR` |

**CI draws the dev instance's and the realization figures on every pull
request and uploads them** as a workflow artifact
(`.github/workflows/figures.yml`); it no longer commits them (#296's commit
step is removed). The realization figures are eight runs of one planted
genome with the likelihood's errors on one run and on the truth (#291).
**They are PNG since #452**: `write_fig` writes one beside each PDF, without
metadata, and the PDFs stay in the run directory.

`studies/` keeps the population study's outputs
(`population_records.jsonl.gz`, `population_summary.json`,
`population_tables.md`, `population_recovery.png`) and
`potts_solvers_table.tex`.

## The dev instance

`M = 4` clones, `K = 10` planted states, `G = 1,000` bins over ten unequal
chromosomes, `S = 1,000` spots. **Five** states are fitted, not the ten
planted: ten does not fit in 15 GB (#90), and asking for fewer states than the
data carries is what a real run does anyway.

State zero is planted diploid and balanced, `mu = 1` and `p = 0.5`. Without
one the run does not reach these figures at all (#106).

Through `run_cnaster_port` with its defaults, the shift included: 54 s end to
end and a peak of 4.25 GB (#304). Plain `cnaster` measured 31 s and 5.89 GB
before #298's normal clone, which made it loop forever (#304).

## What they are, and are not

A figure is a **result**, not a referee. Nothing compares one against a
previous one. A matplotlib PDF carries its creation time, so two
regenerations differed byte for byte with nothing having changed; the PNGs
carry no timestamp, so two runs differ byte for byte only when their pixels
do (#452, toward #103's byte reproduction).

The run they come from is a completion claim, not a correctness one. No number
in these plots has been compared against the planted truth; that is the
component-wise work around `tests/test_run_cnaster_round_trip.py`.

## The order the run draws them

| stage | figures |
| --- | --- |
| before phasing | `pseudobulk_clones_genomic`, `prephasing_clones_genomic`, `prephasing_clones_spatial` |
| after phasing | `postphasing_clones_genomic`, `postphasing_aggr_clones_genomic`, `postphasing_pseudobulk_clones_genomic` |
| initial clones | `initial_clones_spatial` |
| BAF only | `bafonly_clones_genomic`, `bafonly_clones_spatial`, `merged_bafonly_clones_genomic`, `merged_bafonly_clones_spatial` |
| RDR and BAF | `rdr_baf_clones_genomic`, `rdr_baf_clones_spatial`, `merged_rdr_baf_clones_genomic`, `merged_rdr_baf_clones_spatial` |
| final | `real_clones_genomic`, `clones_genomic`, `clones_spatial`, `copy_number_profile` |
| after the run | `genomic`, `spatial`, `combined` |

## `genomic.png`, `spatial.png` and `combined.png`

The final four as two figures at a text column (#309, #280, #339).
`genomic.png` is 122 mm wide and `llncs`'s 193 mm text height less 1.5 in
(`CAPTION_ROOM`), so its caption fits on the page: (a)
`clones_genomic` over (b) `copy_number_profile`, (b)'s axis spanning (a)'s
tracks so their chromosome boundaries line up. `spatial.png` is 122 mm wide
and about a quarter of the block tall: (a) an H&E slide and (b) `clones_spatial`,
square and as large as fit across, (b)'s clones keyed on the right edge
and named by their integer copy profile (#344).
`combined.png` is both on one page, the full 122 by 193 mm: the spatial
figure at the head as (a) and (b), the genomic figure drawn the rest of the
height below as (c) and (d). No captions. `port.extensions.combined_figure` redraws the run's own calls and
writes each page at exactly its size, so it is included at
`width=\linewidth` unscaled.

The slide is **mocked** from the planted labels (`python/port/sim/he_slide.py`) and
read back through `cnaster.he.get_he_image`, as `run_cnaster` reads a slide.
It is
written beside the run's inputs, not into them: in them, `load_input_data`
would pick it up and refine the initial clones by it, and the other figures
would change.

## `sim_qa/`: what a simulated sample planted (#452)

`python -m port.sim.analysis plot sim/generated/dev_tree/r0` writes these to
`sim/generated/dev_tree/r0/qa/`: one realization
of `sim/manifests/dev_tree.toml` (a mutation tree over three clones, two slices
overlapping by half) at seed 0.

- `clone_profiles.png`: the planted `(A, B)` per clone, drawn by `port`'s
  `plot_copy_number_profile` on the truth binned at 1 Mb, the plotter
  `combined.pdf` uses, so a planted and a decoded profile share palette,
  hatching, outlines and key;
- `mutation_tree.png`: along event order, each event `chr::A/B::Mb` (whole Mb) at its time
  on its edge; each node its whole binary barcode, the founder's event the
  leading bit, and over 10 events no events on the edges (PR- #701);
- `spatial.png`: each slice, titled by its `sample_id`, cropped to itself in
  the shared frame; the region the slices share dashed, and a clone on both
  slices inside it; clones named by one legend for every slice, on the left,
  and not on the tissue;
- `phase.png`: switches accumulated along each chromosome per Mb of it, each
  contig's switches per Mb above it, formatted as `plot_clones_genomic`'s tracks;
- `baseline.png`: `log10 lambda` per gene as `normal_baseline.txt.gz` states it,
  the share of genes at 0 as the dropout rate, formatted as
  `plot_clones_genomic`'s tracks;
- `clones_genomic.png`: `plot_clones_genomic` over the true clone labels, on
  1 Mb bins: each clone's pseudobulk RDR against `lambda`, and BAF in the
  planted phase;
- `coverage.png`: `log10` of each nonzero entry the realization wrote, UMI per
  (gene, spot) and SNP-covering UMI per (SNP, spot), the latter against the
  `snp_spot_umi` law it is drawn from. Genes are drawn per spot, so no
  per-entry law is shown for them.

- `truth_combined.png`: `python -m port.sim.truth_figure`, written as
  `truth_combined.pdf` by `plot`: the tree (over 10 events, no events on its edges,
  PR- #701), the profiles, the tracks and the
  spatial map on one page at `combined.pdf`'s 122 mm by 193 mm and 7 pt, the
  profile and tracks on one left and right edge, clones as $m_N$, $m_1$, ...;
  the 10 Mb marks and the chromosome names on the last track alone, no Mb
  numbers, and every chromosome boundary on (b) and each track (PR- #701)

Clones carry `cnaster`'s numerals in every other figure: `Clone 0` is the normal.
Every truth figure numbers the clones down the drawn tree, normal first
(`analysis.tree_order`, which `analysis.read` applies to `Realization.clones`):
$m_k$ and `Clone k` are the tree's k-th tumour clone, which need not be
`clone_{k-1}` in the truth files (PR- #701).
`python -m port.sim.analysis population <sample or manifest>` streams every
realization through the same reading, holding running means only.

## `sim/cna_lengths.png`: the `[cna.length]` laws (#619)

`run_study --cna-lengths`: the density and CDF of the exponential
the `dev_tree*` manifests drew to #619 and the lognormal they draw now, at
`dev_tree`'s mean of 50 Mb and `dev_tree_1s_hard`'s median of 10 Mb. Analytic
(`scipy.stats`); the stamp names each manifest's file hash and `r0_hash`.
