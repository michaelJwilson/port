# `cnamaste.h5` and `truth.h5`

**TL;DR:** a run writes `cnamaste.h5` and `CalicoST`'s own files, nothing else. QA, the audits, the studies and the figures read `cnamaste.h5` alone, and `run_plots` draws every page of a `--sal` run from it, the same bytes. Each quantity is stored once: a page holds references, not data. `port.extensions.cnamaste` is the format; the tables below are its `render()`, which `tests/test_cnamaste.py` holds equal (T- #817, T- #818).

## Conventions
- **Root attributes:**
  - `cnamaste.h5`: `schema` (`cnamaste/1`), `commit`, `port`, `cnaster`, `sal`, `sample_hash`, `flags`, and `stages`, the complete groups in the order written.
  - `truth.h5`: `schema` (`cnamaste-truth/1`) and `sample_hash`, which a scorer joins the two files on.
- **Staged.** Each stage writes its group once, sets `complete = True` on it last, and appends it to `stages`. A reader sees only complete groups, so a run that stops early leaves every finished stage readable. A group rewritten replaces itself whole.
- **Axes.** Every dataset carries `dims`.
  - `n_spots`, `n_genes`, `channel` (2: read depth, B allele) and `xy` (2) take one size across a file; the root records the first two.
  - Every other axis is consistent within its group.
  - Spots are in `/inputs/barcodes`' order everywhere, genes in `/segments/genes`'.
- **Types.** `int64`, `int16` (copy numbers), `float64`, `numeric` (an int64 or float64 array keeps its own), `bool` and UTF-8 `str`. A sparse matrix (`csr`) is a subgroup of `data`, `indices` and `indptr` with `shape`; `/adjacency` is one, stored as the group itself. A value is written only where it casts to its type exactly.
- **Storage.** Every dataset is chunked, byte-shuffled and deflated at level 4. Values are stored as computed, never rounded.

## Each quantity once
| Quantity | Stored as | Read by |
| --- | --- | --- |
| a spot's counts at a level | `/counts/<level>`: `X`, `total_bb_RD`; the baseline as its two factors, `normal_rdr @ coverage`, as `determine_normal_baseline` builds it | every stage and page at that level; `counts/<level>:rdr=0` is the same with read depth and baseline zero, as the BAF stage reads it |
| counts summed over clones | not stored: `/counts` summed over a labelling (`merge_pseudobulk_by_index_mix`) | every stage and page |
| a labelling | the stage that made it: `/initial_clones`, `/baf/assignment`, `/baf_merged/assignment`, `/rdrbaf/assignment`, `/rdrbaf_merged/assignment`, `/clone_assignment`, `/integer_clones/assignment`; a stage's `clone_index` only where it is not `/initial_clones` | the spatial pages and the summed counts |
| a fit | `/phasing`, `/baf`, `/rdrbaf`: one column per clone, with how `cnaster` shaped it (`pred_layout`, `mu_shape`) | a merged stage's fit is its parent's kept columns; the final pages' is `/rdrbaf`'s, columns in `reindex_clones`' order |
| bins along the genome | `/segments`: a label per gene per level; `lengths` derived | every genomic page |
| integer copies | `/integer_copy`, with the bins' `CHR`, `START`, `END` (`START` and `END` include SNP rows, so `/segments`' genes do not give them) | `clones_genomic`, `copy_number_profile`, `/integer_clones` |
| a page | `/figures/<name>`: attributes only, `sources` (which of the above, as JSON), `options` (the plotter's keywords) and `write` (`write_fig`'s) | `run_plots` |

On dev_tree_1s_hard r0 (`9ec90dc2`, 3,000 spots) the file is 14 MB: the three levels' counts are 13 MB of it, and the sample's inputs are 11 MB.

## Groups
- **Segments.** `/segments/levels/<name>` is one level of the run's hierarchy (`segments.Lineage`, #438): a label per gene and the segment ids, appended as the run records them, `order` kept. A stage's `level` names the level its `n_obs` axis is on. A level is named for the step that makes it (`cnamaste.LEVELS`):

  | Level | Made by | Lineage name |
  | --- | --- | --- |
  | `phasing_min_snp_umis` | `assign_initial_blocks`, blocks of `quality.phasing_min_snp_umis` SNP UMIs | `blocks` |
  | `secondary_min_umi` | `create_bin_ranges`, bins of `quality.secondary_min_umi` UMIs | `bins` |
  | `normal_baf_filter` | `normal_baf_bin_filter`, the bins whose normal BAF it keeps | `bins-filtered` |
  | `min_segment_normal_umi` | the floor of #551 under `--sal` | `bins-floored` |
  | `normal_candidates` | `create_bin_ranges` again, on the normal candidates: the RDR+BAF stage's bins | `bins.2` |
- **Inputs.** `/inputs` records each slice's `anndata` path and SNP files (`cell_snp_Aallele`, `cell_snp_Ballele`, `unique_snp_ids`, `snp_barcodes`), absolute, with `sample_sheet` and the configured reference files as `references.*` and `preprocessing.*`.
- **Fields.** `/baf` and `/rdrbaf` each keep their last clone-assignment `field`; `/clone_assignment` is the run's final clones, and its `stage` names the field they were solved on.
- **Final fit only.** A fit is its final iteration; the HMRF keeps no trace, so `llf` and `total_llf` are attributes.
- **Thresholds.** `/integer_copy` carries every `[int_copy_num]` key the decode read, as `int_copy_num.<key>`; `/integer_clones` its `merge_agreement`.

## `cnamaste.h5`
| Group | Dataset | Axes | Type | Holds |
| --- | --- | --- | --- | --- |
| `/inputs` | | | | the sample, its input files (absolute paths), and the run's configuration (YAML) and flags; `references.*` and `preprocessing.*` are the configured reference and annotation files; attributes `config`, `flags`, `sample_sheet`, `references.*`, `preprocessing.*` |
| | `barcodes` | (n_spots) | str | spot barcodes: every spot axis's order |
| | `sample_ids` | (n_spots) | str | sample per spot |
| | `coords` | (n_spots, xy) | numeric | spot positions |
| | `single_tumor_prop` | (n_spots) | float64 | tumour proportion per spot (optional) |
| | `samples` | (n_samples) | str | each slice's sample id, in `sample_sheet`'s order |
| | `anndata` | (n_samples) | str | each slice's count matrix (`<filtered_feature_name>.h5` or `.h5ad`) |
| | `cell_snp_Aallele` | (n_samples) | str | each slice's `cell_snp_Aallele` |
| | `cell_snp_Ballele` | (n_samples) | str | each slice's `cell_snp_Ballele` |
| | `unique_snp_ids` | (n_samples) | str | each slice's `unique_snp_ids` |
| | `snp_barcodes` | (n_samples) | str | each slice's `snp_barcodes` |
| `/adjacency` | | | | the spots' one graph, stored as the group itself; attributes none |
| | `adjacency` | (n_spots, n_spots) | csr | spot adjacency (`adjacency_mat`), every stage's graph |
| `/segments/genes` | | | | the genes every level labels; attributes `excluded_genes`, `floor_min_length`, `floor_min_weight` |
| | `contig` | (n_genes) | str | the root: `df_gene_snp`'s gene rows, sorted |
| | `start` | (n_genes) | int64 | gene start |
| | `end` | (n_genes) | int64 | gene end |
| | `key` | (n_genes) | str | gene index label |
| | `floor_weight` | (n_genes) | float64 | the segment floor's normal UMI per gene (#551) (optional) |
| `/segments/levels/*` | | | | one level of the hierarchy, in the order the run recorded it; attributes `order` |
| | `label` | (n_genes) | int64 | segment per gene, `-1` dropped |
| | `ids` | (n_segments) | int64 | each segment's id, in label order |
| `/counts/*` | | | | the spots' counts at a level, once: every stage's and page's summed counts are these summed over a labelling (`merge_pseudobulk_by_index_mix`). `base` is `zero`, `factors` (`normal_rdr @ coverage`) or `full`; attributes `level`, `base` |
| | `X` | (n_obs, channel, n_spots) | numeric | each spot's counts per bin: read depth, B allele |
| | `total_bb_RD` | (n_obs, n_spots) | numeric | each spot's beta-binomial trials per bin |
| | `normal_rdr` | (n_obs) | float64 | the baseline's per-bin factor (`determine_normal_baseline`) (optional) |
| | `coverage` | (n_spots) | numeric | the baseline's per-spot factor, each spot's read-depth total (optional) |
| | `base_nb_mean` | (n_obs, n_spots) | float64 | the baseline, where its factors do not reproduce it (optional) |
| `/initial_clones` | | | | the run's initial clones, which phasing and the BAF stage start from unless their own `clone_index` says otherwise; attributes none |
| | `clone_index` | (n_spots) | int64 | each spot's initial clone, `-1` none |
| `/phasing` | | | | the phasing fit (`initial_phase_given_partition`) on the initial clones; attributes `level`, `counts`, `pred_layout`, `mu_shape` |
| | `clone_index` | (n_spots) | int64 | the clones phasing is given, where not `/initial_clones` (optional) |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone, a fit stacked along the genome unstacked |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion (optional) |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration (optional) |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| `/baf` | | | | the BAF stage at its final fit; attributes `level`, `counts`, `pred_layout`, `mu_shape`, `n_states`, `t`, `spatial_weight`, `llf`, `total_llf` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clones, where not `/initial_clones` (optional) |
| | `assignment` | (n_spots) | int64 | the stage's clone per spot, before any merge or reindex |
| | `field` | (n_spots, n_field_clones) | float64 | the stage's last clone-assignment field |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone, a fit stacked along the genome unstacked |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion (optional) |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration (optional) |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| `/baf_merged` | | | | the BAF stage after `merge_by_minspots`; its fit is `/baf`'s, its kept clones' columns; attributes `level`, `counts` |
| | `assignment` | (n_spots) | int64 | clone per spot after `merge_by_minspots` |
| `/rdrbaf` | | | | the RDR+BAF stage at its final fit; attributes `level`, `counts`, `pred_layout`, `mu_shape`, `n_states`, `t`, `spatial_weight`, `llf`, `total_llf` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clones, where not `/initial_clones` (optional) |
| | `assignment` | (n_spots) | int64 | the stage's clone per spot, before any merge or reindex |
| | `field` | (n_spots, n_field_clones) | float64 | the stage's last clone-assignment field |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone, a fit stacked along the genome unstacked |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion (optional) |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration (optional) |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| `/rdrbaf_merged` | | | | the RDR+BAF stage after `merge_by_minspots`; its fit is `/rdrbaf`'s, its kept clones' columns; attributes `level`, `counts` |
| | `assignment` | (n_spots) | int64 | clone per spot after `merge_by_minspots` |
| `/clone_assignment` | | | | the run's final clones (`reindex_clones`); `stage` names the group whose `field` they were solved on; attributes `level`, `stage` |
| | `assignment` | (n_spots) | int64 | clone per spot |
| `/integer_copy` | | | | integer copy states (`cnv_seglevel.tsv`'s), and every `[int_copy_num]` key the decode read; attributes `level`, `objective`, `contig_numeric`, `int_copy_num.*` |
| | `clones` | (n_clones) | int64 | each column's clone, as `/clone_assignment` numbers it |
| | `contig` | (n_obs) | str | each bin's `CHR` |
| | `start` | (n_obs) | int64 | each bin's `START`, its first row's, SNP rows included |
| | `end` | (n_obs) | int64 | each bin's `END`, its last row's |
| | `A` | (n_obs, n_clones) | int16 | copies of allele A per bin per clone |
| | `B` | (n_obs, n_clones) | int16 | copies of allele B per bin per clone |
| `/integer_clones` | | | | integer clones; their counts are `counts` summed over `assignment`; attributes `level`, `merge_agreement`, `counts` |
| | `map` | (n_clones) | int64 | integer clone of each `/integer_copy` column |
| | `assignment` | (n_spots) | int64 | integer clone per spot, `-1` none |
| | `integer_ids` | (n_integer_clones) | int64 | each integer clone's id, its smallest member's |
| | `A` | (n_obs, n_integer_clones) | int16 | copies of A per integer clone, its naming clone's |
| | `B` | (n_obs, n_integer_clones) | int16 | copies of B per integer clone |
| `/figures/*` | | | | a page as `run_cnaster_port --sal` drew it, no data of its own: `sources` names the groups it reads (JSON), `options` its plotter's keywords and `write` `write_fig`'s; attributes `kind`, `file`, `sources`, `options`, `write` |

## `truth.h5`
| Group | Dataset | Axes | Type | Holds |
| --- | --- | --- | --- | --- |
| `/inputs` | | | | the sample as drawn, its files, and the manifest that drew it; attributes `manifest`, `sample_sheet` |
| | `barcodes` | (n_spots) | str | spot barcodes: every spot axis's order |
| | `sample_ids` | (n_spots) | str | sample per spot |
| | `coords` | (n_spots, xy) | numeric | spot positions |
| | `samples` | (n_samples) | str | each slice's sample id, in `sample_sheet`'s order |
| | `anndata` | (n_samples) | str | each slice's count matrix (`<filtered_feature_name>.h5` or `.h5ad`) |
| | `cell_snp_Aallele` | (n_samples) | str | each slice's `cell_snp_Aallele` |
| | `cell_snp_Ballele` | (n_samples) | str | each slice's `cell_snp_Ballele` |
| | `unique_snp_ids` | (n_samples) | str | each slice's `unique_snp_ids` |
| | `snp_barcodes` | (n_samples) | str | each slice's `snp_barcodes` |
| `/segments/genes` | | | | the genes the planted copies are at; attributes none |
| | `contig` | (n_genes) | str | the root: `df_gene_snp`'s gene rows, sorted |
| | `start` | (n_genes) | int64 | gene start |
| | `end` | (n_genes) | int64 | gene end |
| | `key` | (n_genes) | str | gene index label |
| `/clone_assignment` | | | | the planted clone per spot; `clones` names them; attributes `clones` |
| | `assignment` | (n_spots) | int64 | clone per spot |
| `/integer_copy` | | | | the planted copies per gene per clone, `level = "genes"`; attributes `level` |
| | `clones` | (n_clones) | int64 | each column's clone, as `/clone_assignment` numbers it |
| | `A` | (n_obs, n_clones) | int16 | copies of allele A per bin per clone |
| | `B` | (n_obs, n_clones) | int16 | copies of allele B per bin per clone |
| `/integer_clones` | | | | the planted clones that share a profile, merged; attributes `level` |
| | `map` | (n_clones) | int64 | integer clone of each `/integer_copy` column |
| | `assignment` | (n_spots) | int64 | integer clone per spot, `-1` none |
| | `integer_ids` | (n_integer_clones) | int64 | each integer clone's id, its smallest member's |
| | `A` | (n_obs, n_integer_clones) | int16 | copies of A per integer clone, its naming clone's |
| | `B` | (n_obs, n_integer_clones) | int16 | copies of B per integer clone |
| `/phase` | | | | the realized phase; attributes none |
| | `snp_ids` | (n_snps) | str | SNPs, in `unique_snp_ids.npy`'s order |
| | `switched` | (n_snps) | bool | where the written A and B are exchanged |
| `/tree` | | | | the planted tree; attributes none |
| | `node` | (n_nodes) | str | clone |
| | `parent` | (n_nodes) | str | its parent, `''` the root |
| | `event_node` | (n_events) | str | the edge an event arose on |
| | `contig` | (n_events) | str | event contig |
| | `start` | (n_events) | int64 | event start |
| | `end` | (n_events) | int64 | event end |
| | `A` | (n_events) | int16 | allele A copies after the event |
| | `B` | (n_events) | int16 | allele B copies after the event |

## Not in either file
`CalicoST`'s files: `clone_labels.tsv`, `cnv_seglevel.tsv`, `cnv_genelevel.tsv` and `rdrbaf_final_*.npz`, under `CalicoST`'s names and layouts. `port` reads none back, but for `cnv_seglevel.tsv` where no page wrote `/integer_copy` (`--no-figure-swaps`).
