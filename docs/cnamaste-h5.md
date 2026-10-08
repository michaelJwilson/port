# `cnamaste.h5` and `truth.h5`

**TL;DR:** a run writes `cnamaste.h5` and `CalicoST`'s own files, nothing else. QA, the audits, the studies and the figures read `cnamaste.h5` alone. A simulated sample's truth is `truth.h5`, under the same names where the quantity exists. `port.extensions.cnamaste` is the format: its `GROUPS` and `TRUTH_GROUPS` declare every group, and the tables below are its `render()`, which `tests/test_cnamaste.py` holds equal (T- #817).

## Conventions
- **Root attributes:**
  - `cnamaste.h5`: `schema` (`cnamaste/1`), `commit`, `port`, `cnaster`, `sal`, `sample_hash`, and `stages`, the complete groups in the order written.
  - `truth.h5`: `schema` (`cnamaste-truth/1`) and `sample_hash`, which a scorer joins the two files on.
- **Staged.** Each stage writes its group once, sets `complete = True` on it last, and appends it to `stages`. A reader sees only complete groups, so a run that stops early leaves every finished stage readable. A stage rewritten replaces its group whole.
- **Axes.** Every dataset carries `dims`.
  - `n_spots`, `n_genes`, `channel` (2: read depth, B allele) and `xy` (2) take one size across a file; the root records the first two.
  - Every other axis is consistent within its group.
  - Spots are in `/inputs/barcodes`' order everywhere, genes in `/segments/genes`'.
- **Types.** `int64`, `int16` (copy numbers), `float64`, `bool` and UTF-8 `str`. A sparse matrix (`csr`) is a subgroup holding `data`, `indices` and `indptr`, with `shape`. A value is written only where it casts to its type exactly.
- **Storage.** Every dataset is chunked and deflated at level 1. Values are stored as computed, never rounded.
- **Segments.** `/segments/levels/<name>` is one level of the run's hierarchy (`segments.Lineage`, #438): a label per gene and the segment ids. Levels are appended as the run records them and keep their `order` attribute. A stage's `level` names the level its `n_obs` axis is on. What a level derives (contig, start, length, `lengths`) is not stored.
- **Inputs.** `/inputs` records each slice's `anndata` path and the SNP files' (`cell_snp_Aallele`, `cell_snp_Ballele`, `unique_snp_ids`, `snp_barcodes`, `sample_sheet`) as the run resolved them, absolute, with the configured reference files as `references.*` and `preprocessing.*`.
- **Graph and fields.** `/adjacency` is the spots' one graph, the CSR matrix stored as the group itself (`data`, `indices`, `indptr`, with `shape`). Each stage keeps its own clone-assignment `field`; `/clone_assignment` is the run's final clones, and its `stage` names the field they were solved on.
- **Final fit only.** `/baf` and `/rdrbaf` keep the final fit and the `llf` trace, not each iteration.
- **Thresholds.** `integer_copy` and `integer_clones` carry every `[int_copy_num]` key they read, as `int_copy_num.<key>`.

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
| `/baf` | | | | the BAF stage at its final fit; attributes `level`, `n_states`, `t`, `spatial_weight`, `llf`, `total_llf` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clone per spot, `-1` none |
| | `X` | (n_obs, channel, n_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_clones) | float64 | beta-binomial trials, pooled |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone, clones unstacked |
| | `field` | (n_spots, n_field_clones) | float64 | the stage's last clone-assignment field: log-likelihood per spot per clone it assigned to |
| | `assignment` | (n_spots) | int64 | the stage's clone per spot, before any merge or reindex |
| `/rdrbaf` | | | | the RDR+BAF stage at its final fit; attributes `level`, `n_states`, `t`, `spatial_weight`, `llf`, `total_llf` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clone per spot, `-1` none |
| | `X` | (n_obs, channel, n_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_clones) | float64 | beta-binomial trials, pooled |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone, clones unstacked |
| | `field` | (n_spots, n_field_clones) | float64 | the stage's last clone-assignment field: log-likelihood per spot per clone it assigned to |
| | `assignment` | (n_spots) | int64 | the stage's clone per spot, before any merge or reindex |
| `/clone_assignment` | | | | the run's final clones (`reindex_clones`); `stage` names the group whose `field` they were solved on; attributes `level`, `stage` |
| | `assignment` | (n_spots) | int64 | clone per spot |
| `/integer_copy` | | | | integer copy states, and every `[int_copy_num]` key the decode read; attributes `level`, `objective`, `int_copy_num.*` |
| | `clones` | (n_clones) | int64 | each column's clone, as `/clone_assignment` numbers it |
| | `A` | (n_obs, n_clones) | int16 | copies of allele A per bin per clone |
| | `B` | (n_obs, n_clones) | int16 | copies of allele B per bin per clone |
| `/integer_clones` | | | | integer clones, and their counts summed over their spots; attributes `level`, `merge_agreement` |
| | `map` | (n_clones) | int64 | integer clone of each `/integer_copy` column |
| | `assignment` | (n_spots) | int64 | integer clone per spot, `-1` none |
| | `integer_ids` | (n_integer_clones) | int64 | each integer clone's id, its smallest member's |
| | `A` | (n_obs, n_integer_clones) | int16 | copies of A per integer clone |
| | `B` | (n_obs, n_integer_clones) | int16 | copies of B per integer clone |
| | `X` | (n_obs, channel, n_integer_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_integer_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_integer_clones) | float64 | beta-binomial trials, pooled |
| `/figures/genomic/*` | | | | a genomic page (`plot_clones_genomic`), as `run_cnaster_port --sal` drew it; attributes `file`, `options`, `write` |
| | `lengths` | (n_contigs) | numeric | bins per contig |
| | `labels` | (n_clones) | str | each row pair's clone label |
| | `sizes` | (n_clones) | int64 | spots per clone |
| | `X` | (n_obs, channel, n_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_clones) | float64 | beta-binomial trials, pooled |
| | `tumor_prop` | (n_clones) | float64 | mean tumour proportion of the pooled spots (optional) |
| | `profile` | (n_obs) | float64 | the baseline summed over spots: the shifted line's lambda |
| | `pred_cnv` | (...) | numeric | the drawn fit's states (optional) |
| | `new_log_mu` | (...) | float64 | the drawn fit's log rates (optional) |
| | `new_p_binom` | (...) | float64 | the drawn fit's B allele probabilities (optional) |
| | `sample_list` | (n_samples) | str | the page's sample names (optional) |
| | `contig` | (n_obs) | str | each bin's `CHR`, named (optional) |
| | `contig_int` | (n_obs) | int64 | each bin's `CHR`, numbered (optional) |
| | `start` | (n_obs) | int64 | each bin's `START` (optional) |
| | `end` | (n_obs) | int64 | each bin's `END` (optional) |
| | `cnv_clones` | (n_cnv_clones) | str | the table's clone ids, in column order (optional) |
| | `float_columns` | (n_cnv_clones) | bool | clones whose A and B columns are floats (optional) |
| | `A` | (n_obs, n_cnv_clones) | float64 | each clone's `A` column (optional) |
| | `B` | (n_obs, n_cnv_clones) | float64 | each clone's `B` column (optional) |
| `/figures/spatial/*` | | | | a spatial page (`plot_clones_spatial`), as `run_cnaster_port --sal` drew it; attributes `file`, `options`, `write` |
| | `coords` | (n_spots, xy) | numeric | spot positions |
| | `assignment` | (n_spots) | str | each spot's label, `''` where missing |
| | `missing` | (n_spots) | bool | spots with no label |
| | `tumor_prop` | (n_spots) | float64 | tumour proportion per spot (optional) |
| | `sample_list` | (n_samples) | str | sample names (optional) |
| | `sample_ids` | (n_spots) | numeric | each spot's sample code (optional) |
| `/figures/profile/*` | | | | a copy-number profile (`plot_copy_number_profile`); attributes `file`, `options`, `write` |
| | `contig` | (n_obs) | str | each bin's `CHR`, named (optional) |
| | `contig_int` | (n_obs) | int64 | each bin's `CHR`, numbered (optional) |
| | `start` | (n_obs) | int64 | each bin's `START` (optional) |
| | `end` | (n_obs) | int64 | each bin's `END` (optional) |
| | `cnv_clones` | (n_cnv_clones) | str | the table's clone ids, in column order (optional) |
| | `float_columns` | (n_cnv_clones) | bool | clones whose A and B columns are floats (optional) |
| | `A` | (n_obs, n_cnv_clones) | float64 | each clone's `A` column (optional) |
| | `B` | (n_obs, n_cnv_clones) | float64 | each clone's `B` column (optional) |

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
| | `A` | (n_obs, n_integer_clones) | int16 | copies of A per integer clone |
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
`CalicoST`'s files: `clone_labels.tsv`, `cnv_seglevel.tsv`, `cnv_genelevel.tsv` and `rdrbaf_final_*.npz`. They are written from these arrays at the end of a run, under `CalicoST`'s names and layouts, and nothing in `port` reads them back.
