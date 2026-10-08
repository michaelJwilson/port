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
- **Final fit only.** `/baf` and `/rdrbaf` keep the final fit and the `llf` trace, not each iteration.
- **Thresholds.** `integer_copy` and `integer_clones` carry every `[int_copy_num]` key they read, as `int_copy_num.<key>`.

## `cnamaste.h5`
| Group | Dataset | Axes | Type | Holds |
| --- | --- | --- | --- | --- |
| `/inputs` | | | | the sample and the run's configuration (YAML) and flags; attributes `config`, `flags` |
| | `barcodes` | (n_spots) | str | spot barcodes: every spot axis's order |
| | `sample_ids` | (n_spots) | str | sample per spot |
| | `coords` | (n_spots, xy) | float64 | spot positions |
| | `single_tumor_prop` | (n_spots) | float64 | tumour proportion per spot (optional) |
| `/segments/genes` | | | | the genes every level labels; attributes `excluded_genes`, `floor_min_length`, `floor_min_weight` |
| | `contig` | (n_genes) | str | the root: `df_gene_snp`'s gene rows, sorted |
| | `start` | (n_genes) | int64 | gene start |
| | `end` | (n_genes) | int64 | gene end |
| | `key` | (n_genes) | str | gene index label |
| | `floor_weight` | (n_genes) | float64 | the segment floor's normal UMI per gene (#551) (optional) |
| `/segments/levels/*` | | | | one level of the hierarchy, in the order the run recorded it; attributes `order` |
| | `label` | (n_genes) | int64 | segment per gene, `-1` dropped |
| | `ids` | (n_segments) | int64 | each segment's id, in label order |
| `/baf` | | | | the BAF stage at its final fit; attributes `level`, `n_states`, `t`, `spatial_weight`, `termination` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clone per spot, `-1` none |
| | `X` | (n_obs, channel, n_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_clones) | float64 | beta-binomial trials, pooled |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone |
| | `llf` | (n_iterations) | float64 | log-likelihood per iteration |
| `/rdrbaf` | | | | the RDR+BAF stage at its final fit; attributes `level`, `n_states`, `t`, `spatial_weight`, `termination` |
| | `clone_index` | (n_spots) | int64 | the stage's initial clone per spot, `-1` none |
| | `X` | (n_obs, channel, n_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_clones) | float64 | beta-binomial trials, pooled |
| | `log_mu` | (n_states) | float64 | final fit: log rate per state |
| | `p_binom` | (n_states) | float64 | final fit: B allele probability per state |
| | `alphas` | (n_states) | float64 | final fit: negative binomial dispersion |
| | `taus` | (n_states) | float64 | final fit: beta-binomial concentration |
| | `logmu_shift` | (n_clones) | float64 | final fit: per-clone log rate shift (#362) (optional) |
| | `pred_cnv` | (n_obs, n_clones) | int64 | final fit: state per bin per clone |
| | `llf` | (n_iterations) | float64 | log-likelihood per iteration |
| `/clone_assignment` | | | | the final clone assignment; attributes `level`, `spatial_weight`, `solver`, `termination` |
| | `field` | (n_spots, n_clones) | float64 | log-likelihood per spot per clone |
| | `adjacency_mat` | (n_spots, n_spots) | csr | spot adjacency |
| | `assignment` | (n_spots) | int64 | clone per spot |
| `/integer_copy` | | | | integer copy states, and every `[int_copy_num]` key the decode read; attributes `level`, `objective`, `int_copy_num.*` |
| | `A` | (n_obs, n_clones) | int16 | copies of allele A per bin per clone |
| | `B` | (n_obs, n_clones) | int16 | copies of allele B per bin per clone |
| `/integer_clones` | | | | integer clones, and their counts summed over their spots; attributes `level`, `int_copy_num.*` |
| | `map` | (n_clones) | int64 | integer clone of each fitted clone |
| | `assignment` | (n_spots) | int64 | integer clone per spot |
| | `A` | (n_obs, n_integer_clones) | int16 | copies of A per integer clone |
| | `B` | (n_obs, n_integer_clones) | int16 | copies of B per integer clone |
| | `X` | (n_obs, channel, n_integer_clones) | float64 | counts pooled over each clone's spots: read depth, B allele |
| | `base_nb_mean` | (n_obs, n_integer_clones) | float64 | negative binomial exposure, pooled |
| | `total_bb_RD` | (n_obs, n_integer_clones) | float64 | beta-binomial trials, pooled |

## `truth.h5`
| Group | Dataset | Axes | Type | Holds |
| --- | --- | --- | --- | --- |
| `/inputs` | | | | the sample and the run's configuration (YAML) and flags; attributes `manifest` |
| | `barcodes` | (n_spots) | str | spot barcodes: every spot axis's order |
| | `sample_ids` | (n_spots) | str | sample per spot |
| | `coords` | (n_spots, xy) | float64 | spot positions |
| `/segments/genes` | | | | the genes every level labels; attributes none |
| | `contig` | (n_genes) | str | the root: `df_gene_snp`'s gene rows, sorted |
| | `start` | (n_genes) | int64 | gene start |
| | `end` | (n_genes) | int64 | gene end |
| | `key` | (n_genes) | str | gene index label |
| `/clone_assignment` | | | | the final clone assignment; attributes `clones` |
| | `assignment` | (n_spots) | int64 | clone per spot |
| `/integer_copy` | | | | integer copy states, and every `[int_copy_num]` key the decode read; attributes `level` |
| | `A` | (n_obs, n_clones) | int16 | copies of allele A per bin per clone |
| | `B` | (n_obs, n_clones) | int16 | copies of allele B per bin per clone |
| `/integer_clones` | | | | integer clones, and their counts summed over their spots; attributes `level` |
| | `map` | (n_clones) | int64 | integer clone of each fitted clone |
| | `assignment` | (n_spots) | int64 | integer clone per spot |
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

`truth.h5`'s `integer_copy` is at the genes (`level = "genes"`). Its `clone_assignment` is the planted clones.

## Not in either file
`CalicoST`'s files: `clone_labels.tsv`, `cnv_seglevel.tsv`, `cnv_genelevel.tsv` and `rdrbaf_final_*.npz`. They are written from these arrays at the end of a run, under `CalicoST`'s names and layouts, and nothing in `port` reads them back.
