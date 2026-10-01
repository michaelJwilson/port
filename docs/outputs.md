# Outputs

**TL;DR:** `run_cnaster_port` writes one file set per pipeline stage beside
`cnaster`'s own files (T- #613): segmentation, core inference (HMM + HMRF),
integer decode, and per-spot labels. Rows are long and join on `clone`,
`bin`, `state` and `gene_index`. The integer-decode files recompute the
decode's per-clone log-likelihood to 7.3e-12 nats on CalicoST easy; the HMM
files recompute their own per-clone log-likelihood exactly, which differs
from `cnaster`'s `llf` by 13.52 nats because `cnaster` keeps a stale
posterior (below).

`port.extensions.outputs.SCHEMA` is the source of every table below, and
`tests/test_output_stages.py` checks this file against it.

## Files by stage

| stage | file | rows |
| --- | --- | --- |
| run | `run.json` | one object |
| segmentation | `cnv_lineage.tsv` | gene |
| segmentation | `cnv_bins.tsv` | bin |
| core inference | `cnv_hmm_states.tsv` | state |
| core inference | `cnv_hmm_transmat.tsv` | (state, state_next) |
| core inference | `cnv_hmm_clones.tsv` | clone |
| core inference | `cnv_hmm_bins.tsv` | (clone, bin) |
| integer decode | `cnv_copy_clones.tsv` | clone |
| integer decode | `cnv_copy_states.tsv` | (clone, state) |
| integer decode | `cnv_copy_bins.tsv` | (clone, bin) |
| integer decode | `cnv_copy_segments.tsv` | (clone, segment) |
| integer decode | `cnv_copy_genes.tsv` | (clone, gene_index) |
| labels | `spot_labels.tsv` | spot |

`cnaster`'s own files -- `cnv_seglevel.tsv`, `cnv_perstate.tsv`,
`cnv_genelevel.tsv`, `clone_labels.tsv`, `baf_clone_labels.tsv` and
`rdrbaf_final_nstates{K}_smp.npz` -- are left as `cnaster` wrote them.

**Replaced** (user-visible): `cnv_binlevel.tsv`, `cnv_states.tsv`,
`cnv_segments.tsv`, `clone_labels_integer.tsv`, `copy_decode.tsv`,
`gene_segments.tsv` and `manifest.json` are no longer written, and
`clone_labels.tsv` and `baf_clone_labels.tsv` are no longer rewritten. Before
T- #613 `port` rewrote `clone_labels.tsv` where #518's merge joined clones
and wherever it recorded the samples (#418). The merged clone is now
`spot_labels.tsv`'s `clone_label_decode`, and each spot's sample is
`spot_labels.tsv`'s `sample`. `port.extensions.outputs.read_run_labels`
returns the clones `clone_labels.tsv` used to carry.

**Why `spot_labels.tsv` and not `clone_labels.tsv`.** `cnaster` writes
`clone_labels.tsv` into the same directory after the integer decode
(`run_cnaster.py:1606`). A `port` file under that name would overwrite it.
The two would then hold different columns under one name, depending on
which ran last.

## Reading

- A `float` column holds 17 significant digits. `pd.read_csv(...,
  float_precision="round_trip")` reads back the value written. An integral
  float such as `20.0` is written as `20`.
- An empty cell is a value the run did not record. For example,
  `tumor_proportion` is empty without an input tumour proportion. An `int`
  column with an empty cell reads as `float` unless read as `Int64`.
- `run.json` is strict JSON with string keys. It holds no NaN; null stands
  for a missing value. Its `samples` maps `sample_id` to the sample name, and
  `clones` maps a clone id to its column of the `.npz`'s `pred_cnv`.

## Completeness

A stage is complete when its files plus the pooled counts in
`cnv_hmm_bins.tsv` (`X_depth`, `X_allele`, `base_nb_mean`, `total_bb_RD`)
recompute that stage's likelihood.
`port.extensions.outputs.hmm_log_likelihoods` and `decode_log_likelihoods`
do the recomputation. The pooled counts are the final assignment's sums over
spots, which `cnaster` also uses for its last fit and for the decode. They
assume no input tumour proportion, which would drop spots below
`tumorprop_threshold` from the HMM's sums.

| stage | recomputed | CalicoST easy `--sal` |
| --- | --- | --- |
| integer decode | each clone's best-path log-likelihood, prior included, from `cnv_copy_clones`, `cnv_copy_bins` and the counts | to 7.3e-12 nats (5.7e-16 relative) of the decode's own |
| core inference | each clone's forward log-likelihood under `cnv_hmm_states`, `cnv_hmm_transmat` and `cnv_hmm_clones.log_mu_shift`, rate `exp(log_mu - log_mu_shift)` | exactly `cnv_hmm_clones.log_likelihood`; sum -50,212.196 against `cnaster`'s `llf` -50,198.673 |

**The HMM's difference is `cnaster`'s, by its own TODO.** After the last
HMRF sweep `run_core_inference` refits the parameters once more and keeps
the earlier fit's `log_gamma`, `pred_cnv` and `llf` (`hmrf.py:784`, "TODO llf
should also technically be updated"). So the `.npz`'s `llf`, `pred_cnv`
(`hmm_state`) and posterior (`hmm_state_probability`) come from parameters no
file holds. On a self-consistent synthetic run the HMM files recompute the
run's `llf`, posterior and path to 1e-10 relative
(`tests/test_output_stages.py`). `run.json` carries both numbers:
`log_likelihood` (the files' sum) and `log_likelihood_cnaster`.

**`mu`.** `cnv_hmm_bins.mu` is `exp(log_mu[hmm_state] - log_mu_shift)`, the
rate the shifted HMM applies to `base_nb_mean`
(`port.patch.hmm_nophasing.shifted_emission`); `cnv_copy_bins.mu` is the
lattice decode's predicted rate in the same convention.

**Not recorded, so not written** (TODO T- #613): the SNPs per bin (the
lineage records genes only), and the HMRF's `total_llf`, which `cnaster`
sets to NaN when it reindexes the clones (`run.json`'s `total_llf` is
null).

**Genes.** `cnv_copy_genes.tsv` lists the genes the final bins hold
(`cnv_lineage.segment_final >= 0`), 7,183 per clone on CalicoST easy.
`cnaster`'s `cnv_genelevel.tsv` has 7,197 rows over the same 7,182 gene
names; one name repeats 16 times there and twice here. At each name's first
row the two agree in `(A, B)`.

## CalicoST-compatible set

`run_cnaster_port --calicost-outputs` also writes CalicoST's file set
(`calicost_supervised.py:360-445`) into `calicost_compatible/` in each run
directory, filled from the stage files above:

- `cnv_seglevel.tsv`: `CHR START END`, then `clone{c} A` and `clone{c} B`
  per clone, the integer decode's pairs;
- `cnv_perstate.tsv`: `clone{c} logmu`, `p`, `A` and `B` per HMM state;
- `cnv_genelevel.tsv`: gene names, sorted, with `clone{s} A` and `clone{s} B`;
- `clone_labels.tsv`: indexed by `BARCODES`, with `clone_label` and, given
  an input tumour proportion, `tumor_proportion`;
- `posterior_clone_probability.npy`: the HMRF's posterior over clones,
  `(n_spots, n_clones)`;
- `normal_candidate_barcodes.txt`: the normal candidates' positions among
  the spots, one per line, as CalicoST writes them under that name;
- `rdrbaf_final_nstates{K}_smp.npz`: CalicoST's nine keys, in its order.

**Why a subdirectory.** The set reuses `cnaster`'s file names with other
contents. A subdirectory named after neither tool keeps both sets, and it
is not `calicost/`, which is what a CalicoST run's own directory is called
in `tests/data/benchmarks`. `port.extensions.outputs.primary` and
`run_directories` skip it, so scoring reads the run's own files.

**Spots** are in the run's order, the order of the HMRF's posterior and
`.npz`. Where the run has several samples, a barcode gets `_<sample>` unless
it already ends with it, as CalicoST's joint runs name them. **Clones** are
`cnaster`'s, before #518's merge, so the posterior's columns are the HMRF's.
A posterior is written only where the HMRF's last labelling is the final
one relabelled. Its columns are then renormalized over the clones that
survived.

**Differences** (`port.extensions.outputs.CALICOST_DIFFERENCES`).
`tests/test_calicost_outputs.py` compares the set with the committed
CalicoST run: file names, headers as clone-free templates, dtypes,
`clone_labels.tsv`'s index, the `.npz`'s keys and dimensions, and the
posterior's shape. It reads `cnv_seglevel.tsv` through CalicoST's
`get_LoH_for_phylogeny`, and fails on any difference not listed here.

| difference | regime | why it is a choice |
| --- | --- | --- |
| `absent cnv_event.tsv` | every run | CalicoST c1abcae's `summary_events` calls `strict_convert_copy_to_states`, which no CalicoST module defines (`utils_IO.py:1257`). It raises `NameError` once any clone has more than 10 bins off 0.5 BAF, so neither tool can write the file at this pin |
| `absent cnv_diploid_seglevel.tsv`, `absent cnv_diploid_perstate.tsv`, `absent cnv_diploid_genelevel.tsv`, `absent cnv_diploid_event.tsv`, `absent cnv_triploid_seglevel.tsv`, `absent cnv_triploid_perstate.tsv`, `absent cnv_triploid_genelevel.tsv`, `absent cnv_triploid_event.tsv`, `absent cnv_tetraploid_seglevel.tsv`, `absent cnv_tetraploid_perstate.tsv`, `absent cnv_tetraploid_genelevel.tsv`, `absent cnv_tetraploid_event.tsv` | every run | port's lattice decode is one decode, so there are no fixed-ploidy passes to write. A copy under the diploid name would claim a constraint that no decode applied |
| `absent mergedallspots_nstates{K}_sp.npz` | every run | CalicoST's BAF-stage checkpoint, which is not an output; port's BAF-stage clones are `spot_labels.tsv`'s `clone_label_baf` |
| `absent calicost_config.txt`, `absent input_filelist.tsv`, `absent run.json` | every run | CalicoST's own run records; port's are the run directory's `run.json` and its configuration |
| `npz values new_log_mu` | shifted runs | one shared table: column `c` is `log_mu - log_mu_shift_c`, the rate the shifted emission applies to `base_nb_mean`. CalicoST fits one table per clone |
| `npz values new_alphas new_p_binom new_taus` | every run | shared by every clone, so they are repeated per column |

## Columns

### `run.json`

| key | meaning |
| --- | --- |
| `n_states`, `n_clones`, `n_obs` | HMM states, clones, bins |
| `lengths` | bins per segment of the chain, which restarts at each |
| `log_likelihood` | sum of `cnv_hmm_clones.log_likelihood` |
| `log_likelihood_cnaster` | the `.npz`'s `llf` |
| `total_llf` | the HMRF's total; null where `cnaster` left NaN |
| `clones` | clone id -> its column of `pred_cnv` |
| `clone_label_decode` | clone id -> its integer clone (#518) |
| `merge_agreement` | the share of bins two profiles must agree at to merge |
| `samples` | `sample_id` -> sample name; null without a recording |
| `integer_decoder` | the decode `run_cnaster_port` ran |
| `max_total_copy` | the lattice decode's cap on `A + B`; null otherwise |
| `config` | the configuration's copy caps, ploidy, state count and output directory |
| `run_cnaster_port` | the flags the outputs depend on |
| `versions` | installed `cnaster` and `port` |

### `cnv_lineage.tsv`

Stage: segmentation.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `gene_index` | int | - | row of the gene among the run's genes, 0-based |
| `gene` | str | - | gene name, `df_gene_snp`'s `gene` |
| `CHR` | int | - | chromosome |
| `START` | int | bp | gene start |
| `END` | int | bp | gene end |
| `segment_block` | int | - | SNP block, -1 where dropped |
| `segment_phased` | int | - | phased bin, -1 where dropped |
| `segment_filtered` | int | - | bin after the normal-BAF filter, -1 where dropped |
| `segment_floored` | int | - | bin after the length and normal-UMI floor, -1 where dropped |
| `segment_final` | int | - | the HMM's bin, `cnv_bins.bin`; -1 where dropped |

### `cnv_bins.tsv`

Stage: segmentation.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `bin` | int | - | bin, 0-based, the HMM's observation index |
| `CHR` | int | - | chromosome |
| `START` | int | bp | first gene's start |
| `END` | int | bp | last gene's end |
| `n_genes` | int | - | genes the bin holds |
| `normal_umi` | float | UMI | UMI over the normal spots, the floor's weight; empty without a floor |

### `cnv_hmm_states.tsv`

Stage: core inference.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `state` | int | - | HMM state, shared by every clone |
| `log_mu` | float | log rate | log negative binomial rate per unit `base_nb_mean`, before the clone's shift |
| `alphas` | float | - | negative binomial dispersion |
| `p_binom` | float | - | beta-binomial B-allele probability |
| `taus` | float | - | beta-binomial concentration |
| `log_startprob` | float | log probability | initial state probability |

### `cnv_hmm_transmat.tsv`

Stage: core inference.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `state` | int | - | state at bin t |
| `state_next` | int | - | state at bin t + 1 |
| `log_transmat` | float | log probability | transition probability |

### `cnv_hmm_clones.tsv`

Stage: core inference.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `log_mu_shift` | float | log rate | the clone's library normalizer `log Z_c`, 0 for the normal clone |
| `tumor_proportion` | float | - | mean input tumour proportion over the clone's spots; empty without one |
| `is_normal` | bool | - | the clone with the largest share of balanced bins |
| `n_spots` | int | - | spots the HMRF assigned the clone |
| `log_likelihood` | float | nats | forward log-likelihood of the clone's counts under these parameters |

### `cnv_hmm_bins.tsv`

Stage: core inference.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `bin` | int | - | bin |
| `hmm_state` | int | - | the run's decoded state, `pred_cnv` |
| `hmm_state_probability` | float | - | the run's posterior probability of `hmm_state` |
| `mu` | float | rate | exp(log_mu[hmm_state] - log_mu_shift), the rate the emission applies |
| `p_binom` | float | - | p_binom[hmm_state] |
| `X_depth` | int | UMI | read depth pooled over the clone's spots, `X` channel 0 |
| `X_allele` | int | UMI | B-allele count pooled over the clone's spots, `X` channel 1 |
| `base_nb_mean` | float | UMI | expected depth at rate one, pooled over the clone's spots |
| `total_bb_RD` | int | UMI | allele-informative depth, pooled over the clone's spots |
| `rdr_observed` | float | rate | X_depth / base_nb_mean; empty where base_nb_mean is 0 |
| `baf_observed` | float | - | X_allele / total_bb_RD; empty where total_bb_RD is 0 |

### `cnv_copy_clones.tsv`

Stage: integer decode.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `log_mu_shift` | float | log rate | the decode's fitted shift |
| `tumour_fraction` | float | - | the decode's tumour share of the clone's spots |
| `alphas` | float | - | the decode's negative binomial dispersion, shared |
| `taus` | float | - | the decode's beta-binomial concentration, shared |
| `parsimony_weight` | float | nats | log-prior per bin per unit of |A + B - 2| |
| `stay` | float | probability | the decode chain's diagonal; the rest even over the other pairs |
| `log_likelihood` | float | nats | the clone's best path's log-likelihood, prior included |
| `ploidy` | int | - | median A + B over the clone's bins |
| `is_normal` | bool | - | the clone the decode holds at (1, 1) |
| `clone_label_decode` | int | - | the clone after merging clones of one profile (#518) |

### `cnv_copy_states.tsv`

Stage: integer decode.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `state` | int | - | HMM state |
| `A` | int | copies | the state's most frequent A on the clone's bins |
| `B` | int | copies | the state's most frequent B on the clone's bins |
| `mu` | float | rate | the decode's predicted rate of (A, B) in this clone |
| `baf` | float | - | the decode's predicted B-allele share of (A, B) in this clone |
| `n_bins` | int | - | the clone's bins in the state |

### `cnv_copy_bins.tsv`

Stage: integer decode.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `bin` | int | - | bin |
| `A` | int | copies | decoded A |
| `B` | int | copies | decoded B |
| `total_copy` | int | copies | A + B |
| `copy_class` | str | - | `copy_class(A, B)` |
| `mu` | float | rate | the decode's predicted rate, exp(log depth - log_mu_shift) |
| `baf` | float | - | the decode's predicted B-allele share |

### `cnv_copy_segments.tsv`

Stage: integer decode.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `segment` | int | - | the clone's run of equal (A, B), 0-based |
| `CHR` | int | - | chromosome |
| `START` | int | bp | first bin's start |
| `END` | int | bp | last bin's end |
| `bin_first` | int | - | first bin |
| `bin_last` | int | - | last bin |
| `n_bins` | int | - | bins in the run |
| `gene_first` | int | - | `cnv_lineage.gene_index` of the first bin's first gene |
| `gene_last` | int | - | `cnv_lineage.gene_index` of the last bin's last gene |
| `A` | int | copies | decoded A |
| `B` | int | copies | decoded B |
| `hmm_states` | str | - | the HMM states the run spans, comma-separated |

### `cnv_copy_genes.tsv`

Stage: integer decode.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `clone` | int | - | clone id |
| `gene_index` | int | - | `cnv_lineage.gene_index` |
| `gene` | str | - | gene name, where known |
| `bin` | int | - | the gene's bin |
| `A` | int | copies | decoded A of the gene's bin |
| `B` | int | copies | decoded B of the gene's bin |

### `spot_labels.tsv`

Stage: labels.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `barcode` | str | - | spot barcode |
| `sample` | str | - | the spot's sample name |
| `sample_id` | int | - | the sample's enum |
| `x` | float | - | spot position, first coordinate |
| `y` | float | - | spot position, second coordinate |
| `n_umi` | int | UMI | the spot's read depth over the binned genes |
| `clone_label_baf` | int | - | the BAF stage's clone, `baf_clone_labels.tsv` |
| `clone_label` | int | - | the HMRF's clone, `clone_labels.tsv` |
| `clone_label_decode` | int | - | the clone after merging clones of one profile (#518) |
