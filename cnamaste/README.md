# cnamaste

`cnaster`'s run path at its pin `4adad4d`, renamed, then fixed and patched PR by PR (T- #836). It is its own uv
project: its own build, lock and environment, sharing nothing with `port`. One command runs it:

```sh
cd cnamaste
uv sync --locked
uv run run_cnamaste config.yaml
```

## Install

Python >= 3.12.2 and [uv](https://docs.astral.sh/uv/). From this directory:

| Command | Does |
| --- | --- |
| `uv sync --locked` | creates `.venv` from `uv.lock`, with the `dev` group (pytest, pytest-xdist) |
| `uv build --wheel` | builds `dist/cnamaste-0.1.0-py3-none-any.whl` |
| `uv lock --upgrade-package <name>` | moves one dependency; commit `uv.lock` with it |

The dependencies are `cnaster`'s at its pin, so `cnamaste` runs wherever `cnaster` does.

## Run

```sh
uv run run_cnamaste CONFIG [-o key=value ...]
```

- `CONFIG`: a YAML configuration. `config.yaml` is a full one, and `zenodo_sim_config.yaml` the one the CalicoST
  simulations run under.
- `-o` / `--over_rides`: overrides one key in dot notation, and repeats, e.g.
  `-o paths.output_dir=/scratch/run -o hmrf.n_clones=4`. The value is stored as text: a numeric key set this way
  reaches the run as a string (`test_smoke.py`'s `-o` row, xfail).

### Configuration

| Key | Is |
| --- | --- |
| `paths.sample_sheet` | the sample sheet below |
| `paths.output_dir` | the run's root; created if absent |
| `references.geneticmap_file`, `hgtable_file`, `filtergenelist_file`, `filterregion_file` | GRCh38 resources (CalicoST's `GRCh38_resources`) |
| `hmrf.n_clones`, `hmrf.random_state`, `hmrf.spatial_weight` | the clone search; together they name the output directory |

### Sample sheet

A TSV, one row per slice:

| Column | Holds |
| --- | --- |
| `bam` | the slice's BAM; not read by the run path |
| `sample_id` | the slice's name |
| `spaceranger_dir` | `filtered_feature_bc_matrix.h5ad`, `spatial/tissue_positions_list.csv`; optionally `spatial/tissue_hires_image.png` and `spatial/scalefactors_json.json`, the H&E slide |
| `snp_dir` | `barcodes.txt`, `unique_snp_ids.npy`, `cell_snp_Aallele.npz`, `cell_snp_Ballele.npz` |

### Outputs

Under `<output_dir>/clone<n_clones>_rectangle<random_state>_w<spatial_weight>/`:

| File | Holds |
| --- | --- |
| `clone_labels.tsv` | each spot's barcode, slice, position and clone |
| `cnv_seglevel.tsv` | per bin: `CHR`, `START`, `END`, and per clone its state, `logmu`, `p` and integer copies `A`, `B` |
| `cnv_genelevel.tsv`, `cnv_perstate.tsv`, `baf_clone_labels.tsv` | per gene, per state, and the BAF-only clones |
| `plots/combined.png` | one page: (a) the H&E slide (a blank frame without one) and the clones; (b) the copy-number profile; (c) each clone's RDR and BAF |
| `plots/he_clones.png` | where a slide is given: each spot's H&E class beside the initial clones |
| `plots/*.pdf` | `cnaster`'s figures, stage by stage |

`<output_dir>/cnamaste.perf` records each stage's time. The pages' face and font size are `pyproject.toml`'s
`[tool.cnamaste.figures]`.

## Test

```sh
uv run pytest -n 2
```

- The suite replays a staged run of CalicoST easy (`2d4ce9a9`, `tests/data/sim_2d4ce9a9.hdf5`) stage by stage, so
  it needs no input files of its own. A row that restages the inputs needs CalicoST's `GRCh38_resources`:
  `$CNAMASTE_GRCH38`, else uv's git checkout; without them it skips.
- `--sims all` selects every supported fixture; `--update-digests` rewrites `tests/data/digests_<hash>.json` from the
  session, deliberately.
- `-n 2` fits a 15 GB host; `-n auto` has run out of memory there.

## Ledger

```sh
uv run python -m cnamaste.ledger tests/data/sim_<hash>.hdf5 [--replace]
```

Appends one row per (capture commit, fixture) to `docs/ledger/ledger.tsv`, scored against the planted truth the
staged file stores. `docs/ledger/README.md` states the columns.

## Provenance

From `port`'s root, not this directory:

- `uv run --locked python -m scripts.vendor_cnamaste --check`, in `port`'s environment: the copy is `cnaster`
  4adad4d renamed, and every departure is declared (`DEPARTED` for an edited file, `ADDED` for one `cnamaste` owns).
- `uv run --locked --project cnamaste python scripts/capture_cnamaste.py --work DIR [--sample ...] [--out FILE]`, in
  `cnamaste`'s: stages a sample and runs `run_cnamaste` with every key stage recorded, into
  `tests/data/sim_<hash>.hdf5`.
