# cnamaste ledger

**TL;DR:** `ledger.tsv` holds one row per (capture commit, fixture hash): what
that staged run scores against the planted truth its file stores. Rows are
appended, never edited; `tests/test_ledger.py` checks each fixture's latest row
against its staged file.

    uv run --locked --project cnamaste python cnamaste/docs/ledger/ledger.py \
        cnamaste/tests/data/sim_<hash>.hdf5 [...] [--replace]

`ledger.py` refuses a row whose commit and fixture hash a row already holds;
`--replace` rewrites that row in place, for a capture re-run at one commit.

## Columns

| Column | Meaning |
| --- | --- |
| `date` | UTC date the row was written (`YYYY-MM-DD`) |
| `commit` | The staged file's `capture_commit`; `-dirty` where `cnamaste/python` had uncommitted edits |
| `fixture` | The fixture's name (`tests/audit/sims.py`) |
| `fixture_hash` | The 8 hex digits naming the data the run read |
| `spots` | Spots the run kept (`load_input_data`'s barcodes) |
| `bins` | Rows of `cnv_seglevel.tsv` |
| `clone_ari` | ARI of the final clone labels against the planted ones, over the spots the run kept |
| `clone_ari_int_099` | The same, after merging clones whose integer `(A, B)` agree at no less than 0.99 of the bins |
| `clone_ari_int_100` | The same, merging only clones whose integer `(A, B)` agree at every bin |
| `copy_ari_pf` | ARI of the fitted against the planted `(A, B)` at each bin midpoint, per matched clone, `(A, B)` and `(B, A)` coded alike |
| `exact_altered_pf` | Share of altered planted clone-bins whose fitted pair equals the planted one up to exchanging the haplotypes |
| `run_seconds` | The run's own wall seconds: the capture's wall time less the time its recording wrappers took |

Clones are matched one to one by `linear_sum_assignment` on the spot overlap
(`tests/audit/scoring.py`). The integer merge takes clones in id order, each
joining the first earlier group it agrees with. Scores carry 4 decimals,
`run_seconds` 1.

`run_seconds` depends on the host and its load; it is recorded, not
compared across hosts.
