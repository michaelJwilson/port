# cnaster's ledger (T- #833)

`cnaster` at its pin, recorded as `docs/metrics/` records port: the same
`ledger.tsv` and `runs.tsv` columns, under the shared `../definitions.tsv`.
Every run is `port.qa.cnaster_arm`: `--no-patch` with #105's row alone.

    run_ledger --record --cnaster --sample PATH --fixture NAME --note "..." -- --no-patch --no-plots

`--benchmark` marks a sweep over `../fixtures.md`'s supported fixtures, as it
does in port's ledger. Fixture names and hashes are that table's. The two
ledgers share no run, and neither counts the other's runs.
