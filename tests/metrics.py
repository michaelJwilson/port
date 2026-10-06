"""`fixture_hash`, at the name `docs/metrics/definitions.tsv` cites (#409).

The ledger moved to `port.qa.ledger` and `run_ledger` (T- #673 G2), and
`fixture_hash` beside the truth it hashes, `port.sim.truth` (G3).
`definitions.tsv` is append-only and names `tests.metrics.fixture_hash` as
`fixture_hash` definition 1, so the name stays importable here.
"""

from port.sim.truth import fixture_hash

__all__ = ["fixture_hash"]
