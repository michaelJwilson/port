"""`run_benchmark`: one tool on one sample, or the patched share (#494, #302).

    run_benchmark --final SAMPLE {port,calicost} [--repeats N] [--timeout S]
                  [--n-clones K] [--root DIR]
    run_benchmark --patched-share

`--final` prints one `BENCH` JSON line; `--patched-share` records the share of
executed `cnaster` lines replaced in `.badges/measurements.json`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark the flags name; print its line."""
    from port.qa import benchmark

    parser = argparse.ArgumentParser(prog="run_benchmark", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--final", nargs=2, metavar=("SAMPLE", "TOOL"))
    mode.add_argument("--patched-share", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=benchmark.TIMEOUT,
                        help="CalicoST's cap [s]; 0 runs uncapped")  # fmt: skip
    parser.add_argument("--n-clones", type=int, default=None,
                        help="CalicoST's n_clones (shipped: 3)")  # fmt: skip
    parser.add_argument("--root", type=Path, default=None,
                        help="CalicoST's working directory, kept so a rerun resumes (default: a new one)")  # fmt: skip
    arguments = parser.parse_args(argv)

    if arguments.patched_share:
        hit, total = benchmark.record_patched_share()
        print(
            f"patched {hit} of {total} executed cnaster lines: {100.0 * hit / total:.2f}%"
        )
        return 0

    sample, tool = arguments.final
    if tool not in ("port", "calicost"):
        parser.error(f"--final's TOOL is port or calicost, not {tool!r}")
    row = (
        benchmark.port(sample, arguments.repeats)
        if tool == "port"
        else benchmark.calicost(
            sample, arguments.timeout, arguments.n_clones, arguments.root
        )
    )
    print("BENCH " + json.dumps({"sample": sample, **row}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
