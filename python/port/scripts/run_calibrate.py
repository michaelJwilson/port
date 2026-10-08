"""`run_calibrate --potts|--copy MANIFEST ...`: measure the settings the studies read, into `configs/` (#749 WP1).

Tunes on held-out realizations and merges the result into
`port.qa.provenance.CONFIGS/<name>.json`, keeping other entries.

    run_calibrate --potts MANIFEST OUT_DIR [--samplers SAMPLER ...] [--held-out 5] [--workers 4] [--states run]
    run_calibrate --copy MANIFEST [--starts NAME ...] [--held-out 5] [--workers 2]

`--potts` writes `potts_sampler_settings.json` (#556); `--copy` writes
`copy_sampler_settings.json` (#540).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate what the first argument names; 0 once its file is written."""
    from port.studies.stream import HELD_OUT

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--potts", action="store_true", help="the Potts samplers")
    kind.add_argument("--copy", action="store_true", help="the copy-state starts")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=None,
        help="--potts: where the held-out realizations are drawn and run",
    )
    parser.add_argument("--held-out", type=int, default=HELD_OUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--samplers", nargs="+", default=None, help="--potts: these alone"
    )
    parser.add_argument("--starts", nargs="+", default=None, help="--copy: these alone")
    parser.add_argument(
        "--states",
        choices=("run", "planted"),
        default="run",
        help="--potts: the Baum-Welch before the field starts from the run's states or the planted ones",
    )
    arguments = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if arguments.potts:
        from port.studies import potts_stream

        if arguments.out_dir is None:
            parser.error("--potts needs OUT_DIR")
        potts_stream.retune(
            arguments.manifest,
            tuple(arguments.samplers or potts_stream.TUNED),
            arguments.held_out,
            arguments.workers or 4,
            arguments.out_dir,
            arguments.states,
        )
        print(potts_stream.SETTINGS)
        return 0

    from port.studies import copy_state_stream

    copy_state_stream.retune(
        arguments.manifest,
        arguments.held_out,
        arguments.workers or 2,
        tuple(arguments.starts or copy_state_stream.GRID),
    )
    print(copy_state_stream.SETTINGS)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script calls main
    raise SystemExit(main())
