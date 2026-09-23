"""Regenerate the committed figures under `docs/plots/`.

Run as `python -m tests.generate_plots`. It writes the dev instance's inputs
the way `tests/test_run_cnaster_round_trip.py` does, runs **`run_cnaster_port`**
on them, and copies what it wrote into the repository. `--cnaster` runs plain
`cnaster` instead, for a comparison.

**CI runs this on every pull request and commits the result** (#296,
`.github/workflows/figures.yml`), so the figures in a pull request are the
figures its code draws.

**The figures are committed because a plot is a result.** #87 asks for them
beside the code rather than in a run directory nobody keeps, so a change to
the pipeline shows up as a change to a picture in a diff.

They are not a referee. Nothing here compares a figure against a previous
one, and a matplotlib PDF carries a creation timestamp, so two runs of this
script differ byte for byte with nothing having changed. Byte reproduction of
figures is the standard `CLAUDE.md` names and reaching it needs the timestamp
pinned first; #103 owns that.
"""

import argparse
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from tests.fixtures import CoreInferenceTruth, dev_instance
from tests.run_config import write_run_cnaster_config
from tests.test_run_cnaster_round_trip import _run
from tests.tmp_inputs import write_tmp_inputs
from tests.unsegment import unsegment

PLOTS = Path(__file__).resolve().parent.parent / "docs" / "plots"
"""Where the figures live in the repository."""

STATES = 5
"""What the run fits, against the ten the dev instance plants.

Five because ten does not fit: the kernel kills the run at 15 GB (#90). At
five it completes in 31 s at a peak of 5.89 GB.
"""


def _run_port(truth: CoreInferenceTruth, root: Path, **config: object) -> Path:
    """The round trip's inputs and configuration, run through `run_cnaster_port`.

    In process, through `port.scripts.run_cnaster.main`, with its defaults:
    `SWAPS`, the figure table and whatever else the entry point installs by
    default. The figures are the ones a user of the entry point would get.
    """
    from port.scripts.run_cnaster import main as run_cnaster_port

    written = write_tmp_inputs(
        truth, unsegment(truth, flip_every=0, unassigned_genes=0), root
    )
    config_path = write_run_cnaster_config(written, truth, **config)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cnaster_port([str(config_path)])

    return written.root / "output"


def main() -> None:
    """Run the pipeline and copy its figures into `docs/plots/`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cnaster",
        action="store_true",
        help="run plain cnaster rather than run_cnaster_port",
    )
    arguments = parser.parse_args()

    run = _run if arguments.cnaster else _run_port
    root = Path(tempfile.mkdtemp())
    output = run(dev_instance(), root, max_iter_outer=1, max_iter=3, n_states=STATES)

    PLOTS.mkdir(parents=True, exist_ok=True)
    for stale in PLOTS.glob("*.pdf"):
        stale.unlink()

    figures = sorted(output.rglob("*.pdf"))
    for figure in figures:
        shutil.copy(figure, PLOTS / figure.name)

    print(f"wrote {len(figures)} figures to {PLOTS}")


if __name__ == "__main__":
    main()
