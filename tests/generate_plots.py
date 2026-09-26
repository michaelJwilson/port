"""Regenerate the committed figures under `docs/plots/`.

Run as `python -m tests.generate_plots`. It writes the dev instance's inputs
through `tests.run_config.run_written`, runs **`run_cnaster_port`**
on them, and copies what it wrote into the repository. `--cnaster` runs plain
`cnaster` instead, for a comparison.

**Two sets.** `docs/plots/` is the dev instance as planted by default;
`docs/plots/lattice/` is the same genome with `copy_lattice=True`, whose states
are integer allele copies `(A, B)` (`tests.fixtures.COPY_LATTICE`), so its
copy-number figures can be read against a truth that is integer (#313).

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
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from port.extensions.combined_figure import (
    Recorded,
    genomic_figure,
    recording,
    spatial_figure,
)

from tests.fixtures import COPY_LATTICE, CoreInferenceTruth, dev_instance
from tests.he_slide import mock_he, write_he_slide
from tests.run_config import run_written

PLOTS = Path(__file__).resolve().parent.parent / "docs" / "plots"
"""Where the figures live in the repository."""

LATTICE_PLOTS = PLOTS / "lattice"
"""Where the copy-lattice set lives."""

STATES = 8
"""What the run fits: the eight states each instance uses of those it plants.

The dev instance plants ten states and its clones use eight; the copy-lattice
instance plants nine and uses eight. At five (#90's figure, set when ten
did not fit) the five planted amplifications could not be separated and
chr7's two events were decoded as one state (#313).
"""


def _write_combined(
    recorded: Recorded, truth: CoreInferenceTruth, root: Path, output: Path
) -> None:
    """The genomic and spatial figures, beside the run's own (#309, #339).

    The slide is mocked from the planted labels and read back through
    `cnaster.he.get_he_image`, as `run_cnaster` reads one. It is written
    beside the run's inputs rather than into them: `load_input_data` would
    otherwise find it and refine the initial clones by it, and the figures
    would stop being the ones the dev instance's run draws.
    """
    from cnaster.he import get_he_image
    from port.patch.utils import write_fig

    slide = mock_he(truth.labels, truth.lattice, seed=truth.seed)
    write_he_slide(slide, root / "slide")
    frame = get_he_image(str(root / "slide"), res="hires", pos=None)

    plots = next(output.rglob("clones_spatial.pdf")).parent
    # NB at its declared size, not a tight box: the page is drawn at the text
    #    width and included at 1:1, so a box that grows past it is rescaled.
    write_fig(str(plots / "genomic.pdf"), genomic_figure(recorded), bbox_inches=None)
    write_fig(
        str(plots / "spatial.pdf"), spatial_figure(recorded, frame), bbox_inches=None
    )


def main() -> None:
    """Run the pipeline and copy its figures into `docs/plots/`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cnaster",
        action="store_true",
        help="run plain cnaster rather than run_cnaster_port",
    )
    arguments = parser.parse_args()

    sets = (
        (dev_instance(), PLOTS),
        (dev_instance(n_states=len(COPY_LATTICE), copy_lattice=True), LATTICE_PLOTS),
    )

    for truth, destination in sets:
        root = Path(tempfile.mkdtemp())

        if arguments.cnaster:
            output = run_written(
                truth, root, port=False, max_iter_outer=1, max_iter=3, n_states=STATES
            )
        else:
            # NB in process, through `port.scripts.run_cnaster.main`, with its
            #    defaults: the figures are the ones a user of the entry point gets.
            with recording() as recorded:
                output = run_written(
                    truth,
                    root,
                    port=True,
                    max_iter_outer=1,
                    max_iter=3,
                    n_states=STATES,
                )

            _write_combined(recorded, truth, root, output)

        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.pdf"):
            stale.unlink()

        figures = sorted(output.rglob("*.pdf"))
        for figure in figures:
            shutil.copy(figure, destination / figure.name)

        print(f"wrote {len(figures)} figures to {destination}")


if __name__ == "__main__":
    main()
