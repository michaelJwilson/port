"""What one genomic figure costs in memory to draw and to write (T- #692 part 2).

**The peak is `write_fig`'s, not `plot_clones_genomic`'s, and port's row
takes it from +1,577 MB to +311 MB, 5.1x.** Measured on CalicoST easy
(`2d4ce9a9`): the `merged_rdr_baf_clones_genomic` call captured from an
unpatched run, replayed alone, peak RSS over the process's own baseline:

| drawn by | written by | draw | draw and write |
| --- | --- | ---: | ---: |
| `cnaster` | `cnaster` (300 dpi, a group per artist) | +116 MB | +1,577 MB |
| `cnaster` | 150 dpi | +115 MB | +486 MB |
| `cnaster` | 300 dpi, a group per axes | +115 MB | +871 MB |
| `cnaster` | port's row (150 dpi, a group per axes) | +116 MB | +311 MB |
| port | `cnaster` | +116 MB | +1,577 MB |
| port | port's row | +116 MB | +311 MB |

Both draws cost the same; the write is the peak, and each of the row's two
defaults removes part of it. Four captured calls replayed twice in one
process leave +42 to +94 MB after each under `cnaster`, +45 to +48 MB under
port, and +22 MB with `malloc_trim`: **no figure is retained**, and none
ratchets. So `cnaster`'s 11.8 to 13.9 GB arms on easy and dev are the run's
own arrays plus `write_fig`'s 1.6 GB, not figures left open, and
`plot_clones_genomic` needs no change. `cnaster` is read only; the defect is
reported on T- #692 with these numbers.

Pinned here at the dev instance's size (2,624 bins, 3,000 spots, 3 clones),
each arm in its own process, since peak RSS is a property of a process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ARM = """
import gc, json, resource, sys
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
from port.patch.plot_genomic import plot_clones_genomic

def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024

rng = np.random.default_rng(3)
n_obs, n_spots, n_clones, n_states = 2624, 3000, 3, 6
total = rng.integers(2, 12, size=(n_obs, n_spots)).astype(float)
X = np.zeros((n_obs, 2, n_spots))
X[:, 0, :] = rng.poisson(5, size=(n_obs, n_spots))
X[:, 1, :] = rng.binomial(total.astype(int), 0.45)
lengths = np.full(22, n_obs // 22)
lengths[-1] += n_obs - lengths.sum()
result = {
    "new_assignment": np.arange(n_spots) % n_clones,
    "pred_cnv": rng.integers(0, n_states, size=(n_obs, n_clones)),
    "new_log_mu": rng.normal(0.0, 0.2, size=(n_states, 1)),
    "new_p_binom": rng.uniform(0.15, 0.85, size=(n_states, 1)),
}
base_nb_mean = rng.uniform(4.0, 6.0, size=(n_obs, n_spots))
if sys.argv[1] == "cnaster":
    from cnaster.utils import write_fig
    options = {}
else:
    from port.patch.utils import write_fig
    options = {"dpi": 150, "group_rasters": True}
gc.collect()
base = rss()
figure = plot_clones_genomic(lengths, X, base_nb_mean, total, res_combine=result)
write_fig(sys.argv[2], figure, transparent=True, bbox_inches="tight", **options)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
print(json.dumps({"over_base_mb": peak - base}))
"""


def _over_base(arm: str, out: Path) -> int:
    done = subprocess.run(
        [sys.executable, "-c", ARM, arm, str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(done.stdout.strip().splitlines()[-1])["over_base_mb"])


@pytest.mark.smoke
@pytest.mark.release
def test_port_writes_a_dev_sized_genomic_figure_in_under_half_cnasters_memory(
    tmp_path: Path,
) -> None:
    """Port's `write_fig` row against `cnaster`'s on the same figure: at least 2x less peak.

    Measured here, twice: +900 and +901 MB under `cnaster`'s writer, +213 MB
    under port's (4.2x). The bar is `CLAUDE.md`'s 2x; the figure is drawn by port's
    `plot_clones_genomic` in both arms, so the writer is the only difference.
    """
    theirs = _over_base("cnaster", tmp_path / "cnaster.pdf")
    ours = _over_base("port", tmp_path / "port.pdf")

    assert ours * 2 <= theirs, f"port +{ours} MB, cnaster +{theirs} MB"
