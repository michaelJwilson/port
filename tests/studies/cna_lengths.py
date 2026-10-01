"""#619: the `[cna.length]` laws, exponential against lognormal, per defined quantity.

`python -m tests.studies.cna_lengths` writes `docs/plots/sim/cna_lengths.png`:
one row per keying -- `dev_tree`'s mean of 50 Mb, `dev_tree_1s_hard`'s median
of 10 Mb -- with the density (left) and the CDF (right) of the exponential the
manifests drew to #619 and the lognormal they draw now. Both laws in a row
hold the manifest's defined quantity: the exponential's mean is the stated
mean, or the stated median over ln 2. The lognormal is the manifest's own
`[cna.length]`; dotted lines mark half the lognormal median and the `minimum`.

The figure is analytic (`scipy.stats`), so its data are the two manifests:
the stamp names each file's SHA-256 (first 8 hex) and its stated `r0_hash`,
and the commit that drew it.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from port.sim.draw import extended, lognormal_median
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "sim" / "manifests"
KEYED = ("dev_tree.toml", "dev_tree_1s_hard.toml")
OUT = ROOT / "docs" / "plots" / "sim" / "cna_lengths.png"
EXPONENTIAL, LOGNORMAL = "#eb6834", "#2a78d6"
"""Categorical slots 2 and 1: the law drawn to #619, and the law drawn now."""


def laws(name: str) -> tuple[Any, Any, dict[str, Any]]:
    """The exponential at the manifest's defined quantity, its lognormal, its law."""
    law = extended(MANIFESTS / name)["cna"]["length"]
    mean = float(law["mean"]) if "mean" in law else float(law["median"]) / np.log(2)
    median = lognormal_median(law)
    return (
        stats.expon(scale=mean),
        stats.lognorm(s=float(law["sigma"]), scale=median),
        law,
    )


def stamp_text() -> str:
    """Each manifest's file hash and `r0_hash`, and the commit (`+` if dirty)."""
    parts = []
    for name in KEYED:
        path = MANIFESTS / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        r0 = tomllib.loads(path.read_text())["sample"]["r0_hash"]
        parts.append(f"{path.stem} {digest} (r0 {r0})")

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return done.stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = "+" if git("status", "--porcelain", "--untracked-files=no") else ""
    return f"data {' · '.join(parts)} · code {commit}{dirty}"


def main(argv: list[str] | None = None) -> int:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from port.extensions.figure_style import figure_rc

    out = Path(argv[0]) if argv else OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context(figure_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), constrained_layout=True)

        for row, name in zip(axes, KEYED, strict=True):
            exponential, lognormal, law = laws(name)
            key = "mean" if "mean" in law else "median"
            top = 3.0 * float(law[key]) / 1e6
            mb = np.linspace(0.0, top, 600)[1:]
            half = lognormal.median() / 2e6
            title = (
                f"{Path(name).stem}: {key} {float(law[key]) / 1e6:.0f} Mb, "
                rf"$\sigma$ = {law['sigma']}"
            )

            for ax, kind in zip(row, ("pdf", "cdf"), strict=True):
                for dist, colour, label in (
                    (exponential, EXPONENTIAL, "exponential (to #619)"),
                    (lognormal, LOGNORMAL, "lognormal (#619)"),
                ):
                    scale = 1e6 if kind == "pdf" else 1.0
                    values = getattr(dist, kind)(mb * 1e6) * scale
                    ax.plot(mb, values, color=colour, lw=2, label=label)

                for x, text, drop in (
                    (half, "½ median", -12),
                    (float(law["minimum"]) / 1e6, "minimum", -26),
                ):
                    ax.axvline(x, color="0.55", lw=1, ls=":")
                    ax.annotate(
                        text, (x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, drop), textcoords="offset points", fontsize=8,
                        color="0.35",
                    )  # fmt: skip

                if kind == "cdf":
                    shares = (exponential.cdf(half * 1e6), lognormal.cdf(half * 1e6))
                    ax.set_ylim(0, 1)
                    ax.text(
                        0.97, 0.05,
                        f"P(< ½ median): {shares[0]:.3f} → {shares[1]:.3f}",
                        transform=ax.transAxes, ha="right", fontsize=9,
                    )  # fmt: skip
                ax.set_xlim(0, top)
                ax.set_xlabel("event length (Mb)")
                ax.set_ylabel("density (per Mb)" if kind == "pdf" else "CDF")
                ax.grid(color="0.9", lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)

            row[0].set_title(title, loc="left", fontsize=10)
            row[0].legend(frameon=False, fontsize=9)

        fig.text(
            0.995, 0.002, stamp_text(), ha="right", va="bottom", fontsize=6.5,
            color="0.35", family="monospace",
        )  # fmt: skip
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))  # type: ignore[union-attr, call-arg]
        fig.savefig(out, dpi=150)
        plt.close(fig)

    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
