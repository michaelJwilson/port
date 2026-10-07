"""`cnaster.he`'s replacement: the H&E gray-level classes in `1..num_labels` (#311, T- #771)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from cnaster.he import get_he_image as _UPSTREAM_HE_IMAGE


def he_image(
    spaceranger_dir: str, res: str = "hires", pos: Any = None, num_labels: int = 4
) -> pd.DataFrame:
    """`cnaster.he.get_he_image`, with every label in `1..num_labels` (#311).

    `cnaster` bins the gray level with `np.digitize` against its 0th to
    100th percentiles; `digitize` puts a value equal to the last edge past
    it, so the brightest pixel is labelled `num_labels + 1`, and
    `run_cnaster` factorizes that label into an initial clone of its own.
    Clipping folds it into the brightest bin, which is where the percentile
    binning means it to be; every other label is `cnaster`'s, and a slide
    without an image is returned as `cnaster` returns it.

    A `SWAPS` row installs it over every caller of `cnaster.he.get_he_image`,
    `run_cnaster`'s figure frame among them (T- #771); `cnaster`'s is reached
    through a private name the rebinding does not follow.
    """
    frame = _UPSTREAM_HE_IMAGE(spaceranger_dir, res=res, pos=pos, num_labels=num_labels)

    # NB without a slide `cnaster` returns the positions unlabelled.
    if "label" in frame.columns:
        frame["label"] = np.clip(frame["label"].to_numpy(), 1, num_labels)

    return frame
