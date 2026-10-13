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

    `cnaster` digitizes against all percentile edges, so the brightest pixel
    gets `num_labels + 1`; binning on the inner edges alone fixes that and is
    otherwise identical. A slide without an image is returned unchanged.
    """
    frame = _UPSTREAM_HE_IMAGE(spaceranger_dir, res=res, pos=pos, num_labels=num_labels)

    # NB without a slide `cnaster` returns the positions unlabelled.
    if "cropped_gray" in frame.columns:
        gray = frame["cropped_gray"].to_numpy()
        # NB `cnaster`'s edges, as it computes them.
        bins = np.percentile(np.sort(gray), np.linspace(0.0, 100.0, 1 + num_labels))
        frame["label"] = np.digitize(gray, bins=bins[1:-1]) + 1

    return frame
