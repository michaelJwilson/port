"""A mock H&E slide over a planted clone labelling, in `cnaster`'s layout (#309).

`run_cnaster` reads a slide through `cnaster.he.get_he_image`, which takes two
files from `{spaceranger_dir}/spatial/`: `tissue_hires_image.png`, an
`(H, W, 3)` image, and `scalefactors_json.json`, whose `tissue_hires_scalef`
maps a position to a pixel. A pixel at image row `i` and column `j` sits at
`x = j / scalef`, `y = i / scalef`, and each spot takes the colour of its
nearest pixel. The image is therefore the **transpose** of the lattice as
`tissue_positions.csv` writes it (`x` a lattice row, `y` a lattice column):
spot `(r, c)` is drawn at image row `c * scalef`, column `r * scalef`.

The stain is Beer--Lambert. Each pixel carries a haematoxylin and an eosin
concentration, and its colour is `exp(-(c_H OD_H + c_E OD_E))` with the
reference optical densities stain normalization uses, so the colours are the ones colour
deconvolution would recover rather than two hand-picked RGB triples.

- **Eosin** stains stroma and cytoplasm pink, with a fibrous texture from
  anisotropically smoothed noise. The normal clone carries the most.
- **Haematoxylin** stains nuclei purple. A tumour clone has more of them,
  larger, more variable in size and darker (hyperchromatic), rising with
  the clone index; the normal clone's are sparse, small and regular.

So a slide reads darker where the clone is further from normal, which is what
`get_he_image`'s grayscale-percentile `label` assumes, and what the
`he_label` refinement in `run_cnaster` conditions on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.ndimage

PIXELS_PER_SPOT = 24
"""`tissue_hires_scalef`: pixels per unit of spot position.

A Visium spot pitch is 100 um, so a pixel is about 4 um and a nucleus of
6 to 12 um is 1.5 to 3 pixels across its radius. 24 is at least what the
page's panel resolves at 300 dpi over 40 spots: 23 for 0.48 of a 6.5 in
column (#280), 13 for 0.36 of `llncs`'s 4.80 in (#339), so the slide is
never the coarser of the two.
"""

OD_HAEMATOXYLIN = np.array([0.5626, 0.7201, 0.4062])
OD_EOSIN = np.array([0.2159, 0.8012, 0.5581])
"""Unit optical density per RGB channel: the reference stain matrix of
Macenko et al. (2009), as stain normalization uses it. Ruifrok and
Johnston's eosin (0.072, 0.990, 0.105) renders magenta rather than pink."""

NUCLEI_PER_SPOT = (12.0, 60.0)
"""Nuclear density, normal clone to the most abnormal, per spot area."""

NUCLEUS_RADIUS = (1.6, 2.8)
"""Mean nuclear radius in pixels, normal to the most abnormal."""

PLEOMORPHISM = (0.1, 0.3)
"""Coefficient of variation of the radius, normal to the most abnormal."""

HAEMATOXYLIN = (1.0, 1.5)
"""Nuclear haematoxylin concentration, normal to the most abnormal."""

EOSIN = (0.45, 0.40)
"""Stromal eosin concentration, normal to the most abnormal.

Nearly flat on purpose: `get_he_image` reads darkness as
`0.2125 R + 0.7154 G + 0.0721 B`, and eosin absorbs green, so a stroma-rich
normal clone would otherwise read darker than the tumour."""


class Slide(NamedTuple):
    """The image, the clone each pixel was stained as, and the scale."""

    image: np.ndarray
    clone: np.ndarray
    scalefactor: float


def _ramp(bounds: tuple[float, float], n_clones: int) -> np.ndarray:
    """One value per clone, from the normal clone's to the most abnormal's."""
    return np.linspace(bounds[0], bounds[1], n_clones)


def mock_he(
    labels: np.ndarray,
    lattice: tuple[int, int],
    *,
    seed: int = 0,
    pixels_per_spot: int = PIXELS_PER_SPOT,
) -> Slide:
    """A slide stained by clone, clone 0 normal, in `get_he_image`'s layout.

    `labels` is row-major over `lattice`, as `CoreInferenceTruth.labels` is.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_columns = lattice
    n_clones = int(labels.max()) + 1
    grid = labels.reshape(lattice)

    height, width = n_columns * pixels_per_spot, n_rows * pixels_per_spot
    image_row, image_column = np.indices((height, width))

    # NB the transpose: image row is lattice column, image column lattice row.
    lattice_row = np.clip(np.rint(image_column / pixels_per_spot), 0, n_rows - 1)
    lattice_column = np.clip(np.rint(image_row / pixels_per_spot), 0, n_columns - 1)
    clone = grid[lattice_row.astype(int), lattice_column.astype(int)]

    density = _ramp(NUCLEI_PER_SPOT, n_clones)[clone] / pixels_per_spot**2
    centres = rng.random((height, width)) < density

    mean_radius = _ramp(NUCLEUS_RADIUS, n_clones)[clone]
    spread = _ramp(PLEOMORPHISM, n_clones)[clone]
    radius = np.clip(
        mean_radius * (1.0 + spread * rng.standard_normal(clone.shape)), 0.8, None
    )

    distance, (nearest_row, nearest_column) = scipy.ndimage.distance_transform_edt(
        ~centres, return_indices=True
    )
    own_radius = radius[nearest_row, nearest_column]
    nucleus = np.clip(own_radius - distance + 0.5, 0.0, 1.0)

    chromatin = _ramp(HAEMATOXYLIN, n_clones)[clone[nearest_row, nearest_column]]
    haematoxylin = 0.05 + nucleus * chromatin * rng.uniform(0.8, 1.2, clone.shape)

    fibres = scipy.ndimage.gaussian_filter(rng.standard_normal(clone.shape), (1.5, 6.0))
    fibres /= fibres.std()
    stroma = _ramp(EOSIN, n_clones)[clone] * np.clip(1.0 + 0.35 * fibres, 0.2, None)
    eosin = stroma * (1.0 - 0.6 * nucleus)

    density_od = haematoxylin[..., None] * OD_HAEMATOXYLIN + eosin[..., None] * OD_EOSIN
    image = np.exp(-density_od)
    image = scipy.ndimage.gaussian_filter(image, (0.7, 0.7, 0.0))
    image += 0.01 * rng.standard_normal(image.shape)

    return Slide(np.clip(image, 0.0, 1.0), clone, float(pixels_per_spot))


def write_he_slide(slide: Slide, spaceranger_dir: Path) -> Path:
    """Write the two files `get_he_image` reads; return the `spatial/` path."""
    import matplotlib.pyplot as plt

    spatial = spaceranger_dir / "spatial"
    spatial.mkdir(parents=True, exist_ok=True)

    plt.imsave(spatial / "tissue_hires_image.png", slide.image)
    (spatial / "scalefactors_json.json").write_text(
        json.dumps(
            {
                "tissue_hires_scalef": slide.scalefactor,
                "tissue_lowres_scalef": slide.scalefactor / 10.0,
            }
        )
    )

    return spatial
