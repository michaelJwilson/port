"""Replaces `cnaster.count_encoder.CountEncoder`, decoding by gather and encoding by `bincount` (T- #799).

Uses the `np.unique` inverse in place of `cnaster`'s one-hot CSR maps; codes
are `cnaster`'s (same rounding, zero-total collapse and order). Decode is
bitwise, encode to 1e-12 relative. CSR maps are built only on read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cnaster.count_encoder import CountEncoder as _UPSTREAM_COUNT_ENCODER

__all__ = ["CountEncoder"]


def _codes(
    obs: np.ndarray, total: np.ndarray, *, common_zero_depth: bool
) -> tuple[np.ndarray, np.ndarray]:
    """One spot's distinct `(obs, total)` pairs and each entry's index into them, as `cnaster` codes them."""
    o, t = obs.copy(), total.copy()
    if common_zero_depth:
        zero = t == 0
        o[zero] = 0
        t[zero] = 0
    counts = np.vstack([o, t]).T
    if not np.issubdtype(total.dtype, np.integer):
        from cnaster.config import get_global_config

        counts = counts.round(decimals=get_global_config().hmm.compression_decimals)
    pairs, inverse = np.unique(counts, axis=0, return_inverse=True)
    return pairs, np.ascontiguousarray(inverse.reshape(-1), dtype=np.int32)


class CountEncoder:
    """`cnaster`'s encoder, its signature and attributes, decoding by gather and encoding by `bincount`."""

    def __init__(
        self, obs_count: Any, total_count: Any, common_zero_depth: bool = False
    ) -> None:
        self.obs_count = obs_count
        self.total_count = total_count
        self.n_obs = obs_count.shape[0]
        self.n_spots = obs_count.shape[1]
        self.common_zero_depth = common_zero_depth
        self.unique_counts: list[np.ndarray] = []
        self.inverses: list[np.ndarray] = []
        for spot in range(self.n_spots):
            pairs, inverse = _codes(
                np.asarray(obs_count[:, spot]),
                np.asarray(total_count[:, spot]),
                common_zero_depth=common_zero_depth,
            )
            self.unique_counts.append(pairs)
            self.inverses.append(inverse)
        self._mapping: list[Any] | None = None

    def encode_array(self, array: Any, spot: int) -> np.ndarray:
        """`[..., n_obs]` or `[n_obs, ...]` summed into `[..., n_unique]` or `[n_unique, ...]` by code."""
        inverse = self.inverses[spot]
        n_unique = self.unique_counts[spot].shape[0]
        values = np.asarray(array, dtype=np.float64)
        if values.shape[-1] == inverse.size:
            rows = values.reshape(-1, inverse.size)
            out = np.empty((rows.shape[0], n_unique))
            for i in range(rows.shape[0]):
                out[i] = np.bincount(inverse, weights=rows[i], minlength=n_unique)
            return out.reshape(*values.shape[:-1], n_unique)
        if values.shape[0] == inverse.size:
            return np.moveaxis(
                self.encode_array(np.moveaxis(values, 0, -1), spot), -1, 0
            )
        msg = f"Array shape {values.shape} not compatible with {inverse.size} observations."
        raise ValueError(msg)

    def decode_array(self, array: Any, spot: int) -> np.ndarray:
        """`[..., n_unique]` or `[n_unique, ...]` gathered to `[..., n_obs]` or `[n_obs, ...]`."""
        inverse = self.inverses[spot]
        n_unique = self.unique_counts[spot].shape[0]
        values = np.asarray(array)
        if values.shape[-1] == n_unique:
            return np.take(values, inverse, axis=-1)
        if values.shape[0] == n_unique:
            return np.take(values, inverse, axis=0)
        msg = f"Array shape {values.shape} not compatible with {n_unique} codes."
        raise ValueError(msg)

    def get_unique_obs(self, spot: int) -> np.ndarray:
        return self.unique_counts[spot][:, 0]

    def get_unique_total(self, spot: int) -> np.ndarray:
        return self.unique_counts[spot][:, 1]

    @property
    def compression_rate(self) -> float:
        total = int(self.n_obs) * int(self.n_spots)
        if total == 0:
            return 0.0
        coded = sum(int(u.shape[0]) for u in self.unique_counts)
        return 1.0 - coded / total

    @property
    def mapping_matrices(self) -> list[Any]:
        """`cnaster`'s one-hot CSR maps, built on first read: no live reader reads them."""
        if self._mapping is None:
            import scipy.sparse

            self._mapping = [
                scipy.sparse.csr_matrix(
                    (np.ones(inverse.size), (np.arange(inverse.size), inverse)),
                    shape=(inverse.size, pairs.shape[0]),
                )
                for pairs, inverse in zip(
                    self.unique_counts, self.inverses, strict=True
                )
            ]
        return self._mapping

    @staticmethod
    def construct_unique_encoding(
        obs_count: Any, total_count: Any, common_zero_depth: bool = True
    ) -> tuple[list[np.ndarray], list[Any]]:
        """`cnaster`'s own: the pairs and the CSR maps, as `hmm_utils.construct_unique_matrix` returns them."""
        pairs, maps = _UPSTREAM_COUNT_ENCODER.construct_unique_encoding(
            obs_count, total_count, common_zero_depth=common_zero_depth
        )
        return list(pairs), list(maps)
