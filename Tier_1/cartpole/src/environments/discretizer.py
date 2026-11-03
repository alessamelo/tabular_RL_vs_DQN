"""Uniform grid discretization of CartPole's continuous observation.

How it works
------------
For dimension ``i`` with range ``[low_i, high_i]`` and ``n_i`` bins we create the
``n_i - 1`` *interior* edges

    edges_i = linspace(low_i, high_i, n_i + 1)[1:-1]

and map a value ``v`` to the bin index ``k = digitize(v, edges_i)``, i.e. the number of
interior edges that are <= v. This gives ``k`` in ``{0, ..., n_i - 1}``:

* values inside the range land in their uniform bin;
* values below ``low_i`` land in bin 0 and values above ``high_i`` in bin ``n_i - 1``.

So out-of-range values are **saturated** into the outermost bins, never discarded: the
edge bins mean "very negative" / "very positive". With an even ``n_i`` and a symmetric
range, 0 is an interior edge, so the sign of every variable is always preserved
(a value of exactly 0.0 goes to the upper bin, because ``digitize`` uses ``edge <= v``).

The four per-dimension indices are combined into one integer state with row-major
(C-order) ravelling, which is a bijection between the grid cells and
``{0, ..., prod(n_i) - 1}`` and lets the Q-table be a plain 2-D array.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class UniformDiscretizer:
    """Map a continuous observation vector to a single integer state id."""

    def __init__(
        self,
        lows: Sequence[float],
        highs: Sequence[float],
        bins: int | Sequence[int],
    ) -> None:
        lows_arr = np.asarray(lows, dtype=np.float64)
        highs_arr = np.asarray(highs, dtype=np.float64)
        if lows_arr.shape != highs_arr.shape or lows_arr.ndim != 1:
            raise ValueError("lows and highs must be 1-D sequences of equal length")
        if np.any(highs_arr <= lows_arr):
            raise ValueError("every high must be strictly greater than its low")

        n_dims = lows_arr.size
        bins_arr = np.full(n_dims, bins, dtype=np.int64) if np.isscalar(bins) else np.asarray(bins, dtype=np.int64)
        if bins_arr.shape != (n_dims,):
            raise ValueError(f"bins must be an int or a sequence of length {n_dims}")
        if np.any(bins_arr < 2):
            raise ValueError("each dimension needs at least 2 bins")

        self.lows = lows_arr
        self.highs = highs_arr
        self.bins = bins_arr
        self.n_dims = n_dims
        # Interior edges only: n_i - 1 edges split the range into n_i bins.
        self.edges: list[np.ndarray] = [
            np.linspace(lo, hi, n + 1)[1:-1] for lo, hi, n in zip(lows_arr, highs_arr, bins_arr)
        ]
        self.n_states = int(np.prod(bins_arr))

    def bin_indices(self, observation: np.ndarray) -> np.ndarray:
        """Return the per-dimension bin index (shape ``(n_dims,)``)."""
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.n_dims,):
            raise ValueError(f"expected observation of shape ({self.n_dims},), got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"observation contains non-finite values: {obs}")
        return np.array([np.digitize(obs[i], self.edges[i]) for i in range(self.n_dims)], dtype=np.int64)

    def out_of_range(self, observation: np.ndarray) -> np.ndarray:
        """Boolean mask of dimensions whose value lies outside ``[low, high]``.

        Used to measure how often information is lost by saturation.
        """
        obs = np.asarray(observation, dtype=np.float64)
        return (obs < self.lows) | (obs > self.highs)

    def __call__(self, observation: np.ndarray) -> int:
        """Return the flat integer state id in ``[0, n_states)``."""
        return int(np.ravel_multi_index(tuple(self.bin_indices(observation)), tuple(self.bins)))

    def __repr__(self) -> str:
        return (
            f"UniformDiscretizer(bins={self.bins.tolist()}, lows={self.lows.tolist()}, "
            f"highs={self.highs.tolist()}, n_states={self.n_states})"
        )
