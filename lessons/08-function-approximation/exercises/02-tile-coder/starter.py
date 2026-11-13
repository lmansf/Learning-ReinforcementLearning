import numpy as np
import rlhud


class TileCoder:
    """Tile coding for a box-shaped continuous space.

    tc = TileCoder(low=[-1.2, -0.07], high=[0.6, 0.07], n_tilings=8, tiles_per_dim=8)
    idx = tc.active([-0.5, 0.01])      # -> array of n_tilings feature indices (one per tiling)
    x   = tc.features([-0.5, 0.01])    # -> binary vector of length tc.n_features

    Each tiling is a grid of `tiles_per_dim` tiles per dimension, shifted by a different
    fraction of a tile. Each grid gets one extra row/column of tiles so the shifted grid
    still covers the whole box.
    """

    def __init__(self, low, high, n_tilings=8, tiles_per_dim=8):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.n_tilings = n_tilings
        self.tiles = tiles_per_dim
        self.dims = len(self.low)
        self.shape = (tiles_per_dim + 1,) * self.dims          # tiles per tiling, per dimension
        self.tiles_per_tiling = (tiles_per_dim + 1) ** self.dims
        self.n_features = n_tilings * self.tiles_per_tiling
        # offsets[i, k] = how far tiling i is shifted along dimension k, in units of one tile.
        # Asymmetric shifts (1, 3, 5, ... × i / n) avoid all tilings lining up on the diagonal.
        k = np.arange(self.dims)
        self.offsets = (np.arange(n_tilings)[:, None] * (2 * k + 1)[None, :] / n_tilings) % 1.0

    def active(self, x):
        """Indices of the n_tilings active tiles for point x (one per tiling)."""
        # TODO
        # 1) rescale x so that one tile has width 1:  (x − low) / (high − low) · tiles_per_dim
        # 2) for each tiling i: coords = floor(scaled + self.offsets[i]), clipped to 0 … self.tiles
        # 3) flatten coords in row-major order (np.ravel_multi_index(coords, self.shape))
        #    and add i · self.tiles_per_tiling so every tiling has its own block of indices
        raise NotImplementedError

    def features(self, x):
        """The same information as a binary feature vector x(s)."""
        v = np.zeros(self.n_features)
        v[self.active(x)] = 1.0
        return v


if __name__ == "__main__":
    # How much do two states "share"? Overlap of active tiles between the centre of the
    # MountainCar box and every other point = how much an update at one moves the other.
    tc = TileCoder(low=[-1.2, -0.07], high=[0.6, 0.07], n_tilings=8, tiles_per_dim=8)
    centre = set(tc.active([-0.3, 0.0]).tolist())
    pos = np.linspace(-1.2, 0.6, 25)
    vel = np.linspace(0.07, -0.07, 15)            # top row = fast to the right
    overlap = [[len(centre & set(tc.active([p, v]).tolist())) for p in pos] for v in vel]
    rlhud.grid(overlap, title="Shared tiles with the point (−0.3, 0.0)", fmt=".0f", cmap="heat")

    # Generalisation in 1-D: fit a sine with a single linear SGD pass over tile features.
    tc1 = TileCoder(low=[0.0], high=[1.0], n_tilings=8, tiles_per_dim=10)
    rng = np.random.default_rng(0)
    w = np.zeros(tc1.n_features)
    alpha = 0.1 / tc1.n_tilings                   # step size per active tile
    for _ in range(3000):
        s = rng.random()
        idx = tc1.active([s])
        w[idx] += alpha * (np.sin(2 * np.pi * s) - w[idx].sum())
    grid_x = np.linspace(0, 1, 101)
    rlhud.plot([np.sin(2 * np.pi * s) for s in grid_x], label="target sin(2πs)", title="Tile-coded fit")
    rlhud.plot([w[tc1.active([s])].sum() for s in grid_x], label="tile coding (8×10)", title="Tile-coded fit")
    rlhud.stat("Features", tc.n_features)
