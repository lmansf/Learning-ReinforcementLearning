import numpy as np

TileCoder = require("TileCoder")

# --- 1-D by hand: 2 tilings of 4 tiles on [0, 1]; tiling 1 is shifted by half a tile.
tc = TileCoder(low=[0.0], high=[1.0], n_tilings=2, tiles_per_dim=4)
check(tc.n_features == 10, "2 tilings × (4+1) tiles = 10 features", f"n_features = {tc.n_features}")
got = list(np.asarray(tc.active([0.3])).astype(int))
check(got == [1, 6], "active([0.3]) == [1, 6]",
      f"got {got}. 0.3 scales to 1.2 tiles → tiling 0: floor(1.2)=1; tiling 1: floor(1.2+0.5)=1, plus its block start 5 → 6")
got = list(np.asarray(tc.active([0.4])).astype(int))
check(got == [1, 7], "active([0.4]) == [1, 7]  (only the shifted tiling changed tile)", f"got {got}")
got = list(np.asarray(tc.active([1.0])).astype(int))
check(got == [4, 9], "The upper edge stays inside the grid (clip to 0 … tiles_per_dim)", f"active([1.0]) = {got}, expected [4, 9]")
got = list(np.asarray(tc.active([-5.0])).astype(int))
check(got == [0, 5], "Points below `low` are clipped to the first tile", f"active([-5]) = {got}, expected [0, 5]")

# --- 2-D by hand: row-major flattening
tc = TileCoder(low=[0.0, 0.0], high=[1.0, 1.0], n_tilings=1, tiles_per_dim=4)
got = list(np.asarray(tc.active([0.6, 0.3])).astype(int))
check(got == [2 * 5 + 1], "2-D tiles are flattened row-major (np.ravel_multi_index)",
      f"(0.6, 0.3) → tile coords (2, 1) → 2·5 + 1 = 11, got {got}")

# --- properties on the MountainCar box
tc = TileCoder(low=[-1.2, -0.07], high=[0.6, 0.07], n_tilings=8, tiles_per_dim=8)
rng = np.random.default_rng(0)
pts = rng.uniform(tc.low, tc.high, size=(300, 2))
ok_len = ok_range = ok_block = True
for p in pts:
    a = np.asarray(tc.active(p)).astype(int)
    ok_len &= len(a) == 8
    ok_range &= bool(np.all((a >= 0) & (a < tc.n_features)))
    ok_block &= bool(np.all(a // tc.tiles_per_tiling == np.arange(8)))
check(ok_len, "Exactly one active tile per tiling")
check(ok_range, "All indices are in 0 … n_features−1")
check(ok_block, "Tiling i's index lies in its own block [i·tiles_per_tiling, (i+1)·tiles_per_tiling)")
a = set(tc.active([-0.5, 0.0]).tolist())
near = len(a & set(tc.active([-0.49, 0.001]).tolist()))
far = len(a & set(tc.active([0.4, 0.06]).tolist()))
check(near >= 6, "Nearby states share most of their tiles (generalisation)", f"shared {near}/8")
check(far == 0, "Distant states share no tiles (discrimination)", f"shared {far}/8")
x = tc.features([-0.5, 0.0])
check(len(x) == tc.n_features and x.sum() == 8, "features() is a binary vector with 8 ones")
