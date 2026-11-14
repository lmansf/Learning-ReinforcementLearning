import numpy as np

nstep_targets = require("nstep_targets")
trace_history = require("trace_history")
Memory = require("Memory")

# ---------------------------------------------------------------- n-step targets
rewards, values = [1.0, 0.0, 2.0], [10.0, 20.0, 40.0]
g1 = np.asarray(nstep_targets(rewards, values, 1, 0.5), dtype=float)
check(approx(g1, [1 + 0.5 * 20, 0 + 0.5 * 40, 2.0]), "n = 1 is the TD(0) target r + γ·V(next)",
      f"expected [11, 20, 2], got {g1.tolist()} — the last step must not bootstrap")
g2 = np.asarray(nstep_targets(rewards, values, 2, 0.5), dtype=float)
check(approx(g2, [1 + 0 + 0.25 * 40, 0 + 0.5 * 2, 2.0]), "n = 2: two rewards, then γ²·V",
      f"expected [11, 1, 2], got {g2.tolist()}")
g9 = np.asarray(nstep_targets(rewards, values, 9, 0.5), dtype=float)
check(approx(g9, [1 + 0 + 0.25 * 2, 0 + 0.5 * 2, 2.0]), "n ≥ episode length gives the Monte Carlo return",
      f"expected [1.5, 1, 2], got {g9.tolist()} — clip the window with min(t + n, T)")

# ---------------------------------------------------------------- traces
E = np.asarray(trace_history([0, 1, 1], 3, 1.0, 0.5), dtype=float)
check(E.shape == (3, 3), "One row per visit, one column per state", f"got shape {E.shape}")
ok = approx(E, [[1, 0, 0], [0.5, 1, 0], [0.25, 1.5, 0]])
check(ok, "Accumulating trace: bump by 1, then fade by γλ",
      "" if ok else f"got {E.tolist()}. If every row is identical, append e.copy(), not e")

# ---------------------------------------------------------------- model memory
m = Memory(seed=0)
m.store(0, 1, 0.0, 1, False)
m.store(0, 1, 5.0, 2, True)
m.store(3, 0, -1.0, 4, False)
check(m.model == {(0, 1): (5.0, 2, True), (3, 0): (-1.0, 4, False)},
      "store keeps the LAST outcome per (s, a) key", f"model = {m.model}")
picks = [m.sample() for _ in range(2000)]
ok = all(isinstance(p, tuple) and len(p) == 5 for p in picks)
check(ok, "sample returns (s, a, r, s2, done)", f"first sample {picks[0]!r}")
if ok:
    n0 = sum(p[:2] == (0, 1) for p in picks)
    check(800 < n0 < 1200, "Remembered pairs are sampled uniformly", f"(0, 1) drawn {n0}/2000 times")
    check(all(p in [(0, 1, 5.0, 2, True), (3, 0, -1.0, 4, False)] for p in picks),
          "Each sample matches what was stored for that pair")
