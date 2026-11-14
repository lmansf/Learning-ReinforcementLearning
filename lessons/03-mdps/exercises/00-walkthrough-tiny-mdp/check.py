import numpy as np
from rlenvs import GridWorld

RobotMDP = require("RobotMDP")
model_arrays = require("model_arrays")
q_values = require("q_values")
evaluate = require("evaluate")

# ---------------------------------------------------------------- model arrays
R, T = map(np.asarray, model_arrays(RobotMDP()))
check(R.shape == (3, 2) and T.shape == (3, 2, 3), "R is (n, k) and T is (n, k, n)",
      f"got {R.shape} and {T.shape}")
ok = approx(R, [[0.0, 0.7 * 1.0 + 0.3 * -3.0], [0.0, 2.0], [0.0, 0.0]])
check(ok, "R[s, a] is the expected immediate reward Σ prob · r",
      "" if ok else f"got {np.round(R, 4).tolist()}; R[low, search] = 0.7·1 + 0.3·(−3) = −0.2. "
                    "Accumulate with += over all outcomes")
ok = approx(T[0, 1], [0.7, 0.0, 0.0]) and approx(T[1, 1], [0.4, 0.6, 0.0])
check(ok, "T keeps only the transitions that do NOT terminate",
      "" if ok else f"T[low, search] = {T[0, 1].tolist()} (expected [0.7, 0, 0]); "
                    f"T[high, search] = {T[1, 1].tolist()} (expected [0.4, 0.6, 0])")
check(approx(T[2], 0), "The terminal state has no successors at all", f"T[dead] = {T[2].tolist()}")

# ---------------------------------------------------------------- vectorised q
V = np.array([1.0, 2.0, 5.0])
Q = np.asarray(q_values(R, T, V, 0.5), dtype=float)
loop = np.zeros((3, 2))
for s in range(3):
    for a in range(2):
        for p, s2, r, d in RobotMDP.P[s][a]:
            loop[s, a] += p * (r + (0.0 if d else 0.5 * V[s2]))
check(Q.shape == (3, 2) and approx(Q, loop), "q_values matches the explicit loop over outcomes",
      f"expected {np.round(loop, 4).tolist()}, got {np.round(Q, 4).tolist()}")

# ---------------------------------------------------------------- exact evaluation
v = np.asarray(evaluate(R, T, np.array([0, 1, 0]), 0.9), dtype=float)
# by hand: v_l = 0.9 v_h ;  v_h = 2 + 0.9 (0.6 v_h + 0.4 v_l)
vh = 2 / (1 - 0.54 - 0.324)
check(approx(v, [0.9 * vh, vh, 0.0], 1e-8), "evaluate solves (I − γT_π) v = r_π exactly",
      f"expected {np.round([0.9 * vh, vh, 0], 4).tolist()}, got {np.round(v, 4).tolist()}")
env = GridWorld(["S.G"], step_reward=-0.1)
Rg, Tg = model_arrays(env)
vg = np.asarray(evaluate(Rg, Tg, np.full(3, 1), 0.9), dtype=float)
check(approx(vg, [-0.1 + 0.9 * 1.0, 1.0, 0.0]), "Works on a GridWorld too (corridor S . G, always RIGHT)",
      f"got {np.round(vg, 4).tolist()}, expected [0.8, 1.0, 0.0]")
