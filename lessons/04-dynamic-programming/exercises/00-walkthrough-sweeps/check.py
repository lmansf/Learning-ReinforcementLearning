import numpy as np
from rlenvs import GridWorld

backup = require("backup")
sweep = require("sweep")
evaluate_until = require("evaluate_until")
greedy_one_hot = require("greedy_one_hot")

RIGHT, LEFT = 1, 3

# ---------------------------------------------------------------- one backup
env = GridWorld(["S.G"], step_reward=-1.0, rewards={"G": 10.0})
V = np.array([2.0, 5.0, 0.0])
b = backup(env, V, 0, RIGHT, 0.5)
check(approx(b, -1 + 0.5 * 5.0), "backup(S, RIGHT) = r + γ·V(next)", f"expected 1.5, got {b}")
b = backup(env, V, 1, RIGHT, 0.5)
check(approx(b, 10.0), "No γ·V(s') after a terminating transition",
      f"expected 10.0, got {b} — use the `done` flag")

# ---------------------------------------------------------------- in place vs two arrays
# corridor S . . G, always RIGHT: only state 2 is next to the goal
env = GridWorld(["S..G"], step_reward=0.0)
acts = np.full(env.n_states, RIGHT)
V = np.zeros(4)
out, d = sweep(env, V, acts, 1.0, in_place=False)
check(approx(out, [0, 0, 1, 0]) and approx(V, 0), "Two-array sweep: new values in a copy, V untouched",
      f"returned {np.round(out, 3).tolist()}, V is now {np.round(V, 3).tolist()} (expected V unchanged)")
check(approx(d, 1.0), "Two-array sweep reports Δ = 1", f"got Δ = {d}")

# mirrored corridor G . . S, always LEFT: states are swept 0, 1, 2, … towards the start
env2 = GridWorld(["G..S"], step_reward=0.0)
acts2 = np.full(env2.n_states, LEFT)
V = np.zeros(4)
out, d = sweep(env2, V, acts2, 1.0, in_place=True)
check(out is V, "In-place sweep writes into the very array it was given")
out2, _ = sweep(env2, np.zeros(4), acts2, 1.0, in_place=False)
check(approx(out2, [0, 1, 0, 0]), "Two arrays: every backup reads only the OLD values",
      f"got {np.round(out2, 3).tolist()}, expected [0, 1, 0, 0] — read from V, write into the copy")
check(approx(out, [0, 1, 1, 1]), "In place, later states already see this sweep's new values",
      f"got {np.round(out, 3).tolist()}, expected [0, 1, 1, 1] (state 1 updated first, then 2 uses it)")

# ---------------------------------------------------------------- convergence loop
env = GridWorld(["...G", ".#.X", "S..."], slip=0.1, step_reward=-0.04)
acts = np.full(env.n_states, RIGHT)
V, deltas = evaluate_until(env, acts, 0.9, theta=1e-10)
V, deltas = np.asarray(V, dtype=float), list(deltas)
n = env.n_states
P, r = np.zeros((n, n)), np.zeros(n)
for s in range(n):
    for p, s2, rr, dn in env.P[s][acts[s]]:
        r[s] += p * rr
        if not dn:
            P[s, s2] += p
exact = np.linalg.solve(np.eye(n) - 0.9 * P, r)
check(approx(V, exact, 1e-8), "evaluate_until converges to the exact v_π",
      f"max error {np.abs(V - exact).max():.2e}")
check(deltas[-1] < 1e-10 and all(x >= 1e-10 for x in deltas[:-1]),
      "It stops at the first sweep with Δ < θ, and records every Δ",
      f"{len(deltas)} sweeps, last Δs {deltas[-3:]}")
V1, d1 = evaluate_until(env, acts, 0.9, max_sweeps=1)
check(len(d1) == 1, "max_sweeps limits the number of sweeps", f"got {len(d1)} deltas")

# ---------------------------------------------------------------- policy extraction
pi = np.asarray(greedy_one_hot(env, V, 0.9))
check(pi.shape == (n, 4) and approx(pi.sum(axis=1), 1), "greedy_one_hot returns one-hot rows",
      f"shape {pi.shape}, row sums {pi.sum(axis=1)[:4]}")
check(pi[env.state_of(2, 0), 0] == 1, "From the start the greedy improvement goes UP (away from the pit row)",
      f"row for the start state: {pi[env.state_of(2, 0)].tolist()}")
