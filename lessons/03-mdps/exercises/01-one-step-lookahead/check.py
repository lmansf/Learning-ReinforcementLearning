import numpy as np
from rlenvs import GridWorld

q_from_v = require("q_from_v")
q_table = require("q_table")
greedy_policy = require("greedy_policy")

# ---------------------------------------------------------------- a 1×3 corridor:  S . G
env = GridWorld(["S.G"], step_reward=-1.0, rewards={"G": 10.0})
V = np.array([2.0, 5.0, 0.0])
q = np.asarray(q_from_v(env, V, 0, 0.5), dtype=float)
# UP/DOWN/LEFT bump into the border and stay in state 0; RIGHT moves to state 1.
expected = [-1 + 0.5 * 2, -1 + 0.5 * 5, -1 + 0.5 * 2, -1 + 0.5 * 2]
ok = approx(q, expected)
check(ok, "q(S, ·) in the corridor S . G with γ = 0.5", f"expected {expected}, got {np.round(q, 4).tolist()}"
      if not ok else f"{np.round(q, 4).tolist()}")

q = np.asarray(q_from_v(env, V, 1, 0.5), dtype=float)
ok = approx(q[1], 10.0)
check(ok, "Entering the terminal goal gives just the reward (no γ·V after termination)",
      "" if ok else f"q(1, RIGHT) should be 10.0, got {q[1]:.4f}. If you got 10 + 0.5·V(G) you bootstrapped "
                    "through a terminal transition — use the `terminated` flag")
check(approx(q[3], -1 + 0.5 * 2.0), "Moving LEFT from the middle backs up V(S)", f"got {q[3]:.4f}")

# ---------------------------------------------------------------- stochastic outcomes
env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
rng = np.random.default_rng(0)
V = rng.normal(size=env.n_states)
s = env.state_of(1, 2)  # next to the pit, below the goal-adjacent cell
q = np.asarray(q_from_v(env, V, s, 0.9), dtype=float)
# hand-expanded: UP → 0.8 to (0,2), 0.1 right into the pit X, 0.1 left into the wall (stay)
up = 0.8 * (-0.04 + 0.9 * V[env.state_of(0, 2)]) + 0.1 * (-1.0) + 0.1 * (-0.04 + 0.9 * V[s])
ok = approx(q[0], up, 1e-9)
check(ok, "Slippery world: q weights every outcome by its probability",
      "" if ok else f"q(s, UP) expected {up:.5f}, got {q[0]:.5f}. Sum prob·(r + γ·V(s')) over ALL outcomes in env.P[s][a]")

Q = np.asarray(q_table(env, V, 0.9), dtype=float)
check(Q.shape == (env.n_states, 4), "q_table has shape (n_states, n_actions)", f"got {Q.shape}")
check(approx(Q[s], q), "q_table row s equals q_from_v(env, V, s, γ)")
check(approx(Q[env.state_of(0, 3)], 0.0), "Terminal states have q = 0 for every action",
      f"got {Q[env.state_of(0, 3)]}")

# ---------------------------------------------------------------- greedy policy
pi = np.asarray(greedy_policy(np.array([[0.0, 2.0, 1.0, 0.0], [5.0, 5.0, 1.0, 0.0], [-3, -2, -1, -4.0]])))
check(pi.tolist() == [1, 0, 2], "greedy_policy picks the argmax per row (ties → lowest index)",
      f"got {pi.tolist()}, expected [1, 0, 2]")

env = GridWorld(["...G", ".#.X", "S..."], step_reward=-0.04)
V = np.array([0.64, 0.75, 0.88, 0.0, 0.54, 0.0, 0.60, 0.0, 0.45, 0.42, 0.49, 0.28])
pi = np.asarray(greedy_policy(q_table(env, V, 0.9)))
check(pi[env.state_of(0, 2)] == 1 and pi[env.state_of(2, 0)] == 0 and pi[env.state_of(1, 2)] == 0,
      "Greedy policy w.r.t. good values heads for the goal and avoids the pit",
      f"actions at (0,2), (2,0), (1,2): {pi[env.state_of(0, 2)]}, {pi[env.state_of(2, 0)]}, {pi[env.state_of(1, 2)]} "
      "(expected RIGHT=1, UP=0, UP=0)")
