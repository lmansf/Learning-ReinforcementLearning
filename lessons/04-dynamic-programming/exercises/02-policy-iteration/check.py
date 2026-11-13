import numpy as np
from rlenvs import GridWorld

policy_improvement = require("policy_improvement")
policy_iteration = require("policy_iteration")


def value_iteration(env, gamma, sweeps=2000):
    V = np.zeros(env.n_states)
    for _ in range(sweeps):
        V = np.array([max(sum(p * (r + (0 if d else gamma * V[s2])) for p, s2, r, d in env.P[s][a])
                          for a in range(4)) for s in range(env.n_states)])
    return V

# ---------------------------------------------------------------- improvement step by hand
env = GridWorld(["S.G"], step_reward=-0.1)
V = np.array([0.0, 0.5, 0.0])
pol = np.asarray(policy_improvement(env, V, 0.9), dtype=float)
check(pol.shape == (3, 4), "policy_improvement returns an (n_states, n_actions) array", f"got {pol.shape}")
check(approx(pol.sum(axis=1), 1.0) and set(np.unique(pol)) <= {0.0, 1.0},
      "Each row is one-hot (a deterministic policy)", f"rows: {pol.tolist()}")
check(int(np.argmax(pol[0])) == 1 and int(np.argmax(pol[1])) == 1,
      "Greedy w.r.t. V picks RIGHT in both corridor cells",
      f"actions {np.argmax(pol, axis=1).tolist()} — q(S,RIGHT) = −0.1 + 0.9·0.5 beats staying (−0.1 + 0)")

# ties: in state S with V = 0 everywhere, every action has q = −0.1
V0 = np.zeros(3)
old = np.zeros((3, 4)); old[:, 2] = 1  # old policy says DOWN
pol = np.asarray(policy_improvement(env, V0, 0.9, old_policy=old), dtype=float)
ok = int(np.argmax(pol[0])) == 2
check(ok, "On a tie, the old action is kept",
      "" if ok else f"chose {int(np.argmax(pol[0]))} in S although DOWN (2) is equally good — keep old actions within tol of the max")
old[1] = 0; old[1, 3] = 1  # old policy goes LEFT in the middle, which is strictly worse than RIGHT
pol = np.asarray(policy_improvement(env, V0, 0.9, old_policy=old), dtype=float)
check(int(np.argmax(pol[1])) == 1, "…but a strictly better action replaces the old one",
      f"middle cell chose {int(np.argmax(pol[1]))}, expected RIGHT (1): entering G pays +1")

# ---------------------------------------------------------------- full policy iteration
env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
pi, V, iters = policy_iteration(env, 0.9)
pi, V = np.asarray(pi, dtype=float), np.asarray(V, dtype=float)
v_star = value_iteration(env, 0.9)
check(approx(V, v_star, 1e-6), "Slippery 4×3 (γ = 0.9): policy iteration reaches v*",
      f"max error {np.abs(V - v_star).max():.2e}")
check(1 < iters <= 10, "Converges in a handful of iterations", f"took {iters}")
greedy_again = np.asarray(policy_improvement(env, V, 0.9, old_policy=pi))
check(np.array_equal(greedy_again, pi), "The returned policy is stable (greedy w.r.t. its own value)")
UP, RIGHT, LEFT = 0, 1, 3
a = np.argmax(pi, axis=1)
check(a[env.state_of(0, 2)] == RIGHT and a[env.state_of(1, 2)] == UP,
      "Optimal actions near the goal: RIGHT into G, UP from the cell beside the pit",
      f"got {a[env.state_of(0, 2)]} and {a[env.state_of(1, 2)]}")

env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
pi1, V1, it1 = policy_iteration(env, 1.0)
check(approx(np.asarray(V1)[env.start], 0.7453, 1e-3),
      "Undiscounted (γ = 1): v*(start) ≈ 0.745 — the classic Russell & Norvig 4×3 world",
      f"got {np.asarray(V1)[env.start]:.4f}")
a1 = np.argmax(np.asarray(pi1), axis=1)
a09 = np.argmax(pi, axis=1)
b = env.state_of(2, 2)
check(a1[b] == LEFT and a09[b] == UP,
      "Bottom row, below the pit: γ = 1 takes the long safe way (LEFT), γ = 0.9 the short risky way (UP)",
      f"action at (2,2): γ=1 → {a1[b]}, γ=0.9 → {a09[b]} (0=UP 1=RIGHT 2=DOWN 3=LEFT). "
      "Discounting makes the long detour less attractive than the small risk of slipping into X")
