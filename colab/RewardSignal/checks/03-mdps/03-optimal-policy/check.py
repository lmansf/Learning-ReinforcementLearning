import itertools
import numpy as np
from rlenvs import GridWorld

brute_force_optimal = require("brute_force_optimal")
is_optimal = require("is_optimal")
solve_v = require("solve_v")
decision_states = require("decision_states")


def value_iteration(env, gamma, sweeps=3000):
    V = np.zeros(env.n_states)
    for _ in range(sweeps):
        V = np.array([max(sum(p * (r + (0 if d else gamma * V[s2])) for p, s2, r, d in env.P[s][a])
                          for a in range(4)) for s in range(env.n_states)])
    return V

# ---------------------------------------------------------------- the optimality test
env = GridWorld(["S.G"], step_reward=-0.1)
v_star = np.array([-0.1 + 0.9 * 1.0, 1.0, 0.0])  # v*(S) = −0.1 + γ·v*(middle)
check(bool(is_optimal(env, v_star, 0.9)),
      "is_optimal accepts the true v* of the corridor S . G", f"v* = {v_star.tolist()}")
v_bad = np.array([0.5, 1.0, 0.0])
check(not is_optimal(env, v_bad, 0.9), "is_optimal rejects values that break max-backup consistency",
      "V(S)=0.5 but max_a q(S, a) = 0.8, so this V is not optimal")
right = np.zeros((3, 4)); right[:, 1] = 1
left = np.zeros((3, 4)); left[:, 3] = 1
check(is_optimal(env, solve_v(env, right, 0.9), 0.9) and not is_optimal(env, solve_v(env, left, 0.9), 0.9),
      "The always-RIGHT policy's value is optimal; always-LEFT's is not")

# ---------------------------------------------------------------- brute force on the corridor
acts, v = brute_force_optimal(env, 0.9)
acts, v = np.asarray(acts), np.asarray(v, dtype=float)
check(acts.shape == (3,) and v.shape == (3,), "brute_force_optimal returns (actions, V) per state",
      f"got shapes {acts.shape}, {v.shape}")
check(acts[0] == 1 and acts[1] == 1, "Corridor: the best policy goes RIGHT in both decision states",
      f"got actions {acts.tolist()}")
check(approx(v, v_star), "…and its value is v*", f"got {np.round(v, 4).tolist()}")

# ---------------------------------------------------------------- slippery 3×3: 4⁶ = 4096 policies
env = GridWorld(["..G", ".#X", "S.."], slip=0.2, step_reward=-0.04)
gamma = 0.9
acts, v = brute_force_optimal(env, gamma)
acts, v = np.asarray(acts), np.asarray(v, dtype=float)
v_vi = value_iteration(env, gamma)
check(approx(v, v_vi, 1e-6), "Slippery 3×3 world: brute-force best value equals v*",
      f"max difference to v* {np.abs(v - v_vi).max():.2e}")
check(approx(v, solve_v(env, np.eye(4)[acts], gamma), 1e-9), "Returned V really is the value of the returned actions")
check(is_optimal(env, v, gamma), "The best policy satisfies the Bellman optimality equation")

rng = np.random.default_rng(0)
states = decision_states(env)
dominated = True
for _ in range(200):
    a = np.zeros(env.n_states, dtype=int)
    a[states] = rng.integers(0, 4, size=len(states))
    if np.any(solve_v(env, np.eye(4)[a], gamma) > v + 1e-9):
        dominated = False
        break
check(dominated, "v* ≥ v_π in EVERY state for 200 random deterministic policies (an optimal policy dominates)")
check(acts[env.state_of(2, 2)] == 3, "Next to the pit the optimal policy steps LEFT, away from X",
      f"action at the bottom-right corner: {acts[env.state_of(2, 2)]} (0=UP 1=RIGHT 2=DOWN 3=LEFT)")
