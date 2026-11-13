import numpy as np
from rlenvs import GridWorld

policy_model = require("policy_model")
solve_v = require("solve_v")

RIGHT = 1

# ---------------------------------------------------------------- corridor S . G, always RIGHT
env = GridWorld(["S.G"], step_reward=0.0, rewards={"G": 1.0})
always_right = np.zeros((3, 4)); always_right[:, RIGHT] = 1
P, r = map(np.asarray, policy_model(env, always_right))
check(P.shape == (3, 3) and r.shape == (3,), "policy_model returns P_π (n×n) and r_π (n)",
      f"got {P.shape} and {r.shape}")
ok = approx(P, [[0, 1, 0], [0, 0, 0], [0, 0, 0]]) and approx(r, [0, 1, 0])
check(ok, "Corridor, always RIGHT: P_π and r_π",
      "" if ok else f"expected P_π=[[0,1,0],[0,0,0],[0,0,0]], r_π=[0,1,0]; got P_π={P.tolist()}, r_π={r.tolist()}. "
                    "A terminating transition gives reward but no successor (its P_π entry stays 0)")
v = np.asarray(solve_v(env, always_right, 0.9), dtype=float)
check(approx(v, [0.9, 1.0, 0.0]), "v_π = [0.9, 1.0, 0.0] for γ = 0.9", f"got {np.round(v, 5).tolist()}")

# ---------------------------------------------------------------- uniform random policy, solved by hand
# From S: RIGHT (1/4) → middle, the other three bump → stay.  From middle: RIGHT → goal (+1), LEFT → S,
# UP/DOWN stay.  v0 = γ(3/4 v0 + 1/4 v1);  v1 = 1/4 + γ(1/4 v0 + 1/2 v1)
uni = np.full((3, 4), 0.25)
g = 0.9
A = np.array([[1 - 0.75 * g, -0.25 * g], [-0.25 * g, 1 - 0.5 * g]])
v_hand = np.linalg.solve(A, [0.0, 0.25])
v = np.asarray(solve_v(env, uni, g), dtype=float)
ok = approx(v[:2], v_hand, 1e-9) and abs(v[2]) < 1e-12
check(ok, "Uniform random policy on the corridor matches the hand-derived system",
      f"expected {np.round(v_hand, 5).tolist()} + [0], got {np.round(v, 5).tolist()}")

P, r = map(np.asarray, policy_model(env, uni))
check(approx(P.sum(axis=1), [1.0, 0.75, 0.0]),
      "Rows of P_π sum to the probability of NOT terminating",
      f"row sums {np.round(P.sum(axis=1), 4).tolist()} (expected [1, 0.75, 0])")

# ---------------------------------------------------------------- slippery 4×3 world
env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
rng = np.random.default_rng(0)
pol = rng.random((env.n_states, 4)); pol /= pol.sum(axis=1, keepdims=True)
v = np.asarray(solve_v(env, pol, 0.95), dtype=float)

# Bellman expectation residual: v(s) − Σ_a π Σ p [r + γ v(s')] must be 0 everywhere
resid = 0.0
for s in range(env.n_states):
    rhs = sum(pol[s, a] * prob * (rr + (0 if d else 0.95 * v[s2]))
              for a in range(4) for prob, s2, rr, d in env.P[s][a])
    resid = max(resid, abs(v[s] - rhs))
check(resid < 1e-9, "Your v_π satisfies the Bellman expectation equation in every state",
      f"largest residual {resid:.2e}")
check(approx(v[list(env.terminal_states)], 0) and approx(v[list(env.wall_states)], 0),
      "Terminal and wall states have value 0")

# compare with many sweeps of plain Bellman backups (the next lesson's method)
vi = np.zeros(env.n_states)
for _ in range(2000):
    vi = np.array([sum(pol[s, a] * prob * (rr + (0 if d else 0.95 * vi[s2]))
                       for a in range(4) for prob, s2, rr, d in env.P[s][a]) for s in range(env.n_states)])
check(approx(v, vi, 1e-6), "Linear solve agrees with repeated Bellman backups",
      f"max difference {np.abs(v - vi).max():.2e}")

v1 = np.asarray(solve_v(env, pol, 1.0), dtype=float)
check(np.all(np.isfinite(v1)) and v1[env.start] < 0,
      "Works with γ = 1 too (every policy here eventually terminates)",
      f"v(start) = {v1[env.start]:.4f}")
