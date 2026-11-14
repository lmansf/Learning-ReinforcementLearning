import numpy as np
from rlenvs import GridWorld

expected_update = require("expected_update")
policy_evaluation = require("policy_evaluation")


def exact_v(env, policy, gamma):
    n = env.n_states
    P, r = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for a in range(4):
            for p, s2, rr, d in env.P[s][a]:
                r[s] += policy[s, a] * p * rr
                if not d:
                    P[s, s2] += policy[s, a] * p
    return np.linalg.solve(np.eye(n) - gamma * P, r)

# ---------------------------------------------------------------- one backup by hand
env = GridWorld(["S.G"], rewards={"G": 1.0})
uni = np.full((3, 4), 0.25)
V = np.array([1.0, 2.0, 0.0])
u = expected_update(env, V, 0, uni, 0.9)
check(approx(u, 0.9 * (0.75 * 1.0 + 0.25 * 2.0)), "Backup of S under the random policy",
      f"expected 0.9·(¾·1 + ¼·2) = 1.125, got {u:.5f}")
u = expected_update(env, V, 1, uni, 0.9)
exp = 0.25 * 1.0 + 0.9 * (0.25 * 1.0 + 0.5 * 2.0)
check(approx(u, exp), "Backup of the middle cell (goal transition contributes reward only)",
      f"expected ¼·1 + 0.9·(¼·V(S) + ½·V(M)) = {exp:.4f}, got {u:.5f}")
right = np.zeros((3, 4)); right[:, 1] = 1
check(approx(expected_update(env, V, 1, right, 0.9), 1.0), "A deterministic policy backs up a single action")
check(approx(expected_update(env, V, 2, uni, 0.9), 0.0), "Terminal state backs up to 0")

# ---------------------------------------------------------------- sweeping
V1, n1 = policy_evaluation(env, uni, 0.9, max_sweeps=1)
ok = approx(V1, [0.0, 0.25, 0.0])
check(ok and n1 == 1, "One in-place sweep from zeros gives [0, 0.25, 0]",
      f"got {np.round(np.asarray(V1), 4).tolist()} after {n1} sweep(s)")
V, n = policy_evaluation(env, uni, 0.9, theta=1e-10)
check(approx(V, exact_v(env, uni, 0.9), 1e-8), "Converges to the exact v_π on the corridor",
      f"got {np.round(np.asarray(V), 5).tolist()}, exact [0.43902, 0.63415, 0]")

env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
rng = np.random.default_rng(1)
pol = rng.random((env.n_states, 4)); pol /= pol.sum(axis=1, keepdims=True)
V, n = policy_evaluation(env, pol, 0.9, theta=1e-9)
ve = exact_v(env, pol, 0.9)
check(approx(V, ve, 1e-7), "Slippery 4×3, random stochastic policy: matches the linear solve",
      f"max error {np.abs(np.asarray(V) - ve).max():.2e}")
check(isinstance(n, (int, np.integer)) and 5 < n < 500, "Reports a sensible number of sweeps", f"sweeps = {n}")
_, n_loose = policy_evaluation(env, pol, 0.9, theta=1e-2)
check(n_loose < n, "A looser θ stops after fewer sweeps", f"θ=1e-2 → {n_loose} sweeps, θ=1e-9 → {n}")

V2, n2 = policy_evaluation(env, pol, 1.0, theta=1e-9)
check(approx(V2, exact_v(env, pol, 1.0), 1e-6), "Works undiscounted (γ = 1) on an episodic task",
      f"v(start) = {np.asarray(V2)[env.start]:.5f}, exact {exact_v(env, pol, 1.0)[env.start]:.5f}")
