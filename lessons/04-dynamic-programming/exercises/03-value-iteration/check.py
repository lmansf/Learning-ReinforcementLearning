import numpy as np
from rlenvs import GridWorld

value_iteration = require("value_iteration")


def reference_v_star(env, gamma, sweeps=3000):
    V = np.zeros(env.n_states)
    for _ in range(sweeps):
        V = np.array([max(sum(p * (r + (0 if d else gamma * V[s2])) for p, s2, r, d in env.P[s][a])
                          for a in range(4)) for s in range(env.n_states)])
    return V


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

# ---------------------------------------------------------------- corridor by hand
env = GridWorld(["S.G"], step_reward=-0.1)
V1, pol1, d1 = value_iteration(env, 0.9, max_sweeps=1)
V1 = np.asarray(V1, dtype=float)
ok = approx(V1, [-0.1, 1.0, 0.0])
check(ok, "One sweep on S . G (step −0.1, goal +1): V = [−0.1, 1, 0]",
      f"got {np.round(V1, 4).tolist()}" + ("" if ok else ". Each state takes the MAX over actions of r + γ·V(s')"))
check(len(d1) == 1, "max_sweeps=1 performs exactly one sweep", f"len(deltas) = {len(d1)}")
V, pol, d = value_iteration(env, 0.9)
check(approx(V, [0.8, 1.0, 0.0], 1e-7), "Converged corridor values: v* = [0.8, 1, 0]",
      f"got {np.round(np.asarray(V), 5).tolist()}")
pol = np.asarray(pol, dtype=float)
check(pol.shape == (3, 4) and approx(pol.sum(axis=1), 1.0), "Returns a one-hot (n_states, n_actions) policy",
      f"shape {pol.shape}")
check(int(np.argmax(pol[0])) == 1 and int(np.argmax(pol[1])) == 1, "Greedy policy goes RIGHT")

# ---------------------------------------------------------------- slippery 4×3
env = GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)
V, pol, deltas = value_iteration(env, 0.9, theta=1e-10)
V, pol = np.asarray(V, dtype=float), np.asarray(pol, dtype=float)
ref = reference_v_star(env, 0.9)
check(approx(V, ref, 1e-8), "Slippery 4×3 (γ = 0.9): V equals v*", f"max error {np.abs(V - ref).max():.2e}")
check(deltas[-1] < 1e-10 and (len(deltas) < 2 or deltas[-2] >= 1e-10),
      "Stops on the first sweep with Δ < θ", f"last deltas {deltas[-2:]}")
check(approx(exact_v(env, pol, 0.9), ref, 1e-6), "The extracted greedy policy actually achieves v*",
      "evaluate your returned policy exactly: it should be worth v* in every state")

# ---------------------------------------------------------------- a bigger slippery maze
env = GridWorld(["......#..G",
                 ".####.#.#X",
                 ".#....#.#.",
                 ".#.##.....",
                 "S#....XX.."], slip=0.2, step_reward=-0.04)
V, pol, deltas = value_iteration(env, 0.99, theta=1e-9)
V, pol = np.asarray(V, dtype=float), np.asarray(pol, dtype=float)
ref = reference_v_star(env, 0.99, sweeps=1500)
check(approx(V, ref, 1e-6), "10×5 slippery maze (γ = 0.99): V equals v*", f"max error {np.abs(V - ref).max():.2e}")
resid = max(abs(V[s] - max(sum(p * (r + (0 if dd else 0.99 * V[s2])) for p, s2, r, dd in env.P[s][a])
                         for a in range(4))) for s in range(env.n_states))
check(resid < 1e-7, "V satisfies the Bellman optimality equation", f"max residual {resid:.2e}")
check(10 < len(deltas) < 500, "Converges in a reasonable number of sweeps", f"{len(deltas)} sweeps")
check(deltas[-1] < deltas[0] / 1000, "Δ shrinks by orders of magnitude", f"first Δ {deltas[0]:.3g}, last {deltas[-1]:.3g}")
