import numpy as np
import rlhud
from rlenvs import GridWorld


def expected_update(env, V, s, policy, gamma):
    """The Bellman expectation backup for one state:

        Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]     (no γ·V(s') after termination)

    policy[s, a] = π(a|s).
    """
    total = 0.0
    for a in range(env.n_actions):
        if policy[s, a] == 0:
            continue
        for prob, s2, r, done in env.P[s][a]:
            total += policy[s, a] * prob * (r + (0.0 if done else gamma * V[s2]))
    return total


def policy_evaluation(env, policy, gamma, theta=1e-8, max_sweeps=10_000, V=None):
    """Iterative policy evaluation, in place.

    Sweep over all states, replacing V[s] by expected_update(...) immediately, and track
    Δ = the largest |change| in the sweep. Stop as soon as Δ < theta.
    Returns (V, sweeps) where sweeps is the number of full sweeps performed.
    """
    V = np.zeros(env.n_states) if V is None else np.array(V, dtype=float)
    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        for s in range(env.n_states):
            v_new = expected_update(env, V, s, policy, gamma)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V, sweep


if __name__ == "__main__":
    env = GridWorld(["....G",
                     ".##.X",
                     "....#",
                     "S...."], slip=0.1, step_reward=-0.04)
    gamma = 0.9
    random_pi = np.full((env.n_states, 4), 0.25)

    # Watch values spread outward from the terminals, one sweep at a time
    V = np.zeros(env.n_states)
    for k in [1, 2, 3]:
        V, _ = policy_evaluation(env, random_pi, gamma, max_sweeps=1, V=V)
        rlhud.grid(env.to_grid(V), title=f"Random policy · after {k} sweep(s)",
                   walls=env.walls_mask(), labels=env.labels())

    V, sweeps = policy_evaluation(env, random_pi, gamma, theta=1e-8)
    rlhud.grid(env.to_grid(V), title=f"v_random converged ({sweeps} sweeps)",
               walls=env.walls_mask(), labels=env.labels())

    # How many sweeps does each accuracy cost?
    thetas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    rows = [[f"{t:g}", policy_evaluation(env, random_pi, gamma, theta=t)[1]] for t in thetas]
    rlhud.table(rows, headers=["θ", "sweeps"], title="Sweeps needed vs tolerance")
    rlhud.stat("v_random(start)", float(V[env.start]))
