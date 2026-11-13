import numpy as np
import rlhud
from rlenvs import GridWorld


def q_from_v(env, V, s, gamma):
    """One-step lookahead (from the MDP lesson): array of q(s, a) for every action."""
    q = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        for prob, s2, r, done in env.P[s][a]:
            q[a] += prob * (r + (0.0 if done else gamma * V[s2]))
    return q


def value_iteration(env, gamma, theta=1e-8, max_sweeps=10_000):
    """Value iteration: sweep  V(s) ← max_a q(s, a)  over all states, in place.

    Track delta (largest change in the sweep) and stop when delta < theta, or after
    max_sweeps sweeps. Then extract the greedy policy as a one-hot array.

    Returns (V, policy, deltas): deltas is the list of per-sweep deltas
    (so len(deltas) is the number of sweeps).
    """
    V = np.zeros(env.n_states)
    deltas = []
    # TODO: in-place sweeps of V[s] = max_a q(s, a), recording each sweep's delta;
    #       stop when delta < theta (or after max_sweeps)
    raise NotImplementedError
    policy = np.zeros((env.n_states, env.n_actions))
    # TODO: one-hot greedy policy with respect to the final V
    return V, policy, deltas


if __name__ == "__main__":
    env = GridWorld(["......#..G",
                     ".####.#.#X",
                     ".#....#.#.",
                     ".#.##.....",
                     "S#....XX.."], slip=0.2, step_reward=-0.04)
    gamma = 0.99
    V, policy, deltas = value_iteration(env, gamma, theta=1e-6)
    rlhud.grid(env.to_grid(V), title=f"v* and π* · slip 0.2 · {len(deltas)} sweeps",
               policy=env.policy_grid(policy), walls=env.walls_mask(), labels=env.labels())
    rlhud.plot(np.log10(np.maximum(deltas, 1e-12)), label="log₁₀ Δ", title="Value iteration convergence",
               xlabel="sweep")

    # Truncated value iteration: how good is the greedy policy after only k sweeps?
    for k in [3, 10, 25]:
        Vk, pk, _ = value_iteration(env, gamma, max_sweeps=k)
        rlhud.grid(env.to_grid(Vk), title=f"After {k} sweeps", id=f"vi-{k}",
                   policy=env.policy_grid(pk), walls=env.walls_mask(), labels=env.labels())

    # Same world without slipping: every value is higher — uncertainty itself costs reward
    env0 = GridWorld(env.layout, slip=0.0, step_reward=-0.04)
    V0, p0, d0 = value_iteration(env0, gamma, theta=1e-6)
    rlhud.grid(env0.to_grid(V0), title="Deterministic version (slip 0)", policy=env0.policy_grid(p0),
               walls=env0.walls_mask(), labels=env0.labels())
    rlhud.stat("v*(start), slip 0.2", float(V[env.start]))
    rlhud.stat("v*(start), slip 0", float(V0[env0.start]))
