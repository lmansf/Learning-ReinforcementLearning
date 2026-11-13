import itertools
import numpy as np
import rlhud
from rlenvs import GridWorld


# ---- helpers from the previous two exercises (complete) ----------------------
def q_from_v(env, V, s, gamma):
    q = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        for prob, s2, r, done in env.P[s][a]:
            q[a] += prob * (r + (0.0 if done else gamma * V[s2]))
    return q


def solve_v(env, policy, gamma):
    """Exact v_π for policy[s, a] = π(a|s) via (I − γ P_π) v = r_π."""
    n = env.n_states
    P_pi, r_pi = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for a in range(env.n_actions):
            if policy[s, a] == 0:
                continue
            for prob, s2, r, done in env.P[s][a]:
                r_pi[s] += policy[s, a] * prob * r
                if not done:
                    P_pi[s, s2] += policy[s, a] * prob
    return np.linalg.solve(np.eye(n) - gamma * P_pi, r_pi)


def decision_states(env):
    """States where the action matters (not terminal, not a wall)."""
    return [s for s in range(env.n_states) if s not in env.terminal_states and s not in env.wall_states]


# ---- your code ------------------------------------------------------------------
def brute_force_optimal(env, gamma):
    """Evaluate EVERY deterministic policy exactly and keep the best one.

    A deterministic policy assigns one action to each decision state (states that are
    terminal or walls can use action 0). Compare policies by the SUM of their state
    values; the optimal policy is at least as good in every state, so it also wins the sum.

    Returns (actions, V): actions is an int array of length n_states, V its exact value.
    """
    states = decision_states(env)
    best_actions, best_v, best_score = None, None, -np.inf
    # TODO: loop over itertools.product(range(env.n_actions), repeat=len(states)),
    #       build the action array and its one-hot policy, evaluate it with solve_v,
    #       and keep the policy with the largest v.sum()
    raise NotImplementedError
    return best_actions, best_v


def is_optimal(env, V, gamma, tol=1e-8):
    """Bellman optimality test: V(s) == max_a q(s, a) in every state (within tol)."""
    # TODO: compare V[s] with the best one-step lookahead value in every state
    raise NotImplementedError


if __name__ == "__main__":
    env = GridWorld(["..G",
                     ".#X",
                     "S.."], slip=0.2, step_reward=-0.04)
    gamma = 0.9
    n_pol = env.n_actions ** len(decision_states(env))
    actions, v_star = brute_force_optimal(env, gamma)
    rlhud.grid(env.to_grid(v_star), title=f"Best of all {n_pol} deterministic policies",
               policy=env.policy_grid(actions), walls=env.walls_mask(), labels=env.labels())

    v_rand = solve_v(env, np.full((env.n_states, 4), 0.25), gamma)
    rlhud.table([["optimal (brute force)", round(float(v_star[env.start]), 4), is_optimal(env, v_star, gamma)],
                 ["uniform random", round(float(v_rand[env.start]), 4), is_optimal(env, v_rand, gamma)]],
                headers=["policy", "v(start)", "satisfies Bellman optimality?"], title="Optimality test")

    # Distribution of v(start) across every deterministic policy
    states = decision_states(env)
    starts = []
    for choice in itertools.product(range(4), repeat=len(states)):
        a = np.zeros(env.n_states, dtype=int); a[states] = choice
        starts.append(solve_v(env, np.eye(4)[a], gamma)[env.start])
    hist, edges = np.histogram(starts, bins=12)
    rlhud.bars(hist, labels=[f"{e:.2f}" for e in edges[:-1]], title="v(start) over all deterministic policies",
               highlight=len(hist) - 1)
