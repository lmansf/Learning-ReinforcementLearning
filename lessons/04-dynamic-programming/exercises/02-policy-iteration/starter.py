import numpy as np
import rlhud
from rlenvs import GridWorld


# ---- helpers from earlier exercises (complete) ---------------------------------
def q_from_v(env, V, s, gamma):
    """One-step lookahead: array of q(s, a) for every action."""
    q = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        for prob, s2, r, done in env.P[s][a]:
            q[a] += prob * (r + (0.0 if done else gamma * V[s2]))
    return q


def policy_evaluation(env, policy, gamma, theta=1e-8, V=None):
    """In-place iterative policy evaluation. policy[s, a] = π(a|s). Returns V."""
    V = np.zeros(env.n_states) if V is None else np.array(V, dtype=float)
    while True:
        delta = 0.0
        for s in range(env.n_states):
            v_new = sum(policy[s, a] * prob * (r + (0.0 if done else gamma * V[s2]))
                        for a in range(env.n_actions) for prob, s2, r, done in env.P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            return V


# ---- your code ------------------------------------------------------------------
def policy_improvement(env, V, gamma, old_policy=None, tol=1e-9):
    """Greedy policy with respect to V, as a one-hot (n_states, n_actions) array.

    In each state pick argmax_a q(s, a). If old_policy is given and its action in s is
    within `tol` of the best q-value, KEEP the old action — otherwise policy iteration can
    flip forever between equally good actions and never report "stable".
    """
    new = np.zeros((env.n_states, env.n_actions))
    # TODO: for every state, compute q with q_from_v, choose the greedy action
    #       (keeping the old one if it is within tol of the best), set new[s, action] = 1
    raise NotImplementedError
    return new


def policy_iteration(env, gamma, theta=1e-8):
    """Alternate evaluation and improvement until the policy stops changing.

    Start from the uniform random policy. Each iteration:
        V      = policy_evaluation(env, policy, gamma, theta, V)   (warm-start from the last V)
        policy = policy_improvement(env, V, gamma, old_policy=policy)
    Stop when the improved policy equals the old one.
    Returns (policy, V, iterations).
    """
    policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    V = np.zeros(env.n_states)
    iterations = 0
    # TODO: loop evaluate → improve until the policy is stable
    raise NotImplementedError


if __name__ == "__main__":
    env = GridWorld(["....G",
                     ".##.X",
                     "....#",
                     "S...."], slip=0.1, step_reward=-0.04)
    gamma = 0.95
    # Show every evaluate → improve round
    policy = np.full((env.n_states, 4), 0.25)
    V = np.zeros(env.n_states)
    values_at_start = []
    for k in range(1, 20):
        V = policy_evaluation(env, policy, gamma, 1e-8, V)
        values_at_start.append(float(V[env.start]))
        new = policy_improvement(env, V, gamma, old_policy=policy)
        rlhud.grid(env.to_grid(V), title=f"Iteration {k}: v_π and its greedy improvement",
                   id=f"pi-{k}", policy=env.policy_grid(new), walls=env.walls_mask(), labels=env.labels())
        if np.array_equal(new, policy):
            break
        policy = new
    rlhud.plot(values_at_start, label="v_π(start)", title="Policy iteration: value of start per iteration",
               xlabel="iteration")

    pi_star, V_star, iters = policy_iteration(env, gamma)
    rlhud.text(f"Policy iteration converged in {iters} iterations. v*(start) = {V_star[env.start]:.4f}",
               title="Done", tone="good")
