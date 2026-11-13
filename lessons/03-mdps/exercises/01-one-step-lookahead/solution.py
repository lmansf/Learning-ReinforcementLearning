import numpy as np
import rlhud
from rlenvs import GridWorld, ACTION_NAMES


def q_from_v(env, V, s, gamma):
    """One-step lookahead: the value of each action in state s, given state values V.

        q(s, a) = Σ_{s', r} p(s', r | s, a) · [ r + γ · V(s') ]

    env.P[s][a] is a list of (prob, next_state, reward, terminated).
    A transition that terminates has no future: do not add γ·V(s') for it.
    Returns a NumPy array of length env.n_actions.
    """
    q = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        for prob, s2, r, done in env.P[s][a]:
            q[a] += prob * (r + (0.0 if done else gamma * V[s2]))
    return q


def q_table(env, V, gamma):
    """q_from_v for every state: an (n_states, n_actions) array."""
    return np.array([q_from_v(env, V, s, gamma) for s in range(env.n_states)])


def greedy_policy(Q):
    """For each state, the action with the largest q-value (ties → lowest index)."""
    return np.argmax(np.asarray(Q), axis=1)


if __name__ == "__main__":
    env = GridWorld(["...G",
                     ".#.X",
                     "S..."], step_reward=-0.04)
    gamma = 0.9
    # The optimal values v*(s) of this world (you will compute them yourself with DP next
    # lesson). Values alone don't say what to DO — the one-step lookahead does.
    V = np.array([0.734, 0.860, 1.000, 0.0,
                  0.621, 0.000, 0.860, 0.0,
                  0.519, 0.621, 0.734, 0.621])
    Q = q_table(env, V, gamma)
    pi = greedy_policy(Q)
    rlhud.grid(env.to_grid(V), title="Given V(s) with the greedy policy",
               policy=env.policy_grid(pi), walls=env.walls_mask(), labels=env.labels())
    s = env.start
    rlhud.bars(Q[s], labels=ACTION_NAMES, title=f"q(start, a) — one-step lookahead from state {s}",
               highlight=int(pi[s]))
    rlhud.table([[ACTION_NAMES[a], round(float(Q[s, a]), 4)] for a in range(4)],
                headers=["action", "q(start, a)"], title="Lookahead at the start state")
