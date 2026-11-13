import numpy as np
import rlhud
from rlenvs import CliffWalking


def q_learning_update(Q, s, a, r, s2, alpha, gamma, terminated):
    """One Q-learning update, in place:  Q(S,A) += alpha * [R + gamma * max_a' Q(S',a') - Q(S,A)].
    If terminated, the target is just R. Returns the TD error delta."""
    # TODO
    raise NotImplementedError


def q_learning(env, n_episodes, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0):
    """Off-policy TD control: behave epsilon-greedily, learn about the greedy policy.
    Returns (Q, returns) where returns[i] is the total reward of episode i."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(n_episodes):
        # TODO: loop: choose A epsilon-greedily, step, q_learning_update, s <- s'
        #       stop on terminated or truncated; append the episode's total reward to `returns`
        raise NotImplementedError
    return Q, returns


# ---------------------------------------------------------------- provided (from the SARSA exercise)
def epsilon_greedy(Q, s, epsilon, rng):
    """Action for state s from the table Q (shape n_states x n_actions).
    With prob. epsilon a uniformly random action, else a greedy one (ties broken at random)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def sarsa_update(Q, s, a, r, s2, a2, alpha, gamma, terminated):
    """One SARSA update, in place:  Q(S,A) += alpha * [R + gamma*Q(S',A') - Q(S,A)].
    If terminated, the target is just R. Returns the TD error delta."""
    target = r if terminated else r + gamma * Q[s2, a2]
    delta = target - Q[s, a]
    Q[s, a] += alpha * delta
    return delta


def sarsa(env, n_episodes, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0):
    """On-policy TD control. Returns (Q, returns) where returns[i] is the total reward of episode i."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        G = 0.0
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            G += r
            a2 = epsilon_greedy(Q, s2, epsilon, rng)    # choose A' FIRST: it is used in the target...
            sarsa_update(Q, s, a, r, s2, a2, alpha, gamma, terminated)
            if terminated or truncated:
                break
            s, a = s2, a2                                # ...and it is the action we actually take next
        returns.append(G)
    return Q, returns


def greedy_rollout(env, Q, max_steps=100):
    """Follow argmax Q from the start (no exploration). Returns (total_reward, visited_states)."""
    s, _ = env.reset()
    total, path = 0.0, [s]
    for _ in range(max_steps):
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        total += r
        path.append(s)
        if term:
            break
    return total, path


def arrows(Q):
    """Greedy action per cell for rlhud.grid, with no arrow on the cliff and the goal."""
    g = np.argmax(Q, axis=1).reshape(4, 12).tolist()
    for c in range(1, 12):
        g[3][c] = None
    return g


if __name__ == "__main__":
    env = CliffWalking()
    q_Q, q_rets = q_learning(env, 500, alpha=0.5, epsilon=0.1, seed=0)
    s_Q, s_rets = sarsa(env, 500, alpha=0.5, epsilon=0.1, seed=0)
    clip = lambda xs: [max(x, -150) for x in xs]
    rlhud.plot(clip(q_rets), label="Q-learning", title="Online return while learning (ε = 0.1)", smooth=20)
    rlhud.plot(clip(s_rets), label="SARSA", title="Online return while learning (ε = 0.1)", smooth=20)

    for name, Q, rets in (("Q-learning", q_Q, q_rets), ("SARSA", s_Q, s_rets)):
        G, path = greedy_rollout(env, Q)
        visits = np.zeros(env.n_states)
        visits[path] = 1
        rlhud.grid(env.to_grid(visits), title=f"{name} greedy path: return {G:.0f}",
                   policy=arrows(Q), labels=env.labels(),
                   cmap="heat", fmt=".0f")
        rlhud.stat(f"{name}: greedy return", G)
        rlhud.stat(f"{name}: mean online return", round(float(np.mean(rets)), 1))
    print("Q-learning's greedy path is optimal (−13) but it falls off the cliff more while exploring.")
