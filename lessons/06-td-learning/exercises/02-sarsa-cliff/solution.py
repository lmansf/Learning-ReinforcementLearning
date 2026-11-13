import numpy as np
import rlhud
from rlenvs import CliffWalking


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


# ---------------------------------------------------------------- provided
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
    live = rlhud.chart("SARSA on the cliff: return per episode", smooth=20, ylabel="return")
    Q, rets = sarsa(env, 500, alpha=0.5, epsilon=0.1, seed=0)
    for G in rets:
        live.add("SARSA (ε = 0.1)", max(G, -150))
    live.flush()

    G, path = greedy_rollout(env, Q)
    visits = np.zeros(env.n_states)
    visits[path] = 1
    rlhud.grid(env.to_grid(Q.max(axis=1)), title="max_a Q(s, a) and greedy policy",
               policy=arrows(Q), labels=env.labels())
    rlhud.grid(env.to_grid(visits), title=f"Greedy path ({len(path) - 1} steps, return {G:.0f})",
               labels=env.labels(), cmap="heat", fmt=".0f")
    rlhud.stat("Greedy path return", G)
    rlhud.stat("Mean online return (last 100)", round(float(np.mean(rets[-100:])), 1))
    print("SARSA learns the SAFE path: it knows its own ε-greedy steps sometimes slip into the cliff.")
