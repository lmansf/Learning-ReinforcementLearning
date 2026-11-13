import numpy as np
import rlhud
from rlenvs import Blackjack


def epsilon_greedy(q, epsilon, rng):
    """Pick an action from the action-value array `q` (one entry per action).

    With probability epsilon: a uniformly random action.
    Otherwise: the greedy action argmax(q) (ties may be broken any way you like).
    `rng` is a numpy Generator (np.random.default_rng).
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


def mc_control(env, n_episodes, epsilon=0.1, gamma=1.0, seed=None):
    """On-policy first-visit Monte Carlo control with an epsilon-soft (epsilon-greedy) policy.

    Returns (Q, returns):
      Q       : dict {state: np.array of shape (env.n_actions,)}
      returns : list with the total (undiscounted) reward of every training episode
    """
    rng = np.random.default_rng(seed)
    n_actions = env.n_actions
    Q = {}
    N = {}  # visit counts per (state) -> array per action, for incremental averaging
    returns = []
    for i in range(n_episodes):
        # 1. generate an episode with the CURRENT epsilon-greedy policy
        episode = []
        obs, _ = env.reset(seed=seed if i == 0 else None)
        while True:
            if obs not in Q:
                Q[obs] = np.zeros(n_actions)
                N[obs] = np.zeros(n_actions)
            a = epsilon_greedy(Q[obs], epsilon, rng)
            next_obs, r, terminated, truncated, _ = env.step(a)
            episode.append((obs, a, r))
            if terminated or truncated:
                break
            obs = next_obs
        returns.append(sum(r for _, _, r in episode))

        # 2. first-visit MC update of Q towards the observed returns
        first = {}
        for t, (s, a, _) in enumerate(episode):
            first.setdefault((s, a), t)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            if first[(s, a)] == t:
                N[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / N[s][a]   # incremental mean
        # 3. policy improvement is implicit: the next episode acts epsilon-greedily on the new Q
    return Q, returns


def evaluate(env, policy, n_episodes, seed=0):
    """Mean return of a deterministic policy(obs) -> action over n_episodes (no learning)."""
    total = 0.0
    obs, _ = env.reset(seed=seed)
    for i in range(n_episodes):
        if i:
            obs, _ = env.reset()
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            total += r
            if term or trunc:
                break
    return total / n_episodes


if __name__ == "__main__":
    env = Blackjack(seed=1)
    N_EPISODES = 200_000
    Q, rets = mc_control(env, N_EPISODES, epsilon=0.1, seed=1)

    # Mean training return per block of 2000 episodes
    block = 2000
    curve = [float(np.mean(rets[i:i + block])) for i in range(0, len(rets), block)]
    rlhud.plot(curve, label="ε-greedy MC (training)", title="Mean return per 2000 episodes",
               xlabel="block", ylabel="mean return")

    greedy = lambda s: int(np.argmax(Q[s])) if s in Q else 0
    naive = lambda s: 0 if s[0] >= 20 else 1
    g_score = evaluate(Blackjack(), greedy, 50_000, seed=123)
    n_score = evaluate(Blackjack(), naive, 50_000, seed=123)
    rlhud.table([["stick on 20+", round(n_score, 4)], ["learned greedy policy", round(g_score, 4)]],
                headers=["policy", "mean return (50k hands)"], title="Head to head")
    rlhud.stat("Learned policy mean return", round(g_score, 4), good=g_score > n_score)

    # Policy maps in S&B Figure 5.2 style: 1 = HIT, 0 = STICK
    sums = list(range(21, 10, -1))
    dealer = list(range(1, 11))
    for usable in (True, False):
        pol = [[greedy((p, d, usable)) for d in dealer] for p in sums]
        labels = [["H" if v else "S" for v in row] for row in pol]
        rlhud.grid(pol, title=f"Greedy policy, {'usable' if usable else 'no usable'} ace "
                              f"(rows 21→11, cols dealer A→10; H = hit)", labels=labels, cmap="heat")
