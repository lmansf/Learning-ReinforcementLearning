import numpy as np
import rlhud
from rlenvs import Blackjack


def epsilon_greedy(q, epsilon, rng):
    """Pick an action from the action-value array `q` (one entry per action).

    With probability epsilon: a uniformly random action.
    Otherwise: the greedy action argmax(q) (ties may be broken any way you like).
    `rng` is a numpy Generator (np.random.default_rng).
    """
    # TODO
    raise NotImplementedError


def mc_control(env, n_episodes, epsilon=0.1, gamma=1.0, seed=None):
    """On-policy first-visit Monte Carlo control with an epsilon-soft (epsilon-greedy) policy.

    Returns (Q, returns):
      Q       : dict {state: np.array of shape (env.n_actions,)}
      returns : list with the total (undiscounted) reward of every training episode
    """
    rng = np.random.default_rng(seed)
    n_actions = env.n_actions
    Q = {}
    N = {}  # visit counts: {state: np.zeros(n_actions)}
    returns = []
    for i in range(n_episodes):
        # TODO 1: generate an episode acting epsilon-greedily on Q
        #         (create Q[obs] / N[obs] the first time you see obs; seed only the first reset)
        # TODO 2: append the episode's total reward to `returns`
        # TODO 3: walk the episode backwards (G = r + gamma * G) and, for the FIRST visit of each
        #         (s, a), update Q[s][a] towards G with an incremental mean
        raise NotImplementedError
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
