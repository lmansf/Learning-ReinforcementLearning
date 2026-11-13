import random

import numpy as np
import rlhud
from rlenvs import Blackjack


def is_estimates(episodes, pi, b, gamma=1.0):
    """Off-policy Monte Carlo estimates of v_pi(start state) from behaviour-policy episodes.

    episodes : list of episodes, each a list of (state, action, reward), all from the SAME start state
    pi(a, s) : probability that the TARGET policy takes action a in state s
    b(a, s)  : probability that the BEHAVIOUR policy (which generated the data) takes a in s

    Returns (ordinary, weighted): two lists of the same length as `episodes`; element k is the
    estimate after the first k+1 episodes.
        ordinary : sum(rho_i * G_i) / (number of episodes)
        weighted : sum(rho_i * G_i) / sum(rho_i)      (0.0 while sum(rho_i) == 0)
    """
    ordinary, weighted = [], []
    sum_wg, sum_w = 0.0, 0.0
    for k, episode in enumerate(episodes, start=1):
        rho, G = 1.0, 0.0
        for t, (s, a, r) in enumerate(episode):
            rho *= pi(a, s) / b(a, s)       # importance-sampling ratio of the whole trajectory
            G += (gamma ** t) * r           # return from the start state
        sum_wg += rho * G
        sum_w += rho
        ordinary.append(sum_wg / k)
        weighted.append(sum_wg / sum_w if sum_w > 0 else 0.0)
    return ordinary, weighted


# ---------------------------------------------------------------- provided
def play_from_soft13_vs_2(env, policy):
    """One Blackjack episode starting in the S&B Example 5.4 state:
    player holds A + 2 (= soft 13, usable ace), dealer shows a 2.
    policy(s) -> action. Returns the list of (state, action, reward)."""
    env.reset()
    env.player = [1, 2]
    env.dealer = [2, env._card()]
    obs, episode = env._obs(), []
    while True:
        a = policy(obs)
        nxt, r, term, trunc, _ = env.step(a)
        episode.append((obs, a, r))
        if term or trunc:
            return episode
        obs = nxt


def target_pi(a, s):
    """Target policy: stick on 20 or 21, otherwise hit (deterministic)."""
    return 1.0 if a == (0 if s[0] >= 20 else 1) else 0.0


def behaviour_b(a, s):
    """Behaviour policy: hit or stick with equal probability."""
    return 0.5


TRUE_VALUE = -0.27726   # v_pi of that state (Sutton & Barto, Example 5.4)


if __name__ == "__main__":
    random.seed(0)
    env = Blackjack(seed=0)
    random_policy = lambda s: random.randrange(2)
    RUNS, EPISODES = 40, 2000
    sq_ord = np.zeros(EPISODES)
    sq_w = np.zeros(EPISODES)
    for run in range(RUNS):
        eps = [play_from_soft13_vs_2(env, random_policy) for _ in range(EPISODES)]
        o, w = is_estimates(eps, target_pi, behaviour_b)
        sq_ord += (np.array(o) - TRUE_VALUE) ** 2
        sq_w += (np.array(w) - TRUE_VALUE) ** 2
        rlhud.progress(run, RUNS, "Runs")
    idx = np.unique(np.logspace(0, np.log10(EPISODES), 60).astype(int)) - 1
    rlhud.plot((sq_ord / RUNS)[idx], x=idx + 1, label="ordinary IS", title="Mean squared error vs episodes (cf. S&B Fig 5.3)",
               xlabel="episodes", ylabel="MSE")
    rlhud.plot((sq_w / RUNS)[idx], x=idx + 1, label="weighted IS", title="Mean squared error vs episodes (cf. S&B Fig 5.3)")
    rlhud.stat("Final MSE, ordinary", round(float(sq_ord[-1] / RUNS), 4))
    rlhud.stat("Final MSE, weighted", round(float(sq_w[-1] / RUNS), 4), good=True)
    print("Weighted IS is biased early on but has far lower variance — it wins on almost every run.")
