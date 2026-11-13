import random

import numpy as np
import rlhud


def nstep_target(rewards, gamma, bootstrap):
    """G = r1 + gamma*r2 + ... + gamma^(k-1)*rk + gamma^k * bootstrap,  where k = len(rewards).
    Pass bootstrap=0.0 when the episode ended inside the window."""
    G = 0.0
    for i, r in enumerate(rewards):
        G += (gamma ** i) * r
    return G + (gamma ** len(rewards)) * bootstrap


def nstep_td(env, policy, n_episodes, n, alpha, gamma=1.0, init=0.0, V=None):
    """n-step TD prediction of v_pi (Sutton & Barto, section 7.1).

    Returns V (numpy array, length env.n_states). If V is given, it is updated in place.
    The update for the state visited at time tau happens once R_{tau+n} (or the episode end)
    is known:  V(S_tau) += alpha * (G_{tau:tau+n} - V(S_tau)).
    An episode ends on terminated or truncated (both treated as the end: no bootstrap past it).
    """
    if V is None:
        V = np.full(env.n_states, float(init))
    for _ in range(n_episodes):
        s, _ = env.reset()
        states, rewards = [s], [0.0]   # rewards[i] holds R_i, so rewards[0] is a dummy
        T = float("inf")
        t = 0
        while True:
            if t < T:
                s, r, terminated, truncated, _ = env.step(policy(states[t]))
                states.append(s)
                rewards.append(r)
                if terminated or truncated:
                    T = t + 1
            tau = t - n + 1               # the time whose estimate is updated now
            if tau >= 0:
                end = min(tau + n, T)
                boot = V[states[tau + n]] if tau + n < T else 0.0
                G = nstep_target(rewards[tau + 1:end + 1], gamma, boot)
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
    return V


# ---------------------------------------------------------------- provided
class RandomWalk:
    """States 1..n are non-terminal, 0 and n+1 terminal; start in the middle.
    Action 0 = left, 1 = right. Exit right -> right_reward, exit left -> left_reward."""

    def __init__(self, n=19, left_reward=-1.0, right_reward=1.0):
        self.n = n
        self.n_states = n + 2
        self.n_actions = 2
        self.left_reward, self.right_reward = left_reward, right_reward

    def reset(self, seed=None):
        self.s = (self.n + 1) // 2
        return self.s, {}

    def step(self, a):
        self.s += 1 if a == 1 else -1
        if self.s == 0:
            return 0, self.left_reward, True, False, {}
        if self.s == self.n + 1:
            return self.s, self.right_reward, True, False, {}
        return self.s, 0.0, False, False, {}

    def true_values(self):
        k = np.arange(1, self.n + 1)
        return self.left_reward + (self.right_reward - self.left_reward) * k / (self.n + 1)


def coin_flip(s):
    return random.randrange(2)


def avg_rms(n, alpha, runs=50, episodes=10, seed=0):
    """Mean RMS error over the first `episodes` episodes and `runs` runs (the S&B Figure 7.2 metric)."""
    env = RandomWalk()
    truth = env.true_values()
    random.seed(seed)
    total = 0.0
    for _ in range(runs):
        V = np.zeros(env.n_states)
        for _ in range(episodes):
            nstep_td(env, coin_flip, 1, n, alpha, V=V)
            total += np.sqrt(np.mean((V[1:-1] - truth) ** 2))
    return total / (runs * episodes)


if __name__ == "__main__":
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    ns = [1, 2, 4, 8, 16, 32]
    rows = []
    for i, n in enumerate(ns):
        errs = [avg_rms(n, a) for a in alphas]
        rlhud.plot(errs, x=alphas, label=f"n = {n}", title="19-state random walk: avg RMS over first 10 episodes",
                   xlabel="α", ylabel="RMS error")
        best = int(np.argmin(errs))
        rows.append([n, alphas[best], round(errs[best], 3)])
        rlhud.progress(i, len(ns), "n values")
    rlhud.table(rows, headers=["n", "best α", "RMS at best α"], title="Best step size per n")
    print("Intermediate n (≈ 4–8) beats both one-step TD (n = 1) and near-Monte-Carlo (n = 32).")
