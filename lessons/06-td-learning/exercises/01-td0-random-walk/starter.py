import random

import numpy as np
import rlhud


def td_error(r, v_s, v_next, gamma, terminated):
    """delta = R + gamma * V(S') - V(S).  A terminal S' has value 0 by definition."""
    # TODO
    raise NotImplementedError


def td0(env, policy, n_episodes, alpha, gamma=1.0, init=0.0, V=None):
    """Tabular TD(0) prediction of v_pi.

    Returns V, a numpy array of length env.n_states. If V is None, start every entry at `init`;
    otherwise keep updating the given array in place (handy to continue training).
    After EVERY step:  V(S) <- V(S) + alpha * delta
    """
    if V is None:
        V = np.full(env.n_states, float(init))
    for _ in range(n_episodes):
        # TODO: reset, then after EVERY step update V[s] with the TD error.
        raise NotImplementedError
    return V


# ---------------------------------------------------------------- provided
class RandomWalk:
    """Sutton & Barto Example 6.2.  Terminal 0  |  A B C D E  |  terminal 6.

    States 1..n are non-terminal; start in the middle. Action 0 = left, 1 = right.
    Stepping off the right end gives `right_reward` (default +1), off the left end `left_reward`.
    """

    def __init__(self, n=5, left_reward=0.0, right_reward=1.0):
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
        """Exact v_pi for the 50/50 policy (non-terminal states 1..n)."""
        k = np.arange(1, self.n + 1)
        return self.left_reward + (self.right_reward - self.left_reward) * k / (self.n + 1)


def coin_flip(s):
    return random.randrange(2)


def constant_alpha_mc(env, policy, n_episodes, alpha, gamma=1.0, init=0.0, V=None):
    """For comparison: V(S) <- V(S) + alpha * (G - V(S)), applied at the END of each episode."""
    if V is None:
        V = np.full(env.n_states, float(init))
    for _ in range(n_episodes):
        s, _ = env.reset()
        traj = []
        while True:
            s2, r, term, trunc, _ = env.step(policy(s))
            traj.append((s, r))
            if term or trunc:
                break
            s = s2
        G = 0.0
        for s, r in reversed(traj):
            G = r + gamma * G
            V[s] += alpha * (G - V[s])
    return V


def rms(V, env):
    return float(np.sqrt(np.mean((V[1:-1] - env.true_values()) ** 2)))


if __name__ == "__main__":
    random.seed(0)
    env = RandomWalk()

    # 1. Estimates after 1, 10, 100 episodes (S&B Figure 6.2, left)
    for n in (1, 10, 100):
        random.seed(n)
        V = td0(env, coin_flip, n, alpha=0.1, init=0.5)
        rlhud.plot(V[1:-1], label=f"{n} episodes", title="TD(0) estimates of states A–E, α = 0.1")
    rlhud.plot(env.true_values(), label="true", title="TD(0) estimates of states A–E, α = 0.1")

    # 2. RMS error per episode, averaged over runs: TD vs MC (Figure 6.2, right)
    RUNS, EPISODES = 50, 100
    for name, fn, alpha in [("TD α=0.05", td0, 0.05), ("TD α=0.15", td0, 0.15),
                            ("MC α=0.01", constant_alpha_mc, 0.01), ("MC α=0.04", constant_alpha_mc, 0.04)]:
        err = np.zeros(EPISODES)
        for run in range(RUNS):
            random.seed(1000 + run)
            V = np.full(env.n_states, 0.5)
            for ep in range(EPISODES):
                fn(env, coin_flip, 1, alpha, V=V)      # one more episode, in place
                err[ep] += rms(V, env) / RUNS
        rlhud.plot(err, label=name, title="RMS error vs episodes (avg of 50 runs)",
                   xlabel="episodes", ylabel="RMS error")
        rlhud.stat(f"Final RMS, {name}", round(float(err[-1]), 3))
    print("TD learns faster than MC on this task at every step size shown.")
