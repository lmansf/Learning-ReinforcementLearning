import random

import numpy as np
import rlhud


def td_lambda(env, policy, n_episodes, lam, alpha, gamma=1.0, init=0.0, V=None):
    """Online TD(lambda) prediction with ACCUMULATING eligibility traces.

    Returns V (numpy array, length env.n_states); if V is given it is updated in place.
    Every step:
        delta = R + gamma * V(S') - V(S)        (V(S') counts as 0 if terminated)
        e(S) += 1                               (bump the trace of the state just left)
        V    += alpha * delta * e               (EVERY state, in proportion to its trace)
        e    *= gamma * lam                     (all traces fade)
    Traces start at zero at the beginning of every episode.
    """
    if V is None:
        V = np.full(env.n_states, float(init))
    for _ in range(n_episodes):
        e = np.zeros(env.n_states)
        s, _ = env.reset()
        # TODO: loop over steps: compute delta, bump e[s], update ALL of V, decay e;
        #       stop on terminated or truncated
        raise NotImplementedError
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


def avg_rms(lam, alpha, runs=50, episodes=10, seed=0):
    """Mean RMS error over the first `episodes` episodes and `runs` runs on the 19-state walk."""
    env = RandomWalk()
    truth = env.true_values()
    random.seed(seed)
    total = 0.0
    for _ in range(runs):
        V = np.zeros(env.n_states)
        for _ in range(episodes):
            td_lambda(env, coin_flip, 1, lam, alpha, V=V)
            total += np.sqrt(np.mean((V[1:-1] - truth) ** 2))
    return total / (runs * episodes)


if __name__ == "__main__":
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    lams = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]
    rows = []
    for i, lam in enumerate(lams):
        with np.errstate(all="ignore"):                       # large α with λ ≈ 1 can diverge
            errs = [avg_rms(lam, a) for a in alphas]
        errs = [e if np.isfinite(e) and e < 0.6 else 0.6 for e in errs]   # clip for a readable plot
        rlhud.plot(errs, x=alphas, label=f"λ = {lam}", title="TD(λ) on the 19-state walk: avg RMS, first 10 episodes",
                   xlabel="α", ylabel="RMS error (clipped at 0.6)")
        best = int(np.argmin(errs))
        rows.append([lam, alphas[best], round(errs[best], 3)])
        rlhud.progress(i, len(lams), "λ values")
    rlhud.table(rows, headers=["λ", "best α", "RMS at best α"], title="Best step size per λ")

    # What a trace looks like: one episode, lam = 0.8, snapshot of e after every step
    env, lam = RandomWalk(), 0.8
    random.seed(3)
    e, s = np.zeros(env.n_states), env.reset()[0]
    snapshots = []
    while True:
        s2, r, term, trunc, _ = env.step(coin_flip(s))
        e[s] += 1.0
        snapshots.append(e[1:-1].copy())
        e *= lam
        if term or len(snapshots) >= 12:
            break
        s = s2
    rlhud.grid(np.array(snapshots), title="Accumulating trace e(s) per step (rows = time, cols = states 1–19)",
               cmap="heat", fmt=".1f")
