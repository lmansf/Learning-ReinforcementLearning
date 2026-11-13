import numpy as np
import rlhud
from rlenvs import MountainCar


# ---------------------------------------------------------------- provided: tile coder (previous exercise)
class TileCoder:
    def __init__(self, low, high, n_tilings=8, tiles_per_dim=8):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.n_tilings, self.tiles, self.dims = n_tilings, tiles_per_dim, len(self.low)
        self.shape = (tiles_per_dim + 1,) * self.dims
        self.tiles_per_tiling = (tiles_per_dim + 1) ** self.dims
        self.n_features = n_tilings * self.tiles_per_tiling
        k = np.arange(self.dims)
        self.offsets = (np.arange(n_tilings)[:, None] * (2 * k + 1)[None, :] / n_tilings) % 1.0

    def active(self, x):
        scaled = (np.asarray(x, float) - self.low) / (self.high - self.low) * self.tiles
        coords = np.clip(np.floor(scaled[None, :] + self.offsets).astype(int), 0, self.tiles)
        return np.arange(self.n_tilings) * self.tiles_per_tiling + np.ravel_multi_index(coords.T, self.shape)


# ---------------------------------------------------------------- your code
def q_values(w, idx):
    """q̂(s, a, w) for every action at once.

    w   : weights, shape (n_actions, n_features) — one linear function per action
    idx : active tile indices of s (the features are binary, so wᵀx(s) = sum of active weights)
    returns an array of shape (n_actions,)
    """
    # TODO: one line of numpy indexing
    raise NotImplementedError


def epsilon_greedy(q, epsilon, rng):
    """Random action with probability epsilon, otherwise a greedy one (ties broken at random)."""
    # TODO: use rng.random(), rng.integers(...) and rng.choice(np.flatnonzero(q == q.max()))
    raise NotImplementedError


def sarsa_update(w, idx, a, r, idx_next, a_next, terminated, alpha, gamma=1.0):
    """One semi-gradient SARSA step. Updates w in place and returns the TD error δ.

    target = r                                  if terminated
             r + γ q̂(S′, A′, w)                 otherwise
    w[a] ← w[a] + α δ ∇q̂  — the gradient is 1 on the active tiles of S and 0 elsewhere.
    """
    # TODO: target, δ, then add α·δ to the active weights of action a only
    raise NotImplementedError


# ---------------------------------------------------------------- provided: the training loop
def train(episodes=150, seed=0, alpha=0.5, epsilon=0.0, gamma=1.0, n_tilings=8, tiles=8,
          max_steps=1000, log=None):
    """Semi-gradient SARSA on MountainCar. Returns (w, tile_coder, steps_per_episode).

    alpha is the *total* step size: each of the n_tilings active weights moves by alpha/n_tilings.
    epsilon=0 works here because q̂ starts at 0 while every real return is negative —
    "optimistic initialisation" makes untried actions look attractive.
    """
    rng = np.random.default_rng(seed)
    env = MountainCar(max_steps=max_steps, seed=seed)
    tc = TileCoder(env.low, env.high, n_tilings, tiles)
    w = np.zeros((env.n_actions, tc.n_features))
    step = alpha / n_tilings
    steps_per_episode = []
    for ep in range(episodes):
        obs, _ = env.reset()
        idx = tc.active(obs)
        a = epsilon_greedy(q_values(w, idx), epsilon, rng)
        n = 0
        while True:
            obs, r, terminated, truncated, _ = env.step(a)
            n += 1
            if terminated:
                sarsa_update(w, idx, a, r, None, None, True, step, gamma)
                break
            idx2 = tc.active(obs)
            a2 = epsilon_greedy(q_values(w, idx2), epsilon, rng)
            # A time-limit cut is NOT a real ending: still bootstrap from q̂(S′, A′).
            sarsa_update(w, idx, a, r, idx2, a2, False, step, gamma)
            idx, a = idx2, a2
            if truncated:
                break
        steps_per_episode.append(n)
        if log:
            log(ep, n)
    return w, tc, steps_per_episode


def greedy_steps(w, tc, seed=123, max_steps=1000):
    """Run one greedy episode and return how many steps it took to reach the flag."""
    env = MountainCar(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    for t in range(1, max_steps + 1):
        obs, _, terminated, truncated, _ = env.step(int(np.argmax(q_values(w, tc.active(obs)))))
        if terminated or truncated:
            return t
    return max_steps


if __name__ == "__main__":
    live = rlhud.chart("Steps to reach the flag (lower is better)", smooth=10, xlabel="episode")
    w, tc, steps = train(episodes=300, seed=0, log=lambda ep, n: (live.add("SARSA", n), rlhud.progress(ep, 300)))
    live.flush()
    rlhud.stat("Mean steps, last 20 episodes", float(np.mean(steps[-20:])))
    rlhud.stat("Greedy run (steps)", greedy_steps(w, tc))

    # The famous "cost-to-go" picture: −max_a q̂(s, a) over the whole state space.
    pos = np.linspace(-1.2, 0.6, 24)
    vel = np.linspace(0.07, -0.07, 16)
    ctg = [[-q_values(w, tc.active([p, v])).max() for p in pos] for v in vel]
    rlhud.grid(ctg, title="Cost-to-go −max q̂  (x: position →, y: velocity, top = +0.07)", fmt=".0f", cmap="heat")
