import numpy as np
import rlhud
from rlenvs import CartPole


# ---------------------------------------------------------------- provided: the policy (exercise 1)
def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def features(obs):
    """x(s) = [x, ẋ, θ, θ̇, 1] — the raw CartPole observation plus a bias term."""
    return np.append(obs, 1.0)


def sample_action(theta, x, rng):
    p = softmax(x @ theta)
    return min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)


# ---------------------------------------------------------------- your code
def returns_to_go(rewards, gamma):
    """[G0, G1, …, G_{T−1}] with Gₜ = Rₜ₊₁ + γ Gₜ₊₁ (computed backwards)."""
    G, out = 0.0, np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def reinforce_update(theta, w, X, A, G, lr, lr_v, normalize=True):
    """One REINFORCE-with-baseline update from a batch of N time steps. Updates theta and w
    in place and returns them.

    X : (N, d) features of every visited state      A : (N,) actions taken
    G : (N,) return-to-go from each step            w : (d,) linear baseline b(s) = wᵀx(s)

    1. advantages   Âₜ = Gₜ − b(Sₜ)
    2. baseline     w ← w + lr_v · mean_t[ Âₜ x(Sₜ) ]                (regress b toward G)
    3. if normalize: Â ← (Â − mean) / (std + 1e-8)
    4. policy       θ ← θ + lr · mean_t[ Âₜ ∇log π(Aₜ|Sₜ) ],  ∇log π = outer(x, onehot(a) − π)
    """
    N = len(A)
    adv = G - X @ w
    w += lr_v * (X.T @ adv) / N
    if normalize:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    P = softmax(X @ theta)                              # (N, n_actions)
    onehot = np.eye(theta.shape[1])[A]
    theta += lr * X.T @ ((onehot - P) * adv[:, None]) / N
    return theta, w


# ---------------------------------------------------------------- provided: training loop
def train(episodes=1000, seed=0, lr=2.0, lr_v=0.1, gamma=0.99, batch_episodes=5, normalize=True, log=None):
    """REINFORCE with a learned linear baseline on CartPole. Returns (theta, w, episode_returns)."""
    rng = np.random.default_rng(seed)
    env = CartPole(seed=seed)
    theta = np.zeros((5, env.n_actions))
    w = np.zeros(5)
    returns, X, A, G = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        rewards, done = [], False
        while not done:                                  # 1) play a whole episode with π_θ
            x = features(obs)
            a = sample_action(theta, x, rng)
            obs, r, terminated, truncated, _ = env.step(a)
            X.append(x); A.append(a); rewards.append(r)
            done = terminated or truncated
        G.append(returns_to_go(rewards, gamma))          # 2) returns from every step
        returns.append(sum(rewards))
        if log:
            log(ep, returns[-1])
        if (ep + 1) % batch_episodes == 0:               # 3) update from a few episodes at once
            reinforce_update(theta, w, np.array(X), np.array(A), np.concatenate(G), lr, lr_v, normalize)
            X, A, G = [], [], []
    return theta, w, returns


def evaluate(theta, episodes=10, seed=1000):
    """Mean return of the STOCHASTIC policy (we sample, as REINFORCE does)."""
    env, rng, total = CartPole(), np.random.default_rng(123), 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(sample_action(theta, features(obs), rng))
            total += r
            done = term or trunc
    return total / episodes


if __name__ == "__main__":
    live = rlhud.chart("REINFORCE on CartPole — episode return", smooth=25, xlabel="episode")
    theta, w, returns = train(episodes=1000, seed=0, log=lambda ep, G: live.add("with baseline", G))
    # Same budget without baseline or normalisation, for comparison:
    def raw(ep, G):
        live.add("no baseline, no normalisation", G)
    train(episodes=1000, seed=0, lr=0.02, lr_v=0.0, normalize=False, log=raw)
    live.flush()
    rlhud.stat("Evaluation (10 episodes)", evaluate(theta))
    rlhud.bars(theta[:, 1] - theta[:, 0], labels=["x", "ẋ", "θ", "θ̇", "bias"],
               title="Preference for RIGHT over LEFT, per feature")
