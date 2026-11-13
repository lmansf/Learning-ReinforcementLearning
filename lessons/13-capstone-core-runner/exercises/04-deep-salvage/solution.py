import os
import time
import numpy as np
import rlhud
from corerunner import CoreRunner

WEIGHTS = "deep_salvage.npz"          # saved in your workspace folder


# ================================================================ provided: network + optimiser
class MLP:
    """Fully-connected ReLU network with a linear output layer, trained with Adam."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]
        self._t = 0

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self._acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            self._acts.append(x)
        return x

    def backward(self, grad_out, lr):
        """Backprop dLoss/dOutput through the last forward() call and take one Adam step."""
        gW, gb, g = [], [], grad_out
        for i in reversed(range(len(self.W))):
            gW.insert(0, self._acts[i].T @ g)
            gb.insert(0, g.sum(0))
            if i > 0:
                g = (g @ self.W[i].T) * (self._acts[i] > 0)
        self._t += 1
        for k, (p, gr) in enumerate(zip(self.params(), gW + gb)):
            self._m[k] = 0.9 * self._m[k] + 0.1 * gr
            self._v[k] = 0.999 * self._v[k] + 0.001 * gr * gr
            p -= lr * (self._m[k] / (1 - 0.9 ** self._t)) / (np.sqrt(self._v[k] / (1 - 0.999 ** self._t)) + 1e-8)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def save(self, path):
        np.savez(path, *self.W, *self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        arrs = [data[f"arr_{i}"] for i in range(len(data.files))]
        net = cls.__new__(cls)
        half = len(arrs) // 2
        net.W, net.b = arrs[:half], arrs[half:]
        return net


class ReplayBuffer:
    """Stores n-step transitions (s, a, R_n, s_n, done, discount=γ^k)."""

    def __init__(self, capacity, obs_size):
        self.S = np.zeros((capacity, obs_size), np.float32)
        self.S2 = np.zeros((capacity, obs_size), np.float32)
        self.A = np.zeros(capacity, np.int64)
        self.R = np.zeros(capacity)
        self.D = np.zeros(capacity)
        self.G = np.zeros(capacity)
        self.cap, self.i, self.n = capacity, 0, 0

    def add(self, s, a, r, s2, done, disc):
        j = self.i
        self.S[j], self.A[j], self.R[j], self.S2[j], self.D[j], self.G[j] = s, a, r, s2, done, disc
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k, rng):
        idx = rng.integers(self.n, size=k)
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx], self.G[idx]


# ================================================================ your part
def epsilon_at(step, total_steps, explore_frac=0.4, eps_end=0.05):
    """Linear decay from 1.0 to eps_end over the first explore_frac of training."""
    return max(eps_end, 1.0 - (1.0 - eps_end) * step / (explore_frac * total_steps))


def nstep_return(rewards, gamma):
    """Discounted sum of a short list of rewards, and the discount to apply to the bootstrap.
    nstep_return([r0, r1, r2], γ) -> (r0 + γ r1 + γ² r2, γ³)"""
    G = 0.0
    for k, r in enumerate(rewards):
        G += gamma ** k * r
    return G, gamma ** len(rewards)


def double_dqn_targets(online, target, R, S2, D, disc):
    """y = R + disc · (1 − D) · Q_target(s', argmaxₐ Q_online(s', a))."""
    a_star = online.forward(S2).argmax(axis=1)
    q_next = target.forward(S2)[np.arange(len(S2)), a_star]
    return R + disc * (1.0 - D) * q_next


def huber_grad(pred, y, delta=1.0):
    """d/dpred of the Huber loss, averaged over the batch: clip(pred − y, −δ, δ) / N."""
    return np.clip(pred - y, -delta, delta) / len(pred)


def train(total_steps=60_000, seed=0, hidden=128, gamma=0.97, lr=5e-4, n_step=3,
          batch=64, train_every=2, target_every=1000, chart=None):
    """Train a DQN on procedurally generated wrecks. Returns the online network."""
    rng = np.random.default_rng(seed)
    env = CoreRunner(level=4, obs="vector", seed=seed)
    q = MLP([env.obs_size, hidden, 5], rng)
    q_target = MLP([env.obs_size, hidden, 5], rng)
    q_target.copy_from(q)
    buf = ReplayBuffer(60_000, env.obs_size)
    obs, _ = env.reset()
    pending, ep_return = [], 0.0          # last n (obs, action, reward) for n-step returns
    for step in range(total_steps):
        if rng.random() < epsilon_at(step, total_steps):
            a = int(rng.integers(5))
        else:
            a = int(np.argmax(q.forward(obs[None])[0]))
        obs2, r, term, trunc, _ = env.step(a)
        ep_return += r
        pending.append((obs, a, r))
        if len(pending) == n_step:        # oldest transition now has its full n-step return
            R, disc = nstep_return([p[2] for p in pending], gamma)
            buf.add(pending[0][0], pending[0][1], R, obs2, float(term), disc)
            pending.pop(0)
        obs = obs2
        if term or trunc:                 # flush shorter returns at the end of an episode
            while pending:
                R, disc = nstep_return([p[2] for p in pending], gamma)
                buf.add(pending[0][0], pending[0][1], R, obs2, float(term), disc)
                pending.pop(0)
            if chart is not None:
                chart.add("episode return", ep_return)
            obs, _ = env.reset()
            ep_return = 0.0
        if buf.n >= 2000 and step % train_every == 0:
            S, A, R, S2, D, G = buf.sample(batch, rng)
            y = double_dqn_targets(q, q_target, R, S2, D, G)
            qs = q.forward(S)
            grad = np.zeros_like(qs)
            grad[np.arange(batch), A] = huber_grad(qs[np.arange(batch), A], y)
            q.backward(grad, lr)
        if step % target_every == 0:
            q_target.copy_from(q)
        rlhud.progress(step, total_steps, "Training DQN")
    return q


def load_agent(path=WEIGHTS):
    """Load saved weights and return a greedy policy obs -> action."""
    net = MLP.load(path)
    return lambda obs: int(np.argmax(net.forward(np.asarray(obs, np.float32)[None])[0]))


# ================================================================ evaluation (provided)
def evaluate(policy, episodes=50, seed=50_000, record=0):
    env = CoreRunner(level=4, obs="vector")
    returns, escapes, replays = [], 0, []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        frames, G = [env.snapshot()], 0.0
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            G += r
            if i < record:
                frames.append(env.snapshot())
            if term or trunc:
                break
        returns.append(G)
        escapes += info["event"] == "escaped"
        if i < record:
            replays.append(frames)
    return float(np.mean(returns)), escapes / episodes, replays


if __name__ == "__main__":
    t0 = time.time()
    chart = rlhud.chart("Deep Salvage · training return", smooth=100)
    net = train(total_steps=80_000, seed=0, chart=chart)
    chart.flush()
    net.save(WEIGHTS)
    print(f"trained in {time.time() - t0:.0f}s, saved weights to {os.path.abspath(WEIGHTS)}")
    mean, success, replays = evaluate(load_agent(), record=3)
    rlhud.stat("Mean return (50 unseen wrecks)", mean)
    rlhud.stat("Extraction rate", f"{success:.0%}", good=success >= 0.45)
    for k, frames in enumerate(replays):
        rlhud.replay(frames, title=f"Unseen wreck #{k + 1}")
