import numpy as np
import rlhud


class ReplayBuffer:
    """Fixed-size circular buffer of transitions (s, a, r, s', terminated).

    buf = ReplayBuffer(capacity=50_000, obs_dim=4, seed=0)
    buf.add(s, a, r, s2, terminated)
    S, A, R, S2, D = buf.sample(64)     # numpy arrays with a leading batch dimension
    len(buf)                            # number of stored transitions (≤ capacity)

    Storage is pre-allocated numpy arrays — much faster than a Python list of tuples
    once you sample tens of thousands of batches.
    """

    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0          # where the next transition goes
        self.size = 0         # how many slots are filled
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminated):
        """Store one transition, overwriting the oldest one when full."""
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = s, a, r
        self.next_obs[i], self.terminated[i] = s2, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Uniformly sample `batch_size` stored transitions (with replacement)."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.terminated[idx])

    def __len__(self):
        return self.size


def huber(x, delta=1.0):
    """Huber loss, element-wise: ½x² for |x| ≤ δ, δ(|x| − ½δ) beyond."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def huber_grad(x, delta=1.0):
    """d huber / dx, element-wise: x inside [−δ, δ], ±δ outside — i.e. clip(x, −δ, δ)."""
    return np.clip(np.asarray(x, dtype=float), -delta, delta)


if __name__ == "__main__":
    # 1) Why replay? Consecutive CartPole states are almost identical. Compare the correlation
    #    between neighbours in a trajectory with neighbours in a replay batch.
    from rlenvs import CartPole
    env, rng = CartPole(seed=0), np.random.default_rng(0)
    buf = ReplayBuffer(10_000, 4, seed=0)
    s, _ = env.reset()
    traj = []
    for t in range(5000):
        a = int(rng.integers(2))
        s2, r, term, trunc, _ = env.step(a)
        buf.add(s, a, r, s2, term)
        traj.append(s[2])
        s = s2 if not (term or trunc) else env.reset()[0]
    traj = np.array(traj)
    batch_theta = buf.sample(5000)[0][:, 2]
    corr = lambda v: float(np.corrcoef(v[:-1], v[1:])[0, 1])
    rlhud.table([["consecutive steps", f"{corr(traj):.3f}"], ["replay sample", f"{corr(batch_theta):.3f}"]],
                headers=["pole angle θ, neighbour pairs from…", "correlation"], title="Replay breaks correlations")
    rlhud.stat("Transitions stored", len(buf))

    # 2) Huber vs squared loss and their gradients
    xs = np.linspace(-3, 3, 61)
    rlhud.plot(0.5 * xs ** 2, x=xs, label="½x² (MSE)", title="Loss")
    rlhud.plot(huber(xs), x=xs, label="Huber δ=1", title="Loss")
    rlhud.plot(xs, x=xs, label="x (MSE gradient)", title="Gradient")
    rlhud.plot(huber_grad(xs), x=xs, label="Huber gradient", title="Gradient")
