import numpy as np
import rlhud
from rlenvs import CartPole


# ---------------------------------------------------------------- provided: network, optimiser, replay
class MLP:
    """Linear → ReLU → … → Linear, with manual backprop (exercise 1)."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.params += [rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)), np.zeros(n_out)]
        self.n_layers = len(sizes) - 1

    def forward(self, x):
        h, self._inputs = np.asarray(x, dtype=float), []
        for i in range(self.n_layers):
            self._inputs.append(h)
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)
        return h

    def backward(self, dout):
        """Gradients for the inputs of the MOST RECENT forward() call."""
        grads, d = [None] * len(self.params), np.asarray(dout, dtype=float)
        for i in reversed(range(self.n_layers)):
            grads[2 * i], grads[2 * i + 1] = self._inputs[i].T @ d, d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[2 * i].T) * (self._inputs[i] > 0)
        return grads

    def copy_from(self, other):
        """Overwrite these weights with a copy of `other`'s (used for the target network)."""
        for p, q in zip(self.params, other.params):
            p[...] = q


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


class ReplayBuffer:
    """Circular buffer (exercise 2)."""

    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim)); self.next_obs = np.zeros((capacity, obs_dim))
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity); self.terminated = np.zeros(capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i], self.next_obs[i], self.terminated[i] = s, a, r, s2, terminated
        self.pos, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.terminated[idx]

    def __len__(self):
        return self.size


# ---------------------------------------------------------------- your code: the DQN pieces
def epsilon_by_step(t, start=1.0, end=0.05, decay_steps=10_000):
    """Linear ε schedule: `start` at t = 0, `end` from t = decay_steps on."""
    # TODO
    raise NotImplementedError


def select_action(q_net, obs, epsilon, rng):
    """ε-greedy on the network's Q-values for a single observation."""
    # TODO: with probability epsilon a random action (rng.random(), rng.integers(n_actions)),
    #       otherwise argmax of q_net.forward(obs[None, :])[0]. n_actions = q_net.params[-1].shape[0]
    raise NotImplementedError


def td_targets(rewards, terminated, q_next_target, gamma, q_next_online=None):
    """Bootstrapped targets y for a batch.

    rewards, terminated : (B,)
    q_next_target       : (B, n_actions) — target network's Q(s′, ·)
    q_next_online       : (B, n_actions) or None — online network's Q(s′, ·)

    DQN:         y = r + γ (1 − done) · max_a Q_target(s′, a)
    Double DQN:  y = r + γ (1 − done) · Q_target(s′, argmax_a Q_online(s′, a))   (when q_next_online is given)
    """
    # TODO: pick next_v per row (max for DQN; target's value at the online argmax for Double DQN),
    #       then y = r + γ (1 − terminated) next_v
    raise NotImplementedError


def dqn_loss_and_grad(q_pred, actions, targets, delta=1.0):
    """Mean Huber loss between Q(s, a) for the TAKEN actions and the targets.

    q_pred  : (B, n_actions) online network output for the batch states
    actions : (B,) ints;  targets : (B,)
    returns (loss, dq) where dq = ∂loss/∂q_pred has q_pred's shape and is zero
    for every action that was not taken.
    """
    # TODO
    # 1) err = Q(s, a_taken) − target        (use q_pred[np.arange(B), actions])
    # 2) loss = mean Huber(err)
    # 3) dq = zeros like q_pred; at the taken actions put clip(err, −δ, δ) / B
    raise NotImplementedError


# ---------------------------------------------------------------- provided: training loop
def evaluate(q_net, episodes=10, seed=1000):
    """Mean return of the GREEDY policy (no exploration) over fixed start states."""
    env, total = CartPole(), 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(int(np.argmax(q_net.forward(obs[None, :])[0])))
            total += r
            done = term or trunc
    return total / episodes


def train(total_steps=70_000, seed=0, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=50_000,
          learning_starts=1_000, target_sync=500, eps_decay=10_000, hidden=64, double=True,
          eval_every=2_000, log=None):
    """Train DQN on CartPole. Returns (best_q_net, episode_returns)."""
    rng = np.random.default_rng(seed)
    env = CartPole(seed=seed)
    q = MLP([4, hidden, hidden, 2], seed=seed)
    q_target = MLP([4, hidden, hidden, 2], seed=seed + 1)
    q_target.copy_from(q)
    opt = Adam(q.params, lr=lr)
    buf = ReplayBuffer(buffer_size, 4, seed=seed)
    best, best_score = MLP([4, hidden, hidden, 2]), -1.0

    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []
    for t in range(total_steps):
        # 1) act ε-greedily and store the transition
        eps = epsilon_by_step(t, decay_steps=eps_decay)
        a = select_action(q, obs, eps, rng)
        obs2, r, terminated, truncated, _ = env.step(a)
        buf.add(obs, a, r, obs2, terminated)      # store `terminated`, NOT truncated
        obs, ep_return = obs2, ep_return + r
        if terminated or truncated:
            returns.append(ep_return)
            if log:
                log(len(returns), ep_return, eps)
            obs, _ = env.reset()
            ep_return = 0.0

        # 2) learn from a random minibatch of past experience
        if t >= learning_starts:
            S, A, R, S2, D = buf.sample(batch_size)
            q_next_t = q_target.forward(S2)
            q_next_o = q.forward(S2) if double else None
            y = td_targets(R, D, q_next_t, gamma, q_next_o)
            q_pred = q.forward(S)                 # forward(S) LAST, so backward() uses S
            loss, dq = dqn_loss_and_grad(q_pred, A, y)
            opt.step(q.backward(dq))

        # 3) every so often, copy the online weights into the frozen target network
        if t % target_sync == 0:
            q_target.copy_from(q)

        # 4) deep RL is noisy: keep the best greedy policy we have seen ("checkpointing")
        if t >= learning_starts and t % eval_every == 0 and returns and np.mean(returns[-5:]) > 100:
            score = evaluate(q, 5, seed=10_000 + t)
            if score >= best_score:
                best_score = score
                best.copy_from(q)
    if best_score < 0 or evaluate(q, 5, seed=99_999) > best_score:
        best.copy_from(q)
    return best, returns


if __name__ == "__main__":
    live = rlhud.chart("DQN on CartPole — episode return", smooth=20, xlabel="episode")
    eps_chart = rlhud.chart("ε (exploration)", xlabel="episode")

    def log(ep, G, eps):
        live.add("return", G)
        eps_chart.add("ε", eps)

    q, returns = train(total_steps=70_000, seed=0, log=log)
    live.flush(); eps_chart.flush()
    score = evaluate(q, 10)
    rlhud.stat("Greedy evaluation (10 episodes)", score, good=score >= 150)
    rlhud.stat("Episodes trained", len(returns))
    # What did it learn? Q(s, ·) as the pole angle sweeps from left to right (cart centred, still).
    thetas = np.linspace(-0.2, 0.2, 41)
    Q = q.forward(np.array([[0.0, 0.0, th, 0.0] for th in thetas]))
    rlhud.plot(Q[:, 0], x=thetas, label="Q(push left)", title="Q-values vs pole angle θ", xlabel="θ (rad)")
    rlhud.plot(Q[:, 1], x=thetas, label="Q(push right)", title="Q-values vs pole angle θ", xlabel="θ (rad)")
