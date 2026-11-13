import numpy as np
import rlhud
from rlenvs import CartPole


# ---------------------------------------------------------------- provided: critic network + optimiser
class MLP:
    """Linear → tanh → … → Linear with manual backprop (like lesson 9, tanh instead of ReLU)."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.params += [rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out)), np.zeros(n_out)]
        self.n_layers = len(sizes) - 1

    def forward(self, x):
        h, self._inputs = np.asarray(x, dtype=float), []
        for i in range(self.n_layers):
            self._inputs.append(h)
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < self.n_layers - 1:
                h = np.tanh(h)
        return h

    def backward(self, dout):
        grads, d = [None] * len(self.params), np.asarray(dout, dtype=float)
        for i in reversed(range(self.n_layers)):
            grads[2 * i], grads[2 * i + 1] = self._inputs[i].T @ d, d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[2 * i].T) * (1.0 - self._inputs[i] ** 2)   # tanh′ = 1 − tanh²
        return grads


class Adam:
    """Minimises: pass the gradient of a LOSS."""

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


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def features(obs):
    return np.append(obs, 1.0)


# ---------------------------------------------------------------- your code
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """(advantages, returns) — exactly as in the previous exercise."""
    rewards, values, dones = (np.asarray(x, dtype=float) for x in (rewards, values, dones))
    next_values = np.append(values[1:], last_value)
    deltas = rewards + gamma * (1.0 - dones) * next_values - values
    adv, running = np.zeros(len(deltas)), 0.0
    for t in reversed(range(len(deltas))):
        running = deltas[t] + gamma * lam * (1.0 - dones[t]) * running
        adv[t] = running
    return adv, adv + values


def actor_loss_grad(theta, X, A, adv):
    """Gradient w.r.t. theta of the actor LOSS  L(θ) = −mean_t[ Âₜ log π(Aₜ|Sₜ) ].

    X: (N, d) features, A: (N,) actions, adv: (N,) advantages (treated as constants).
    Minimising this loss = ascending the policy gradient. Returns an array shaped like theta.
    """
    P = softmax(X @ theta)
    onehot = np.eye(theta.shape[1])[A]
    return -X.T @ ((onehot - P) * adv[:, None]) / len(A)


def critic_loss_grad(v_pred, targets):
    """Gradient w.r.t. v_pred (shape (N, 1)) of the critic loss  ½ · mean (v_pred − target)²."""
    return (v_pred - np.asarray(targets).reshape(-1, 1)) / len(v_pred)


# ---------------------------------------------------------------- provided: A2C training loop
def train(total_steps=250_000, seed=0, n_steps=32, gamma=0.99, lam=0.95, lr_actor=5e-3,
          lr_critic=1e-3, hidden=32, value_scale=0.01, log=None):
    """Advantage actor-critic: a linear-softmax actor and a small MLP critic.

    value_scale: the critic network predicts V(s)·value_scale. CartPole values reach ~100;
    networks are happiest with targets around ±1, so we shrink them (and grow them back).
    Returns (theta, critic, episode_returns).
    """
    rng = np.random.default_rng(seed)
    env = CartPole(seed=seed)
    theta = np.zeros((5, env.n_actions))
    critic = MLP([4, hidden, 1], seed=seed)
    actor_opt, critic_opt = Adam([theta], lr=lr_actor), Adam(critic.params, lr=lr_critic)
    V = lambda obs: critic.forward(np.atleast_2d(obs))[:, 0] / value_scale

    obs, _ = env.reset(seed=seed)
    ep_return, returns, t = 0.0, [], 0
    while t < total_steps:
        # 1) collect a short rollout of n_steps with the current policy
        X, OBS, A, R, D = [], [], [], [], []
        for _ in range(n_steps):
            x = features(obs)
            p = softmax(x @ theta)
            a = int(rng.random() < p[1])
            obs2, r, terminated, truncated, _ = env.step(a)
            if truncated and not terminated:
                r += gamma * V(obs2)[0]       # time limit: fold the bootstrap value into r …
            X.append(x); OBS.append(obs); A.append(a); R.append(r)
            D.append(float(terminated or truncated))   # … and then cut the episode like a done
            ep_return += 1.0; t += 1
            obs = obs2
            if terminated or truncated:
                returns.append(ep_return)
                if log:
                    log(len(returns), ep_return)
                ep_return = 0.0
                obs, _ = env.reset()
        X, OBS, A = np.array(X), np.array(OBS), np.array(A)

        # 2) advantages from the critic
        adv, targets = compute_gae(R, V(OBS), D, V(obs)[0], gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 3) actor step (policy gradient with the critic's advantage) and critic step (regression)
        actor_opt.step([actor_loss_grad(theta, X, A, adv)])
        v_pred = critic.forward(OBS)
        critic_opt.step(critic.backward(critic_loss_grad(v_pred, targets * value_scale)))
    return theta, critic, returns


def evaluate(theta, episodes=10, seed=1000):
    """Mean return of the stochastic policy."""
    env, rng, total = CartPole(), np.random.default_rng(123), 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            p = softmax(features(obs) @ theta)
            obs, r, term, trunc, _ = env.step(int(rng.random() < p[1]))
            total += r
            done = term or trunc
    return total / episodes


if __name__ == "__main__":
    live = rlhud.chart("A2C on CartPole — episode return", smooth=25, xlabel="episode")
    theta, critic, returns = train(seed=0, log=lambda ep, G: live.add("A2C", G))
    live.flush()
    rlhud.stat("Evaluation (10 episodes)", evaluate(theta))
    # What does the critic believe? V(s) as the pole angle sweeps (cart centred, at rest).
    thetas = np.linspace(-0.2, 0.2, 41)
    vs = critic.forward(np.array([[0.0, 0.0, th, 0.0] for th in thetas]))[:, 0] / 0.01
    rlhud.plot(vs, x=thetas, label="V(s)", title="Critic: value vs pole angle θ", xlabel="θ (rad)")
