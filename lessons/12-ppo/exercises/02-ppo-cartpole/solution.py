import numpy as np
import rlhud
from rlenvs import CartPole


# ---------------------------------------------------------------- provided: networks, optimiser, helpers
class MLP:
    """Linear → tanh → … → Linear with manual backprop."""

    def __init__(self, sizes, seed=0, out_scale=1.0):
        rng = np.random.default_rng(seed)
        self.params = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(1.0 / n_in) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.params += [rng.normal(0.0, scale, size=(n_in, n_out)), np.zeros(n_out)]
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
                d = (d @ self.params[2 * i].T) * (1.0 - self._inputs[i] ** 2)
        return grads


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


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def entropy_loss_grad(P):
    """Gradient w.r.t. the logits of  −mean(H(π)),  H = −Σ p log p.  Shape like P.
    Adding  ent_coef · this  to the actor's logit-gradient implements the entropy bonus."""
    logP = np.log(P + 1e-12)
    H = -(P * logP).sum(axis=1, keepdims=True)
    return P * (logP + H) / len(P)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    rewards, values, dones = (np.asarray(x, dtype=float) for x in (rewards, values, dones))
    deltas = rewards + gamma * (1.0 - dones) * np.append(values[1:], last_value) - values
    adv, running = np.zeros(len(deltas)), 0.0
    for t in reversed(range(len(deltas))):
        running = deltas[t] + gamma * lam * (1.0 - dones[t]) * running
        adv[t] = running
    return adv, adv + values


# ---------------------------------------------------------------- your code
def ppo_clip_loss(logp_new, logp_old, adv, clip_eps=0.2):
    """(loss, dloss/dlogp_new) — paste your solution from the previous exercise."""
    ratio = np.exp(logp_new - logp_old)
    loss = -np.mean(np.minimum(ratio * adv, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv))
    flat = ((adv > 0) & (ratio > 1 + clip_eps)) | ((adv < 0) & (ratio < 1 - clip_eps))
    return float(loss), np.where(flat, 0.0, -ratio * adv) / len(adv)


def ppo_update(actor, critic, actor_opt, critic_opt, batch, rng, epochs=4, minibatch_size=64,
               clip_eps=0.2, ent_coef=0.01, value_scale=0.01):
    """Several epochs of minibatch updates on one rollout. Returns the mean approx-KL of the
    last epoch (a health statistic).

    batch: dict of arrays with N rows — "obs" (N, 4), "actions" (N,), "logp_old" (N,),
           "advantages" (N,), "returns" (N,)
    """
    N = len(batch["actions"])
    rows = np.arange(minibatch_size)
    for epoch in range(epochs):
        kls = []
        order = rng.permutation(N)
        for start in range(0, N, minibatch_size):
            mb = order[start:start + minibatch_size]
            obs, act = batch["obs"][mb], batch["actions"][mb]
            adv = batch["advantages"][mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # per-minibatch normalisation

            # --- actor: clipped surrogate + entropy bonus, back-propagated through the softmax
            P = softmax(actor.forward(obs))
            logp = np.log(P[rows[:len(mb)], act] + 1e-12)
            _, dlogp = ppo_clip_loss(logp, batch["logp_old"][mb], adv, clip_eps)
            onehot = np.eye(P.shape[1])[act]
            dlogits = dlogp[:, None] * (onehot - P)                 # ∂ log π(a) / ∂ logits = onehot − π
            dlogits += ent_coef * entropy_loss_grad(P)
            actor_opt.step(actor.backward(dlogits))

            # --- critic: regression on the (scaled) λ-returns
            v = critic.forward(obs)
            target = batch["returns"][mb][:, None] * value_scale
            critic_opt.step(critic.backward((v - target) / len(mb)))
            kls.append(np.mean((np.exp(logp - batch["logp_old"][mb]) - 1) - (logp - batch["logp_old"][mb])))
    return float(np.mean(kls))


# ---------------------------------------------------------------- provided: rollouts + training loop
def train(total_steps=80_000, seed=0, n_steps=512, gamma=0.99, lam=0.95, lr=1e-3, hidden=32,
          epochs=4, minibatch_size=64, clip_eps=0.2, ent_coef=0.01, value_scale=0.01, log=None):
    """PPO on CartPole. Returns (actor, critic, episode_returns)."""
    rng = np.random.default_rng(seed)
    env = CartPole(seed=seed)
    actor = MLP([4, hidden, env.n_actions], seed=seed, out_scale=0.01)   # near-uniform initial policy
    critic = MLP([4, hidden, 1], seed=seed + 1)
    actor_opt, critic_opt = Adam(actor.params, lr), Adam(critic.params, lr)
    V = lambda o: critic.forward(np.atleast_2d(o))[:, 0] / value_scale

    obs, _ = env.reset(seed=seed)
    ep_return, returns, t = 0.0, [], 0
    while t < total_steps:
        # 1) collect n_steps with the current ("old") policy, remembering log π_old(a|s)
        O, A, LP, R, D = [], [], [], [], []
        for _ in range(n_steps):
            p = softmax(actor.forward(obs[None, :]))[0]
            a = min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)
            obs2, r, terminated, truncated, _ = env.step(a)
            if truncated and not terminated:
                r += gamma * V(obs2)[0]                  # time-limit bootstrap
            O.append(obs); A.append(a); LP.append(np.log(p[a])); R.append(r)
            D.append(float(terminated or truncated))
            ep_return += 1.0; t += 1
            obs = obs2
            if terminated or truncated:
                returns.append(ep_return)
                if log:
                    log(len(returns), ep_return, None)
                ep_return = 0.0
                obs, _ = env.reset()
        O = np.array(O)
        # 2) advantages and value targets
        adv, rets = compute_gae(R, V(O), D, V(obs)[0], gamma, lam)
        batch = {"obs": O, "actions": np.array(A), "logp_old": np.array(LP), "advantages": adv, "returns": rets}
        # 3) several epochs of clipped updates on this one batch
        kl = ppo_update(actor, critic, actor_opt, critic_opt, batch, rng, epochs, minibatch_size,
                        clip_eps, ent_coef, value_scale)
        if log:
            log(len(returns), None, kl)
    return actor, critic, returns


def evaluate(actor, episodes=10, seed=1000):
    """Mean return of the stochastic policy."""
    env, rng, total = CartPole(), np.random.default_rng(123), 0.0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            p = softmax(actor.forward(obs[None, :]))[0]
            a = min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
    return total / episodes


if __name__ == "__main__":
    live = rlhud.chart("PPO on CartPole — episode return", smooth=10, xlabel="episode")
    kl_chart = rlhud.chart("Approx KL per update (keep it small: ~0.001–0.02)", xlabel="update")

    def log(ep, G, kl):
        if G is not None:
            live.add("PPO", G)
        if kl is not None:
            kl_chart.add("KL", kl)

    actor, critic, returns = train(total_steps=80_000, seed=0, log=log)
    live.flush(); kl_chart.flush()
    score = evaluate(actor)
    rlhud.stat("Evaluation (10 episodes)", score, good=score >= 200)
    rlhud.stat("Episodes", len(returns))
