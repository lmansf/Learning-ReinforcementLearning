import time
import numpy as np

ppo_clip_loss = require("ppo_clip_loss")
ppo_update = require("ppo_update")

L = np.log
loss, g = ppo_clip_loss(L([0.75, 0.25]), L([0.5, 0.5]), np.array([1.0, 1.0]), 0.2)
check(approx(loss, -(1.2 + 0.5) / 2) and approx(g, [0.0, -0.25]), "ppo_clip_loss (from the previous exercise)",
      f"expected loss −0.85 and grad [0, −0.25], got {loss}, {np.asarray(g).tolist()}")

# --- one ppo_update on a synthetic batch: action 1 is good in every state
rng = np.random.default_rng(0)
actor = MLP([4, 16, 2], seed=0, out_scale=0.01)
critic = MLP([4, 16, 1], seed=1)
aopt, copt = Adam(actor.params, 1e-2), Adam(critic.params, 1e-2)
N = 256
obs = rng.normal(size=(N, 4))
P0 = softmax(actor.forward(obs))
acts = rng.integers(0, 2, N)
batch = {"obs": obs, "actions": acts, "logp_old": np.log(P0[np.arange(N), acts]),
         "advantages": np.where(acts == 1, 1.0, -1.0) + 0.1 * rng.normal(size=N),
         "returns": np.full(N, 50.0)}
v0 = float(np.mean((critic.forward(obs)[:, 0] - 0.5) ** 2))
kl = ppo_update(actor, critic, aopt, copt, batch, np.random.default_rng(1), epochs=4, minibatch_size=64,
                clip_eps=0.2, ent_coef=0.0, value_scale=0.01)
P1 = softmax(actor.forward(obs))
p_before, p_after = P0[:, 1].mean(), P1[:, 1].mean()
check(p_after > p_before + 0.05, "ppo_update makes positive-advantage actions more likely",
      f"mean π(good action) {p_before:.3f} → {p_after:.3f}. Check the sign chain: Adam minimises the loss.")
ratio = P1[np.arange(N), acts] / P0[np.arange(N), acts]
check(np.mean(ratio[acts == 1]) < 1.6, "…but clipping keeps the new policy close to the old one",
      f"mean ratio for the good action = {np.mean(ratio[acts == 1]):.2f}; with ε = 0.2 it should stay near 1.2")
v1 = float(np.mean((critic.forward(obs)[:, 0] - 0.5) ** 2))
check(v1 < v0 * 0.5, "ppo_update trains the critic toward returns · value_scale",
      f"critic MSE vs target 0.5: {v0:.3f} → {v1:.3f}")
check(isinstance(kl, float) and 0 <= kl < 0.5, "ppo_update returns the approx KL (a small non-negative float)",
      f"got {kl!r}")

t0 = time.time()
actor, critic, returns = train(total_steps=80_000, seed=0)
dt = time.time() - t0
score = evaluate(actor)
check(score >= 250, "PPO balances the pole: mean return ≥ 250 over 10 episodes",
      f"got {score:.0f} after {len(returns)} episodes ({dt:.0f}s). Print the KL per update: if it is > 0.1 the "
      "updates are too large; if it is ~0 the actor isn't learning (check dlogits).")
