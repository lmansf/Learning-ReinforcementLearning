import numpy as np

epsilon_greedy = require("epsilon_greedy")
sarsa_update = require("sarsa_update")
sarsa = require("sarsa")
greedy_rollout = require("greedy_rollout")

# ---- epsilon_greedy
Q = np.array([[0.0, 2.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
rng = np.random.default_rng(0)
picks = np.array([epsilon_greedy(Q, 0, 0.2, rng) for _ in range(20_000)])
f = [(picks == a).mean() for a in range(4)]
check(abs(f[1] - 0.85) < 0.015 and all(abs(f[a] - 0.05) < 0.01 for a in (0, 2, 3)),
      "ε = 0.2, 4 actions: greedy action 0.8 + 0.2/4 = 0.85, others 0.05",
      f"observed {[round(x, 3) for x in f]}")
picks = {epsilon_greedy(Q, 1, 0.0, rng) for _ in range(400)}
check(picks == {0, 1, 2, 3}, "Ties are broken at random (all-zero row → every action gets picked)",
      f"with ε = 0 and Q[s] all equal you only ever picked {sorted(picks)}. np.argmax always returns the "
      "first maximum; use rng.choice(np.flatnonzero(Q[s] == Q[s].max())).")

# ---- sarsa_update (hand-computed)
Q = np.zeros((3, 2))
Q[2, 0] = 4.0
d = sarsa_update(Q, 0, 1, -1.0, 2, 0, 0.5, 0.9, False)
check(approx(Q[0, 1], 1.3) and approx(d, 2.6), "sarsa_update: target −1 + 0.9·Q(2,0)=2.6 → Q(0,1) = 0.5·2.6 = 1.3, δ = 2.6",
      f"got Q(0,1) = {Q[0, 1]}, returned δ = {d}")
check(approx(Q.sum(), 5.3), "sarsa_update changes only Q(S, A)", f"Q is now {Q.tolist()}")
d = sarsa_update(Q, 0, 1, 1.0, 2, 0, 0.5, 0.9, True)
check(approx(Q[0, 1], 1.3 + 0.5 * (1.0 - 1.3)), "Terminal transition: target is just R (no Q(S', A') term)",
      f"got Q(0,1) = {Q[0, 1]}, expected 1.15")

# ---- CliffWalking: SARSA finds the SAFE path
from rlenvs import CliffWalking

env = CliffWalking()
results = []
for seed in range(5):
    Q, rets = sarsa(env, 500, alpha=0.1, epsilon=0.1, seed=seed)
    G, path = greedy_rollout(env, Q)
    results.append((G, float(np.mean(rets[-100:]))))
check(isinstance(Q, np.ndarray) and Q.shape == (48, 4), "sarsa returns Q with shape (48, 4)", f"got {getattr(Q, 'shape', Q)}")
check(len(rets) == 500, "sarsa returns one total reward per episode", f"len(returns) = {len(rets)}")
safe = sum(-19 <= G <= -15 for G, _ in results)
check(safe >= 3, "Greedy policy reaches the goal by the safe route (return −15 … −19) for most seeds",
      f"greedy returns over 5 seeds: {[G for G, _ in results]}. −13 means you learned the cliff-edge path — "
      "that is Q-learning (max over A'), not SARSA (the A' you actually take). −100 means the greedy policy loops.")
online = [m for _, m in results]
check(np.mean(online) > -30, "Online return while exploring (last 100 episodes) is better than −30",
      f"mean online return per seed: {[round(m, 1) for m in online]}")
