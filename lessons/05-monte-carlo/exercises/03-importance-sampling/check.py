import random

import numpy as np

is_estimates = require("is_estimates")

# ---- hand-computed case
# target: always action 1.  behaviour: 50/50.
pi = lambda a, s: 1.0 if a == 1 else 0.0
b = lambda a, s: 0.5
ep1 = [("s", 1, 0.0), ("t", 1, 2.0)]   # rho = 2·2 = 4, G = 2
ep2 = [("s", 0, 5.0)]                  # rho = 0 (target never takes action 0)
ep3 = [("s", 1, 1.0)]                  # rho = 2, G = 1

o, w = is_estimates([ep1], pi, b)
check(len(o) == 1 and len(w) == 1, "Returns two lists with one estimate per episode", f"got {o!r}, {w!r}")
check(approx(o[0], 8.0), "Ordinary IS after ep1: ρG/1 = 4·2 = 8",
      f"got {o[0]}. ρ is the PRODUCT of π(a|s)/b(a|s) over every step of the episode.")
check(approx(w[0], 2.0), "Weighted IS after ep1: ρG/ρ = 2 (just the observed return)", f"got {w[0]}")

o, w = is_estimates([ep1, ep2, ep3], pi, b)
check(approx(o, [8.0, 4.0, 10 / 3]), "Ordinary IS: [8, 4, 10/3] — divides by the number of episodes",
      f"got {np.round(o, 4).tolist()}. An episode with ρ = 0 still counts in the denominator.")
check(approx(w, [2.0, 2.0, 10 / 6]), "Weighted IS: [2, 2, 1.667] — divides by the sum of ratios",
      f"got {np.round(w, 4).tolist()}. (4·2 + 0 + 2·1) / (4 + 0 + 2) = 10/6.")

o, w = is_estimates([ep2, ep1], pi, b)
check(approx(w[0], 0.0) and approx(w[1], 2.0), "Weighted IS is 0.0 while all ratios are 0 (no division by zero)",
      f"got {w}")

o, w = is_estimates([ep1], pi, b, gamma=0.5)
check(approx(o[0], 4.0) and approx(w[0], 1.0), "γ = 0.5 discounts the second reward: G = 0 + 0.5·2 = 1",
      f"got ordinary {o[0]}, weighted {w[0]}")

# ---- Blackjack, Sutton & Barto Example 5.4
from rlenvs import Blackjack


def play(env, policy):
    env.reset()
    env.player = [1, 2]
    env.dealer = [2, env._card()]
    obs, episode = env._obs(), []
    while True:
        a = policy(obs)
        nxt, r, term, trunc, _ = env.step(a)
        episode.append((obs, a, r))
        if term or trunc:
            return episode
        obs = nxt


target = lambda a, s: 1.0 if a == (0 if s[0] >= 20 else 1) else 0.0
behaviour = lambda a, s: 0.5
TRUE = -0.27726

random.seed(1)
env = Blackjack(seed=1)
RUNS, N = 30, 1000
se_o, se_w = np.zeros(N), np.zeros(N)
in_range = True
for _ in range(RUNS):
    eps = [play(env, lambda s: random.randrange(2)) for _ in range(N)]
    o, w = is_estimates(eps, target, behaviour)
    o, w = np.asarray(o, dtype=float), np.asarray(w, dtype=float)
    in_range &= bool(np.all((w >= -1 - 1e-9) & (w <= 1 + 1e-9)))
    se_o += (o - TRUE) ** 2
    se_w += (w - TRUE) ** 2
mse_o, mse_w = se_o / RUNS, se_w / RUNS
check(in_range, "Weighted estimates stay within the range of possible returns [−1, 1]",
      "a weighted average of returns can never leave their range — check the denominator is Σρ")
check(mse_w[-1] < 0.03 and mse_o[-1] < 0.03,
      "Both estimators approach v_π = −0.277 after 1000 episodes (MSE < 0.03)",
      f"MSE ordinary {mse_o[-1]:.4f}, weighted {mse_w[-1]:.4f}")
check(mse_w[:100].mean() < 0.75 * mse_o[:100].mean(),
      "Weighted IS has much lower error than ordinary IS in the first 100 episodes",
      f"mean MSE over episodes 1–100: ordinary {mse_o[:100].mean():.3f}, weighted {mse_w[:100].mean():.3f}")
