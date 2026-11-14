import numpy as np
from rlenvs import GridWorld

egreedy_probs = require("egreedy_probs")
expected_sarsa_update = require("expected_sarsa_update")
train = require("train")

# ---------------------------------------------------------------- policy probabilities
p = np.asarray(egreedy_probs(np.array([0.0, 3.0, 1.0, 2.0]), 0.2), dtype=float)
check(approx(p, [0.05, 0.85, 0.05, 0.05]), "ε-greedy probabilities for ε = 0.2, k = 4",
      f"expected [0.05, 0.85, 0.05, 0.05], got {np.round(p, 4).tolist()} — ε/k each, +1−ε on the greedy")
check(approx(p.sum(), 1.0), "The probabilities sum to 1")

# ---------------------------------------------------------------- one update by hand
Q = np.zeros((3, 2))
Q[1] = [2.0, 4.0]
d = expected_sarsa_update(Q, 0, 1, 1.0, 1, False, 0.5, 0.9, 0.5)
# E[Q(1, ·)] = 0.25·2 + 0.75·4 = 3.5 → target 1 + 0.9·3.5 = 4.15
check(approx(d, 4.15), "δ = r + γ·E[Q(S', ·)] − Q(S, A)", f"expected 4.15, got {d}")
check(approx(Q[0, 1], 0.5 * 4.15) and approx(Q[0, 0], 0.0) and approx(Q[1], [2.0, 4.0]),
      "Only Q[s, a] moves, by α·δ", f"Q is now {np.round(Q, 4).tolist()}")
Q = np.zeros((3, 2))
Q[2] = [100.0, 100.0]
d = expected_sarsa_update(Q, 0, 0, -1.0, 2, True, 1.0, 0.9, 0.1)
check(approx(d, -1.0) and approx(Q[0, 0], -1.0), "A terminal S' contributes no bootstrap (target = r)",
      f"got δ = {d} — ignore Q[s2] when terminated")

# ---------------------------------------------------------------- learning
env = GridWorld(["S...", ".XX.", "...G"], step_reward=-1.0, rewards={"G": 0.0, "X": -20.0})
Q, rets = train(env, 300, alpha=0.5, epsilon=0.1, seed=0)
Q = np.asarray(Q)
check(Q.shape == (env.n_states, 4) and len(rets) == 300, "train returns a (n_states, n_actions) table and one return per episode",
      f"Q {Q.shape}, {len(rets)} returns")
s, _ = env.reset()
G = 0.0
for _ in range(20):
    s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
    G += r
    if term:
        break
check(term and G == -4.0, "The greedy policy walks the shortest safe path (return −4)",
      f"greedy return {G}, reached the goal: {term}")
check(np.mean(rets[-50:]) > np.mean(rets[:20]), "Returns improve with training",
      f"first 20 episodes {np.mean(rets[:20]):.1f}, last 50 {np.mean(rets[-50:]):.1f}")
