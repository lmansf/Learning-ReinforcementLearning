import numpy as np

q_learning_update = require("q_learning_update")
q_learning = require("q_learning")
sarsa = require("sarsa")
greedy_rollout = require("greedy_rollout")

# ---- q_learning_update (hand-computed)
Q = np.zeros((3, 2))
Q[2] = [4.0, -2.0]
d = q_learning_update(Q, 0, 1, -1.0, 2, 0.5, 0.9, False)
check(approx(Q[0, 1], 1.3) and approx(d, 2.6),
      "q_learning_update: target −1 + 0.9·max(4, −2) = 2.6 → Q(0,1) = 1.3, δ = 2.6",
      f"got Q(0,1) = {Q[0, 1]}, δ = {d}")
Q = np.zeros((3, 2))
Q[2] = [-3.0, 1.0]
q_learning_update(Q, 0, 0, 0.0, 2, 1.0, 1.0, False)
check(approx(Q[0, 0], 1.0), "Bootstraps from the BEST next action, whichever you take next",
      f"Q(2) = [−3, 1], α = γ = 1, r = 0 → Q(0,0) should be 1, got {Q[0, 0]}")
Q = np.zeros((3, 2))
Q[2] = [5.0, 5.0]
q_learning_update(Q, 0, 0, 2.0, 2, 0.5, 0.9, True)
check(approx(Q[0, 0], 1.0), "Terminal transition: target is just R", f"expected 0.5·2 = 1, got {Q[0, 0]}")

# ---- CliffWalking
from rlenvs import CliffWalking

env = CliffWalking()
greedy, q_online, s_online = [], [], []
for seed in range(3):
    Q, rets = q_learning(env, 500, alpha=0.5, epsilon=0.1, seed=seed)
    G, path = greedy_rollout(env, Q)
    greedy.append(G)
    q_online.append(float(np.mean(rets)))
    _, srets = sarsa(env, 500, alpha=0.5, epsilon=0.1, seed=seed)
    s_online.append(float(np.mean(srets)))
check(isinstance(Q, np.ndarray) and Q.shape == (48, 4) and len(rets) == 500,
      "q_learning returns (Q of shape (48, 4), list of 500 episode returns)",
      f"got Q shape {getattr(Q, 'shape', None)}, {len(rets)} returns")
check(sum(G == -13 for G in greedy) >= 2, "Greedy policy is optimal: the cliff-edge path, return −13",
      f"greedy returns over 3 seeds: {greedy}. Q-learning's target uses max over Q(S', ·); −15/−17 suggests "
      "you are bootstrapping from the action actually taken (that is SARSA).")
check(np.mean(q_online) < np.mean(s_online) - 5,
      "…yet its online return while exploring is WORSE than SARSA's",
      f"mean online return: Q-learning {np.mean(q_online):.1f}, SARSA {np.mean(s_online):.1f}. "
      "Q-learning should behave ε-greedily (ε = 0.1) during training.")
