import time
import numpy as np

q_values = require("q_values")
epsilon_greedy = require("epsilon_greedy")
sarsa_update = require("sarsa_update")

w = np.arange(12, dtype=float).reshape(3, 4)          # 3 actions, 4 features
q = np.asarray(q_values(w, np.array([0, 2])))
check(q.shape == (3,) and approx(q, [2, 10, 18]), "q_values sums the active weights of each action",
      f"w = arange(12).reshape(3,4), idx=[0,2] → expected [2, 10, 18], got {q}")

rng = np.random.default_rng(0)
picks = [epsilon_greedy(np.array([0.0, 5.0, 1.0]), 0.0, rng) for _ in range(50)]
check(set(picks) == {1}, "ε = 0 → always the greedy action", f"picked {sorted(set(picks))}")
picks = [epsilon_greedy(np.array([3.0, 3.0, 1.0]), 0.0, rng) for _ in range(200)]
check(set(picks) == {0, 1}, "Ties are broken at random (matters with all-zero initial weights!)",
      f"with q=[3,3,1] you picked {sorted(set(picks))}; use rng.choice(np.flatnonzero(q == q.max()))")
picks = np.array([epsilon_greedy(np.array([0.0, 5.0, 1.0]), 0.3, rng) for _ in range(6000)])
frac = np.mean(picks == 1)
check(abs(frac - (0.7 + 0.1)) < 0.03, "ε = 0.3 → greedy action ≈ 80% of the time (70% + its share of random)",
      f"greedy fraction = {frac:.3f}")

w = np.zeros((3, 4)); w[2, [1, 3]] = [1.0, 2.0]
d = sarsa_update(w, np.array([0, 1]), 0, -1.0, np.array([1, 3]), 2, False, alpha=0.5, gamma=0.9)
check(approx(d, -1 + 0.9 * 3), "SARSA TD error δ = r + γ q̂(S′,A′) − q̂(S,A)", f"expected 1.7, got {d}")
check(approx(w[0], [0.85, 0.85, 0, 0]) and approx(w[2], [0, 1, 0, 2]),
      "Only the active weights of the taken action change", f"w after update:\n{w}")
w = np.ones((3, 4))
d = sarsa_update(w, np.array([2, 3]), 1, -1.0, None, None, True, alpha=0.25)
check(approx(d, -3) and approx(w[1], [1, 1, 0.25, 0.25]), "Terminal step: target is just r",
      f"q̂(S,A)=2 so δ = −1 − 2 = −3; got δ={d}, w[1]={w[1]}")

t0 = time.time()
_, tc, steps = train(episodes=150, seed=0)
dt = time.time() - t0
early, late = float(np.mean(steps[:5])), float(np.mean(steps[-10:]))
check(late < 200, "After 150 episodes the car reaches the flag in < 200 steps on average (last 10 episodes)",
      f"mean of last 10 episodes = {late:.0f} steps (first 5 episodes: {early:.0f}). "
      "Check the bootstrap on truncation and that α is split across tilings.")
check(late < early / 2, "Learning curve goes down", f"first 5: {early:.0f}, last 10: {late:.0f}")
check(dt < 30, "Training is reasonably fast", f"took {dt:.1f}s")
