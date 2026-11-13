import numpy as np
from collections import Counter

incremental_update = require("incremental_update")
epsilon_greedy = require("epsilon_greedy")
EpsilonGreedyAgent = require("EpsilonGreedyAgent")

# ---------------------------------------------------------------- incremental update
v = incremental_update(2.0, 6.0, 0.25)
check(approx(v, 3.0), "incremental_update(2, 6, 0.25) == 3.0",
      "" if approx(v, 3.0) else f"got {v}. Old + α·(Target − Old) = 2 + 0.25·4")
check(approx(incremental_update(5.0, 1.0, 1.0), 1.0), "Step size 1 jumps straight to the target")
check(approx(incremental_update(5.0, 1.0, 0.0), 5.0), "Step size 0 leaves the estimate unchanged")

# ---------------------------------------------------------------- epsilon-greedy
rng = np.random.default_rng(0)
picks = {epsilon_greedy([0.1, 0.9, 0.3], 0.0, rng) for _ in range(200)}
check(picks == {1}, "ε = 0 always picks the highest estimate", f"picked {sorted(picks)}")

rng = np.random.default_rng(1)
c = Counter(epsilon_greedy([1.0, 5.0, 5.0, 0.0], 0.0, rng) for _ in range(2000))
ok = set(c) == {1, 2} and 800 < c[1] < 1200
check(ok, "Ties between greedy arms are broken at random",
      "" if ok else f"counts {dict(c)} — use np.flatnonzero(Q == Q.max()) and rng.choice, not np.argmax")

rng = np.random.default_rng(2)
c = Counter(epsilon_greedy([0.0, 10.0, 0.0, 0.0], 1.0, rng) for _ in range(4000))
check(set(c) == {0, 1, 2, 3} and all(800 < n < 1200 for n in c.values()),
      "ε = 1 picks uniformly among ALL arms", f"counts {dict(c)}")

rng = np.random.default_rng(3)
n = 20000
best = sum(epsilon_greedy([0.0, 0.0, 3.0, 0.0, 0.0], 0.2, rng) == 2 for _ in range(n)) / n
# P(best) = (1 − ε) + ε / k = 0.8 + 0.04
ok = abs(best - 0.84) < 0.015
check(ok, "ε = 0.2, k = 5 picks the greedy arm with probability 0.84",
      f"measured {best:.3f}" + ("" if ok else ". Exploration chooses among all k arms, including the greedy one"))

# ---------------------------------------------------------------- agent updates
ag = EpsilonGreedyAgent(k=3, epsilon=0.0, seed=0)
for r in [1.0, 2.0, 3.0, 6.0]:
    ag.update(0, r)
ok = approx(ag.Q[0], 3.0) and ag.N[0] == 4
check(ok, "Sample-average: rewards 1,2,3,6 → Q = 3.0, N = 4",
      "" if ok else f"got Q={ag.Q[0]:.4f}, N={ag.N[0]}. With alpha=None the step size must be 1/N(a) (after incrementing N)")
check(approx(ag.Q[1:], [0.0, 0.0]), "Updating one arm leaves the others alone", f"Q = {ag.Q}")

ag = EpsilonGreedyAgent(k=2, epsilon=0.0, alpha=0.5, seed=0)
for r in [4.0, 4.0]:
    ag.update(1, r)
check(approx(ag.Q[1], 3.0), "Constant α = 0.5: rewards 4, 4 from Q₀ = 0 → 2 → 3",
      f"got {ag.Q[1]:.4f}")

ag = EpsilonGreedyAgent(k=4, epsilon=0.0, q0=5.0, seed=0)
check(approx(ag.Q, [5, 5, 5, 5]), "q0 sets the initial estimates")
ag.update(2, 1.0)
ok = approx(ag.Q[2], 1.0)
check(ok, "With sample averages the first reward replaces q0 entirely",
      "" if ok else f"got {ag.Q[2]:.4f} — 1/N with N = 1 is a step size of 1")

# ---------------------------------------------------------------- behaviour on a real testbed
from rlenvs import KArmedBandit

def pct_optimal(eps, runs=40, steps=1000):
    hits = 0
    for i in range(runs):
        b = KArmedBandit(k=10, seed=100 + i)
        ag = EpsilonGreedyAgent(k=10, epsilon=eps, seed=i)
        for t in range(steps):
            a = ag.act()
            ag.update(a, b.pull(a))
            if t >= steps // 2:
                hits += a == b.optimal_action
    return hits / (runs * (steps - steps // 2))

p = pct_optimal(0.1)
check(p > 0.6, "ε = 0.1 agent finds the best arm on a 10-armed testbed",
      f"picked the optimal arm {100 * p:.0f}% of the time in steps 500–1000 (expected > 60%)")
