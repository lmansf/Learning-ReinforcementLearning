import numpy as np

ucb_scores = require("ucb_scores")
ucb_action = require("ucb_action")
run_ucb = require("run_ucb")
run_eps_greedy = require("run_eps_greedy")

# ---------------------------------------------------------------- the bound itself
s = np.asarray(ucb_scores([1.0, 0.5], [10, 1], 11, 1.0), dtype=float)
expected = [1.0 + np.sqrt(np.log(11) / 10), 0.5 + np.sqrt(np.log(11) / 1)]
ok = approx(s, expected, 1e-6)
check(ok, "Scores for Q=[1, 0.5], N=[10, 1], t=11, c=1",
      "" if ok else f"expected {np.round(expected, 4)}, got {np.round(s, 4)}. Bound = Q + c·sqrt(ln t / N) (natural log)")
a = ucb_action([1.0, 0.5], [10, 1], 11, 1.0)
check(a == 1, "The rarely tried arm wins despite a lower estimate", f"picked arm {a}")
a = ucb_action([1.0, 0.5], [10, 1], 11, 0.0)
check(a == 0, "With c = 0 UCB is plain greedy", f"picked arm {a}")
s = np.asarray(ucb_scores([3.0, 0.0, 1.0], [5, 0, 2], 8, 2.0), dtype=float)
check(np.isinf(s[1]) and s[1] > 0 and np.isfinite(s[0]) and np.isfinite(s[2]),
      "Untried arms score +inf (and only those)", f"got {s}")
check(ucb_action([3.0, 0.0, 1.0], [5, 0, 2], 8, 2.0) == 1, "An untried arm is always chosen first")
s1 = np.asarray(ucb_scores([0.0, 0.0], [4, 4], 10, 1.0), dtype=float)
s2 = np.asarray(ucb_scores([0.0, 0.0], [4, 4], 1000, 1.0), dtype=float)
check(s2[0] > s1[0], "The bonus of an un-pulled arm grows (slowly) with time t",
      f"t=10 → {s1[0]:.3f}, t=1000 → {s2[0]:.3f}")


# ---------------------------------------------------------------- the loop
class Recorder:
    """Deterministic fake bandit: arm a always pays a. Records the pulls."""
    k = 4
    optimal_action = 3
    def __init__(self):
        self.pulls = []
    def pull(self, a):
        self.pulls.append(int(a))
        return float(a)

b = Recorder()
rewards, optimal = run_ucb(b, 30, c=1.0)
check(len(rewards) == 30 and len(optimal) == 30, "run_ucb returns one reward and one flag per step",
      f"got {len(rewards)} rewards, {len(optimal)} flags")
ok = b.pulls[:4] == [0, 1, 2, 3]
check(ok, "The first k pulls try every arm once (0, 1, 2, 3)",
      "" if ok else f"first pulls: {b.pulls[:4]}. Did you start t at 1 and give untried arms +inf?")
check(list(rewards) == [float(a) for a in b.pulls], "rewards[i] is what the bandit paid at step i")
check(list(map(bool, optimal)) == [a == 3 for a in b.pulls], "optimal[i] flags pulls of the optimal arm")
ok = b.pulls.count(3) > 15
check(ok, "UCB settles on the best arm of the fake bandit",
      f"arm 3 pulled {b.pulls.count(3)}/30 times" + ("" if ok else " — are you updating Q and N after each pull?"))

# ---------------------------------------------------------------- testbed comparison
from rlenvs import KArmedBandit
runs, steps = 50, 1000
u = e = 0.0
for i in range(runs):
    r, _ = run_ucb(KArmedBandit(k=10, seed=500 + i), steps, c=2.0)
    u += np.mean(r) / runs
    r, _ = run_eps_greedy(KArmedBandit(k=10, seed=500 + i), steps, eps=0.1, seed=i)
    e += np.mean(r) / runs
check(u > e + 0.02, "UCB (c = 2) earns more than ε-greedy (ε = 0.1) over 1000 steps",
      f"mean reward UCB {u:.3f} vs ε-greedy {e:.3f} over {runs} seeded 10-armed bandits")
