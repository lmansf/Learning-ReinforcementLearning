import numpy as np
from rlenvs import KArmedBandit

run_testbed = require("run_testbed")
compare_epsilons = require("compare_epsilons")
EpsilonGreedyAgent = require("EpsilonGreedyAgent")


class FixedArm:
    """Always pulls the same arm; records how it was built and updated."""
    log = []
    def __init__(self, seed, arm=0):
        self.seed, self.arm, self.updates = seed, arm, []
        FixedArm.log.append(self)
    def act(self):
        return self.arm
    def update(self, a, r):
        self.updates.append((a, r))

# ---------------------------------------------------------------- mechanics with a scripted agent
FixedArm.log = []
rew, opt = run_testbed(lambda s: FixedArm(s), n_runs=6, n_steps=20, seed=40)
rew, opt = np.asarray(rew, dtype=float), np.asarray(opt, dtype=float)
check(rew.shape == (20,) and opt.shape == (20,), "Returns two arrays of length n_steps",
      f"got shapes {rew.shape} and {opt.shape}")
seeds = [a.seed for a in FixedArm.log]
check(seeds == [40, 41, 42, 43, 44, 45], "make_agent is called once per run with seed + i",
      f"seeds passed to make_agent: {seeds}")
check(all(len(a.updates) == 20 for a in FixedArm.log) and all(u[0] == 0 for a in FixedArm.log for u in a.updates),
      "agent.update(a, r) is called after every step", f"update counts: {[len(a.updates) for a in FixedArm.log]}")

# Reproduce what the testbed should have measured, independently.
exp_r = np.zeros(20)
exp_o = 0.0
for i in range(6):
    b = KArmedBandit(k=10, seed=40 + i)
    exp_o += 100.0 * (b.optimal_action == 0) / 6
    exp_r += np.array([b.pull(0) for _ in range(20)]) / 6
ok = approx(rew, exp_r, 1e-9)
check(ok, "avg_reward[t] is the mean over runs of the reward at step t",
      "" if ok else f"first steps expected {np.round(exp_r[:3], 3)}, got {np.round(rew[:3], 3)}. "
                    "Build each bandit with KArmedBandit(k=k, seed=seed + i, nonstationary=nonstationary)")
ok = approx(opt, np.full(20, exp_o), 1e-9)
check(ok, "pct_optimal is a percentage (0–100) of runs that chose the optimal arm",
      "" if ok else f"expected {exp_o:.2f} at every step, got {np.round(opt[:3], 2)}…")

r1, o1 = run_testbed(lambda s: EpsilonGreedyAgent(10, 0.1, seed=s), n_runs=5, n_steps=50, seed=3)
r2, o2 = run_testbed(lambda s: EpsilonGreedyAgent(10, 0.1, seed=s), n_runs=5, n_steps=50, seed=3)
check(approx(r1, r2, 0) and approx(o1, o2, 0), "Same seed → identical results (fully reproducible)")

FixedArm.log = []
rn, _ = run_testbed(lambda s: FixedArm(s), n_runs=2, n_steps=30, seed=0, nonstationary=True)
b = KArmedBandit(k=10, seed=0, nonstationary=True)
first = [b.pull(0) for _ in range(30)]
check(approx(np.asarray(rn)[0] * 2 - first[0], KArmedBandit(k=10, seed=1, nonstationary=True).pull(0), 1e-9),
      "The nonstationary flag is passed on to the bandit")

# ---------------------------------------------------------------- the classic result
res = compare_epsilons([0.0, 0.1], n_runs=100, n_steps=1000, seed=7)
check(set(res) == {0.0, 0.1}, "compare_epsilons returns one entry per ε", f"keys {list(res)}")
g_r, g_o = map(np.asarray, res[0.0])
e_r, e_o = map(np.asarray, res[0.1])
check(e_o[-500:].mean() > g_o[-500:].mean() + 20,
      "ε = 0.1 picks the optimal arm far more often than greedy (steps 500–1000)",
      f"% optimal: ε=0.1 → {e_o[-500:].mean():.1f}%, greedy → {g_o[-500:].mean():.1f}%")
check(e_r[-500:].mean() > g_r[-500:].mean() + 0.15,
      "ε = 0.1 also earns more reward than greedy in the long run",
      f"mean reward: ε=0.1 → {e_r[-500:].mean():.3f}, greedy → {g_r[-500:].mean():.3f}")
check(g_r[:20].mean() > 0 and g_r[0] < g_r[-500:].mean(), "Both learning curves start low and improve",
      f"greedy step 0 → {g_r[0]:.3f}, late → {g_r[-500:].mean():.3f}")
