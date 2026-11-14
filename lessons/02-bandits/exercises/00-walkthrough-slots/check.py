import numpy as np
from collections import Counter

SlotMachines = require("SlotMachines")
greedy_arm = require("greedy_arm")
GreedyAgent = require("GreedyAgent")
run_many = require("run_many")

# ---------------------------------------------------------------- the slot machines
env = SlotMachines([0.2, 0.9, 0.5], seed=0)
check(env.best_arm == 1, "SlotMachines knows its best arm",
      f"best_arm={env.best_arm} (expected 1)")
pays = [env.pull(1) for _ in range(4000)]
ok = set(pays) <= {0.0, 1.0} and abs(np.mean(pays) - 0.9) < 0.03
check(ok, "Arm 1 pays 1.0 about 90% of the time (and otherwise 0.0)",
      f"mean payout {np.mean(pays):.3f}, values seen {sorted(set(pays))}")
a = [SlotMachines([0.5] * 3, seed=7).pull(0) for _ in range(5)]
check(len(set(a)) == 1, "Same seed → same first pull (the generator is seeded)")

# ---------------------------------------------------------------- tie-breaking
rng = np.random.default_rng(0)
c = Counter(greedy_arm(np.array([1.0, 3.0, 3.0, 0.0]), rng) for _ in range(2000))
ok = set(c) == {1, 2} and 850 < c[1] < 1150
check(ok, "greedy_arm breaks ties at random among the best arms",
      f"counts {dict(c)} — use np.flatnonzero(Q == Q.max()) and rng.choice")
check(isinstance(greedy_arm(np.array([0.0, 1.0]), rng), int), "greedy_arm returns a plain int")

# ---------------------------------------------------------------- the agent
ag = GreedyAgent(3, q0=5.0, seed=0)
check(approx(ag.Q, [5.0, 5.0, 5.0]) and ag.N.dtype.kind == "i", "q0 fills Q; N is an int array",
      f"Q={ag.Q}, N dtype={ag.N.dtype}")
for r in [1.0, 0.0, 1.0, 1.0]:
    ag.update(2, r)
ok = approx(ag.Q[2], 0.75) and ag.N[2] == 4
check(ok, "Rewards 1, 0, 1, 1 → sample average 0.75, N = 4",
      f"got Q={ag.Q[2]:.4f}, N={ag.N[2]} — increment N before dividing by it")
check(approx(ag.Q[:2], [5.0, 5.0]), "Updating one arm leaves the others alone")

# ---------------------------------------------------------------- many runs
counts, totals = run_many(lambda s: GreedyAgent(3, q0=2.0, seed=s), [0.1, 0.2, 0.9], 20, 100)
counts, totals = np.asarray(counts), np.asarray(totals)
check(counts.shape == (20, 3) and totals.shape == (20,), "run_many returns (n_runs, k) counts and n_runs totals",
      f"got {counts.shape} and {totals.shape}")
check(approx(counts.sum(axis=1), np.full(20, 100)), "Each row of counts adds up to n_steps",
      f"row sums {counts.sum(axis=1)[:5]}…")
check(totals.mean() > 60, "Optimistic greedy finds the 90% arm on an easy problem",
      f"mean total reward {totals.mean():.1f} out of 100")
