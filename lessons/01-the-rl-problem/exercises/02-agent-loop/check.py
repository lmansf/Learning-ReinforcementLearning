run_episode = require("run_episode")
random_policy = require("random_policy")


class ScriptedEnv:
    """Tiny fake env: rewards 1,2,3 then terminates. Records what the agent did."""
    def __init__(self, trunc_at=None):
        self.trunc_at = trunc_at
    def reset(self, seed=None):
        self.t = 0
        self.seed_seen = seed
        self.actions = []
        return 100, {}
    def step(self, a):
        self.actions.append(a)
        self.t += 1
        term = self.t >= 3 and self.trunc_at is None
        trunc = self.trunc_at is not None and self.t >= self.trunc_at
        return 100 + self.t, float(self.t), term, trunc, {}


env = ScriptedEnv()
seen = []
total, steps = run_episode(env, lambda o: (seen.append(o), 7)[1], seed=42)
check(approx(total, 6) and steps == 3, "Sums rewards until terminated (1+2+3 = 6 in 3 steps)",
      f"got total={total}, steps={steps}")
check(seen == [100, 101, 102], "Policy is called with the latest observation each step",
      f"policy saw {seen}")
check(env.seed_seen == 42, "Passes `seed` to env.reset", f"reset got seed={env.seed_seen}")

env = ScriptedEnv(trunc_at=2)
total, steps = run_episode(env, lambda o: 0)
check(steps == 2, "Also stops on truncated", f"ran {steps} steps")

import random
from collections import Counter
random.seed(0)
counts = Counter(random_policy(None) for _ in range(5000))
check(set(counts) == {0, 1, 2, 3, 4}, "random_policy uses all five actions", f"saw {sorted(counts)}")
check(all(800 < n < 1200 for n in counts.values()), "random_policy is roughly uniform", str(dict(counts)))

from corerunner import CoreRunner
G, n = run_episode(CoreRunner(level=1), random_policy, seed=0)
check(isinstance(n, int) and 1 <= n <= 40, "Runs a real Core Runner episode", f"steps={n}")
