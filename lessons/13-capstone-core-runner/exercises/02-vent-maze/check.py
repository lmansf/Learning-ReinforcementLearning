import numpy as np
from corerunner import CoreRunner, run_episode

solve_vent_maze = require("solve_vent_maze")
greedy_policy = require("greedy_policy")


class _Budget:
    """The checker's own budget wrapper (your StepBudget class is not trusted)."""
    def __init__(self, env, budget):
        self.env, self.budget, self.used = env, budget, 0
        self.n_states, self.n_actions = env.n_states, env.n_actions
    @property
    def steps_left(self):
        return self.budget - self.used
    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    def step(self, a):
        if self.used >= self.budget:
            raise RuntimeError("step budget exhausted — your loop must stop when env.steps_left == 0")
        self.used += 1
        return self.env.step(a)


BUDGET = 1500
results = []
for seed in (0, 1, 2):
    env = _Budget(CoreRunner(level=2), BUDGET)
    try:
        Q = solve_vent_maze(env, seed=seed)
    except NotImplementedError:
        raise
    except RuntimeError as e:
        check(False, f"Seed {seed}: stays within the {BUDGET}-step budget", str(e))
        continue
    check(env.used <= BUDGET, f"Seed {seed}: used {env.used} ≤ {BUDGET} real steps")
    check(env.used >= BUDGET * 0.5, f"Seed {seed}: actually uses the budget", f"only {env.used} steps used — keep training while steps_left > 0")
    G, _, info = run_episode(CoreRunner(level=2), greedy_policy(Q))
    results.append(G)
    check(G >= 46.0, f"Seed {seed}: greedy return ≥ 46 (par)",
          f"greedy return {G:.1f}, episode ended with {info['event']!r}. Plain Q-learning cannot do this in {BUDGET} steps — add planning with a learned model.")
