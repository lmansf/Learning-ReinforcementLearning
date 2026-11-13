import numpy as np
import rlhud
from corerunner import CoreRunner, run_episode


class StepBudget:
    """Wraps an env and refuses to step more than `budget` times in total."""

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
            raise RuntimeError(f"Step budget of {self.budget} exhausted — check env.steps_left before stepping")
        self.used += 1
        return self.env.step(a)


def choose_action(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))


def solve_vent_maze(env, seed=0, alpha=0.5, gamma=0.99, n_planning=30, curve=None):
    """Learn a Q table for level 2 using at most env.steps_left real environment steps.

    Plain Q-learning is NOT sample-efficient enough for this budget. The environment is
    deterministic, so a learned model is perfect: remember (s, a) -> (r, s', terminated)
    and replay it (Dyna-Q). Append each finished episode's return to `curve` if given."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    # TODO: loop over episodes while env.steps_left > 0
    #   - inside an episode also stop when env.steps_left hits 0
    #   - ε-greedy action, real step, Q-learning update
    #   - store the transition in a model, then do n_planning simulated updates
    raise NotImplementedError
    return Q


def greedy_policy(Q):
    return lambda obs: int(np.argmax(Q[obs]))


if __name__ == "__main__":
    BUDGET = 1500
    for n_plan in (0, 10, 30):
        env = StepBudget(CoreRunner(level=2), BUDGET)
        curve = []
        Q = solve_vent_maze(env, seed=0, n_planning=n_plan, curve=curve)
        G, frames, _ = run_episode(CoreRunner(level=2), greedy_policy(Q), record=True)
        rlhud.plot(curve, label=f"n_planning={n_plan}", title="Vent Maze · return per episode (1,500-step budget)")
        rlhud.stat(f"Greedy return (n={n_plan})", G, good=G >= 46)
    rlhud.replay(frames, title="Greedy agent · Level 2")
