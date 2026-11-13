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
    """Dyna-Q: learn from each real step, then replay n_planning imagined steps from a learned model.

    env has a hard step budget (env.steps_left). Returns the Q table."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    model = {}          # (s, a) -> (r, s', terminated) — the level is deterministic
    seen = []           # list of (s, a) keys for O(1) random sampling
    budget = env.steps_left
    while env.steps_left > 0:
        s, _ = env.reset()
        G = 0.0
        while env.steps_left > 0:
            # ε decays with the fraction of the budget used, not with episodes
            eps = max(0.05, 1.0 - (budget - env.steps_left) / (0.5 * budget))
            a = choose_action(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            G += r
            Q[s, a] += alpha * ((r if term else r + gamma * Q[s2].max()) - Q[s, a])
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, term)
            for _ in range(n_planning):      # planning: free updates from the model
                ps, pa = seen[rng.integers(len(seen))]
                pr, ps2, pterm = model[(ps, pa)]
                Q[ps, pa] += alpha * ((pr if pterm else pr + gamma * Q[ps2].max()) - Q[ps, pa])
            s = s2
            if term or trunc:
                if curve is not None:
                    curve.append(G)
                break
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
