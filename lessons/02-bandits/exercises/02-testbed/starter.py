import numpy as np
import rlhud
from rlenvs import KArmedBandit


class EpsilonGreedyAgent:
    """The agent from exercise 1, complete. alpha=None → sample averages."""

    def __init__(self, k, epsilon=0.1, alpha=None, q0=0.0, seed=None):
        self.k, self.epsilon, self.alpha = k, epsilon, alpha
        self.Q = np.full(k, float(q0))
        self.N = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(seed)

    def act(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(self.rng.choice(np.flatnonzero(self.Q == self.Q.max())))

    def update(self, a, r):
        self.N[a] += 1
        step = self.alpha if self.alpha is not None else 1.0 / self.N[a]
        self.Q[a] += step * (r - self.Q[a])


def run_testbed(make_agent, n_runs=200, n_steps=1000, k=10, seed=0, nonstationary=False):
    """Average an agent's learning curve over many independent bandit problems.

    For run i (0 … n_runs-1):
        bandit = KArmedBandit(k=k, seed=seed + i, nonstationary=nonstationary)
        agent  = make_agent(seed + i)
    then play n_steps:  a = agent.act(); r = bandit.pull(a); agent.update(a, r)

    Returns (avg_reward, pct_optimal): NumPy arrays of length n_steps.
        avg_reward[t]  = mean over runs of the reward received at step t
        pct_optimal[t] = 100 × fraction of runs whose action at step t was
                         bandit.optimal_action (checked *before* the pull)
    """
    rewards = np.zeros((n_runs, n_steps))
    optimal = np.zeros((n_runs, n_steps))
    # TODO: for each run, build the bandit and the agent exactly as described above,
    #       play n_steps, and fill rewards[i, t] and optimal[i, t]
    raise NotImplementedError
    return rewards.mean(axis=0), 100.0 * optimal.mean(axis=0)


def compare_epsilons(epsilons, n_runs=200, n_steps=1000, seed=0):
    """Run the testbed once per ε (sample-average agents, q0 = 0).

    Returns {epsilon: (avg_reward, pct_optimal)}.
    """
    results = {}
    for eps in epsilons:
        # TODO: results[eps] = run_testbed(...) with a make_agent that builds
        #       EpsilonGreedyAgent(10, epsilon=eps, seed=s)
        raise NotImplementedError
    return results


if __name__ == "__main__":
    runs = 100
    res = compare_epsilons([0.0, 0.01, 0.1], n_runs=runs, n_steps=1000)
    rows = []
    for eps, (rew, opt) in res.items():
        name = "greedy" if eps == 0 else f"ε = {eps}"
        rlhud.plot(rew, label=name, title=f"Average reward · {runs} runs", smooth=10)
        rlhud.plot(opt, label=name, title="% optimal action", smooth=10)
        rows.append([name, round(float(rew[-500:].mean()), 3), round(float(opt[-500:].mean()), 1)])

    # Optimism as an exploration strategy: greedy, but every arm starts at Q₀ = 5
    rew, opt = run_testbed(lambda s: EpsilonGreedyAgent(10, epsilon=0.0, alpha=0.1, q0=5.0, seed=s),
                           n_runs=runs, n_steps=1000)
    rlhud.plot(opt, label="optimistic greedy (Q₀=5, α=0.1)", title="% optimal action", smooth=10)
    rows.append(["optimistic Q₀=5", round(float(rew[-500:].mean()), 3), round(float(opt[-500:].mean()), 1)])
    rlhud.table(rows, headers=["agent", "reward (last 500)", "% optimal (last 500)"], title="Testbed summary")

    # Nonstationary: arms drift. Constant α keeps tracking; sample averages freeze.
    for alpha, name in [(None, "sample average"), (0.1, "constant α = 0.1")]:
        rew, opt = run_testbed(lambda s, a=alpha: EpsilonGreedyAgent(10, epsilon=0.1, alpha=a, seed=s),
                               n_runs=30, n_steps=3000, nonstationary=True)
        rlhud.plot(opt, label=name, title="Nonstationary bandit · % optimal", smooth=50)
