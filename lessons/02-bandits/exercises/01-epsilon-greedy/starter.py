import numpy as np
import rlhud
from rlenvs import KArmedBandit


def incremental_update(old, target, step_size):
    """NewEstimate ← OldEstimate + StepSize · [Target − OldEstimate]"""
    # TODO: one line
    raise NotImplementedError


def epsilon_greedy(Q, epsilon, rng):
    """With probability ε pick a uniformly random arm, otherwise a greedy one.

    Ties between equally good greedy arms are broken uniformly at random —
    np.argmax would always pick the lowest index, which silently biases the agent.
    rng is a numpy Generator: use rng.random(), rng.integers(n), rng.choice(arr).
    """
    Q = np.asarray(Q, dtype=float)
    # TODO: explore with probability epsilon, otherwise pick among the argmax arms at random
    raise NotImplementedError


class EpsilonGreedyAgent:
    """Action-value bandit agent.

    alpha=None  -> sample-average estimates (step size 1/N(a))
    alpha=0.1   -> constant step size (recency-weighted average)
    q0          -> initial estimate for every arm (set it high for optimism)
    """

    def __init__(self, k, epsilon=0.1, alpha=None, q0=0.0, seed=None):
        self.k, self.epsilon, self.alpha = k, epsilon, alpha
        self.Q = np.full(k, float(q0))
        self.N = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(seed)

    def act(self):
        return epsilon_greedy(self.Q, self.epsilon, self.rng)

    def update(self, a, r):
        # TODO: count the pull, choose the step size (1/N(a) or alpha),
        #       then move Q[a] toward r with incremental_update
        raise NotImplementedError


if __name__ == "__main__":
    bandit = KArmedBandit(k=10, seed=0)
    agent = EpsilonGreedyAgent(k=10, epsilon=0.1, seed=0)
    rewards, optimal = [], []
    for t in range(2000):
        a = agent.act()
        r = bandit.pull(a)
        agent.update(a, r)
        rewards.append(r)
        optimal.append(100.0 * (a == bandit.optimal_action))

    rlhud.plot(rewards, label="ε = 0.1", title="Reward per step", smooth=100)
    rlhud.plot(optimal, label="ε = 0.1", title="% optimal action", smooth=100)
    rlhud.bars(agent.Q, labels=[f"a{i}" for i in range(10)], title="Estimated Q(a)",
               highlight=int(np.argmax(agent.Q)))
    rlhud.bars(bandit.q_star, labels=[f"a{i}" for i in range(10)], title="True q*(a) (hidden from the agent)",
               highlight=bandit.optimal_action)
    rlhud.stat("Pulls of best arm", int(agent.N[bandit.optimal_action]))
    rlhud.stat("Mean reward", float(np.mean(rewards)))
