import numpy as np
import rlhud
from corerunner import CoreRunner, run_episode


def epsilon_by_episode(ep, episodes, eps_start=1.0, eps_end=0.05, decay_frac=0.6):
    """Linear decay over the first decay_frac of training, then constant eps_end."""
    # TODO
    raise NotImplementedError


def choose_action(Q, s, epsilon, rng):
    """ε-greedy with random tie-breaking."""
    # TODO
    raise NotImplementedError


def train_q(level=1, episodes=400, alpha=0.5, gamma=0.99,
            eps_start=1.0, eps_end=0.05, decay_frac=0.6, seed=0):
    """Tabular Q-learning. Returns (Q, list_of_episode_returns)."""
    rng = np.random.default_rng(seed)
    env = CoreRunner(level=level)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for ep in range(episodes):
        # TODO: one episode of Q-learning
        #   - pick epsilon for this episode
        #   - reset, then loop: choose, step, update Q[s, a], accumulate the return
        #   - stop on terminated or truncated
        raise NotImplementedError
    return Q, returns


def greedy_policy(Q):
    # TODO: return a function obs -> argmax action
    raise NotImplementedError


if __name__ == "__main__":
    Q, returns = train_q(level=1, episodes=400)
    rlhud.plot(returns, label="Q-learning", title="Boot Sequence · training return", smooth=20)
    env = CoreRunner(level=1)
    G, frames, info = run_episode(env, greedy_policy(Q), record=True)
    rlhud.stat("Greedy return", G, good=G >= 28)
    rlhud.stat("Par", 28.0)
    rlhud.stat("States visited", int((np.abs(Q).sum(1) > 0).sum()))
    rlhud.replay(frames, title="Greedy agent · Level 1")
