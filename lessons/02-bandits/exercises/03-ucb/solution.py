import numpy as np
import rlhud
from rlenvs import KArmedBandit


def ucb_scores(Q, N, t, c):
    """Upper confidence bound of every arm:  Q(a) + c · sqrt( ln t / N(a) ).

    Arms that were never tried (N = 0) get +inf: they are maximally uncertain.
    t is the number of the current step, counting from 1.
    """
    Q = np.asarray(Q, dtype=float)
    N = np.asarray(N, dtype=float)
    scores = np.full(len(Q), np.inf)
    tried = N > 0
    scores[tried] = Q[tried] + c * np.sqrt(np.log(t) / N[tried])
    return scores


def ucb_action(Q, N, t, c):
    """The arm with the highest upper bound (ties → lowest index, like np.argmax)."""
    return int(np.argmax(ucb_scores(Q, N, t, c)))


def run_ucb(bandit, steps, c=2.0):
    """Play `steps` pulls with UCB and sample-average estimates.

    Returns (rewards, optimal) — two lists of length `steps`, where optimal[i]
    is True if the arm chosen at step i was the bandit's optimal arm.
    """
    k = bandit.k
    Q, N = np.zeros(k), np.zeros(k)
    rewards, optimal = [], []
    for t in range(1, steps + 1):
        a = ucb_action(Q, N, t, c)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
        optimal.append(a == bandit.optimal_action)
    return rewards, optimal


def run_eps_greedy(bandit, steps, eps=0.1, seed=0):
    """Baseline for comparison (already written for you)."""
    rng = np.random.default_rng(seed)
    k = bandit.k
    Q, N = np.zeros(k), np.zeros(k)
    rewards, optimal = [], []
    for t in range(steps):
        a = int(rng.integers(k)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q == Q.max())))
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
        optimal.append(a == bandit.optimal_action)
    return rewards, optimal


if __name__ == "__main__":
    runs, steps = 100, 1000
    curves = {"UCB c=2": np.zeros(steps), "ε-greedy 0.1": np.zeros(steps)}
    for i in range(runs):
        r, _ = run_ucb(KArmedBandit(k=10, seed=i), steps, c=2.0)
        curves["UCB c=2"] += np.array(r) / runs
        r, _ = run_eps_greedy(KArmedBandit(k=10, seed=i), steps, eps=0.1, seed=i)
        curves["ε-greedy 0.1"] += np.array(r) / runs
        rlhud.progress(i, runs, "Testbed runs")
    for name, y in curves.items():
        rlhud.plot(y, label=name, title=f"Average reward ({runs} runs)", smooth=10)
        rlhud.stat(f"{name} · mean reward", float(y.mean()))

    # What UCB "thinks" halfway through one run
    bandit = KArmedBandit(k=10, seed=0)
    Q, N = np.zeros(10), np.zeros(10)
    for t in range(1, 201):
        a = ucb_action(Q, N, t, 2.0)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
    rlhud.bars(ucb_scores(Q, N, 200, 2.0), labels=[f"a{i}" for i in range(10)],
               title="UCB scores after 200 steps", highlight=bandit.optimal_action)
    rlhud.bars(N, labels=[f"a{i}" for i in range(10)], title="Pull counts after 200 steps",
               highlight=bandit.optimal_action)
