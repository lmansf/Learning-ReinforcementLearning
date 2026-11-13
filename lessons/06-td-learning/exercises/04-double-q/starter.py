import numpy as np
import rlhud


def double_q_update(Q1, Q2, s, a, r, s2, alpha, gamma, terminated, update_first):
    """One Double Q-learning update, in place.

    Q1, Q2 : dicts {state: np.array of action values}
    update_first=True : pick a* = argmax Q1[s2], evaluate it with Q2, update Q1[s][a]
    update_first=False: the mirror image (argmax with Q2, evaluate with Q1, update Q2[s][a])
    If terminated, the target is just r. Returns the TD error delta.
    """
    # TODO
    raise NotImplementedError


def double_q_learning(env, n_episodes, alpha=0.1, gamma=1.0, epsilon=0.1, seed=0):
    """Double Q-learning on `env`. Behaviour: epsilon-greedy on Q1 + Q2.
    Returns a list of booleans: did the agent go LEFT from state A in episode i?"""
    rng = np.random.default_rng(seed)
    Q1 = {s: np.zeros(env.n_actions(s)) for s in range(env.n_states)}
    Q2 = {s: np.zeros(env.n_actions(s)) for s in range(env.n_states)}
    went_left = []
    for _ in range(n_episodes):
        # TODO: act epsilon-greedily on Q1[s] + Q2[s]; record whether you went LEFT from env.A;
        #       step; double_q_update with a fair coin flip; stop on terminated or truncated
        raise NotImplementedError
    return went_left


# ---------------------------------------------------------------- provided
class MaximizationBiasMDP:
    """Sutton & Barto Example 6.7.

        terminal <--LEFT-- [B] <--LEFT-- [A] --RIGHT--> terminal (r = 0)
                  (r = 0)

    From A: RIGHT ends the episode with 0; LEFT moves to B with 0.
    From B: any of `n_b` actions ends the episode with reward ~ Normal(-0.1, 1).
    So LEFT is worse on average (−0.1 < 0) — but the max over noisy estimates in B looks positive.
    """
    A, B, T = 0, 1, 2
    LEFT, RIGHT = 0, 1
    n_states = 3

    def __init__(self, n_b=10, seed=None):
        self.n_b = n_b
        self.rng = np.random.default_rng(seed)

    def n_actions(self, s):
        return 2 if s == self.A else (self.n_b if s == self.B else 1)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = self.A
        return self.s, {}

    def step(self, a):
        if self.s == self.A:
            if a == self.LEFT:
                self.s = self.B
                return self.B, 0.0, False, False, {}
            self.s = self.T
            return self.T, 0.0, True, False, {}
        self.s = self.T
        return self.T, float(self.rng.normal(-0.1, 1.0)), True, False, {}


def epsilon_greedy(q, epsilon, rng):
    """q: array of action values for one state. Random ties."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(rng.choice(np.flatnonzero(q == q.max())))


def q_learning_bias(env, n_episodes, alpha=0.1, gamma=1.0, epsilon=0.1, seed=0):
    """Plain Q-learning on the same MDP, for comparison. Same return format."""
    rng = np.random.default_rng(seed)
    Q = {s: np.zeros(env.n_actions(s)) for s in range(env.n_states)}
    went_left = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        while True:
            a = epsilon_greedy(Q[s], epsilon, rng)
            if s == env.A:
                went_left.append(a == env.LEFT)
            s2, r, term, trunc, _ = env.step(a)
            target = r if term else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            if term or trunc:
                break
            s = s2
    return went_left


if __name__ == "__main__":
    RUNS, EPISODES = 500, 300
    curves = {"Q-learning": np.zeros(EPISODES), "Double Q-learning": np.zeros(EPISODES)}
    for run in range(RUNS):
        env = MaximizationBiasMDP(seed=run)
        curves["Q-learning"] += np.array(q_learning_bias(env, EPISODES, seed=run)) / RUNS
        curves["Double Q-learning"] += np.array(double_q_learning(env, EPISODES, seed=run)) / RUNS
        rlhud.progress(run, RUNS, "Runs")
    for name, c in curves.items():
        rlhud.plot(100 * c, label=name, title="% LEFT actions from A (avg of 500 runs)",
                   xlabel="episode", ylabel="% left")
        rlhud.stat(f"{name}: % left, episodes 1–50", round(100 * float(c[:50].mean()), 1))
    rlhud.plot([5.0] * EPISODES, label="optimal (ε/2 = 5 %)", title="% LEFT actions from A (avg of 500 runs)")
    print("Q-learning is fooled by max over noisy estimates; Double Q-learning is not.")
