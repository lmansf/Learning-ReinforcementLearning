import numpy as np
import rlhud


# ---------------------------------------------------------------- your code
def v_hat(w, x):
    """Linear value estimate: v̂(s, w) = wᵀ x(s)."""
    # TODO: one line
    raise NotImplementedError


def gradient_mc_update(w, xs, rewards, alpha, gamma=1.0):
    """Gradient Monte Carlo for ONE finished episode. Updates w in place and returns it.

    xs      : feature vectors [x(S0), x(S1), ..., x(S_{T-1})]
    rewards : [R1, R2, ..., RT]   (rewards[t] follows the action taken in S_t)
    For every t:  w ← w + α [Gₜ − v̂(Sₜ, w)] ∇v̂(Sₜ, w),  and for a linear v̂, ∇v̂ = x(Sₜ).
    """
    # TODO: 1) compute the return Gₜ for every t (walk backwards: G = r + γ G)
    #       2) for each (x, Gₜ): nudge w toward Gₜ along x
    raise NotImplementedError


def td0_update(w, x, r, x_next, terminated, alpha, gamma=1.0):
    """Semi-gradient TD(0) for one transition S --r--> S'. Updates w in place, returns δ.

    target = r                      if the episode terminated at S'
             r + γ v̂(S', w)         otherwise          (no gradient flows through the target)
    """
    # TODO: compute the target, the TD error δ = target − v̂(S, w), then w += α δ x
    raise NotImplementedError


# ---------------------------------------------------------------- provided: the 1000-state random walk
class RandomWalk1000:
    """States 0..999, start in the middle. Each step jumps 1..100 states left or right
    (uniformly). Falling off the left edge ends the episode with reward −1, the right edge +1.
    All other rewards are 0. Sutton & Barto, Example 9.1."""

    n_states = 1000

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.s = 499
        return self.s

    def step(self):
        jump = int(self.rng.integers(1, 101)) * (1 if self.rng.random() < 0.5 else -1)
        s = self.s + jump
        if s < 0:
            return None, -1.0, True
        if s >= self.n_states:
            return None, 1.0, True
        self.s = s
        return s, 0.0, False


def aggregate(s, n_groups=10, n_states=1000):
    """State aggregation: a one-hot vector saying which block of states s falls in."""
    x = np.zeros(n_groups)
    x[s * n_groups // n_states] = 1.0
    return x


def run_gradient_mc(episodes=4000, alpha=0.0005, seed=0):
    env, w = RandomWalk1000(seed), np.zeros(10)
    for _ in range(episodes):
        s, done, xs, rs = env.reset(), False, [], []
        while not done:
            xs.append(aggregate(s))
            s, r, done = env.step()
            rs.append(r)
        gradient_mc_update(w, xs, rs, alpha)
    return w


def run_td0(episodes=2000, alpha=0.01, seed=0):
    env, w = RandomWalk1000(seed), np.zeros(10)
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            s2, r, done = env.step()
            td0_update(w, aggregate(s), r, None if done else aggregate(s2), done, alpha)
            s = s2
    return w


def true_values():
    """Exact v(s) for the random walk under its (only) policy, by solving v = r + P v."""
    n = RandomWalk1000.n_states
    P, r = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for jump in range(1, 101):
            for s2 in (s + jump, s - jump):
                if s2 < 0:
                    r[s] -= 1 / 200
                elif s2 >= n:
                    r[s] += 1 / 200
                else:
                    P[s, s2] += 1 / 200
    return np.linalg.solve(np.eye(n) - P, r)


if __name__ == "__main__":
    v_true = true_values()
    w_mc = run_gradient_mc()
    w_td = run_td0()
    states = range(0, 1000, 10)
    rlhud.plot([v_true[s] for s in states], label="true v(s)", title="Random walk: value vs state (every 10th state)")
    rlhud.plot([v_hat(w_mc, aggregate(s)) for s in states], label="gradient MC", title="Random walk: value vs state (every 10th state)")
    rlhud.plot([v_hat(w_td, aggregate(s)) for s in states], label="semi-gradient TD(0)", title="Random walk: value vs state (every 10th state)")
    rmse = lambda w: float(np.sqrt(np.mean([(v_hat(w, aggregate(s)) - v_true[s]) ** 2 for s in range(1000)])))
    rlhud.stat("RMSE gradient MC", rmse(w_mc))
    rlhud.stat("RMSE TD(0)", rmse(w_td))
    rlhud.bars(w_mc, labels=[f"{100*i}-{100*i+99}" for i in range(10)], title="Gradient MC weights (one per group)")
