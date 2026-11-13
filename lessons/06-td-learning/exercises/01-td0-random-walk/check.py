import random

import numpy as np

td_error = require("td_error")
td0 = require("td0")
RandomWalk = require("RandomWalk")
constant_alpha_mc = require("constant_alpha_mc")

# ---- td_error
check(approx(td_error(1.0, 0.5, 2.0, 0.9, False), 1.0 + 0.9 * 2.0 - 0.5),
      "td_error = r + γ·V(S') − V(S)", f"td_error(1, 0.5, 2, 0.9, False) gave {td_error(1.0, 0.5, 2.0, 0.9, False)}, expected 2.3")
check(approx(td_error(1.0, 0.5, 2.0, 0.9, True), 0.5),
      "A terminal S' contributes no bootstrap: δ = r − V(S)",
      f"got {td_error(1.0, 0.5, 2.0, 0.9, True)}, expected 0.5 — ignore v_next when terminated")


class Chain:
    """Deterministic chain 0 → 1 → 2 → 3 (terminal). Reward 1 on the last step, 0 otherwise."""
    n_states, n_actions = 4, 1
    def reset(self, seed=None):
        self.s = 0
        return 0, {}
    def step(self, a):
        self.s += 1
        return self.s, float(self.s == 3), self.s == 3, False, {}


pol = lambda s: 0
V = td0(Chain(), pol, 1, alpha=0.5, gamma=1.0, init=0.0)
check(isinstance(V, np.ndarray) and V.shape == (4,), "td0 returns a numpy array of length env.n_states",
      f"got {type(V).__name__} {getattr(V, 'shape', '')}")
check(approx(V, [0, 0, 0.5, 0]), "Chain, 1 episode, α=0.5: only the last state learns → V = [0, 0, 0.5, 0]",
      f"got {np.round(V, 4).tolist()}. TD updates after every step using the CURRENT estimate of V(S'); "
      "if V(0) or V(1) moved, you are using Monte Carlo returns.")
V = td0(Chain(), pol, 2, alpha=0.5, gamma=1.0, init=0.0)
check(approx(V, [0, 0.25, 0.75, 0]), "Chain, 2 episodes: the reward propagates back one state per episode → [0, 0.25, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}")
V = td0(Chain(), pol, 1, alpha=0.5, gamma=1.0, init=0.5)
check(approx(V[:3], [0.5, 0.5, 0.75]), "init=0.5: the terminal state is NOT bootstrapped from → V(2) = 0.5 + 0.5·(1 − 0.5) = 0.75",
      f"got {np.round(V[:3], 4).tolist()}. When terminated, the target is just r.")
V = td0(Chain(), pol, 2, alpha=1.0, gamma=0.5, init=0.0)
check(approx(V[:3], [0, 0.5, 1.0]), "γ=0.5, α=1, 2 episodes → V = [0, 0.5, 1]", f"got {np.round(V[:3], 4).tolist()}")

V0 = np.zeros(4)
out = td0(Chain(), pol, 1, alpha=0.5, V=V0)
check(out is V0 and approx(V0, [0, 0, 0.5, 0]), "td0(..., V=array) keeps training that array in place",
      "if V is given, do not create a new array — update and return the one passed in")

# ---- the 5-state random walk
env = RandomWalk()
random.seed(0)
coin = lambda s: random.randrange(2)
td_err, mc_err = [], []
truth = env.true_values()
for run in range(20):
    V = td0(env, coin, 100, alpha=0.05, init=0.5)
    td_err.append(np.sqrt(np.mean((V[1:-1] - truth) ** 2)))
    V = constant_alpha_mc(env, coin, 100, alpha=0.02, init=0.5)
    mc_err.append(np.sqrt(np.mean((V[1:-1] - truth) ** 2)))
td_rms, mc_rms = float(np.mean(td_err)), float(np.mean(mc_err))
check(td_rms < 0.06, "Random walk: after 100 episodes (α=0.05) RMS error vs true values [1/6 … 5/6] < 0.06",
      f"mean RMS over 20 runs = {td_rms:.3f}")
check(td_rms < mc_rms, "TD(0) beats constant-α MC on the random walk", f"TD {td_rms:.3f} vs MC {mc_rms:.3f}")
