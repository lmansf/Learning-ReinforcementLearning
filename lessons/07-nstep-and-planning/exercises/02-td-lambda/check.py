import numpy as np

td_lambda = require("td_lambda")
avg_rms = require("avg_rms")


class Chain:
    """Deterministic 0 → 1 → 2 → 3 (terminal); reward 1 on the final step only."""
    n_states, n_actions = 4, 1
    def reset(self, seed=None):
        self.s = 0
        return 0, {}
    def step(self, a):
        self.s += 1
        return self.s, float(self.s == 3), self.s == 3, False, {}


pol = lambda s: 0
V = td_lambda(Chain(), pol, 2, 0.0, 0.5)
check(isinstance(V, np.ndarray) and approx(V, [0, 0.25, 0.75, 0]),
      "λ = 0 is TD(0): chain after 2 episodes → [0, 0.25, 0.75, 0]",
      f"got {np.round(V, 4).tolist() if hasattr(V, 'tolist') else V}")
V = td_lambda(Chain(), pol, 1, 0.5, 0.5)
check(approx(V, [0.125, 0.25, 0.5, 0]),
      "λ = 0.5, 1 episode: the final δ = 1 reaches back with traces [0.25, 0.5, 1] → [0.125, 0.25, 0.5, 0]",
      f"got {np.round(V, 4).tolist()}. Bump e[s] BEFORE updating V, and decay e by γλ AFTER.")
V = td_lambda(Chain(), pol, 2, 0.5, 0.5)
check(approx(V, [0.3125, 0.5, 0.75, 0]), "λ = 0.5, 2 episodes → [0.3125, 0.5, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}. Traces must restart at zero each episode.")
V = td_lambda(Chain(), pol, 2, 1.0, 0.5)
check(approx(V, [0.75, 0.75, 0.75, 0]), "λ = 1 on this chain matches Monte Carlo → [0.75, 0.75, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}")
V = td_lambda(Chain(), pol, 2, 0.5, 0.5, gamma=0.5, init=0.5)
check(approx(V[:3], [0.2890625, 0.4375, 0.875]),
      "γ = 0.5, init = 0.5: traces decay by γλ and terminal states are never bootstrapped from",
      f"got {np.round(V[:3], 5).tolist()}, expected [0.28906, 0.4375, 0.875]")
V0 = np.zeros(4)
check(td_lambda(Chain(), pol, 1, 0.5, 0.5, V=V0) is V0, "td_lambda(..., V=array) updates the given array in place",
      "return the array you were given")

e0 = avg_rms(0.0, 0.4, runs=30)
e8 = avg_rms(0.8, 0.4, runs=30)
check(e8 < 0.32 and e8 < e0 - 0.05, "19-state walk, α = 0.4: λ = 0.8 clearly beats λ = 0 over the first 10 episodes",
      f"avg RMS: λ=0 {e0:.3f}, λ=0.8 {e8:.3f} (expect ≈ 0.41 vs 0.27)")
