import numpy as np

nstep_target = require("nstep_target")
nstep_td = require("nstep_td")
avg_rms = require("avg_rms")

# ---- nstep_target
check(approx(nstep_target([1.0, 2.0, 3.0], 0.5, 8.0), 1 + 1 + 0.75 + 0.125 * 8),
      "nstep_target([1,2,3], γ=0.5, bootstrap=8) = 1 + 0.5·2 + 0.25·3 + 0.125·8 = 3.75",
      f"got {nstep_target([1.0, 2.0, 3.0], 0.5, 8.0)}")
check(approx(nstep_target([], 0.9, 5.0), 5.0), "With no rewards the target is just the bootstrap value",
      f"got {nstep_target([], 0.9, 5.0)}")
check(approx(nstep_target([2.0], 0.9, 0.0), 2.0), "bootstrap = 0 (episode ended) leaves only the rewards",
      f"got {nstep_target([2.0], 0.9, 0.0)}")


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
V = nstep_td(Chain(), pol, 2, 1, 0.5)
check(approx(V, [0, 0.25, 0.75, 0]), "n = 1 is TD(0): after 2 episodes V = [0, 0.25, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}")
V = nstep_td(Chain(), pol, 1, 2, 0.5)
check(approx(V, [0, 0.5, 0.5, 0]), "n = 2, 1 episode: the reward reaches TWO states back → [0, 0.5, 0.5, 0]",
      f"got {np.round(V, 4).tolist()}. V(S_τ) uses R_τ+1 … R_τ+n plus γⁿ·V(S_τ+n) if the episode has not ended by then.")
V = nstep_td(Chain(), pol, 2, 2, 0.5)
check(approx(V, [0.25, 0.75, 0.75, 0]), "n = 2, 2 episodes → [0.25, 0.75, 0.75, 0]", f"got {np.round(V, 4).tolist()}")
V = nstep_td(Chain(), pol, 2, 10, 0.5)
check(approx(V, [0.75, 0.75, 0.75, 0]), "n ≥ episode length is constant-α Monte Carlo → [0.75, 0.75, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}. Do the remaining updates after the episode ends (until τ = T − 1).")
V = nstep_td(Chain(), pol, 2, 2, 0.5, gamma=0.5)
check(approx(V, [0.0625, 0.375, 0.75, 0]), "n = 2, γ = 0.5, 2 episodes → [0.0625, 0.375, 0.75, 0]",
      f"got {np.round(V, 4).tolist()}")
V = nstep_td(Chain(), pol, 1, 2, 0.5, init=0.5)
check(approx(V[:3], [0.5, 0.75, 0.75]), "Never bootstraps from beyond the end of the episode (init = 0.5)",
      f"got {np.round(V[:3], 4).tolist()}. If τ + n ≥ T the target is just the remaining rewards.")
V0 = np.zeros(4)
out = nstep_td(Chain(), pol, 1, 2, 0.5, V=V0)
check(out is V0, "nstep_td(..., V=array) updates the given array in place", "return the same array you were given")

# ---- 19-state random walk: intermediate n wins
e1 = avg_rms(1, 0.4, runs=30)
e4 = avg_rms(4, 0.4, runs=30)
e32 = avg_rms(32, 0.4, runs=30)
check(e4 < 0.32, "19-state walk, n = 4, α = 0.4: average RMS over the first 10 episodes < 0.32",
      f"got {e4:.3f} (expect ≈ 0.26)")
check(e4 < e1 - 0.05 and e4 < e32 - 0.05, "n = 4 beats both n = 1 (TD) and n = 32 (≈ MC) at α = 0.4",
      f"RMS: n=1 {e1:.3f}, n=4 {e4:.3f}, n=32 {e32:.3f}")
