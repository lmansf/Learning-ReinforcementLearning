import numpy as np

v_hat = require("v_hat")
gradient_mc_update = require("gradient_mc_update")
td0_update = require("td0_update")

# --- v_hat
check(approx(v_hat(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, 1.0])), 3.5),
      "v_hat is the dot product wᵀx", "w=[1,2,3], x=[0.5,0,1] should give 3.5")

# --- TD(0) by hand
w = np.array([0.0, 1.0])
d = td0_update(w, np.array([1.0, 0.0]), 0.5, np.array([0.0, 1.0]), False, alpha=0.1, gamma=0.9)
check(approx(d, 1.4), "TD error δ = r + γ v̂(S') − v̂(S)", f"expected 0.5 + 0.9·1 − 0 = 1.4, got {d}")
check(approx(w, [0.14, 1.0]), "TD(0) moves only the weights of active features (w ← w + αδx)",
      f"expected w = [0.14, 1.0], got {w} — did you update w in place?")
w = np.array([2.0, 5.0])
d = td0_update(w, np.array([1.0, 0.0]), 1.0, np.array([0.0, 1.0]), True, alpha=0.5, gamma=0.9)
check(approx(d, -1.0) and approx(w, [1.5, 5.0]), "Terminal transitions do not bootstrap (target = r)",
      f"expected δ = 1 − 2 = −1 and w = [1.5, 5], got δ={d}, w={w}")

# --- gradient MC by hand: two states, rewards 0 then 1, gamma 0.5
w = np.zeros(2)
xs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
out = gradient_mc_update(w, xs, [0.0, 1.0], alpha=0.5, gamma=0.5)
check(out is not None and approx(w, [0.25, 0.5]), "Gradient MC uses the return Gₜ from every step",
      f"G0 = 0 + 0.5·1 = 0.5, G1 = 1 → w should be [0.25, 0.5], got {w}")
w = np.array([0.0, 0.0])
gradient_mc_update(w, [np.array([1.0, 1.0])], [2.0], alpha=0.25)
check(approx(w, [0.5, 0.5]), "Every active feature moves by α (G − v̂) xᵢ",
      f"one state with x=[1,1], G=2, α=0.25 → w should be [0.5, 0.5], got {w}")

# --- the real thing: 1000-state random walk with state aggregation
v_true = true_values()
def rmse(w):
    return float(np.sqrt(np.mean([(v_hat(w, aggregate(s)) - v_true[s]) ** 2 for s in range(1000)])))

w_mc = run_gradient_mc(seed=0)
e = rmse(w_mc)
check(e < 0.1, "Gradient MC learns the random-walk values (RMSE < 0.10)",
      f"RMSE = {e:.3f}. The best possible with 10 groups is ≈ 0.053.")
check(w_mc[0] < -0.6 and w_mc[-1] > 0.6 and np.all(np.diff(w_mc) > -0.05),
      "MC values rise from −1 (left) to +1 (right)", f"weights = {np.round(w_mc, 2)}")
w_td = run_td0(seed=0)
e = rmse(w_td)
check(e < 0.2, "Semi-gradient TD(0) learns the random-walk values (RMSE < 0.20)",
      f"RMSE = {e:.3f}. TD converges to the TD fixed point, which is a little worse than MC here — but not this bad.")
