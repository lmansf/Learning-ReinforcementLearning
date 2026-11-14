import numpy as np

shifted_values = require("shifted_values")
rollout_returns = require("rollout_returns")
critic_step = require("critic_step")
column_error = require("column_error")

vals = [3.0, 1.0, 2.0]
nv = np.asarray(shifted_values(vals, 7.0))
check(approx(nv, [1.0, 2.0, 7.0]), "shifted_values([3, 1, 2], 7) == [1, 2, 7]", f"got {nv.tolist()}")
check(vals == [3.0, 1.0, 2.0], "shifted_values does not modify its input", f"values became {vals}")

G = np.asarray(rollout_returns([1.0, 2.0, 0.0], [0, 0, 0], last_value=8.0, gamma=0.5))
# G2 = 0 + .5·8 = 4 ; G1 = 2 + .5·4 = 4 ; G0 = 1 + .5·4 = 3
check(approx(G, [3.0, 4.0, 4.0]), "rollout_returns bootstraps from last_value at the end",
      f"expected [3, 4, 4], got {G.tolist()}")
G = np.asarray(rollout_returns([1.0, 2.0, 0.0], [0, 1, 0], last_value=8.0, gamma=0.5))
# G2 = 4 ; G1 = 2 (episode ended) ; G0 = 1 + .5·2 = 2
check(approx(G, [2.0, 2.0, 4.0]), "rollout_returns: nothing leaks across a done",
      f"expected [2, 2, 4], got {G.tolist()} — multiply the carried return by (1 − doneₜ)")
G = np.asarray(rollout_returns(np.ones(4), [False, False, False, True], last_value=99.0, gamma=1.0))
check(approx(G, [4.0, 3.0, 2.0, 1.0]), "A done on the last step ignores last_value (boolean dones work too)",
      f"got {G.tolist()}")

w = np.array([1.0, 0.0])
X = np.array([[1.0, 1.0], [2.0, 1.0]])
loss = critic_step(w, X, np.array([2.0, 0.0]), lr=0.5)
# v = [1, 2], err = [−1, 2], loss = ½·mean([1, 4]) = 1.25 ; grad = Xᵀerr/2 = [3, 1]/2
check(approx(loss, 1.25), "critic_step returns ½·mean(err²) (before the update)", f"expected 1.25, got {loss}")
check(approx(w, [1.0 - 0.75, -0.25]), "critic_step: w −= lr · Xᵀ(Xw − target) / N, in place",
      f"expected w = [0.25, −0.25], got {w}. Sign: err = prediction − target, and we subtract.")
rng = np.random.default_rng(0)
X = np.c_[rng.normal(size=50), np.ones(50)]
t = X @ np.array([2.0, -1.0])
w = np.zeros(2)
for _ in range(300):
    last = critic_step(w, X, t, lr=0.3)
check(approx(w, [2.0, -1.0], 1e-3) and last < 1e-6, "Repeated critic steps recover the true weights",
      f"w = {np.round(w, 4)}, loss = {last:.2e}")

e = np.asarray(column_error(np.array([[1.0], [3.0], [0.0]]), np.array([2.0, 2.0, 2.0])))
check(e.shape == (3, 1) and approx(e, [[-1.0], [1.0], [-2.0]]), "column_error keeps shape (N, 1)",
      f"got shape {e.shape} — reshape the targets with .reshape(-1, 1) before subtracting")
