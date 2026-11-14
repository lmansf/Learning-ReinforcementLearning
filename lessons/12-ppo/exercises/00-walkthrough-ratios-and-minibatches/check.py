import numpy as np

taken_logp = require("taken_logp")
ratios = require("ratios")
capped_estimate = require("capped_estimate")
capped_estimate_grad = require("capped_estimate_grad")
fraction_outside = require("fraction_outside")
minibatch_fit = require("minibatch_fit")

P = np.array([[0.2, 0.8], [0.6, 0.4], [1.0, 0.0]])
lp = np.asarray(taken_logp(P, np.array([1, 0, 1])))
check(lp.shape == (3,) and approx(lp[:2], np.log([0.8, 0.6])), "taken_logp picks log P[i, actions[i]]",
      f"got {lp.tolist()} — use P[np.arange(N), actions]")
check(np.isfinite(lp[2]) and lp[2] < -20, "taken_logp is finite even for a zero probability (+1e-12)",
      f"log of P = 0 gave {lp[2]}")

r = np.asarray(ratios(np.log([0.3, 0.5]), np.log([0.6, 0.25])))
check(approx(r, [0.5, 2.0]), "ratios = exp(logp_new − logp_old)", f"expected [0.5, 2.0], got {r.tolist()}")

r = np.array([0.5, 1.0, 3.0]); R = np.array([2.0, -1.0, 1.0])
est = capped_estimate(r, R, 1.5)
check(approx(est, (1.0 - 1.0 + 1.5) / 3), "capped_estimate = mean(min(ratio, cap) · reward)",
      f"expected 0.5, got {est}. Use np.minimum (element-wise), not np.min.")
g = np.asarray(capped_estimate_grad(r, R, 1.5))
check(approx(g, [1.0 / 3, -1.0 / 3, 0.0]), "capped_estimate_grad: ratio·reward/N, zero where the cap binds",
      f"expected [0.333, −0.333, 0], got {g.tolist()}")
rng = np.random.default_rng(0)
lo = np.log(rng.uniform(0.2, 0.8, 30)); ln = lo + rng.normal(0, 0.4, 30); R = rng.normal(size=30)
rr = np.exp(ln - lo)
ok = np.abs(rr - 1.3) > 1e-3
num = np.zeros(30)
for i in range(30):
    e = np.zeros(30); e[i] = 1e-6
    num[i] = (capped_estimate(np.exp(ln + e - lo), R, 1.3) - capped_estimate(np.exp(ln - e - lo), R, 1.3)) / 2e-6
check(approx(np.asarray(capped_estimate_grad(rr, R, 1.3))[ok], num[ok], 1e-6),
      "capped_estimate_grad matches finite differences w.r.t. logp_new")

f = fraction_outside(np.array([0.7, 0.9, 1.0, 1.25, 1.1]), 0.8, 1.2)
check(approx(f, 0.4), "fraction_outside counts ratios below lo OR above hi", f"expected 0.4, got {f}")

# minibatch_fit against a reference that walks the same shuffled order
X = np.c_[np.arange(10.0) / 10, np.ones(10)]
batch = {"x": X, "y": X @ np.array([1.0, 2.0])}
ref = np.zeros(2)
rr = np.random.default_rng(5)
for _ in range(2):
    order = rr.permutation(10)
    for s in range(0, 10, 4):
        mb = order[s:s + 4]
        ref -= 0.5 * X[mb].T @ (X[mb] @ ref - batch["y"][mb]) / len(mb)
w = np.zeros(2)
out = minibatch_fit(batch, w, 0.5, 2, 4, np.random.default_rng(5))
check(out is not None and approx(np.asarray(out), ref), "minibatch_fit: rng.permutation each epoch, slices of `size`, mean over len(mb)",
      f"expected {np.round(ref, 6)}, got {None if out is None else np.round(np.asarray(out), 6)}. "
      "10 rows in minibatches of 4 → the last one has 2 rows: divide by len(mb).")
check(approx(w, ref), "minibatch_fit updates w in place (and returns it)", f"w = {w}")
w = minibatch_fit(batch, np.zeros(2), 0.5, 300, 4, np.random.default_rng(0))
check(approx(np.asarray(w), [1.0, 2.0], 1e-3), "Enough epochs recover the true weights", f"got {np.round(w, 4)}")
