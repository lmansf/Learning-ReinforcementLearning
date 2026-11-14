import numpy as np

softmax = require("softmax")
policy_probs = require("policy_probs")
grad_log_pi = require("grad_log_pi")
sample_action = require("sample_action")

p = np.asarray(softmax(np.array([1.0, 2.0, 3.0])))
check(approx(p, [0.09003057, 0.24472847, 0.66524096]), "softmax([1, 2, 3])", f"got {p}")
check(approx(np.asarray(softmax(np.array([0.0, 0.0]))), [0.5, 0.5]), "Equal preferences → equal probabilities")
p = np.asarray(softmax(np.array([1000.0, 1001.0])))
check(np.all(np.isfinite(p)) and approx(p, [0.26894142, 0.73105858]), "softmax is stable for huge preferences",
      f"softmax([1000, 1001]) gave {p} — subtract the max before exp()")
P = np.asarray(softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])))
check(P.shape == (2, 3) and approx(P.sum(axis=1), [1, 1]) and approx(P[1], [1 / 3] * 3),
      "softmax works row-wise on a batch (axis=-1)", f"got {P}")

theta = np.array([[1.0, -1.0], [0.5, 0.5], [0.0, 2.0]])
x = np.array([1.0, 2.0, 0.5])
p = np.asarray(policy_probs(theta, x))
# h = xᵀθ = [2.0, 1.0]
check(approx(p, softmax(np.array([2.0, 1.0]))), "policy_probs = softmax(xᵀθ)", f"got {p}")

g = np.asarray(grad_log_pi(theta, x, 0))
check(g.shape == theta.shape, "grad_log_pi has theta's shape", f"got {g.shape}, expected {theta.shape}")
p0 = float(softmax(np.array([2.0, 1.0]))[0])
check(approx(g, np.outer(x, [1 - p0, -(1 - p0)])), "grad_log_pi = outer(x, onehot(a) − π) by hand", f"got\n{g}")

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(5):
    th = rng.normal(size=(4, 3)); xx = rng.normal(size=4); a = int(rng.integers(3))
    an = np.asarray(grad_log_pi(th, xx, a))
    num = np.zeros_like(th)
    for i in range(4):
        for j in range(3):
            tp = th.copy(); tp[i, j] += 1e-6
            tm = th.copy(); tm[i, j] -= 1e-6
            num[i, j] = (np.log(policy_probs(tp, xx)[a]) - np.log(policy_probs(tm, xx)[a])) / 2e-6
    worst = max(worst, float(np.abs(num - an).max()))
check(worst < 1e-6, "grad_log_pi matches finite differences of log π", f"max abs error {worst:.2e}")

rng = np.random.default_rng(1)
th = np.array([[0.0, np.log(3.0)]])          # π = [0.25, 0.75]
draws = np.array([sample_action(th, np.array([1.0]), rng) for _ in range(4000)])
check(abs(draws.mean() - 0.75) < 0.03, "sample_action draws from π", f"P(a=1) ≈ {draws.mean():.3f}, expected 0.75")
check(isinstance(sample_action(th, np.array([1.0]), rng), (int, np.integer)), "sample_action returns an int")
