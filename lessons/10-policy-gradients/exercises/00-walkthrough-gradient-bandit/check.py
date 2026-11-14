import numpy as np

softmax = require("softmax")
sample = require("sample")
one_hot = require("one_hot")
bandit_step = require("bandit_step")
batch_step = require("batch_step")

p = np.asarray(softmax(np.array([0.0, np.log(3.0)])))
check(approx(p, [0.25, 0.75]), "softmax([0, ln 3]) == [0.25, 0.75]", f"got {p}")
p = np.asarray(softmax(np.array([800.0, 800.0, 801.0])))
check(np.all(np.isfinite(p)) and approx(p.sum(), 1.0) and p[2] > p[0],
      "softmax is stable for huge preferences", f"softmax([800, 800, 801]) gave {p} — subtract the max before exp()")
P = np.asarray(softmax(np.array([[0.0, 0.0], [0.0, np.log(3.0)], [5.0, 5.0]])))
check(P.shape == (3, 2) and approx(P, [[0.5, 0.5], [0.25, 0.75], [0.5, 0.5]]),
      "softmax works row by row on a batch (axis=-1, keepdims=True)", f"got {P.tolist()}")

rng = np.random.default_rng(0)
draws = np.array([sample(np.array([0.2, 0.0, 0.8]), rng) for _ in range(4000)])
check(set(draws.tolist()) <= {0, 2} and abs(np.mean(draws == 2) - 0.8) < 0.03,
      "sample draws from p", f"P(a=2) ≈ {np.mean(draws == 2):.3f}, expected 0.8; saw actions {sorted(set(draws.tolist()))}")
check(isinstance(sample(np.array([0.5, 0.5]), rng), int), "sample returns a plain int")

oh = np.asarray(one_hot(np.array([2, 0, 2]), 3))
check(oh.tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 1]], "one_hot([2, 0, 2], 3)", f"got {oh.tolist()}")

H = np.zeros(2)
p = bandit_step(H, 0, 3.0, 1.0, 0.5)
check(approx(H, [0.5, -0.5]), "bandit_step: H += α (r − b)(onehot(a) − π), in place",
      f"π = [.5, .5], r − b = 2 → expected H = [0.5, −0.5], got {H}")
check(approx(np.asarray(p), [0.5, 0.5]), "bandit_step returns the probabilities used for the update", f"got {p}")
H = np.zeros(3)
bandit_step(H, 1, 0.0, 1.0, 0.3)
check(H[1] < 0 and H[0] > 0 and approx(H.sum(), 0.0), "A worse-than-baseline reward makes the action LESS likely",
      f"got H = {H}")

# batch: two contexts, π uniform at the start
H = np.zeros((2, 2))
C = np.array([0, 0, 1, 1]); A = np.array([0, 1, 1, 1]); R = np.array([2.0, 0.0, 1.0, 1.0])
adv = (R - R.mean()) / (R.std() + 1e-8)
G = (np.eye(2)[A] - 0.5) * adv[:, None]
expected = np.zeros((2, 2))
for c, g in zip(C, G):
    expected[c] += 0.4 * g / 4
batch_step(H, C, A, R, 0.4)
check(approx(H, expected, 1e-6), "batch_step: mean over the batch of normalised advantage × (onehot − π), per context",
      f"expected\n{np.round(expected, 4)}\ngot\n{np.round(H, 4)}")

rng = np.random.default_rng(1)
H = np.zeros((2, 3))
for _ in range(100):
    C = rng.integers(0, 2, 32)
    A = np.array([sample(p, rng) for p in softmax(H[C])])
    batch_step(H, C, A, (A == 2 * C).astype(float), 0.5)
P = softmax(H)
check(P[0, 0] > 0.8 and P[1, 2] > 0.8, "Repeated batch steps learn the best arm of each context",
      f"π(·|ctx 0) = {np.round(P[0], 2)}, π(·|ctx 1) = {np.round(P[1], 2)}")
