import numpy as np

bucket = require("bucket")
one_hot_features = require("one_hot_features")
predict = require("predict")
sgd_step = require("sgd_step")
sparse_predict = require("sparse_predict")
sparse_step = require("sparse_step")
grid_index = require("grid_index")

got = [int(bucket(v, 0.0, 1.0, 10)) for v in (0.35, 0.0, 0.999)]
check(got == [3, 0, 9], "bucket: 0.35 → 3, 0.0 → 0, 0.999 → 9 (10 bins on [0, 1])",
      f"got {got}. Rescale with (x − low) / (high − low) · n, then np.floor.")
got = [int(bucket(v, 0.0, 1.0, 10)) for v in (1.0, -2.0, 7.0)]
check(got == [9, 0, 9], "bucket clips the edges into 0 … n−1",
      f"bucket(1.0), bucket(−2), bucket(7) gave {got}, expected [9, 0, 9] — use np.clip(idx, 0, n − 1)")
got = np.asarray(bucket(np.array([-0.5, 2.5]), -1.0, 3.0, 4))
check(got.tolist() == [0, 3], "bucket works on arrays (vectorised)", f"got {got.tolist()}")

x = np.asarray(one_hot_features(0.6, 0.0, 1.0, 4))
check(x.tolist() == [0.0, 0.0, 1.0, 0.0], "one_hot_features(0.6, 0, 1, 4) == [0, 0, 1, 0]", f"got {x.tolist()}")
check(approx(predict(np.array([2.0, -1.0, 4.0]), np.array([1.0, 1.0, 0.5])), 3.0),
      "predict is the dot product wᵀx", "w=[2, −1, 4], x=[1, 1, 0.5] should give 3.0")

w = np.array([1.0, 1.0, 0.0])
err = sgd_step(w, np.array([0.0, 1.0, 0.0]), 3.0, 0.5)
check(approx(err, 2.0), "sgd_step returns the error y − wᵀx (before the update)", f"expected 2.0, got {err}")
check(approx(w, [1.0, 2.0, 0.0]), "sgd_step updates w in place, only where xᵢ ≠ 0",
      f"expected w = [1, 2, 0], got {w}. Use `w += …`, not `w = w + …`.")

W = np.arange(12, dtype=float).reshape(3, 4)
q = np.asarray(sparse_predict(W, np.array([1, 3])))
check(q.shape == (3,) and approx(q, [4.0, 12.0, 20.0]), "sparse_predict sums the active weights of every row",
      f"W = arange(12).reshape(3, 4), idx = [1, 3] → expected [4, 12, 20], got {q.tolist()}")
W = np.zeros((2, 4))
err = sparse_step(W, np.array([1, 3]), 1, 2.0, 0.25)
check(approx(err, 2.0) and approx(W, [[0, 0, 0, 0], [0, 0.5, 0, 0.5]]),
      "sparse_step moves only the active weights of row k by α·err", f"err={err}, W=\n{W}")

got = [grid_index(p, [0, 0], [1, 1], 4) for p in ([0.6, 0.3], [0.0, 0.0], [1.0, 1.0])]
check(got == [9, 0, 15], "grid_index flattens row-major: (2, 1) → 2·4 + 1 = 9",
      f"(0.6, 0.3), (0, 0), (1, 1) gave {got}, expected [9, 0, 15]")
