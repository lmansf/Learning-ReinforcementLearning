import numpy as np

init_params = require("init_params")
forward = require("forward")
backward = require("backward")
numeric_grad = require("numeric_grad")
pick = require("pick")

ps = init_params(3, 5, 2, seed=0)
shapes = [np.shape(p) for p in ps]
check(shapes == [(3, 5), (5,), (5, 2), (2,)], "init_params returns [W1 (n_in, n_hidden), b1, W2, b2]",
      f"got shapes {shapes}")

# --- forward by hand: 1 → 2 → 1
params = [np.array([[1.0, -2.0]]), np.array([0.5, 1.0]), np.array([[2.0], [1.0]]), np.array([-1.0])]
y, cache = forward(params, np.array([[1.0], [-1.0]]))
y = np.asarray(y)
# row 1: z = [1.5, −1] → h = [1.5, 0] → 2 ; row 2: z = [−0.5, 3] → h = [0, 3] → 2
check(y.shape == (2, 1), "forward returns shape (batch, n_out)", f"got {y.shape}")
check(approx(y, [[2.0], [2.0]]), "forward: Linear → ReLU → Linear, by hand",
      f"expected [[2], [2]], got {y.tolist()}. ReLU on the hidden layer only.")

g = backward(params, cache, np.ones((2, 1)))
check(isinstance(g, (list, tuple)) and len(g) == 4 and all(np.shape(a) == np.shape(b) for a, b in zip(g, params)),
      "backward returns one gradient per parameter, each with its parameter's shape",
      f"got {[np.shape(a) for a in g] if isinstance(g, (list, tuple)) else type(g).__name__}")
check(approx(g[2], [[1.5], [3.0]]) and approx(g[3], [2.0]), "∂L/∂W2 = hᵀ·dy and ∂L/∂b2 = Σ_batch dy",
      f"got dW2 = {np.asarray(g[2]).tolist()}, db2 = {np.asarray(g[3]).tolist()}")
check(approx(g[1], [2.0, 1.0]) and approx(g[0], [[2.0, -1.0]]), "Gradients pass the ReLU only where z > 0",
      f"expected db1 = [2, 1], dW1 = [[2, −1]], got {np.asarray(g[1]).tolist()}, {np.asarray(g[0]).tolist()}")

# --- numeric_grad on a function with a known gradient
p = np.array([[1.0, -2.0], [0.5, 3.0]])
before = p.copy()
ng = np.asarray(numeric_grad(lambda: float(np.sum(p ** 3)), p))
check(approx(ng, 3 * before ** 2, 1e-4), "numeric_grad of Σp³ is 3p²", f"got {ng.tolist()}")
check(np.array_equal(p, before), "numeric_grad restores every weight afterwards", f"p changed to {p.tolist()}")

# --- backward against an independent finite-difference check
rng = np.random.default_rng(3)
params = init_params(3, 6, 2, seed=4)
x = rng.normal(size=(5, 3)); R = rng.normal(size=(5, 2))
y, cache = forward(params, x)
an = backward(params, cache, R)
worst = 0.0
for pa, ga in zip(params, an):
    flat = pa.reshape(-1)
    for j in range(flat.size):
        old = flat[j]
        flat[j] = old + 1e-6; lp = float(np.sum(forward(params, x)[0] * R))
        flat[j] = old - 1e-6; lm = float(np.sum(forward(params, x)[0] * R))
        flat[j] = old
        worst = max(worst, abs((lp - lm) / 2e-6 - np.asarray(ga).reshape(-1)[j]))
check(worst < 1e-6, "backward matches finite differences on a random batch", f"worst error {worst:.2e}")

q = np.asarray(pick(np.array([[1.0, 9.0], [7.0, 2.0], [0.0, 5.0]]), np.array([1, 0, 0])))
check(q.shape == (3,) and approx(q, [9.0, 7.0, 0.0]), "pick takes out[i, cols[i]] for every row",
      f"got {q.tolist()} — use out[np.arange(len(cols)), cols]")
