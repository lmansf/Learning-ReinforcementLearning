import numpy as np

MLP = require("MLP")

# --- forward by hand: 2 → 2 → 1 with chosen weights
net = MLP([2, 2, 1], seed=0)
net.params[0][...] = [[1.0, -1.0], [2.0, 1.0]]
net.params[1][...] = [0.0, -0.5]
net.params[2][...] = [[1.0], [3.0]]
net.params[3][...] = [0.25]
y = np.asarray(net.forward(np.array([[1.0, 1.0], [-1.0, 0.0]])))
# row 1: h = relu([3, -0.5]) = [3, 0] → 3.25 ; row 2: h = relu([-1, 0.5]) = [0, 0.5] → 1.75
check(y.shape == (2, 1), "forward returns shape (batch, n_out)", f"got shape {y.shape}")
check(approx(y, [[3.25], [1.75]]), "forward: Linear → ReLU → Linear by hand",
      f"expected [[3.25], [1.75]], got {y.tolist()}. ReLU on hidden layers only, no activation on the output.")

# --- backward by hand on the same net, loss = sum(y)
g = net.backward(np.ones((2, 1)))
check(isinstance(g, (list, tuple)) and len(g) == 4, "backward returns one gradient per parameter",
      f"got {type(g).__name__} of length {len(g) if hasattr(g, '__len__') else '?'}")
check(all(np.shape(gi) == np.shape(p) for gi, p in zip(g, net.params)), "Each gradient has its parameter's shape",
      str([np.shape(gi) for gi in g]))
check(approx(g[3], [2.0]), "∂L/∂b_out = sum of upstream gradient over the batch", f"got {g[3]}")
check(approx(g[2], [[3.0], [0.5]]), "∂L/∂W_out = hiddenᵀ · upstream", f"expected [[3], [0.5]], got {np.asarray(g[2]).tolist()}")
check(approx(g[1], [1.0, 3.0]), "∂L/∂b_hidden passes through the ReLU mask",
      f"expected [1, 3] (each hidden unit was active in exactly one row), got {g[1]}")

# --- numerical gradient check on a deeper net
rng = np.random.default_rng(1)
net = MLP([3, 8, 8, 2], seed=3)
x = rng.normal(size=(5, 3))
R = rng.normal(size=(5, 2))
def loss():
    return float(np.sum(net.forward(x) * R))
loss()
analytic = net.backward(R)
worst = 0.0
for pi, p in enumerate(net.params):
    flat = p.reshape(-1)
    for j in range(0, flat.size, max(1, flat.size // 12)):
        old = flat[j]
        flat[j] = old + 1e-6; lp = loss()
        flat[j] = old - 1e-6; lm = loss()
        flat[j] = old
        num = (lp - lm) / 2e-6
        an = np.asarray(analytic[pi]).reshape(-1)[j]
        worst = max(worst, abs(num - an) / max(1.0, abs(num)))
check(worst < 1e-5, "Gradients match finite differences on a 3-layer net",
      f"worst relative error {worst:.2e}. Check the order of layers and the ReLU mask (use the *input* to each layer).")

# --- it can learn
net = MLP([1, 32, 1], seed=0)
xs = np.linspace(-1, 1, 64)[:, None]; ys = xs ** 2
lr = 0.05
first = None
for _ in range(500):
    pred = net.forward(xs)
    l = float(np.mean((pred - ys) ** 2))
    first = l if first is None else first
    for p, gp in zip(net.params, net.backward(2 * (pred - ys) / len(xs))):
        p -= lr * gp
check(l < first * 0.2, "Plain gradient descent with your gradients fits y = x²",
      f"MSE went from {first:.4f} to {l:.4f}")
