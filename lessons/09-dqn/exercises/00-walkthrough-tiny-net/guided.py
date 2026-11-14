import numpy as np
import rlhud


#> A one-hidden-layer network is just four arrays: W1, b1, W2, b2.
def init_params(n_in, n_hidden, n_out, seed=0):
    #> One seeded Generator → the same "random" weights on every run.
    rng = np.random.default_rng(seed)
    #> Weights are `(n_in, n_out)` so a batch `x @ W` maps `(B, n_in)` → `(B, n_out)`.
    #> He initialisation (std √(2/n_in)) keeps ReLU activations at a sensible scale.
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
    W2 = rng.normal(0.0, np.sqrt(1.0 / n_hidden), size=(n_hidden, n_out))
    #> Biases are 1-D, `(n_out,)`: broadcasting adds them to every row of the batch.
    return [W1, np.zeros(n_hidden), W2, np.zeros(n_out)]


#> Forward pass for a batch `x` of shape `(B, n_in)`. Returns the output AND a cache.
def forward(params, x):
    W1, b1, W2, b2 = params
    #> Hidden pre-activation, shape `(B, n_hidden)`.
    z = x @ W1 + b1
    #> ReLU, element-wise. `np.maximum(z, 0)` — the built-in `max(z, 0)` fails on arrays.
    h = np.maximum(z, 0.0)
    #> No activation on the output layer: Q-values and targets can be any real number.
    y = h @ W2 + b2
    #> Backward needs each layer's INPUT (`x`, `h`) and where the ReLU was open (`z > 0`).
    return y, (x, z, h)


#> Backward pass. `dy = ∂L/∂y` has y's shape `(B, n_out)`.
#> Returns the gradients in the same order as `params`.
def backward(params, cache, dy):
    W1, b1, W2, b2 = params
    x, z, h = cache
    #> ∂L/∂W2 = hᵀ·dy: `(n_hidden, B) @ (B, n_out)` → `(n_hidden, n_out)` — W2's shape.
    #> The matrix product sums over the batch for you.
    dW2 = h.T @ dy
    #> A bias is added to every row, so its gradient is the upstream gradient summed
    #> over the rows of the batch.
    db2 = dy.sum(axis=0)
    #> Chain rule back through W2: `(B, n_out) @ (n_out, n_hidden)` → `(B, n_hidden)`.
    dh = dy @ W2.T
    #> Through the ReLU: the gradient passes where z > 0, and is 0 elsewhere
    #> (a bool array times a float array gives floats).
    dz = dh * (z > 0)
    dW1 = x.T @ dz
    db1 = dz.sum(axis=0)
    return [dW1, db1, dW2, db2]


#> **Finite differences** — the universal test for a hand-written gradient.
def numeric_grad(loss_fn, p, eps=1e-6):
    g = np.zeros_like(p)
    #> `reshape(-1)` returns a VIEW of the same memory: writing `flat[j]` changes `p`
    #> itself, which is what lets `loss_fn()` see the nudged weight.
    flat, gflat = p.reshape(-1), g.reshape(-1)
    for j in range(flat.size):
        old = flat[j]
        flat[j] = old + eps
        up = loss_fn()
        flat[j] = old - eps
        down = loss_fn()
        #> Always restore the weight, or later entries are measured at the wrong point.
        flat[j] = old
        gflat[j] = (up - down) / (2 * eps)
    return g


#> Pick ONE output per row — e.g. Q(sᵢ, aᵢ) for the action actually taken in row i.
def pick(out, cols):
    #> Pair row i with column `cols[i]` → shape `(B,)`.
    #> Beware: `out[:, cols]` would give every column for every row, shape `(B, B)`.
    return out[np.arange(len(cols)), cols]


if __name__ == "__main__":
    #= given
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(512, 1))       # a "V" no linear model can fit
    Y = np.abs(X) - 1.0
    params = init_params(1, 32, 1, seed=0)

    def loss():                                      # MSE on the first 8 rows
        return float(np.mean((forward(params, X[:8])[0] - Y[:8]) ** 2))

    y, cache = forward(params, X[:8])
    grads = backward(params, cache, 2 * (y - Y[:8]) / 8)
    #= end
    #> Gradient check: the closure `loss` reads `params`, so it sees every nudge.
    #> A correct backward agrees with finite differences to ~1e-8 or better.
    err = max(np.abs(numeric_grad(loss, p) - g).max() for p, g in zip(params, grads))
    #= given
    rlhud.stat("Gradient check: max |analytic − numeric|", float(err), good=err < 1e-6)
    live = rlhud.chart("Mini-batch MSE (log10)", smooth=20, xlabel="step")
    #= end
    for step in range(1500):
        #> A random mini-batch: 64 row indices drawn WITH replacement — exactly how a
        #> replay buffer samples. One index array keeps each x with its own y.
        idx = rng.integers(0, len(X), size=64)
        y, cache = forward(params, X[idx])
        #> ∂/∂y of mean((y − Y)²) is 2(y − Y)/B. Forgetting the /B scales every gradient.
        grads = backward(params, cache, 2 * (y - Y[idx]) / len(idx))
        for p, g in zip(params, grads):
            #> `-=` edits the arrays inside `params` in place; the list never changes.
            p -= 0.05 * g
        #= given
        live.add("train", float(np.log10(np.mean((y - Y[idx]) ** 2))))
    live.flush()
    xs = np.linspace(-2.0, 2.0, 81)[:, None]
    title = "Tiny net fit"
    rlhud.plot(np.abs(xs[:, 0]) - 1.0, x=xs[:, 0], label="target |x| − 1", title=title)
    rlhud.plot(forward(params, xs)[0][:, 0], x=xs[:, 0], label="network", title=title)
    print("pick:", pick(np.array([[1, 2], [3, 4], [5, 6]]), np.array([1, 0, 1])))
    #= end
