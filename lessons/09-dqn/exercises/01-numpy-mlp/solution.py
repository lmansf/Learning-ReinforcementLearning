import numpy as np
import rlhud


class MLP:
    """A fully connected network: Linear → ReLU → Linear → ReLU → … → Linear.

    net = MLP([4, 64, 64, 2], seed=0)
    y = net.forward(x)             # x: (batch, 4)  ->  y: (batch, 2)
    grads = net.backward(dy)       # dy = ∂loss/∂y, same shape as y
                                   # grads[i] = ∂loss/∂params[i]
    net.params = [W0, b0, W1, b1, W2, b2]   with Wi: (n_in, n_out), bi: (n_out,)
    """

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation keeps activations at a sensible scale through ReLU layers.
            self.params.append(rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.params.append(np.zeros(n_out))
        self.n_layers = len(sizes) - 1

    def forward(self, x):
        """Compute the output and remember what backward() will need."""
        h = np.asarray(x, dtype=float)
        self._inputs = []                    # input to each Linear layer
        for i in range(self.n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            self._inputs.append(h)
            h = h @ W + b                    # Linear
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)       # ReLU on every hidden layer, not on the output
        return h

    def backward(self, dout):
        """Back-propagate ∂loss/∂output. Returns grads in the same order as self.params."""
        grads = [None] * len(self.params)
        d = np.asarray(dout, dtype=float)    # ∂loss/∂(output of layer i)
        for i in reversed(range(self.n_layers)):
            W = self.params[2 * i]
            h_in = self._inputs[i]
            grads[2 * i] = h_in.T @ d        # ∂L/∂W = inputᵀ · upstream
            grads[2 * i + 1] = d.sum(axis=0) # ∂L/∂b = upstream summed over the batch
            if i > 0:
                d = d @ W.T                  # ∂L/∂(layer input)
                d = d * (h_in > 0)           # through the ReLU that produced h_in
        return grads


class Adam:
    """Adam optimiser. Updates the arrays in `params` in place."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            m_hat, v_hat = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    # Supervised warm-up: fit y = sin(3x) with mean-squared error. No autograd anywhere —
    # the gradient of the loss w.r.t. the output is (pred − y)·2/N, and backward() does the rest.
    rng = np.random.default_rng(0)
    net = MLP([1, 64, 64, 1], seed=0)
    opt = Adam(net.params, lr=3e-3)
    live = rlhud.chart("MSE loss (log10)", smooth=20, xlabel="step")
    for step in range(2000):
        x = rng.uniform(-2, 2, size=(64, 1))
        y = np.sin(3 * x)
        pred = net.forward(x)
        loss = np.mean((pred - y) ** 2)
        opt.step(net.backward(2 * (pred - y) / len(x)))
        live.add("train", float(np.log10(loss)))
    live.flush()
    xs = np.linspace(-2, 2, 81)[:, None]
    rlhud.plot(np.sin(3 * xs[:, 0]), label="sin(3x)", title="MLP fit")
    rlhud.plot(net.forward(xs)[:, 0], label="MLP", title="MLP fit")
    rlhud.stat("Final MSE", float(loss))
