import numpy as np
import rlhud


def softmax(z):
    """Numerically stable softmax over the last axis. Works for a vector or a batch (rows)."""
    # TODO: subtract the max along the last axis (keepdims=True), exponentiate, normalise
    raise NotImplementedError


def policy_probs(theta, x):
    """Linear-softmax policy π(·|s) = softmax(x(s)ᵀ θ).   theta: (d, n_actions), x: (d,)"""
    # TODO: one line
    raise NotImplementedError


def grad_log_pi(theta, x, a):
    """∇_θ log π(a|s) for the linear-softmax policy, same shape as theta.

    log π(a|s) = h_a − log Σ_b exp(h_b),  h = xᵀθ
    ∂ log π(a|s) / ∂θ[:, b] = x · (1[a = b] − π(b|s))      →  outer(x, onehot(a) − π)
    """
    # TODO
    raise NotImplementedError


def sample_action(theta, x, rng):
    """Draw an action from π(·|s)."""
    # TODO: rng.choice(n_actions, p=...)
    raise NotImplementedError


if __name__ == "__main__":
    # A one-state "bandit" with 3 actions and rewards 1, 2, 5 (+ noise). Gradient ascent on
    # E[r] using the score-function estimate r · ∇log π(a) — REINFORCE in its smallest form.
    rng = np.random.default_rng(0)
    theta = np.zeros((1, 3))
    x = np.array([1.0])
    means = np.array([1.0, 2.0, 5.0])
    live = rlhud.chart("π(a) while learning", xlabel="step")
    for step in range(600):
        a = sample_action(theta, x, rng)
        r = means[a] + rng.normal()
        theta += 0.02 * r * grad_log_pi(theta, x, a)
        if step % 5 == 0:
            p = policy_probs(theta, x)
            for i in range(3):
                live.add(f"π(a{i})", float(p[i]))
    live.flush()
    rlhud.bars(policy_probs(theta, x), labels=["a0 (r≈1)", "a1 (r≈2)", "a2 (r≈5)"],
               title="Final policy", highlight=2)
