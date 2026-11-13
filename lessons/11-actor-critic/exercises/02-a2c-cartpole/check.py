import time
import numpy as np

compute_gae = require("compute_gae")
actor_loss_grad = require("actor_loss_grad")
critic_loss_grad = require("critic_loss_grad")

adv, ret = compute_gae([1.0, 0.0, 2.0], [1.0, 2.0, 1.0], [0, 1, 0], 4.0, gamma=0.5, lam=0.5)
check(approx(adv, [0.5, -2.0, 3.0]) and approx(ret, [1.5, 0.0, 4.0]), "compute_gae (from the previous exercise)",
      f"expected advantages [0.5, −2, 3] and returns [1.5, 0, 4], got {np.asarray(adv).tolist()}, {np.asarray(ret).tolist()}")

# --- actor gradient by hand: theta = 0 → π = [.5, .5]
theta = np.zeros((2, 2))
X = np.array([[1.0, 2.0], [1.0, 0.0]]); A = np.array([1, 0]); adv = np.array([1.0, -1.0])
g = np.asarray(actor_loss_grad(theta, X, A, adv))
expected = -(1 * np.outer([1, 2], [-0.5, 0.5]) + -1 * np.outer([1, 0], [0.5, -0.5])) / 2
check(g.shape == theta.shape and approx(g, expected), "actor_loss_grad = −mean[Â · outer(x, onehot − π)]",
      f"expected\n{expected}\ngot\n{g}")

# numerical check of the loss gradient
rng = np.random.default_rng(0)
theta = rng.normal(size=(3, 2)); X = rng.normal(size=(6, 3)); A = rng.integers(0, 2, 6); adv = rng.normal(size=6)
def L(th):
    P = softmax(X @ th)
    return -np.mean(adv * np.log(P[np.arange(6), A]))
num = np.zeros_like(theta)
for i in range(3):
    for j in range(2):
        tp = theta.copy(); tp[i, j] += 1e-6; tm = theta.copy(); tm[i, j] -= 1e-6
        num[i, j] = (L(tp) - L(tm)) / 2e-6
check(approx(actor_loss_grad(theta, X, A, adv), num, 1e-6), "actor_loss_grad matches finite differences of −mean(Â log π)")

g = np.asarray(critic_loss_grad(np.array([[1.0], [3.0]]), np.array([2.0, 2.0])))
check(g.shape == (2, 1) and approx(g, [[-0.5], [0.5]]), "critic_loss_grad = (v_pred − target) / N, shape (N, 1)",
      f"got {g.tolist()} with shape {g.shape}")

t0 = time.time()
theta, critic, returns = train(seed=0)
dt = time.time() - t0
score = evaluate(theta)
check(score >= 150, "A2C balances the pole: mean return ≥ 150 over 10 episodes",
      f"got {score:.0f} after {len(returns)} episodes ({dt:.0f}s). If returns fall over time, check signs: "
      "Adam MINIMISES, so the actor gradient must be the gradient of −Â·log π.")
