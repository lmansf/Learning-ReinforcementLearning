import time
import numpy as np

returns_to_go = require("returns_to_go")
reinforce_update = require("reinforce_update")

g = np.asarray(returns_to_go([1.0, 1.0, 1.0], 0.5), dtype=float)
check(approx(g, [1.75, 1.5, 1.0]), "returns_to_go([1,1,1], 0.5) == [1.75, 1.5, 1.0]", f"got {g}")

# --- hand case without normalisation: one state, theta = 0 so π = [0.5, 0.5]
theta = np.zeros((2, 2)); w = np.array([1.0, 0.0])
X = np.array([[1.0, 2.0], [1.0, 0.0]]); A = np.array([1, 0]); G = np.array([3.0, 0.0])
# b = [1, 1] → adv = [2, −1]; w += 0.5 · Xᵀadv / 2 = 0.5·[1, 4]/2 → [1.25, 1.0]
# policy: row 0: adv 2 · outer([1,2], [−.5, .5]); row 1: −1 · outer([1,0], [.5, −.5])  → mean
out = reinforce_update(theta, w, X, A, G, lr=1.0, lr_v=0.5, normalize=False)
check(out is not None and len(out) == 2, "reinforce_update returns (theta, w)")
check(approx(w, [1.25, 1.0]), "Baseline regression: w ← w + lr_v · mean[(G − b) x]",
      f"expected w = [1.25, 1.0], got {w}. Use the advantages computed BEFORE updating w.")
expected = (2 * np.outer([1, 2], [-0.5, 0.5]) + -1 * np.outer([1, 0], [0.5, -0.5])) / 2
check(approx(theta, expected), "Policy step: θ ← θ + lr · mean[Â ∇log π]  (normalize=False)",
      f"expected\n{expected}\ngot\n{theta}")

# --- with normalisation the advantages become [+1, −1]
theta = np.zeros((2, 2)); w = np.array([1.0, 0.0])
reinforce_update(theta, w, X, A, G, lr=1.0, lr_v=0.5, normalize=True)
expected = (1 * np.outer([1, 2], [-0.5, 0.5]) + -1 * np.outer([1, 0], [0.5, -0.5])) / 2
check(approx(theta, expected, 1e-6), "Advantage normalisation: (Â − mean) / (std + 1e-8) before the policy step",
      f"expected\n{expected}\ngot\n{theta}")

# --- the gradient points uphill: actions with positive advantage become more likely
rng = np.random.default_rng(0)
theta = np.zeros((3, 2)); w = np.zeros(3)
X = np.c_[rng.normal(size=(50, 2)), np.ones(50)]; A = rng.integers(0, 2, 50)
G = np.where(A == 1, 1.0, 0.0) + 0.1 * rng.normal(size=50)
for _ in range(20):
    reinforce_update(theta, w, X, A, G, lr=0.5, lr_v=0.1)
p1 = softmax(X @ theta)[:, 1].mean()
check(p1 > 0.7, "Repeated updates make the better action (here: action 1) more likely",
      f"mean π(1|s) = {p1:.2f} — is the sign of the update right (gradient ASCENT)?")

t0 = time.time()
theta, w, returns = train(episodes=1000, seed=0)
dt = time.time() - t0
score = evaluate(theta)
early, late = float(np.mean(returns[:50])), float(np.mean(returns[-50:]))
check(late > 2 * early, "Learning curve rises", f"first 50 episodes: {early:.0f}, last 50: {late:.0f}")
check(score >= 150, "REINFORCE balances the pole: mean return ≥ 150 over 10 sampled episodes",
      f"got {score:.0f} ({dt:.0f}s of training)")
