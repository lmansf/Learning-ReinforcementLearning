import time
import numpy as np

epsilon_by_step = require("epsilon_by_step")
select_action = require("select_action")
td_targets = require("td_targets")
dqn_loss_and_grad = require("dqn_loss_and_grad")

# --- ε schedule
vals = [epsilon_by_step(t, 1.0, 0.1, 100) for t in (0, 50, 100, 500)]
check(approx(vals, [1.0, 0.55, 0.1, 0.1]), "epsilon_by_step decays linearly, then stays at `end`",
      f"for start=1, end=0.1, decay=100 at t=0,50,100,500 expected [1, .55, .1, .1], got {vals}")

# --- action selection
net = MLP([4, 8, 2], seed=0)
net.params[-1][...] = [0.0, 100.0]          # output bias makes action 1 clearly best
rng = np.random.default_rng(0)
obs = np.zeros(4, dtype=np.float32)
picks = [select_action(net, obs, 0.0, rng) for _ in range(30)]
check(set(picks) == {1}, "ε = 0 → greedy action", f"picked {sorted(set(picks))}")
picks = np.array([select_action(net, obs, 1.0, rng) for _ in range(2000)])
check(0.4 < picks.mean() < 0.6 and all(isinstance(p, (int, np.integer)) for p in picks[:5]),
      "ε = 1 → uniformly random action (returned as an int)", f"fraction of action 1 = {picks.mean():.2f}")

# --- targets
r = np.array([1.0, 1.0, 0.0])
d = np.array([0.0, 1.0, 0.0])
qt = np.array([[2.0, 5.0], [9.0, 9.0], [-1.0, -3.0]])
y = np.asarray(td_targets(r, d, qt, 0.5))
check(approx(y, [3.5, 1.0, -0.5]), "DQN target r + γ(1−done)·max Q_target(s′)",
      f"expected [3.5, 1.0, −0.5], got {y.tolist()} — terminated transitions must not bootstrap")
qo = np.array([[3.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
y = np.asarray(td_targets(r, d, qt, 0.5, q_next_online=qo))
check(approx(y, [2.0, 1.0, -1.5]), "Double DQN: online net picks a′, target net evaluates it",
      f"expected [2.0, 1.0, −1.5], got {y.tolist()}")

# --- loss and gradient
q_pred = np.array([[1.0, 2.0], [0.0, 4.0]])
loss, dq = dqn_loss_and_grad(q_pred, np.array([1, 0]), np.array([1.5, 3.0]))
# errors: 2 − 1.5 = 0.5 (quadratic: 0.125) ; 0 − 3 = −3 (linear: 2.5) → mean 1.3125
check(approx(loss, 1.3125), "Loss = mean Huber over Q(s, a_taken) − y", f"expected 1.3125, got {loss}")
check(approx(dq, [[0.0, 0.25], [-0.5, 0.0]]), "Gradient flows only into the taken actions, clipped and divided by B",
      f"expected [[0, 0.25], [−0.5, 0]], got {np.asarray(dq).tolist()}")

# --- one update on a fixed batch must reduce the loss (end-to-end wiring)
rng = np.random.default_rng(1)
net = MLP([4, 16, 2], seed=1); opt = Adam(net.params, lr=1e-2)
S = rng.normal(size=(32, 4)); A = rng.integers(0, 2, 32); Y = rng.normal(size=32)
l0, _ = dqn_loss_and_grad(net.forward(S), A, Y)
for _ in range(50):
    l, g = dqn_loss_and_grad(net.forward(S), A, Y)
    opt.step(net.backward(g))
check(l < 0.5 * l0, "Your loss gradient trains the network (50 Adam steps on a fixed batch)",
      f"loss {l0:.3f} → {l:.3f}. If it went UP, the sign of err is flipped (use prediction − target).")

# --- the real thing
t0 = time.time()
q, returns = train(total_steps=70_000, seed=0)
dt = time.time() - t0
score = evaluate(q, 10)
check(score >= 150, "Trained DQN balances the pole: greedy mean return ≥ 150 over 10 episodes",
      f"got {score:.1f} after {len(returns)} episodes ({dt:.0f}s). Typical bugs: bootstrapping from terminal "
      "states, using the online net for max Q(s′) in plain DQN, a wrong gradient sign, ε never decaying.")
