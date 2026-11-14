import numpy as np

q_update = require("q_update")
plan = require("plan")
dyna_q = require("dyna_q")
make_maze = require("make_maze")

# ---- q_update
Q = np.zeros((3, 2))
Q[2] = [0.0, 3.0]
q_update(Q, 0, 1, 1.0, 2, False, 0.5, 0.9)
check(approx(Q[0, 1], 1.85), "q_update: Q(0,1) = 0.5·(1 + 0.9·3) = 1.85", f"got {Q[0, 1]}")
q_update(Q, 1, 0, 2.0, 2, True, 0.5, 0.9)
check(approx(Q[1, 0], 1.0), "q_update on a terminal transition uses target = r", f"got {Q[1, 0]}")

# ---- plan: replays remembered transitions
rng = np.random.default_rng(0)
Q = np.zeros((3, 2))
Q[2] = [0.0, 3.0]
plan(Q, {(0, 1): (1.0, 2, False)}, 2, rng, 0.5, 0.9)
check(approx(Q[0, 1], 1.85 + 0.5 * (3.7 - 1.85)),
      "plan(n=2) with one remembered transition applies the same update twice → Q(0,1) = 2.775",
      f"got {Q[0, 1]}. Each planning step: pick (s, a) from the model, look up (r, s2, terminated), q_update.")
Q = np.zeros((3, 2))
plan(Q, {}, 5, rng, 0.5, 0.9)
check(approx(Q, 0), "plan with an empty model does nothing (and does not crash)")
Q = np.zeros((4, 2))
model = {(0, 0): (1.0, 3, True), (1, 1): (2.0, 3, True)}
plan(Q, model, 200, np.random.default_rng(1), 1.0, 0.9)
check(approx(Q[0, 0], 1.0) and approx(Q[1, 1], 2.0) and approx(Q.sum(), 3.0),
      "plan samples among ALL remembered pairs and touches nothing else",
      f"Q = {Q.tolist()}")

# ---- dyna_q on the maze
res = {}
for n_plan in (0, 10):
    later, final = [], []
    for seed in range(5):
        env = make_maze()
        Q, steps, model = dyna_q(env, 30, n_plan, seed=seed)
        later.append(sum(steps[1:10]))
        final.append(np.mean(steps[-5:]))
    res[n_plan] = (np.mean(later), np.mean(final))
check(len(steps) == 30 and Q.shape == (env.n_states, 4), "dyna_q returns Q (n_states × 4) and one step count per episode",
      f"Q shape {getattr(Q, 'shape', None)}, {len(steps)} step counts")
ok_model = isinstance(model, dict) and len(model) > 20 and all(
    env.P[s][a][0][1:] == (s2, r, d) for (s, a), (r, s2, d) in model.items())
check(ok_model, "model maps (s, a) → (r, s2, terminated) exactly as the (deterministic) maze behaves",
      f"{len(model) if isinstance(model, dict) else model!r} entries; store model[(s, a)] = (r, s2, terminated) after each real step")
check(res[10][0] < 0.5 * res[0][0],
      "10 planning steps per real step: episodes 2–10 take less than half the steps of plain Q-learning",
      f"mean total steps in episodes 2–10: n=0 → {res[0][0]:.0f}, n=10 → {res[10][0]:.0f}. "
      "Call plan(...) after every real step, using the same Q table.")
check(res[10][1] < 22, "With planning, the agent runs the maze in ≈ 14–20 steps by episode 30",
      f"mean steps over episodes 26–30: {res[10][1]:.1f} (shortest path is 14; ε = 0.1 adds a few)")
