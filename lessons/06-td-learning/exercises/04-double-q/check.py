import numpy as np

double_q_update = require("double_q_update")
double_q_learning = require("double_q_learning")
q_learning_bias = require("q_learning_bias")
MaximizationBiasMDP = require("MaximizationBiasMDP")


def tables():
    Q1 = {0: np.zeros(2), 1: np.array([1.0, 3.0, 2.0]), 2: np.zeros(1)}
    Q2 = {0: np.zeros(2), 1: np.array([5.0, -1.0, 0.0]), 2: np.zeros(1)}
    return Q1, Q2


# ---- hand-computed updates
Q1, Q2 = tables()
d = double_q_update(Q1, Q2, 0, 0, 0.5, 1, 0.5, 1.0, False, True)
check(approx(Q1[0][0], -0.25) and approx(d, -0.5),
      "update_first=True: a* = argmax Q1(B) = 1, evaluated by Q2(B, 1) = −1 → Q1(A, 0) = 0.5·(0.5 − 1) = −0.25",
      f"got Q1(A,0) = {Q1[0][0]}, δ = {d}. (1.75 would mean you evaluated with Q1 itself — plain Q-learning; "
      "2.75 means you took max of Q2.)")
check(approx(Q2[0], [0, 0]) and approx(Q2[1], [5, -1, 0]), "…and Q2 is left untouched", f"Q2 = {Q2}")

Q1, Q2 = tables()
double_q_update(Q1, Q2, 0, 0, 0.5, 1, 0.5, 1.0, False, False)
check(approx(Q2[0][0], 0.75) and approx(Q1[0], [0, 0]),
      "update_first=False: a* = argmax Q2(B) = 0, evaluated by Q1(B, 0) = 1 → Q2(A, 0) = 0.5·1.5 = 0.75",
      f"got Q2(A,0) = {Q2[0][0]}, Q1(A) = {Q1[0].tolist()}")

Q1, Q2 = tables()
double_q_update(Q1, Q2, 1, 2, -1.0, 2, 0.5, 1.0, True, True)
check(approx(Q1[1][2], 2.0 + 0.5 * (-1.0 - 2.0)), "Terminal transition: target is just r",
      f"expected Q1(B,2) = 0.5, got {Q1[1][2]}")

# ---- the maximization-bias experiment (S&B Figure 6.5)
RUNS, EPISODES = 200, 100
q_left, d_left = np.zeros(EPISODES), np.zeros(EPISODES)
for run in range(RUNS):
    left = double_q_learning(MaximizationBiasMDP(seed=run), EPISODES, seed=run)
    if run == 0:
        check(len(left) == EPISODES, "double_q_learning returns one LEFT/not-LEFT flag per episode",
              f"got {len(left)} entries for {EPISODES} episodes")
    d_left += np.asarray(left, dtype=float) / RUNS
    q_left += np.asarray(q_learning_bias(MaximizationBiasMDP(seed=run), EPISODES, seed=run), dtype=float) / RUNS
dq, qq = float(d_left.mean()), float(q_left.mean())
check(qq > 0.5, "(sanity) plain Q-learning goes LEFT most of the time early on", f"{100 * qq:.0f}% left")
check(dq < 0.35 and dq < 0.5 * qq,
      "Double Q-learning goes LEFT far less often than Q-learning (first 100 episodes)",
      f"LEFT from A: Double Q {100 * dq:.0f}% vs Q-learning {100 * qq:.0f}% (expect ≈ 20% vs 75%). "
      "Act ε-greedily on Q1 + Q2, and flip a fair coin each step to decide which table to update.")
check(d_left[50:].mean() < 0.2, "Double Q-learning approaches the optimal 5% LEFT by episodes 50–100",
      f"{100 * d_left[50:].mean():.0f}% left in episodes 51–100")
