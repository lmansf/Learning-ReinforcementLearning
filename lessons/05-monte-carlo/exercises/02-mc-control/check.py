import numpy as np

epsilon_greedy = require("epsilon_greedy")
mc_control = require("mc_control")
evaluate = require("evaluate")

# ---- epsilon_greedy
rng = np.random.default_rng(0)
q = np.array([0.1, 0.9, 0.3])
picks = [epsilon_greedy(q, 0.0, rng) for _ in range(200)]
check(all(int(p) == 1 for p in picks), "ε = 0 always picks argmax(q)", f"picked {sorted(set(picks))}")
rng = np.random.default_rng(1)
picks = np.array([epsilon_greedy(q, 0.3, rng) for _ in range(20_000)])
frac = [(picks == a).mean() for a in range(3)]
check(abs(frac[1] - 0.8) < 0.02 and abs(frac[0] - 0.1) < 0.02 and abs(frac[2] - 0.1) < 0.02,
      "ε = 0.3 over 3 actions → greedy 0.7 + 0.3/3 = 0.8, others 0.1 each",
      f"observed frequencies {[round(f, 3) for f in frac]}. The random branch should pick among ALL actions "
      "(including the greedy one).")
picks = [epsilon_greedy(q, 1.0, rng) for _ in range(3000)]
check(set(int(p) for p in picks) == {0, 1, 2}, "ε = 1 explores every action", f"saw {sorted(set(picks))}")


# ---- tiny scripted envs with exact answers
class TwoStep:
    """S0 --(any action, r=0)--> S1 --(action 1: r=+1, action 0: r=0)--> end."""
    n_actions = 2
    def reset(self, seed=None):
        self.s = "S0"
        return "S0", {}
    def step(self, a):
        if self.s == "S0":
            self.s = "S1"
            return "S1", 0.0, False, False, {}
        return "END", float(a == 1), True, False, {}


Q, rets = mc_control(TwoStep(), 600, epsilon=0.2, gamma=0.9, seed=0)
check(isinstance(Q, dict) and "S1" in Q and np.shape(Q["S1"]) == (2,),
      "Q is a dict {state: array of length n_actions}", f"got {type(Q).__name__} with keys {list(Q)[:5]}")
check(len(rets) == 600, "returns has one entry per training episode", f"len = {len(rets)}")
check(approx(Q["S1"], [0.0, 1.0], 1e-9), "Deterministic rewards ⇒ Q(S1) = [0, 1] exactly",
      f"got {np.round(Q['S1'], 4)}. Q(s,a) should be the AVERAGE return after taking a in s.")


class Delayed:
    """S0 --(any action, r=0)--> S1 --(any action, r=1)--> end. Every return from S0 is exactly γ."""
    n_actions = 2
    def reset(self, seed=None):
        self.s = "S0"
        return "S0", {}
    def step(self, a):
        if self.s == "S0":
            self.s = "S1"
            return "S1", 0.0, False, False, {}
        return "END", 1.0, True, False, {}


Qd, _ = mc_control(Delayed(), 200, epsilon=0.5, gamma=0.9, seed=0)
check(approx(Qd["S0"], [0.9, 0.9], 1e-9) and approx(Qd["S1"], [1.0, 1.0], 1e-9),
      "γ = 0.9 discounts a reward one step later: Q(S0, ·) = 0.9, Q(S1, ·) = 1",
      f"got Q(S0) = {np.round(Qd['S0'], 4)}, Q(S1) = {np.round(Qd['S1'], 4)}. Walk backwards with G = r + γ·G.")
check(abs(np.mean(rets[-300:]) - 0.9) < 0.08, "Training returns record the episode reward (≈ 0.9 = 1 − ε/2)",
      f"mean of last 300 = {np.mean(rets[-300:]):.3f}")

# ---- Blackjack: beat the naive baseline
from rlenvs import Blackjack

Q, rets = mc_control(Blackjack(), 50_000, epsilon=0.1, seed=3)
greedy = lambda s: int(np.argmax(Q[s])) if s in Q else 0
naive = lambda s: 0 if s[0] >= 20 else 1
check(all(greedy((s, d, False)) == 0 for s in (20, 21) for d in range(1, 11)),
      "Learned policy sticks on hard 20 and 21", "hitting on 20 busts on almost every card")
hits11 = sum(greedy((11, d, False)) == 1 for d in range(1, 11))
check(hits11 >= 8, "Learned policy hits on hard 11 (you cannot bust)", f"hits against only {hits11}/10 dealer cards")
g_score = evaluate(Blackjack(), greedy, 20_000, seed=7)
n_score = evaluate(Blackjack(), naive, 20_000, seed=7)
check(g_score > n_score + 0.15, "Greedy policy from 50k episodes clearly beats 'stick on 20+'",
      f"learned {g_score:.3f} vs naive {n_score:.3f} (expect ≈ −0.07 vs −0.35). Check that the policy you "
      "ACT with is ε-greedy w.r.t. the latest Q, and that each Q(s,a) is the mean of its returns.")
