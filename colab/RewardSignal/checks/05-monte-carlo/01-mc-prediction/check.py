generate_episode = require("generate_episode")
mc_prediction = require("mc_prediction")


class LoopEnv:
    """Deterministic scripted env:  A --(r=1)--> A --(r=0)--> B --(r=2)--> end.
    State A is visited twice, so first-visit and every-visit MC disagree on it."""
    def reset(self, seed=None):
        self.t, self.seed_seen = 0, seed
        return "A", {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            return "A", 1.0, False, False, {}
        if self.t == 2:
            return "B", 0.0, False, False, {}
        return "END", 2.0, True, False, {}


class CoinEnv:
    """One-step episodes from state 'S'. Rewards alternate 1, 0, 1, 0, ... across episodes,
    so the correct average after an even number of episodes is exactly 0.5."""
    def __init__(self):
        self.k = 0
    def reset(self, seed=None):
        return "S", {}
    def step(self, action):
        self.k += 1
        return "T", float(self.k % 2), True, False, {}


class TruncEnv:
    """Never terminates on its own — it is truncated after 4 steps with reward 1 each."""
    def reset(self, seed=None):
        self.t = 0
        return 0, {}
    def step(self, action):
        self.t += 1
        return self.t, 1.0, False, self.t >= 4, {}


# ---- generate_episode
env = LoopEnv()
seen = []
ep = generate_episode(env, lambda s: (seen.append(s), 7)[1], seed=11)
check(isinstance(ep, list) and len(ep) == 3, "generate_episode returns one triple per step (3 here)",
      f"got {ep!r}")
check(list(ep) == [("A", 7, 1.0), ("A", 7, 0.0), ("B", 7, 2.0)],
      "Triples are (state, action, reward-after-action)",
      f"expected [('A', 7, 1.0), ('A', 7, 0.0), ('B', 7, 2.0)], got {ep!r}. The state must be the one "
      "the action was taken IN, not the next state.")
check(seen == ["A", "A", "B"], "Policy is called with the current observation", f"policy saw {seen}")
check(env.seed_seen == 11, "seed is passed to env.reset", f"reset got seed={env.seed_seen}")
ep = generate_episode(TruncEnv(), lambda s: 0)
check(len(ep) == 4, "generate_episode also stops on truncated", f"got {len(ep)} steps")

# ---- first-visit vs every-visit on the scripted loop (hand-computed)
V = mc_prediction(LoopEnv(), lambda s: 0, 1, gamma=1.0, first_visit=True)
check(isinstance(V, dict), "mc_prediction returns a dict {state: value}", f"got {type(V).__name__}")
check(approx(V.get("A", 99), 3.0) and approx(V.get("B", 99), 2.0),
      "First-visit, γ=1: V(A) = 1+0+2 = 3, V(B) = 2",
      f"got V(A)={V.get('A')}, V(B)={V.get('B')}. G₀ = 3 is the return from the FIRST visit of A.")
check("END" not in V, "Terminal observation gets no value entry",
      "the observation returned on the final step is never a state you act in")

V = mc_prediction(LoopEnv(), lambda s: 0, 1, gamma=1.0, first_visit=False)
check(approx(V.get("A", 99), 2.5), "Every-visit, γ=1: V(A) = mean(3, 2) = 2.5",
      f"got V(A)={V.get('A')}. Returns from both visits of A are 3 and 2.")

V = mc_prediction(LoopEnv(), lambda s: 0, 1, gamma=0.5, first_visit=True)
check(approx(V.get("A", 99), 1.5) and approx(V.get("B", 99), 2.0),
      "First-visit, γ=0.5: V(A) = 1 + 0.5·0 + 0.25·2 = 1.5",
      f"got V(A)={V.get('A')}, V(B)={V.get('B')}. Use G = r + γ·G walking backwards.")
V = mc_prediction(LoopEnv(), lambda s: 0, 1, gamma=0.5, first_visit=False)
check(approx(V.get("A", 99), 1.25), "Every-visit, γ=0.5: V(A) = mean(1.5, 1.0) = 1.25", f"got {V.get('A')}")

# ---- averaging across episodes
V = mc_prediction(CoinEnv(), lambda s: 0, 10)
check(approx(V.get("S", 99), 0.5), "Averages returns over all episodes (1,0,1,0,… → 0.5)",
      f"got {V.get('S')}. Are you keeping only the last return, or dividing by the wrong count?")

# ---- sanity on Blackjack with the 'stick on 20' policy
from rlenvs import Blackjack

def stick20(obs):
    return 0 if obs[0] >= 20 else 1

V = mc_prediction(Blackjack(seed=0), stick20, 30_000, seed=0)
v21 = [V[(21, d, False)] for d in range(1, 11) if (21, d, False) in V]
v20 = [V[(20, d, False)] for d in range(2, 11) if (20, d, False) in V]
v15 = [V[(15, d, False)] for d in range(1, 11) if (15, d, False) in V]
n_main = sum(1 for k in V if isinstance(k, tuple) and 12 <= k[0] <= 21)
check(n_main >= 190, "Blackjack: (almost) all 200 states with sums 12–21 get visited",
      f"only {n_main} of them — pass the seed to the FIRST reset only, otherwise every episode deals the same cards")
check(len(v21) == 10 and min(v21) > 0.4, "Blackjack: holding 21 (no usable ace) is worth > 0.4 against every dealer card",
      f"V(21, ·, False) = {[round(v, 2) for v in v21]}")
check(len(v20) == 9 and sum(v20) / 9 > 0.5, "Blackjack: holding 20 vs dealer 2–10 is worth ≈ +0.65 on average",
      f"mean = {sum(v20) / max(1, len(v20)):.3f}")
check(len(v15) == 10 and max(v15) < -0.3, "Blackjack: hitting on 15 under this policy is a losing position",
      f"V(15, ·, False) = {[round(v, 2) for v in v15]}")
