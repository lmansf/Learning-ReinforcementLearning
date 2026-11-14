import numpy as np
from collections import defaultdict
from rlenvs import GridWorld

wander = require("wander")
record_episode = require("record_episode")
returns_to_go = require("returns_to_go")
add_first_visits = require("add_first_visits")
averages = require("averages")

# ---------------------------------------------------------------- recording an episode
env = GridWorld(["S..G"], step_reward=-0.1)
ep = record_episode(env, lambda obs, rng: 1, np.random.default_rng(0), seed=0)
ok = isinstance(ep, list) and len(ep) == 3 and all(len(x) == 3 for x in ep)
check(ok, "record_episode returns a list of (state, action, reward) triples",
      f"got {ep!r}")
if ok:
    check([x[0] for x in ep] == [0, 1, 2], "States are the ones the actions were taken IN (0, 1, 2)",
          f"states {[x[0] for x in ep]} — append `obs` before overwriting it with the next state")
    check(approx([x[2] for x in ep], [-0.1, -0.1, 1.0]), "Rewards follow their actions: [-0.1, -0.1, 1.0]",
          f"rewards {[x[2] for x in ep]}")
env = GridWorld(["S.G"], max_steps=5)
ep = record_episode(env, lambda obs, rng: 3, np.random.default_rng(0))
check(len(ep) == 5, "A truncated episode stops at the time limit too", f"got {len(ep)} steps (max_steps=5)")
rng = np.random.default_rng(2)
acts = {wander(0, rng) for _ in range(400)}
check(acts == {0, 1, 2, 3}, "wander picks all four actions", f"saw {sorted(acts)}")

# ---------------------------------------------------------------- returns, backwards
ep = [("A", 0, 1.0), ("B", 0, 0.0), ("A", 1, 2.0)]
g = returns_to_go(ep, 0.5)
check(approx(g, [1.0 + 0.5 * 0.0 + 0.25 * 2.0, 0.0 + 0.5 * 2.0, 2.0]), "returns_to_go(…, γ = 0.5) = [1.5, 1.0, 2.0]",
      f"got {g}")
check(approx(returns_to_go(ep, 1.0), [3.0, 2.0, 2.0]), "With γ = 1 the returns are plain suffix sums")

# ---------------------------------------------------------------- first visits
ret = defaultdict(list)
add_first_visits(ep, 0.5, ret)
check(dict(ret) == {"A": [1.5], "B": [1.0]}, "Only the FIRST visit of A counts (return 1.5, not 2.0)",
      f"got {dict(ret)}")
ret_q = defaultdict(list)
add_first_visits(ep, 0.5, ret_q, by_action=True)
check(dict(ret_q) == {("A", 0): [1.5], ("B", 0): [1.0], ("A", 1): [2.0]},
      "by_action=True keys by (state, action): (A, 1) is a different pair, so it counts",
      f"got {dict(ret_q)}")
add_first_visits([("A", 0, -1.0)], 0.5, ret)
avg = averages(ret)
check(approx(avg["A"], 0.25) and approx(avg["B"], 1.0) and isinstance(avg["A"], float),
      "averages: A has returns [1.5, -1.0] → 0.25", f"got {avg}")
