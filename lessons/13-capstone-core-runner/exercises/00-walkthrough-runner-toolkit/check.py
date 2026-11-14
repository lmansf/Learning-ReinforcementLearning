import numpy as np
from corerunner import CoreRunner

linear = require("linear")
argmax_random = require("argmax_random")
play = require("play")
evaluate = require("evaluate")

vals = [linear(t, 100, 1.0, 0.2) for t in (0, 25, 100, 250)]
check(approx(vals, [1.0, 0.8, 0.2, 0.2]), "linear(t, 100, 1.0, 0.2) at t = 0, 25, 100, 250",
      f"expected [1, 0.8, 0.2, 0.2], got {vals}")
check(approx(linear(5, 0, 0.5, 0.0), 0.0), "linear with n = 0 jumps straight to `end` (no division by zero)",
      f"got {linear(5, 0, 0.5, 0.0)}")

rng = np.random.default_rng(0)
picks = {argmax_random(np.array([1.0, 3.0, 3.0, 0.0, 3.0]), rng) for _ in range(300)}
check(picks == {1, 2, 4}, "argmax_random returns only maxima, each of them sometimes", f"saw {sorted(picks)}")
check(isinstance(argmax_random(np.zeros(5), rng), int), "argmax_random returns a plain int")

env = CoreRunner(level=1)
seen = []
def wait(s):
    seen.append(s)
    return 4                                    # WAIT until the battery dies
G, frames, info, visits = play(env, wait)
check(approx(G, -4.0) and info["event"] == "battery", "play: 40 × WAIT on level 1 → return −4.0, truncated by the battery",
      f"got return {G}, event {info.get('event')!r}")
check(len(frames) == 41 and frames[0]["t"] == 0 and frames[-1]["t"] == 40,
      "play records the start frame plus one frame per step", f"got {len(frames)} frames")
check(all(isinstance(s, (int, np.integer)) and 0 <= s < env.n_states for s in seen),
      "the policy receives the tabular state (an int in [0, n_states))", f"first observations: {seen[:3]}")
v = np.asarray(visits)
check(v.shape == (env.h, env.w) and v[env.start] == 40 and v.sum() == 40,
      "visits counts the drone's cell after every step, shape (h, w)", f"visits =\n{v}")

ROUTE = [1, 1, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3]     # a hand-written escape route for level 1
G, _, info, _ = play(env, lambda s: ROUTE[env.t])
check(info["event"] == "escaped" and approx(G, 28.8), "play: a scripted route collects the core and escapes (28.8)",
      f"got return {G}, event {info['event']!r}")
mean, rate = evaluate(env, lambda s: ROUTE[env.t], seeds=[0, 1, 2])
check(approx(mean, 28.8) and approx(rate, 1.0), "evaluate returns (mean return, escape rate) for the route",
      f"expected (28.8, 1.0), got ({mean}, {rate})")
mean, rate = evaluate(env, lambda s: 4, seeds=[0, 1])
check(approx(mean, -4.0) and approx(rate, 0.0), "evaluate for WAIT-forever: (−4.0, 0.0)", f"got ({mean}, {rate})")
