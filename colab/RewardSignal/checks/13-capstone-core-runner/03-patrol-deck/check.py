import random
import numpy as np
from corerunner import CoreRunner

n_states_for = require("n_states_for")
encode_state = require("encode_state")
train_patrol = require("train_patrol")
play_greedy = require("play_greedy")

env = CoreRunner(level=3)
env.reset()
N = n_states_for(env)
check(isinstance(N, (int, np.integer)) and N > 0, "n_states_for returns a positive integer", f"got {N!r}")

# Enumerate every combination of (cell, collected mask, phase) and make sure the encoding is
# in range and never maps two different situations to the same integer.
seen = {}
collision = None
out_of_range = None
cells = [(r, c) for r in range(env.h) for c in range(env.w)]
nc = len(env.core_list)
random.seed(0)
for (r, c) in cells:
    for mask in range(2 ** nc):
        for phase in range(env.period):
            env.pos = (r, c)
            env.collected = [bool(mask >> i & 1) for i in range(nc)]
            env.t = phase + env.period * random.randint(0, 3)   # same phase, different absolute time
            s = int(encode_state(env))
            key = (r, c, mask, phase)
            if not (0 <= s < N):
                out_of_range = out_of_range or (key, s)
            if s in seen and seen[s] != key:
                collision = collision or (seen[s], key)
            seen[s] = key
check(out_of_range is None, "Every encoded state is in [0, n_states)", f"(row, col, mask, phase)={out_of_range and out_of_range[0]} gave {out_of_range and out_of_range[1]}")
check(collision is None, "Distinct situations never share a state (position, cores, sentry phase)",
      f"{collision} map to the same integer" if collision else "")
check(len(seen) <= N, "n_states is large enough")
check(N <= 4 * env.h * env.w * 2 ** nc * env.period, "n_states is not absurdly large", f"n_states={N}")

ok = 0
for seed in (0, 1):
    Q = train_patrol(3000, seed=seed)
    G, event, _ = play_greedy(Q)
    ok += G >= 44
    check(G >= 44.0, f"Seed {seed}: greedy return ≥ 44 (par)", f"got {G:.1f}, episode ended with {event!r}")
