import numpy as np
from corerunner import CoreRunner, run_episode

epsilon_by_episode = require("epsilon_by_episode")
choose_action = require("choose_action")
train_q = require("train_q")
greedy_policy = require("greedy_policy")

e0 = epsilon_by_episode(0, 100, 1.0, 0.1, 0.5)
e_mid = epsilon_by_episode(25, 100, 1.0, 0.1, 0.5)
e_end = epsilon_by_episode(80, 100, 1.0, 0.1, 0.5)
check(approx(e0, 1.0), "ε starts at eps_start", f"got {e0}")
check(approx(e_mid, 0.55), "ε is halfway at 25% of training when decay_frac=0.5", f"got {e_mid}, expected 0.55")
check(approx(e_end, 0.1), "ε stays at eps_end after the decay window", f"got {e_end}")

rng = np.random.default_rng(0)
Q = np.zeros((3, 5)); Q[1] = [0, 0, 5, 0, 0]
check(all(choose_action(Q, 1, 0.0, rng) == 2 for _ in range(50)), "ε = 0 always picks the argmax")
picks = {choose_action(Q, 0, 0.0, rng) for _ in range(200)}
check(picks == {0, 1, 2, 3, 4}, "Ties are broken randomly (all-zero row uses every action)", f"saw actions {sorted(picks)}")
picks = [choose_action(Q, 1, 1.0, rng) for _ in range(1000)]
check(500 > picks.count(2) > 100, "ε = 1 is uniformly random", f"action 2 chosen {picks.count(2)}/1000 times")

Qt, returns = train_q(level=1, episodes=400, seed=0)
env = CoreRunner(level=1)
check(getattr(Qt, "shape", None) == (env.n_states, 5), "Q has shape (n_states, 5)", f"got {getattr(Qt, 'shape', None)}")
check(len(returns) == 400, "One return per episode", f"got {len(returns)}")
G, _, info = run_episode(env, greedy_policy(Qt))
check(info["event"] == "escaped", "Greedy agent reaches the hatch", f"episode ended with event {info['event']!r}")
check(G >= 28.0 - 1e-9, "Greedy return ≥ 28.0 (par)", f"greedy return was {G:.1f}")
