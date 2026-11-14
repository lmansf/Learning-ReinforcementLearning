import os
import numpy as np
from corerunner import CoreRunner

epsilon_at = require("epsilon_at")
nstep_return = require("nstep_return")
double_dqn_targets = require("double_dqn_targets")
huber_grad = require("huber_grad")
load_agent = require("load_agent")
MLP = require("MLP")
WEIGHTS = require("WEIGHTS")

check(approx(epsilon_at(0, 1000), 1.0), "ε starts at 1.0", f"got {epsilon_at(0, 1000)}")
check(approx(epsilon_at(200, 1000, 0.4, 0.05), 0.525), "ε is halfway at 20% when explore_frac=0.4", f"got {epsilon_at(200, 1000, 0.4, 0.05)}")
check(approx(epsilon_at(900, 1000, 0.4, 0.05), 0.05), "ε stays at eps_end afterwards", f"got {epsilon_at(900, 1000, 0.4, 0.05)}")

G, disc = nstep_return([1.0, 2.0, 3.0], 0.5)
check(approx(G, 2.75) and approx(disc, 0.125), "nstep_return([1,2,3], 0.5) == (2.75, 0.125)", f"got ({G}, {disc})")
G, disc = nstep_return([4.0], 0.9)
check(approx(G, 4.0) and approx(disc, 0.9), "One-step return bootstraps with γ", f"got ({G}, {disc})")


def linear_net(W):
    net = MLP.__new__(MLP)
    net.W, net.b = [np.asarray(W, float)], [np.zeros(np.shape(W)[1])]
    return net

# s' is one-hot; online prefers action 0, target values action 0 at 1.0 and action 1 at 9.0
online = linear_net([[5.0, 1.0]])
target = linear_net([[1.0, 9.0]])
S2 = np.array([[1.0], [1.0]])
y = double_dqn_targets(online, target, R=np.array([0.5, 0.5]), S2=S2, D=np.array([0.0, 1.0]), disc=np.array([0.9, 0.9]))
check(approx(y, [0.5 + 0.9 * 1.0, 0.5]), "Double-DQN target: online net picks a*, target net evaluates it; done masks the bootstrap",
      f"got {np.round(y, 3).tolist()}, expected [1.4, 0.5] (plain max over the target net would give 8.6)")

g = huber_grad(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.5, -0.5, 3.0, -3.0]))
check(approx(g, [-0.125, 0.125, -0.25, 0.25]), "huber_grad clips errors to ±1 and averages over the batch", f"got {np.round(g, 4).tolist()}")

path = WEIGHTS
check(os.path.exists(path), "Trained weights found in your workspace",
      f"{os.path.abspath(path)} does not exist yet — press RUN to train and save your agent, then CHECK")
if os.path.exists(path):
    policy = load_agent(path)
    env = CoreRunner(level=4, obs="vector")
    returns, escapes = [], 0
    for i in range(50):                       # 50 wrecks your agent has never seen
        obs, _ = env.reset(seed=777_000 + i)
        G = 0.0
        while True:
            obs, r, term, trunc, info = env.step(int(policy(obs)))
            G += r
            if term or trunc:
                break
        returns.append(G)
        escapes += info["event"] == "escaped"
    mean = float(np.mean(returns))
    check(mean >= 15.0, "Mean return ≥ 15 on 50 unseen wrecks", f"mean return {mean:.1f} (a random agent scores about −20; the hand-coded autopilot about +31)")
    check(escapes >= 23, "Extracts from at least 45% of the unseen wrecks", f"escaped {escapes}/50")
