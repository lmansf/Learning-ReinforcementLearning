from rlenvs import GridWorld

rollout = require("rollout")
discounted = require("discounted")
always_right = require("always_right")

env = GridWorld(["S..G", ".#..", "...."], step_reward=-0.1)
rewards = rollout(env, always_right)
check(len(rewards) == 3, "Rollout reaches the goal in 3 steps", f"got {len(rewards)} rewards: {rewards}")
check(approx(rewards, [-0.1, -0.1, 1.0]), "Rewards are [-0.1, -0.1, 1.0]", f"got {rewards}")
g = discounted(rewards, 0.9)
check(approx(g, -0.1 - 0.09 + 0.81), "discounted(rewards, 0.9) == 0.62", f"got {g}")
check(isinstance(g, float), "discounted returns a plain float", f"got {type(g).__name__}")
