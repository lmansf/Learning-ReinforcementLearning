import random
import rlhud
from corerunner import CoreRunner


def run_episode(env, policy, seed=None):
    """Play one episode. Returns (total_reward, n_steps)."""
    obs, info = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps


def random_policy(obs):
    """Ignore the observation, pick one of the 5 actions uniformly at random."""
    return random.randrange(5)


if __name__ == "__main__":
    random.seed(0)
    env = CoreRunner(level=1)
    totals = [run_episode(env, random_policy)[0] for _ in range(500)]
    rlhud.plot(totals, label="random agent", title="Random baseline · Core Runner L1", smooth=25)
    rlhud.stat("Mean return", sum(totals) / len(totals))
    rlhud.stat("Best return", max(totals))

    # Record one episode so you can watch it in the Visuals tab.
    obs, _ = env.reset(seed=1)
    frames = [env.snapshot()]
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(random_policy(obs))
        frames.append(env.snapshot())
        done = term or trunc
    rlhud.replay(frames, title="Random agent replay")
