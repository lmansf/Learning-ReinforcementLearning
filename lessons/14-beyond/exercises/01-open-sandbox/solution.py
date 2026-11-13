import importlib.util
import numpy as np
import rlhud

HAS = {m: importlib.util.find_spec(m) is not None for m in ("gymnasium", "torch", "stable_baselines3")}
rlhud.table([[k, "installed" if v else "—"] for k, v in HAS.items()], headers=["library", "status"], title="Optional libraries")

if HAS["gymnasium"]:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    source = "gymnasium CartPole-v1"
else:
    from rlenvs import CartPole
    env = CartPole()
    source = "bundled rlenvs.CartPole (pip install gymnasium for the real thing)"

rng = np.random.default_rng(0)
returns = []
for ep in range(50):
    obs, info = env.reset(seed=ep)
    G, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(int(rng.integers(2)))
        G += r
        done = term or trunc
    returns.append(G)

rlhud.plot(returns, label="random agent", title=f"Random baseline · {source}")
rlhud.stat("Mean return", float(np.mean(returns)))
print("Your playground. Replace the random agent with something smarter!")
