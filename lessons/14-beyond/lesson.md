---
title: Beyond the Training Ground
chapter: Capstone
summary: Where to go next — real libraries, bigger problems, and the open frontiers of RL.
minutes: 15
---

# Beyond the training ground

You have built every core idea of reinforcement learning by hand. That is the hard part.
Everything from here is scale, engineering, and a handful of new ideas layered on the same
foundations: **values, policies, bootstrapping, exploration**.

## The map you now hold

| Family | You built | Industrial-strength descendants |
|---|---|---|
| Bandits | ε-greedy, UCB, gradient bandit | Thompson sampling, contextual bandits (recommendation, ads, A/B testing) |
| Planning with a model | policy & value iteration, Dyna-Q | MCTS, AlphaZero, MuZero, model-based RL (Dreamer) |
| Model-free value-based | MC, SARSA, Q-learning, DQN | Rainbow, distributional RL (C51, QR-DQN), R2D2 |
| Policy gradient | REINFORCE, actor-critic, PPO | TRPO, IMPALA, RLHF (PPO / DPO for language models) |
| Continuous control | — | DDPG, TD3, **SAC** |

> [!KEY] The one sentence to remember
> Almost every RL algorithm is generalised policy iteration: evaluate the current behaviour
> a little, improve it a little, repeat — with different choices for *how* to estimate and *how*
> to improve.

## Use real libraries

Your own implementations are perfect for understanding. For real projects, reach for battle-tested
code. Install into the same Python that runs this app, then restart `server.py`:

```bash
pip install gymnasium torch stable-baselines3
```

**Gymnasium** is the standard environment API — the one you have been using all along:

```python
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Stable-Baselines3** gives you tuned PPO / DQN / SAC in three lines:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=0).learn(total_timesteps=50_000)
```

> [!TIP] Wrap Core Runner
> Core Runner already speaks the Gymnasium step API. Wrapping it as a real `gym.Env`
> (declare `observation_space` / `action_space`) lets you throw SB3's PPO at level 4 and compare
> it with your own DQN — a great weekend project. The sandbox below is a good place to start.

## Directions worth exploring

- **Continuous actions** — robotics-style control with SAC on `Pendulum-v1` or MuJoCo tasks.
- **Partial observability** — recurrent policies (LSTM) when the state is hidden.
- **Exploration bonuses** — count-based and curiosity methods for sparse rewards.
- **Offline RL** — learning from a fixed dataset without interacting (CQL, IQL).
- **Multi-agent RL** — self-play, cooperation, emergent behaviour.
- **RL from human feedback** — how language models are fine-tuned with a learned reward model.

## Reading list

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — free online; this course
  follows its structure through chapter 13.
- OpenAI *Spinning Up in Deep RL* — concise deep RL theory and clean reference implementations.
- David Silver's RL lecture series (UCL) — the classic video course.
- The CleanRL project — single-file, readable implementations of modern algorithms.

```quiz
{"q": "You want to control a robot arm with continuous joint torques. Which algorithm family from the table is the natural starting point?",
 "options": ["Tabular Q-learning", "DQN", "SAC / TD3 (continuous-control actor-critic)", "UCB bandits"],
 "answer": 2,
 "explain": "DQN needs a max over a discrete action set. Continuous actions call for actor-critic methods that output actions directly, such as SAC or TD3 (PPO works too)."}
```

```exercise
01-open-sandbox
```

## Signing off

You started with a random drone bumping into walls. You end with agents that plan, learn from
experience and generalise to worlds they have never seen. Keep the reward signal strong.
