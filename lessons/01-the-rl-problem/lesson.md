---
title: The RL Problem
chapter: Foundations
summary: Agents, environments, rewards, returns and discounting — the vocabulary everything else is built on.
minutes: 25
---

# The reinforcement learning problem

Supervised learning is told the right answer for every example. Reinforcement learning is
only told *how good* the outcome was — often much later, and never why. That single difference
creates every hard problem in this course: **delayed credit**, **exploration**, and
**non-stationary data** produced by the learner's own behaviour.

## The agent–environment loop

At each time step *t*:

1. the agent observes a **state** $Sₜ$
2. it picks an **action** $Aₜ$ according to its **policy** $π(a | s)$
3. the environment returns a **reward** $Rₜ₊₁$ and the next state $Sₜ₊₁$

$$
S₀, A₀, R₁, S₁, A₁, R₂, S₂, A₂, R₃, …
$$

The interaction is often split into **episodes** (a game, a maze run) that end in a
*terminal* state. Some tasks are **continuing** and never end.

> [!NOTE] Gymnasium-style API
> Every environment in this course uses the same interface as the popular Gymnasium library:
> `obs, info = env.reset()` then `obs, reward, terminated, truncated, info = env.step(action)`.
> `terminated` means the episode truly ended (goal, crash); `truncated` means we cut it off
> (time limit). The difference matters later when we bootstrap.

## Rewards and the reward hypothesis

> [!KEY] The reward hypothesis
> All of what we mean by goals can be thought of as maximising the expected cumulative sum of a
> scalar reward signal.

The reward says **what** you want, not **how** to get it. Reward a chess agent for capturing
pieces and it will learn to capture pieces — even if that loses the game.

```quiz
{"q": "You reward a robot vacuum +1 for every piece of dust it sucks up. What might go wrong?",
 "options": ["Nothing — that is exactly the goal", "It may learn to dump dust out and re-collect it", "It will refuse to move"],
 "answer": 1,
 "explain": "Reward hacking: the agent maximises what you measured, not what you meant. Reward 'clean floor', not 'dust collected'."}
```

## Returns and discounting

The agent maximises the **return** — the sum of future rewards. With a **discount factor**
γ ∈ [0, 1], rewards further in the future count for less:

$$
Gₜ = Rₜ₊₁ + γ Rₜ₊₂ + γ² Rₜ₊₃ + … = Σₖ γᵏ Rₜ₊ₖ₊₁
$$

The return obeys a beautiful recursion that the rest of the course leans on heavily:

$$
Gₜ = Rₜ₊₁ + γ Gₜ₊₁
$$

γ near 0 makes a **myopic** agent; γ near 1 a **far-sighted** one. For continuing tasks, γ < 1
keeps the infinite sum finite. Drag the slider and watch how much a distant reward is worth.

```widget
{"type": "discount", "gamma": 0.9, "rewards": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5]}
```

```quiz
{"q": "With γ = 0.5, rewards [1, 1, 1] (R₁, R₂, R₃) give a return G₀ of…",
 "options": ["3", "1.75", "1.5", "2.5"],
 "answer": 1,
 "explain": "1 + 0.5·1 + 0.25·1 = 1.75"}
```

```exercise
01-discounted-return
```

## Policies and value functions

A **policy** π maps states to (probabilities of) actions. A **value function** says how good
it is to be somewhere *if you follow π afterwards*:

$$
v_π(s) = E_π[ Gₜ | Sₜ = s ]          q_π(s, a) = E_π[ Gₜ | Sₜ = s, Aₜ = a ]
$$

Nearly every method you will build either **estimates these values** (value-based: DP, TD,
Q-learning, DQN), **optimises π directly** (policy gradient), or does both (actor-critic, PPO).

## Exploration vs exploitation

To get reward the agent should pick what has worked before (**exploit**). To find out what
works it has to try things it is unsure about (**explore**). Doing only one of these fails.
The next lesson attacks this trade-off in its purest form.

## Your first agent

Time to drive the actual capstone environment with the dumbest possible policy — random
actions — and measure how badly it does. That number is your **baseline**: every algorithm
from here on has to beat it.

```exercise
02-agent-loop
```

> [!TIP] Look at the replay
> After **Run**, open the *Visuals* tab: the demo records an episode of your random agent in
> Core Runner. Notice how often it wanders into a wall or a vent. We will fix that.
