---
title: Capstone — Core Runner
chapter: Capstone
summary: Four missions in an original salvage game. Design the state, pick the algorithm, train an agent, and beat par.
minutes: 120
---

# CORE RUNNER

A derelict station drifts in a dead orbit. Its power cores are still warm. You command a
salvage drone with a limited battery: **collect every core, then reach the extraction hatch**.
Scalding vents and patrolling sentries end the run instantly.

You have already met the environment — your random agent stumbled through it in lesson 01.
Now you will build agents that *master* it, escalating from a lookup table to a neural network.

## Mission briefing

Try it yourself first. Click the game, then use the keyboard. Getting a feel for the
optimal route makes it much easier to judge whether your agent is any good.

```widget
{"type": "corerunner", "level": 1}
```

| Element | | Reward | Effect |
|---|---|---|---|
| Each step | | −0.1 | battery −1 |
| Bump a wall | `#` | −0.5 | you stay put |
| Energy core | ◆ | **+10** | collected once |
| Extraction hatch | ▢ | **+20** | ends the episode — **locked until every core is collected** |
| Vent | ≋ | −20 | ends the episode |
| Sentry | ▲ | −20 | ends the episode (they move every step, ping-ponging along a track) |
| Battery empty | | — | episode is *truncated* |

Actions: `0 UP`, `1 RIGHT`, `2 DOWN`, `3 LEFT`, `4 WAIT`.

```python
from corerunner import CoreRunner
env = CoreRunner(level=2)               # 1..4
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(1)
print(env.render())                      # ASCII view
frames = [env.snapshot()]                # JSON frames for rlhud.replay(frames)
```

> [!KEY] Par
> Every level has a **par** score: the return of a near-optimal run. Your missions are scored
> against it. Matching par means your agent found (essentially) the shortest safe route.

## What is "the state"?

The Markov property says the state must contain everything needed to predict the future.
In Core Runner that is more than the drone's position:

- **position** — obviously
- **which cores are already collected** — the same cell has very different value before and after
  you grab a core, and the hatch's behaviour depends on it
- **the sentries' phase** — where the sentries are *now* (and so where they will be next step)

`obs="tabular"` (the default) packs exactly these three into one integer in `[0, env.n_states)`:

$$
state = ((row · W + col) · 2ⁿᶜᵒʳᵉˢ + collected_mask) · period + (t mod period)
$$

```quiz
{"q": "Suppose the tabular state only encoded the drone's position, not which cores were collected. What breaks?",
 "options": ["Nothing — position is enough", "The same position needs different actions before and after a core is collected, so no single Q-value per cell can be right", "Q-learning becomes slower but still converges to the optimal policy"],
 "answer": 1,
 "explain": "The process is no longer Markov in that representation. Values before/after collecting a core differ hugely, so a position-only table averages incompatible situations."}
```

`env.n_states` is 50 on level 1, 392 on level 2 and 31,360 on level 3 (sentry phases multiply the
count). Level 4 generates a **new random wreck every episode**, so a table is hopeless there —
that is where function approximation comes in.

## Mission 1 — Boot Sequence

Build a tabular Q-learning agent. This is the template you will reuse for the next two
missions, so write it cleanly.

Before you start the missions, type along once. The walkthrough builds the toolkit all four
missions reuse — an ε schedule, tie-breaking argmax, the Core Runner episode loop with replays,
and an evaluation helper — with an explanation for each line. It does not solve any mission.

```exercise
00-walkthrough-runner-toolkit
```

Now the same tools, no ghost text:

```exercise
01-boot-sequence
```

> [!TIP] Debugging RL
> If the greedy return is stuck around −10 to −20 the agent has learned to *die quickly* —
> a sign it never discovered the hatch. More exploration, or a longer decay, usually fixes it.
> Always look at a replay before you look at hyperparameters.

## Mission 2 — Vent Maze

Three cores, a field of vents, and a hatch at the far corner. The reward for the full run is
over 40 steps away and the obvious shortcuts are lethal. Exploration is now the whole game.

```widget
{"type": "corerunner", "level": 2}
```

```quiz
{"q": "Early in training ε is high and the agent often walks into vents. Why is that actually useful?",
 "options": ["It isn't — vents should be removed during training", "Dying teaches the agent the large negative value of vent cells, which Q-learning then propagates backwards", "It lowers the learning rate automatically"],
 "answer": 1,
 "explain": "Q-learning only knows a vent is bad after experiencing it. Those −20 targets bootstrap into neighbouring states so the greedy policy learns to steer clear."}
```

```exercise
02-vent-maze
```

## Mission 3 — Patrol Deck

Two sentries sweep the corridors. Sometimes the right move is to **wait**. Because the tabular
state includes the sentry phase, plain Q-learning can still solve this level exactly — but the
table is now 80× bigger, so learning efficiency matters.

```widget
{"type": "corerunner", "level": 3}
```

> [!NOTE] Algorithm choice
> Anything tabular from the course is fair game: Q-learning, SARSA, Expected SARSA, n-step,
> Dyna-Q. Dyna-Q is a particularly good fit — the environment is deterministic, so a learned
> model is perfect and every planning step is free experience.

```exercise
03-patrol-deck
```

## Mission 4 — Deep Salvage

Every episode is a freshly generated 6×6 wreck. You cannot memorise a map; you have to learn a
*skill*. Switch to `obs="vector"`: a 133-number observation made of

- an egocentric **5×5 window** around the drone with five channels (wall, vent, core, sentry, exit) — 125 numbers
- **8 globals**: the relative position of the current target (nearest core, or the hatch once
  unlocked), four direction flags, whether the hatch is open, and the battery fraction

That is exactly the setting for a **Deep Q-Network**: a neural network maps the observation to
five Q-values, trained from a replay buffer against a slowly-updated target network.

```widget
{"type": "corerunner", "level": 4}
```

This mission works like real RL research: you **train in Run** (as long as you like — raise the
time limit in Settings → System if you need to), save the weights to your workspace, and
**Check evaluates your saved agent** on 50 wrecks it has never seen.

```exercise
04-deep-salvage
```

> [!WARN] Deep RL is noisy
> Two runs with different seeds can differ a lot. If training collapses, try: a smaller learning
> rate, a slower ε decay, Double-DQN targets, gradient/TD-error clipping, or simply a different seed.
> Plot the training return and watch a replay of a failure case before changing anything.

## Debrief

You took one environment from a random walk to a trained neural agent:

| Mission | Representation | Method | Why it works |
|---|---|---|---|
| 1–2 | exact tabular state | Q-learning + ε schedule | small, Markov, deterministic |
| 3 | tabular with sentry phase | Q-learning / Dyna-Q | time folded into the state keeps it Markov |
| 4 | local features | DQN | generalises across layouts it has never seen |

Ideas to push further on your own:

- **Reward shaping** with a potential function Φ(s) (e.g. negative distance to target) —
  shaping of the form γΦ(s') − Φ(s) provably keeps the optimal policy unchanged.
- A **policy-gradient** agent (REINFORCE / PPO) on level 4 — compare sample efficiency with DQN.
- Use `obs="grid"` with a small convolutional network if you install PyTorch.
- Make your own levels: `CoreRunner(layout=[...])` accepts any ASCII map.
