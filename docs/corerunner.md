---
title: Core Runner
---

## CoreRunner
sig: `CoreRunner(level=1, obs="tabular", seed=None, battery=None, layout=None)`
match: CoreRunner(
tags: capstone, environment, drone, levels

The capstone game: pilot a salvage drone, collect every energy core, then reach the exit before the battery dies. Vents and patrol sentries end the run. Gymnasium-style `reset` / `step`; `n_actions = 5`.

```python
from corerunner import CoreRunner
env = CoreRunner(level=1)
obs, info = env.reset(seed=0)       # obs == 0 (tabular state id)
obs, r, terminated, truncated, info = env.step(1)   # RIGHT: r == -0.1
env                                  # CoreRunner(level=1, 5x5, obs='tabular')
```

> [!WARN] Reset after the end
> Calling `step` after an episode ended raises `RuntimeError: Episode is over — call reset() first.`

## CoreRunner levels
sig: `level=1` Boot Sequence · `2` Vent Maze · `3` Patrol Deck · `4` Deep Salvage (procedural)
match: level=
tags: levels, difficulty, procedural, battery

| level | name | size | cores | battery | notes |
|---|---|---|---|---|---|
| 1 | Boot Sequence | 5×5 | 1 | 40 | walls only |
| 2 | Vent Maze | 7×7 | 3 | 70 | vents are deadly |
| 3 | Patrol Deck | 7×8 | 3 | 80 | 2 sentries, `period = 70` |
| 4 | Deep Salvage | 6×6 | 2 | 50 | new random wreck every `reset` |

`battery=` overrides the step budget; `layout=` plays your own ASCII map.

```python
from corerunner import CoreRunner, LEVELS
LEVELS[3]["name"], LEVELS[3]["par"]     # ('Patrol Deck', 44.0)  par = return of a strong policy
env = CoreRunner(level=2, battery=100)
```

> [!WARN] Level 4 changes every episode
> A tabular Q-table can't generalise across layouts — use `obs="vector"` and a function approximator.

## CoreRunner actions
sig: `0 = UP · 1 = RIGHT · 2 = DOWN · 3 = LEFT · 4 = WAIT` · `CoreRunner.ACTIONS`
match: WAIT, CoreRunner.ACTIONS, env.ACTIONS
tags: actions, wait, move

Four moves plus **WAIT** (stay put — essential to let a sentry pass). Moving into a wall or the border costs a bump penalty and leaves you in place.

```python
from corerunner import CoreRunner
CoreRunner.ACTIONS          # ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
CoreRunner.n_actions        # 5
```

> [!WARN] Actions must be 0..4
> Anything else raises `ValueError`. Pass a Python `int` (e.g. `int(np.argmax(q))`).

## CoreRunner rewards
sig: `R_STEP=-0.1 · R_BUMP=-0.5 · R_CORE=+10 · R_EXIT=+20 · R_DEATH=-20`
match: R_STEP, R_BUMP, R_CORE, R_EXIT, R_DEATH
tags: reward, penalty, shaping

Every step costs −0.1; bumping a wall adds −0.5; each new core +10; reaching the **unlocked** exit +20 (terminates); a vent or sentry −20 (terminates). An empty battery **truncates** the episode. The exit is locked until all cores are collected.

```python
from corerunner import R_STEP, R_CORE, R_EXIT
best_level1 = R_CORE + R_EXIT + 20 * R_STEP     # ≈ 28 — level 1's par
```

> [!WARN] Walking onto a locked exit
> Stepping on the exit before collecting every core does nothing — no reward, no termination.

## CoreRunner observation modes
sig: `obs="tabular"` → int · `obs="vector"` → float32 (133,) · `obs="grid"` → float32 (6, H, W)
match: obs="vector", obs="grid", obs="tabular"
tags: observation, features, state, tabular

- **tabular** (default): an int in `[0, env.n_states)` = position × collected-cores × sentry phase.
- **vector**: egocentric 5×5 view in 5 channels (wall, vent, core, sentry, exit) + 8 global features — `env.obs_size == 133`.
- **grid**: full-map one-hot channels (wall, vent, core, sentry, exit, agent), shape `(6, h, w)`.

```python
from corerunner import CoreRunner
env = CoreRunner(level=4, obs="vector")
obs, _ = env.reset(seed=0)
obs.shape, obs.dtype             # ((133,), dtype('float32'))
CoreRunner(level=1, obs="grid").reset()[0].shape    # (6, 5, 5)
```

## CoreRunner.n_states / state_id
sig: `env.n_states` = h·w · 2^n_cores · period · `env.state_id()` → int
match: .state_id(
tags: tabular state, encoding, Q-table

The tabular observation is `state_id()`: `((r·w + c) · 2^n_cores + core_mask) · period + t % period`. `n_states` is the matching table size (level 1: 50, level 2: 392, level 3: 31 360).

```python
from corerunner import CoreRunner
import numpy as np
env = CoreRunner(level=2)
Q = np.zeros((env.n_states, env.n_actions))      # (392, 5)
env.reset(); env.state_id()                       # same as the obs returned by reset
```

> [!WARN] Level 4 n_states is per layout
> A new layout means the same `n_states` but *different meanings* for each id.

## CoreRunner vector layout
sig: `obs[:125]` = 5×5×5 local view · `obs[125:]` = 8 globals
match: .vector(, .obs_size
tags: features, vector, egocentric

The first 125 numbers are `v.ravel()` of a `(5 channels, 5, 5)` window centred on the drone (channels: wall/out-of-bounds, vent, uncollected core, sentry now-or-next-step, exit). The last 8: `dr, dc` to the target (nearest core, or exit when open) scaled by map size, four direction flags (target is up / right / down / left), `exit_open`, battery fraction.

```python
from corerunner import CoreRunner
env = CoreRunner(level=1, obs="vector")
obs, _ = env.reset()
local = obs[:125].reshape(5, 5, 5)        # [channel, row, col]
local[0, 2, 2]                            # 0.0 — the drone's own cell is never a wall
obs[-1]                                   # 1.0 — full battery
```

## CoreRunner.grid
sig: `env.grid()` → float32 (6, h, w)
match: env.grid(
tags: image, channels, full map

Whole-map observation as six one-hot planes: walls, vents, uncollected cores, sentries, exit (1 if open, 0.5 if locked), agent. Use it for CNN-style or fully global features.

```python
from corerunner import CoreRunner
env = CoreRunner(level=1, obs="grid"); env.reset()
g = env.grid()
g[5].nonzero()          # (array([0]), array([0]))  — the drone at (0, 0)
```

## CoreRunner info dict
sig: `info = {"event", "cores", "cores_total", "battery", "score", "t"}`
match: ["event"], ["cores"], ["cores_total"], info["battery"], ["score"]
tags: info, event, diagnostics

`step` and `reset` return an info dict. `event` is one of `start`, `up/right/down/left/wait`, `bump`, `core`, `escaped`, `vent`, `caught`, `battery`.

```python
from corerunner import CoreRunner
env = CoreRunner(level=1); env.reset(seed=0)
_, r, term, trunc, info = env.step(3)      # LEFT into the border
info["event"], r                            # ('bump', -0.6)
won = info["event"] == "escaped"
```

## CoreRunner.snapshot
sig: `env.snapshot()` → JSON-friendly dict of the current frame
match: .snapshot(
tags: replay, frame, animation

One frame for `rlhud.replay`: map size, walls, vents, remaining cores, exit (+ open), sentries, agent position, `t`, battery, score, last event.

```python
import rlhud
from corerunner import CoreRunner
env = CoreRunner(level=1)
env.reset(seed=0); frames = [env.snapshot()]
for a in [1, 1, 2, 2]:
    env.step(a); frames.append(env.snapshot())
rlhud.replay(frames, title="Four steps")
```

> [!WARN] Snapshot after reset too
> Record the first frame right after `reset`, otherwise the replay starts one step in.

## CoreRunner.render
sig: `env.render()` → multi-line ASCII string
match: env.render(
tags: ascii, debug, print

Text picture of the current state: `@` drone, `!` sentry, `#` wall, `X` vent, `C` core, `E`/`e` exit open/locked, plus a status line.

```python
from corerunner import CoreRunner
env = CoreRunner(level=1); env.reset()
print(env.render())
# @...#
# .##..
# ...#C
# #.#..
# e....
# t=0 battery=40 score=0.0 event=start
```

## run_episode
sig: `run_episode(env, policy, seed=None, record=False, max_steps=10_000)` → `(total_reward, frames, info)`
match: run_episode(
tags: rollout, evaluate, replay, policy

Rolls out `policy(obs) -> action` for one episode. With `record=True`, `frames` holds snapshots ready for `rlhud.replay`.

```python
from corerunner import CoreRunner, run_episode
env = CoreRunner(level=1)
total, frames, info = run_episode(env, lambda obs: 4, seed=0)   # WAIT until the battery dies
round(total, 1), info["event"]          # (-4.0, 'battery')
```

> [!WARN] Greedy policy for evaluation
> Pass a *greedy* policy (`lambda s: int(np.argmax(Q[s]))`) when measuring performance; an ε-greedy one under-reports what the agent has learned.

## generate_layout
sig: `generate_layout(seed, size=(6, 6), n_cores=2, n_vents=2)` → list of strings
match: generate_layout(, layout=
tags: procedural, map, level 4, custom level

Builds a random wreck in which every core and the exit are reachable without crossing a vent. Level 4 calls it on every `reset`; you can pass the result to `CoreRunner(layout=...)` to fix one map.

```python
from corerunner import CoreRunner, generate_layout
rows = generate_layout(0)
rows[0]                          # 'S.....'
env = CoreRunner(layout=rows, battery=50)
```

> [!WARN] Custom layouts have no sentries
> With `layout=` the level's sentry paths are dropped.

## CoreRunner internals (pos, collected, t, period, core_list)
sig: `env.pos` (r, c) · `env.collected` list[bool] · `env.t` · `env.period` · `env.core_list` · `env.h`, `env.w` · `env.battery`
match: env.pos =, c = env.pos, env.pos), env.collected, env.period, env.core_list, * env.w, env.h *, env.t =, env.t %, env.battery
tags: state design, raw state, Markov, internals

The raw game state — what you combine into your own Markov state in *Patrol Deck*. Sentries repeat every `period` steps, so only `t % period` matters for their positions.

```python
from corerunner import CoreRunner
env = CoreRunner(level=3); env.reset()
env.pos, env.collected, env.t, env.period      # ((0, 0), [False, False, False], 0, 70)
mask = sum(int(g) << i for i, g in enumerate(env.collected))
phase = env.t % env.period
```

> [!WARN] Absolute time is not the state
> Using `env.t` itself (instead of `t % period`) makes the table enormous and never repeats a state. Ignoring time entirely makes sentries invisible to the agent.

## CoreRunner.sentries / exit_open
sig: `env.sentries(t=None)` → list of (r, c) · `env.exit_open` → bool
match: .sentries(, .exit_open
tags: sentry, patrol, exit, lookahead

Sentry positions at time `t` (default: now) — they ping-pong along fixed paths, so you can look ahead. `exit_open` is True once every core is collected.

```python
from corerunner import CoreRunner
env = CoreRunner(level=3); env.reset()
env.sentries()            # [(2, 0), (4, 2)]
env.sentries(5)           # [(2, 5), (4, 7)]
env.exit_open             # False
```

> [!WARN] Swapping places counts as caught
> Moving *through* a sentry (you and it swap cells in one step) ends the run, just like landing on it.
