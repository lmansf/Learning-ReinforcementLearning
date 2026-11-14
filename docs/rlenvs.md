---
title: rlenvs
---

## env.reset / env.step
sig: `obs, info = env.reset(seed=None)` · `obs, reward, terminated, truncated, info = env.step(action)`
match: env.reset(, env.step(
tags: gym API, episode loop, interface

Every bundled environment (rlenvs, `CoreRunner`, `InventoryEnv`) follows the Gymnasium API. `reset` starts an episode; `step` applies one action and returns **five** values.

```python
from rlenvs import GridWorld
env = GridWorld()
obs, info = env.reset(seed=0)          # (8, {})
G = 0.0
while True:
    obs, r, terminated, truncated, info = env.step(1)   # always RIGHT
    G += r
    if terminated or truncated:
        break
```

> [!WARN] Seed once
> Pass `seed=` on the *first* `reset` only (or the episodes become identical copies). Later resets continue the same random stream.

## terminated vs truncated
sig: `terminated` → reached a terminal state · `truncated` → stopped by a time/battery limit
match: truncated, trunc
tags: done, bootstrap, time limit, terminal

Two different reasons an episode stops. **terminated**: the MDP really ended (goal, pit, pole fell) — the future value is 0. **truncated**: we just ran out of steps — the state still has value. End the loop on either, but only zero the bootstrap on `terminated`.

```python
gamma, r, v_next = 0.99, 1.0, 5.0
for terminated in (True, False):
    target = r if terminated else r + gamma * v_next
    print(terminated, target)          # True 1.0 / False 5.95
done = True or False                    # loop condition: terminated or truncated
```

> [!WARN] Bootstrapping through a time limit
> Using `done = terminated or truncated` as the mask in the TD target teaches the agent that the last CartPole state (500 steps, pole still up) is worth 0. Mask with `terminated` only.

## env.n_states / env.n_actions
sig: `env.n_states` (tabular envs) · `env.n_actions` (all envs)
match: env.n_states, env.n_actions, .n_actions
tags: size, Q-table shape, spaces

Sizes of the discrete spaces — use them to allocate tables: `Q = np.zeros((env.n_states, env.n_actions))`.

| env | n_states | n_actions |
|---|---|---|
| `GridWorld(layout)` | rows × cols (walls included) | 4 |
| `CliffWalking()` | 48 | 4 |
| `KArmedBandit(k)` | – (one state, obs 0) | k |
| `Blackjack()` | – (tuple obs) | 2 |
| `CartPole()` / `MountainCar()` | – (`obs_size` 4 / 2) | 2 / 3 |
| `CoreRunner(level)` | `env.n_states` | 5 |

```python
import numpy as np
from rlenvs import CliffWalking
env = CliffWalking()
Q = np.zeros((env.n_states, env.n_actions))
Q.shape                  # (48, 4)
```

## env.obs_size
sig: `env.obs_size` → length of a vector observation
match: .obs_size
tags: input size, network, features

Number of floats in a continuous observation: `CartPole` 4, `MountainCar` 2, `CoreRunner(obs="vector")` 133. Use it as the network's input size.

```python
from rlenvs import CartPole
env = CartPole()
sizes = [env.obs_size, 64, 64, env.n_actions]    # [4, 64, 64, 2]
```

## KArmedBandit
sig: `KArmedBandit(k=10, seed=None, nonstationary=False, drift=0.01, mean=0.0)`
match: KArmedBandit(
tags: bandit, arms, exploration, testbed

The k-armed Gaussian bandit: pulling arm `a` pays `N(q*[a], 1)`. True means `q_star` are drawn from `N(mean, 1)`; with `nonstationary=True` they start equal and random-walk by `N(0, drift)` after every pull.

```python
from rlenvs import KArmedBandit
bandit = KArmedBandit(k=3, seed=0)
bandit.q_star            # array([ 0.12573022, -0.13210486,  0.64042265])
bandit.optimal_action    # 2
r = bandit.pull(2)       # one noisy reward around 0.64
```

> [!WARN] Fresh testbed per run
> For a testbed average, create a *new* bandit (different seed) for each run; reusing one bandit averages over noise but not over problems.

## bandit.pull / optimal_action / q_star
sig: `bandit.pull(a)` → float · `bandit.optimal_action` → int · `bandit.q_star` → ndarray (k,)
match: .pull(, .optimal_action, .q_star
tags: reward, best arm, % optimal

`pull` returns one reward. `optimal_action` is the index of the arm with the highest *true* mean (re-computed each time — it can change in a nonstationary bandit). Use it to plot "% optimal action". Gym-style `reset()`/`step(a)` also exist (one state, never terminates).

```python
from rlenvs import KArmedBandit
b = KArmedBandit(k=5, seed=1)
hits = sum(int(a == b.optimal_action) for a in [0, 1, 2, 3, 4])   # 1
obs, r, term, trunc, info = b.step(0)    # obs == 0, term == False
```

## GridWorld
sig: `GridWorld(layout=None, slip=0.0, step_reward=0.0, rewards=None, max_steps=200, seed=None)`
match: GridWorld(, env.layout
tags: gridworld, MDP, tabular, dynamic programming

A tabular grid MDP from an ASCII map: `S` start, `.` floor, `#` wall, `G` goal (+1, terminal), `X` pit (−1, terminal). Moves into walls/borders stay put. `slip` = probability the move goes to one of the two perpendicular directions (split equally). Default layout: `["...G", ".#.X", "S..."]`.

```python
from rlenvs import GridWorld
env = GridWorld(["...G",
                 ".#.X",
                 "S..."], slip=0.1, step_reward=-0.04)
env.n_states, env.n_actions, env.start     # (12, 4, 8)
env.rewards                                # {'G': 1.0, 'X': -1.0}
```

> [!WARN] Wall states exist
> States are numbered row-major *including* walls, so `n_states` counts unreachable wall cells. Skip `env.wall_states` (and `env.terminal_states`) when you loop over states to update.

## GridWorld.P (the model)
sig: `env.P[s][a]` → list of `(prob, next_state, reward, terminated)`
match: env.P[, .P[s]
tags: model, transitions, dynamic programming, Bellman

The full transition model, for dynamic programming. Each outcome is a tuple; probabilities for one `(s, a)` sum to 1. Terminal and wall states have a single self-loop `(1.0, s, 0.0, True)`.

```python
from rlenvs import GridWorld
env = GridWorld(slip=0.2)
env.P[8][0]      # [(0.8, 4, 0.0, False), (0.1, 9, 0.0, False), (0.1, 8, 0.0, False)]
gamma, V = 0.9, [0.0] * env.n_states
q = sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in env.P[8][0])
```

> [!WARN] Terminal next states
> Multiply the bootstrap by `(not done)` (or keep `V[terminal] == 0`). Otherwise a stale value in a terminal state leaks into its neighbours.

## GridWorld.state_of / pos_of
sig: `env.state_of(r, c)` → s · `env.pos_of(s)` → (r, c)
match: .state_of(, .pos_of(
tags: index, row, column, coordinates

Convert between `(row, col)` and the row-major state number `s = r * cols + c`. Also on `CliffWalking`.

```python
from rlenvs import GridWorld
env = GridWorld()
env.state_of(2, 0)       # 8   (the S cell)
env.pos_of(7)            # (1, 3)
```

## GridWorld.start / terminal_states / wall_states
sig: `env.start` → int · `env.terminal_states` → set · `env.wall_states` → set
match: env.start, .terminal_states, .wall_states
tags: start, terminal, walls, skip

The start state and the sets of terminal (`G`, `X`) and wall (`#`) states.

```python
from rlenvs import GridWorld
env = GridWorld()
env.start, env.terminal_states, env.wall_states    # (8, {3, 7}, {5})
active = [s for s in range(env.n_states)
          if s not in env.terminal_states | env.wall_states]
len(active)                                         # 9
```

> [!WARN] Sets, not lists
> Test membership with `in`; don't index them (`env.terminal_states[0]` raises `TypeError`).

## GridWorld.to_grid
sig: `env.to_grid(v)` → 2-D float array (rows, cols)
match: .to_grid(
tags: reshape, heatmap, display

Reshapes a per-state vector into the grid layout for `rlhud.grid`; wall cells are set to 0. `CliffWalking.to_grid` reshapes to `(4, 12)`.

```python
from rlenvs import GridWorld
env = GridWorld()
env.to_grid(range(12))
# array([[ 0.,  1.,  2.,  3.],
#        [ 4.,  0.,  6.,  7.],
#        [ 8.,  9., 10., 11.]])
```

## GridWorld.policy_grid
sig: `env.policy_grid(policy)` → 2-D list of actions (None on terminal/wall cells)
match: .policy_grid(
tags: arrows, policy display, greedy

Turns a policy — an action per state, *or* an `(n_states, n_actions)` probability table (argmax is taken) — into the arrow grid `rlhud.grid(..., policy=...)` expects.

```python
from rlenvs import GridWorld
env = GridWorld()
env.policy_grid([1] * env.n_states)
# [[1, 1, 1, None], [1, None, 1, None], [1, 1, 1, 1]]
```

> [!WARN] Pass actions, not values
> Give it `Q.argmax(axis=1)` (or `Q` itself), not `V`. Only on `GridWorld` — `CliffWalking` has none; use `np.asarray(Q.argmax(1)).reshape(4, 12)`.

## GridWorld.walls_mask
sig: `env.walls_mask()` → 2-D list of bools
match: .walls_mask(
tags: walls, display

Boolean grid, True on `#` cells — pass it as `rlhud.grid(..., walls=env.walls_mask())`.

```python
from rlenvs import GridWorld
GridWorld().walls_mask()[1]     # [False, True, False, False]
```

## GridWorld.labels
sig: `env.labels()` → 2-D list of strings ("S", "G", "X" or "")
match: .labels(
tags: labels, display, start, goal

Cell captions for `rlhud.grid(..., labels=env.labels())`. On `CliffWalking` the cliff cells are labelled `≈`.

```python
from rlenvs import GridWorld
GridWorld().labels()     # [['', '', '', 'G'], ['', '', '', 'X'], ['S', '', '', '']]
```

## Action constants
sig: `UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3` · `ACTION_NAMES` · CoreRunner adds `4 = WAIT`
match: ACTION_NAMES, UP, RIGHT, DOWN, LEFT
tags: actions, directions, names

Grid actions are integers in clockwise order starting at up. The same numbering is used by `rlhud.grid` arrows.

```python
from rlenvs import ACTION_NAMES, UP, LEFT
ACTION_NAMES[LEFT]       # 'LEFT'
UP                       # 0
```

## CliffWalking
sig: `CliffWalking(max_steps=500)` — 4×12, states 0..47, actions UP RIGHT DOWN LEFT
match: CliffWalking(
tags: cliff, SARSA, Q-learning, tabular

The classic cliff (Sutton & Barto Ex. 6.6). Start bottom-left (36), goal bottom-right (47). Every step −1; stepping into the cliff gives **−100 and teleports you back to the start** (the episode does *not* end; `info["fell"]` is True). Deterministic, no `P` model.

```python
from rlenvs import CliffWalking
env = CliffWalking()
s, _ = env.reset()                # 36
env.step(1)                       # (36, -100.0, False, False, {'fell': True})
env.goal                          # 47
```

> [!WARN] Falling is not terminal
> Don't wait for a terminal flag on the fall; the agent continues from the start. Episodes end at the goal or when truncated at `max_steps`.

## Blackjack
sig: `Blackjack(seed=None)` — obs `(player_sum, dealer_card, usable_ace)`, actions 0 = stick, 1 = hit
match: Blackjack(
tags: blackjack, Monte Carlo, episodic, tuple state

Simplified blackjack with an infinite deck (S&B Example 5.1). `player_sum` 4–21, `dealer_card` 1–10 (ace = 1), `usable_ace` bool. Reward only at the end: +1 win, 0 draw, −1 loss. Hitting above 21 loses immediately; sticking plays out the dealer (hits below 17).

```python
from rlenvs import Blackjack
env = Blackjack(seed=0)
obs, _ = env.reset()              # ((17, 7, False), {})
obs, r, terminated, truncated, _ = env.step(0)   # stick → (…, -1.0, True, False, {})
```

> [!WARN] Tuple states
> Observations are tuples — index dict-based tables (`Q[obs]`), or map them yourself, e.g. `Q[p_sum, dealer, int(ace)]` in a `(22, 11, 2, 2)` array.

## Blackjack internals
sig: `env.player` / `env.dealer` (lists of cards) · `env._card()` · `env._obs()`
match: env.player, env.dealer, ._card(, ._obs(
tags: hands, exploring starts, internals

The hands are plain lists of card values (ace = 1, faces = 10). Setting them directly gives *exploring starts* — begin from any state you like.

```python
from rlenvs import Blackjack
env = Blackjack(seed=0); env.reset()
env.player, env.dealer = [10, 3], [7, 5]
env._obs()                         # (13, 7, False)
```

> [!WARN] Private API
> Underscore methods are internal and only used here to construct start states; normal agents should stick to `reset`/`step`.

## CartPole
sig: `CartPole(max_steps=500, seed=None)` — obs `[x, ẋ, θ, θ̇]` (float32), actions 0 = left, 1 = right
match: CartPole(
tags: cartpole, DQN, policy gradient, continuous state

Balance a pole on a cart. +1 reward per step; **terminates** when |θ| > 12° or |x| > 2.4, **truncates** at `max_steps` (500 = solved). `obs_size = 4`, `n_actions = 2`.

```python
from rlenvs import CartPole
env = CartPole(seed=0)
obs, _ = env.reset()
obs.shape, obs.dtype               # ((4,), dtype('float32'))
obs, r, terminated, truncated, _ = env.step(1)    # r == 1.0
```

> [!WARN] Truncation at 500
> Reaching 500 steps is `truncated`, not `terminated`. Bootstrap from that last state (see *terminated vs truncated*).

## MountainCar
sig: `MountainCar(max_steps=200, seed=None)` — obs `[position, velocity]`, actions 0 = left, 1 = none, 2 = right
match: MountainCar(
tags: mountain car, tile coding, sparse reward

An under-powered car in a valley must rock back and forth to reach `position ≥ 0.5`. −1 per step, terminates at the flag, truncates at `max_steps`. Position ∈ [−1.2, 0.6], velocity ∈ [−0.07, 0.07]; start position uniform in [−0.6, −0.4].

```python
from rlenvs import MountainCar
env = MountainCar(seed=0)
obs, _ = env.reset()               # array([-0.47260767,  0.  ], dtype=float32)
obs, r, terminated, truncated, _ = env.step(2)   # r == -1.0
```

> [!WARN] Sparse, flat reward
> Every step is −1 until the goal, so a random policy almost never succeeds in 200 steps. Optimistic initial values (Q = 0 with tile coding) are what drives exploration.

## MountainCar.low / high
sig: `MountainCar.low` → `array([-1.2, -0.07])` · `MountainCar.high` → `array([0.6, 0.07])`
match: env.low, env.high, MountainCar.low, MountainCar.high
tags: bounds, normalise, tile coder

Observation bounds — feed them to a tile coder or scale features into [0, 1].

```python
from rlenvs import MountainCar
import numpy as np
env = MountainCar()
x = np.array([-0.5, 0.0])
(x - env.low) / (env.high - env.low)    # array([0.38888889, 0.5       ])
```

## Gymnasium compatibility
sig: `import gymnasium as gym; env = gym.make("CartPole-v1")` (optional)
match: gymnasium, gym.make(
tags: gymnasium, gym, optional

The bundled envs mimic Gymnasium's API, so the same agent code runs on `gym.make("CartPole-v1")` if you have it installed. It is never required here; checks always use `rlenvs`.

```python
try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n        # Gymnasium uses spaces, not n_actions
except ImportError:
    from rlenvs import CartPole
    env = CartPole(); n_actions = env.n_actions
```
