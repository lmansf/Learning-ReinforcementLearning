---
title: RL patterns
---

## ε-greedy with random tie-breaking
sig: `a = rng.integers(n) if rng.random() < eps else rng.choice(flatnonzero(q == q.max()))`
match: rng.random() < eps, rng.random() < epsilon, def epsilon_greedy(, self.rng.random() < self.epsilon
tags: exploration, epsilon greedy, ties, action selection

Explore with probability ε (uniformly random action), otherwise exploit — choosing **randomly among all tied best actions**.

```python
import numpy as np
def epsilon_greedy(q, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(len(q)))
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))

rng = np.random.default_rng(0)
epsilon_greedy(np.zeros(4), 0.1, rng)    # any of 0..3 — ties broken at random
```

> [!WARN] argmax ties
> `int(np.argmax(q))` always picks the first maximum, so a fresh all-zero table means "always action 0". With ε small, the agent barely explores and may look stuck.

## ε decay schedule
sig: `eps = max(eps_end, eps_start - (eps_start - eps_end) * t / decay_steps)`
match: epsilon_at(, eps_end, eps_start
tags: schedule, annealing, exploration

Start exploratory, become greedy. A linear schedule with a floor is the usual DQN choice.

```python
def epsilon_at(t, total, start=1.0, end=0.05, frac=0.5):
    return max(end, start - (start - end) * t / (frac * total))
epsilon_at(0, 1000), epsilon_at(250, 1000), epsilon_at(900, 1000)   # (1.0, 0.525, 0.05)
```

## Incremental mean update
sig: `N[a] += 1; Q[a] += (r - Q[a]) / N[a]`
match: / N[, (r - Q[a])
tags: sample average, bandit, Monte Carlo, running mean

Keeps the exact average of all rewards seen without storing them: new = old + (target − old) / n.

```python
import numpy as np
Q, N = np.zeros(2), np.zeros(2)
for r in [1.0, 3.0, 2.0]:
    a = 0
    N[a] += 1
    Q[a] += (r - Q[a]) / N[a]
Q[0]            # 2.0  — the mean of 1, 3, 2
```

> [!WARN] Update N first
> Divide by the count *including* the new sample. With `N` still 0 you divide by zero; with it lagging by one, the estimate is biased.

## Constant step-size update
sig: `V[s] += alpha * (target - V[s])`
match: alpha * (target, += alpha *
tags: TD, step size, exponential average, nonstationary

Moves an estimate a fraction α of the way to a target. It's an exponential recency-weighted average — good for nonstationary problems and the core of every TD method.

```python
v, alpha = 0.0, 0.5
for target in [1.0, 1.0, 1.0]:
    v += alpha * (target - v)
v               # 0.875
```

> [!WARN] Sign of the error
> It is `target - estimate`. Swapping them makes the estimate run *away* from the target and diverge.

## TD target with terminal masking
sig: `target = r if terminated else r + gamma * Q[s2].max()` · vectorised: `R + gamma * (1 - D) * Q2.max(1)`
match: if terminated else, (1.0 - d, (1 - d, not terminated
tags: TD target, bootstrap, Q-learning, terminal, done mask

Bootstrap from the next state **unless the episode terminated** — a terminal state has no future value.

```python
import numpy as np
Q = np.array([[0., 0.], [1., 3.]])
r, s2, gamma = 1.0, 1, 0.9
for terminated in (False, True):
    target = r if terminated else r + gamma * Q[s2].max()
    print(target)                        # 3.7, then 1.0
R, D = np.array([1., 1.]), np.array([0., 1.])
R + gamma * (1.0 - D) * Q.max(axis=1)    # array([1., 1.])
```

> [!WARN] terminated, not done
> Mask with `terminated` only; a *truncated* episode (time limit) still has future value. And keep the mask float (`1.0 - D`) so it broadcasts.

## SARSA vs Q-learning target
sig: SARSA `r + γ Q[s2, a2]` · Q-learning `r + γ max Q[s2]` · Expected SARSA `r + γ Σ π(a|s2) Q[s2, a]`
match: Q[s2, a2], Q[s2].max(, Q[s2][a2]
tags: on-policy, off-policy, control

SARSA bootstraps from the action you will actually take (on-policy, so it learns the safe path on the cliff); Q-learning from the greedy action (off-policy, learns the optimal but risky path).

```python
import numpy as np
Q = np.array([[0., 0.], [2., 5.]])
r, s2, a2, g = -1.0, 1, 0, 0.9
r + g * Q[s2, a2]        # SARSA:      0.8
r + g * Q[s2].max()      # Q-learning: 3.5
```

> [!WARN] Choose a2 first
> In SARSA pick `a2` (ε-greedy) *before* the update and then actually take it next step; choosing a fresh action later breaks on-policy learning.

## Returns-to-go (backward loop)
sig: `G = 0; for t in reversed(range(T)): G = r[t] + gamma * G; returns[t] = G`
match: gamma * G, reversed(range(
tags: return, Monte Carlo, REINFORCE, discount

Computes every discounted return Gₜ = rₜ + γGₜ₊₁ in one backward pass — O(T) instead of O(T²).

```python
import numpy as np
rewards, gamma = [1.0, 0.0, 2.0], 0.9
G, out = 0.0, np.zeros(len(rewards))
for t in reversed(range(len(rewards))):
    G = rewards[t] + gamma * G
    out[t] = G
out             # array([2.62, 1.8 , 2.  ])
```

> [!WARN] Reset G per episode
> Start `G = 0.0` for each episode; carrying it over leaks reward between episodes.

## Discounted return (forward)
sig: `G = Σ γᵗ rₜ` → `sum(gamma**t * r for t, r in enumerate(rewards))`
match: gamma ** 
tags: return, discount, evaluation

The return from the start state, summed forward with powers of γ.

```python
import numpy as np
rewards, gamma = [1.0, 0.0, 2.0], 0.9
sum(gamma ** t * r for t, r in enumerate(rewards))     # 2.62
float(np.dot(gamma ** np.arange(3), rewards))          # 2.62
```

> [!WARN] γ⁰ = 1
> The first reward is not discounted. Starting the exponent at 1 shrinks every return by a factor γ.

## Bellman backup over a model
sig: `q = Σ p · (r + γ · V[s2] · (not done))` for `(p, s2, r, done)` in `env.P[s][a]`
match: in env.P[, env.P[s][a]
tags: dynamic programming, value iteration, policy evaluation, one-step lookahead

The one-step lookahead that powers policy evaluation and value iteration: expected reward plus discounted next-state value.

```python
from rlenvs import GridWorld
env, gamma = GridWorld(), 0.9
V = [0.0] * env.n_states
def q_value(s, a):
    return sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in env.P[s][a])
max(q_value(2, a) for a in range(env.n_actions))   # 1.0 — next to the goal
```

> [!WARN] In-place vs two arrays
> Updating `V` in place during a sweep (Gauss–Seidel) is fine and usually faster, but then you must measure convergence with the per-state change, not by comparing to a copy you never made.

## Softmax with max-subtraction
sig: `e = np.exp(z - z.max(axis=-1, keepdims=True)); p = e / e.sum(axis=-1, keepdims=True)`
match: def softmax, z.max(, np.exp(z
tags: softmax, policy, numerical stability, probabilities

Turns preferences into probabilities. Subtracting the max changes nothing mathematically but prevents `exp` overflow.

```python
import numpy as np
def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
softmax([1000.0, 1001.0])          # array([0.26894142, 0.73105858])
softmax(np.zeros((2, 3))).sum(axis=1)   # array([1., 1.])
```

> [!WARN] Without keepdims
> For a batch `(N, A)`, `z.max(axis=-1)` has shape `(N,)` and won't broadcast against `(N, A)`. Use `keepdims=True`.

## Sampling from probabilities
sig: `a = rng.choice(len(p), p=p)` · fast: `min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)`
match: searchsorted(np.cumsum, p=p
tags: stochastic policy, sample, categorical

Draw an action from a categorical distribution. `rng.choice` is simple; the cumsum/searchsorted version is several times faster in tight loops.

```python
import numpy as np
rng = np.random.default_rng(0)
p = np.array([0.1, 0.7, 0.2])
a1 = int(rng.choice(len(p), p=p))
a2 = min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(p) - 1)
```

## One-hot encoding
sig: `np.eye(n)[i]` · `x = np.zeros(n); x[i] = 1.0`
match: onehot, one_hot, np.eye(n
tags: features, categorical, policy gradient

A vector of zeros with a single 1 at index `i`. Used for tabular-as-linear features and for `∇ log π = x ⊗ (onehot(a) − π)`.

```python
import numpy as np
np.eye(4)[2]                  # array([0., 0., 1., 0.])
A = np.array([0, 2])
np.eye(3)[A]                  # rows for a batch of actions, shape (2, 3)
```

## Replay buffer sampling
sig: `idx = rng.integers(0, size, size=batch); S, A, R, S2, D = obs[idx], actions[idx], ...`
match: ReplayBuffer, .sample(
tags: experience replay, DQN, minibatch, circular buffer

Store transitions in preallocated arrays, overwrite the oldest circularly, and sample uniform random indices — breaking the correlation between consecutive steps.

```python
import numpy as np
cap, pos, size = 4, 0, 0
obs = np.zeros((cap, 2), dtype=np.float32)
for t in range(6):                       # write 6 transitions into 4 slots
    obs[pos] = t; pos = (pos + 1) % cap; size = min(size + 1, cap)
rng = np.random.default_rng(0)
idx = rng.integers(0, size, size=3)
obs[idx].shape                           # (3, 2)
```

> [!WARN] Sample only filled slots
> Draw from `[0, size)`, not `[0, capacity)` — before the buffer fills up the rest is zeros.

## Target network copy
sig: `target.params = [p.copy() for p in online.params]` · every `target_every` steps
match: copy_from(, target_every
tags: DQN, target network, stability

A frozen copy of the Q-network supplies the bootstrap targets and is refreshed every few thousand steps, so targets don't chase the network you're training.

```python
import numpy as np
online = [np.ones((2, 2)), np.zeros(2)]
target = [p.copy() for p in online]       # a real copy
online[0] += 1.0
target[0][0, 0]                           # 1.0 — unaffected
```

> [!WARN] Aliasing
> `target.params = online.params` (no copy) makes them the *same arrays*; the target then moves with every update and the stabilising effect vanishes.

## Huber loss gradient
sig: `dL/dq = np.clip(q - y, -delta, delta)` (per element; divide by batch size for a mean)
match: huber
tags: DQN loss, smooth L1, gradient clipping

Huber is quadratic for small errors and linear for large ones, so its gradient is simply the error clipped to `[−δ, δ]` — outliers can't blow up the update.

```python
import numpy as np
err = np.array([-3.0, 0.5, 2.0])                    # q_pred - target
np.clip(err, -1.0, 1.0)                             # array([-1. ,  0.5,  1. ])
np.where(np.abs(err) <= 1, 0.5 * err**2, np.abs(err) - 0.5).mean()   # loss
```

> [!WARN] Only the taken action
> The gradient w.r.t. the network output is zero except at `[np.arange(B), A]` — build a zero `(B, n_actions)` array and fill those entries.

## Advantage normalisation
sig: `adv = (adv - adv.mean()) / (adv.std() + 1e-8)`
match: adv.std(, adv.mean(
tags: PPO, A2C, variance reduction, normalise

Rescaling advantages to mean 0 and std 1 per batch keeps policy-gradient step sizes consistent across tasks and over training.

```python
import numpy as np
adv = np.array([1.0, 2.0, 3.0, 6.0])
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
round(float(adv.mean()), 6), round(float(adv.std()), 6)     # (0.0, 1.0)
```

> [!WARN] The epsilon
> A minibatch of identical advantages has std 0; without `+ 1e-8` you divide by zero and fill the network with `nan`.

## GAE (generalised advantage estimation)
sig: `δₜ = rₜ + γ(1−dₜ)V(sₜ₊₁) − V(sₜ)` · `Aₜ = δₜ + γλ(1−dₜ)Aₜ₊₁` · `returns = A + V`
match: compute_gae(, gamma * lam, lam *
tags: advantage, actor critic, PPO, lambda

A backward pass over δ-errors mixes n-step advantages with weight λ: λ = 0 is one-step TD, λ = 1 is Monte Carlo minus a baseline.

```python
import numpy as np
r = np.array([1., 1., 1.]); v = np.array([0.5, 0.5, 0.5]); d = np.array([0., 0., 1.])
gamma, lam, last_v = 0.9, 0.95, 0.0
deltas = r + gamma * (1.0 - d) * np.append(v[1:], last_v) - v
adv, running = np.zeros(3), 0.0
for t in reversed(range(3)):
    running = deltas[t] + gamma * lam * (1.0 - d[t]) * running
    adv[t] = running
returns = adv + v
```

> [!WARN] Reset at episode boundaries
> The `(1 − dₜ)` factor must also cut the *running* sum, or advantages leak across episodes in a rollout.

## PPO clipped objective
sig: `ratio = exp(logp_new − logp_old)` · `L = −mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))`
match: clip_eps, ratio
tags: PPO, trust region, policy gradient

The probability ratio is clipped so that a single batch can't move the policy too far; once the ratio leaves `[1−ε, 1+ε]` in the advantageous direction the gradient is zero.

```python
import numpy as np
ratio = np.array([0.7, 1.0, 1.5]); adv = np.array([1.0, 1.0, 1.0]); eps = 0.2
obj = np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
obj          # array([0.7, 1. , 1.2])
loss = -obj.mean()
```

> [!WARN] logp_old is a constant
> Store the old log-probs when collecting the rollout and never recompute them during the update epochs.

## Seeding for reproducibility
sig: `rng = np.random.default_rng(seed)` · `env.reset(seed=seed)` (first episode only) · `random.seed(seed)`
match: random.seed(, default_rng(seed
tags: seed, reproducible, debugging, variance

Seed every source of randomness once — your agent's `rng`, the env's first `reset`, and Python's `random` if you use it — so a run can be repeated exactly. Compare methods over several seeds.

```python
import numpy as np, random
from rlenvs import GridWorld
def run(seed):
    rng = np.random.default_rng(seed); random.seed(seed)
    env = GridWorld(slip=0.2)
    s, _ = env.reset(seed=seed)
    return [env.step(int(rng.integers(4)))[0] for _ in range(5)]
run(3) == run(3)       # True
```

> [!WARN] One seed is an anecdote
> RL results vary a lot between seeds. Report the mean (and spread) over several seeds before concluding one method is better.

## Episode loop skeleton
sig: `obs, _ = env.reset(); while True: a = policy(obs); obs, r, term, trunc, _ = env.step(a); ...; if term or trunc: break`
match: term or trunc, terminated or truncated
tags: loop, rollout, training loop

The shape of nearly every tabular training loop: act, step, learn from the transition, move on, stop on either end condition.

```python
import numpy as np
from rlenvs import CliffWalking
env, rng = CliffWalking(max_steps=50), np.random.default_rng(0)
Q, alpha, gamma = np.zeros((48, 4)), 0.5, 1.0
s, _ = env.reset(seed=0)
while True:
    a = int(rng.integers(4))
    s2, r, terminated, truncated, _ = env.step(a)
    target = r if terminated else r + gamma * Q[s2].max()
    Q[s, a] += alpha * (target - Q[s, a])
    s = s2
    if terminated or truncated:
        break
```

> [!WARN] Forgetting `s = s2`
> If you never advance the state, every update is applied to the start state and nothing is learned.
