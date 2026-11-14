---
title: n-step Bootstrapping & Planning
chapter: Tabular Methods
summary: The dial between TD and Monte Carlo (n-step returns, eligibility traces), then learned models and Dyna-Q — imagining experience to learn faster.
minutes: 50
---

# n-step bootstrapping and planning

The last two lessons gave you two extremes. Monte Carlo waits for the whole return; TD(0)
looks one reward ahead and then trusts its own guess. Neither is best in general. This lesson
does two things with that observation:

1. It builds a **dial** between them — n-step returns and eligibility traces — so you can pick
   how far to look before bootstrapping.
2. It asks a different question: if every real step is precious, can the agent **reuse** it?
   Learning a model of the world and planning with it — **Dyna** — answers yes.

## n-step returns

Write out the targets of the methods you know, for the state visited at time t:

$$
TD(0):     Gₜ:ₜ₊₁ = Rₜ₊₁ + γ V(Sₜ₊₁)
2-step:    Gₜ:ₜ₊₂ = Rₜ₊₁ + γ Rₜ₊₂ + γ² V(Sₜ₊₂)
n-step:    Gₜ:ₜ₊ₙ = Rₜ₊₁ + γ Rₜ₊₂ + … + γⁿ⁻¹ Rₜ₊ₙ + γⁿ V(Sₜ₊ₙ)
MC:        Gₜ     = Rₜ₊₁ + γ Rₜ₊₂ + … + γᵀ⁻ᵗ⁻¹ R_T
$$

The **n-step return** uses n real rewards, then bootstraps from the estimate n steps later.
If the episode ends before that (t + n ≥ T), there is nothing to bootstrap from and the n-step
return is just the ordinary return. n = 1 is TD(0); n = ∞ is Monte Carlo.

$$
V(Sₜ) ← V(Sₜ) + α [ Gₜ:ₜ₊ₙ − V(Sₜ) ]
$$

There is one practical wrinkle: Gₜ:ₜ₊ₙ needs Rₜ₊ₙ, which only arrives at time t + n. So the
update for time τ is made at time τ + n − 1, and after the episode ends you must flush the last
n − 1 pending updates. Store the states and rewards of the episode and index carefully.

> [!NOTE] The bias–variance dial
> Each real reward in the target adds a little variance (it is a random sample); each bootstrap
> adds a little bias (V is only a guess). Small n: low variance, high bias. Large n: the reverse.
> On most problems the error is lowest somewhere in the middle — and the best α shrinks as n grows,
> because the target gets noisier.

### Worked example

The deterministic chain from lesson 6 (0 → 1 → 2 → terminal, reward 1 at the end, γ = 1,
α = 0.5, V = 0). With **n = 2**, in the very first episode:

| τ | update made at time | target | new V(S_τ) |
|---|---|---|---|
| 0 | 1 | R₁ + R₂ + V(S₂) = 0 + 0 + 0 | 0 |
| 1 | 2 | R₂ + R₃ = 0 + 1 (episode ended) | 0.5 |
| 2 | after the end | R₃ = 1 | 0.5 |

The reward reaches two states back in one episode — TD(0) managed one, MC all three.

```quiz
{"q": "Rewards R₁ = 1, R₂ = 0, R₃ = 2, γ = 0.5, V(S₃) = 4, episode not over. What is the 3-step return G₀:₃?",
 "options": ["3.0", "2.0", "1.5", "7.0"],
 "answer": 1,
 "explain": "1 + 0.5·0 + 0.25·2 + 0.125·4 = 1 + 0 + 0.5 + 0.5 = 2.0"}
```

Before you write these on your own, type along once. The walkthrough builds three small tools,
one per exercise: n-step reward windows on a finished episode, an eligibility-trace vector, and
a model dict you can sample from for planning, with an explanation for each line.

```exercise
00-walkthrough-windows-traces-models
```

Now the same tools, no ghost text:

```exercise
01-nstep-td
```

### n-step SARSA

Control works the same way with action values. Bootstrap from Q(Sₜ₊ₙ, Aₜ₊ₙ):

$$
Gₜ:ₜ₊ₙ = Rₜ₊₁ + … + γⁿ⁻¹ Rₜ₊ₙ + γⁿ Q(Sₜ₊ₙ, Aₜ₊ₙ)        Q(Sₜ, Aₜ) ← Q(Sₜ, Aₜ) + α [ Gₜ:ₜ₊ₙ − Q(Sₜ, Aₜ) ]
$$

Picture an agent that wanders a grid and finally finds the goal. One-step SARSA strengthens only
the very last action of that path. 10-step SARSA strengthens the last ten. The longer the reward
is delayed, the bigger the win. (n-step *off-policy* methods need importance-sampling ratios over
the n actions again, or the tree-backup algorithm that avoids them.)

## Eligibility traces and TD(λ)

Choosing one n is arbitrary. Why not average **all** of them? The **λ-return** weights the
n-step return by (1 − λ)λⁿ⁻¹ — weights that sum to 1 and fade geometrically:

$$
Gₜ^λ = (1 − λ) Σₙ λⁿ⁻¹ Gₜ:ₜ₊ₙ
$$

λ = 0 gives TD(0); λ = 1 gives Monte Carlo. That is the **forward view**: elegant, but it needs
the future, so it cannot be computed online.

The **backward view** gets (almost) the same updates online. Each state keeps an
**eligibility trace** e(s): a short-term memory of how recently and how often it was visited.

$$
δₜ = Rₜ₊₁ + γ V(Sₜ₊₁) − V(Sₜ)
e(Sₜ) ← e(Sₜ) + 1
V(s) ← V(s) + α δₜ e(s)        for every s
e(s) ← γ λ e(s)                for every s
$$

When something surprising happens (a large δ), *all* recently visited states share the credit,
in proportion to their traces. Think of a rat that gets a shock after a bell rang twice and a
light flashed once: the bell is more "eligible" for blame. This is the trick that later lets
actor-critic and GAE assign credit across many steps.

> [!KEY] One dial, two mechanisms
> n-step returns and TD(λ) both interpolate between TD (small n, λ = 0) and Monte Carlo
> (n = ∞, λ = 1). n-step does it with a fixed window; traces do it with an exponentially
> fading memory, updated online with one TD error per step.

```quiz
{"q": "With accumulating traces, γ = 1 and λ = 0.5, a state was visited 1 and 3 steps ago (nothing else). What is its trace now, just before the current update?",
 "options": ["1.5", "0.625", "0.75", "0.5"],
 "answer": 1,
 "explain": "Each visit adds 1, then decays by γλ = 0.5 per step: 0.5¹ + 0.5³ = 0.5 + 0.125 = 0.625."}
```

```exercise
02-td-lambda
```

> [!WARN] Accumulating traces can blow up
> A state revisited many times in quick succession builds a trace above 1, and with large α the
> updates overshoot — you will see λ = 1 diverge in the demo. *Replacing* traces (set e(s) = 1
> instead of adding 1) and the modern *true online TD(λ)* fix this. With function approximation
> (next chapter) the same idea uses a trace vector over weights.

## Models and planning

So far every update came straight from real experience. Now suppose the agent also learns a
**model**: a prediction of what the environment does.

| kind | gives you | example |
|---|---|---|
| **distribution model** | all outcomes and their probabilities, p(s', r ∣ s, a) | what DP needed in lesson 4 |
| **sample model** | one plausible outcome (s', r) when asked | a simulator; a table of what happened last time |

**Planning** means computing values or policies from a model instead of from the real world. DP
is planning with a distribution model. But planning does not need sweeps: you can **sample a
transition from the model and apply an ordinary Q-learning update to it**. To the learning rule,
a simulated transition looks exactly like a real one.

That unifies everything: *learning* and *planning* are the same backup applied to different
sources of experience.

## Dyna-Q

Dyna-Q (Sutton, 1990) runs both, interleaved, on one Q table. After every real step:

1. **act** ε-greedily in the real environment and observe R, S'
2. **direct RL** — Q-learning update from the real transition
3. **model learning** — remember `model[S, A] = (R, S')` (deterministic world: just a table)
4. **planning** — repeat n times: pick a previously seen (s, a) at random, ask the model for
   (r, s'), apply the Q-learning update

```
            ┌──────────── real experience ────────────┐
            ▼                                          │
   environment ──► direct RL update ──► Q ──► ε-greedy action
            │                           ▲
            ▼                           │
       model learning ──► model ──► planning updates (× n)
```

Why does this help so much? In a maze with a single reward at the goal, Q-learning's first
episode ends with **one** useful update: the step into the goal. In the second episode, the value
creeps back one more step, and so on. With planning, that single rewarding transition is
replayed, and the states leading to it — all remembered from the first episode — get updated in
**imagination**. Every real step of the second episode triggers n more replays, so the value
spreads back over most of the path within that single episode.

Here is plain Q-learning on a small maze — watch how long it takes for values to reach the start:

```widget
{"type": "qlearning", "map": ["......#G", "..#...#.", "S.#.....", "..#..#..", "........"], "algo": "q", "alpha": 0.5, "gamma": 0.95, "epsilon": 0.1}
```

```exercise
03-dyna-q
```

```quiz
{"q": "Dyna-Q with n = 50 planning steps reaches good performance in far fewer episodes than n = 0. What does it cost?",
 "options": ["Nothing — it is strictly better", "More computation per real step (51 updates instead of 1), and it relies on the model being right", "It needs the true transition probabilities"],
 "answer": 1,
 "explain": "Planning trades compute for data. That is a great deal when real experience is expensive (robots, users) — and dangerous if the model is wrong."}
```

### When the model is wrong

Real worlds change. If a shortcut opens up or a path closes, the model still remembers the old
world, and planning confidently reinforces stale beliefs. **Dyna-Q+** adds an exploration bonus
to planning rewards, r + κ√τ, where τ is how long ago (s, a) was last tried for real. Pairs
not tested for a while start to look attractive, so the agent goes back and checks.

### Prioritized sweeping

Uniformly random planning wastes most updates: early on, almost every remembered transition has
δ = 0 because nothing around it has value yet. **Prioritized sweeping** keeps a priority queue of
(s, a) pairs keyed by the size of their expected update |δ|. After a real step with a big δ, it
updates that pair, then pushes its **predecessors** (pairs the model says lead into s) whose
priority is above a threshold, and keeps popping the largest. Value changes ripple backwards
from where something actually changed — the same "work backwards from the goal" intuition as DP,
but focused. On mazes it often needs 5–10× fewer updates than Dyna-Q.

## Pitfalls

> [!WARN] Common bugs in this lesson
> - **Off-by-one in n-step indexing.** The rewards for τ are R_τ+1 … R_min(τ+n, T). Store a dummy
>   R₀ so list indices match the maths.
> - **Forgetting the tail.** After the episode ends, the last n − 1 states still need updates.
>   Keep looping until τ = T − 1.
> - **Bootstrapping past the end.** If τ + n ≥ T there is no V(S_τ+n) term.
> - **Traces not reset.** Eligibility traces belong to one episode; zero them at every reset.
> - **Planning on a separate table.** Dyna's planning updates must hit the *same* Q the agent acts on.
> - **argmax ties in a sparse-reward maze.** All-zero Q plus `np.argmax` means "always UP": break ties at random.

## Summary

- **n-step returns** use n real rewards then bootstrap; n is a bias–variance dial between TD and MC,
  and intermediate n usually learns fastest.
- **TD(λ)** averages all n-step returns; **eligibility traces** implement it online by sharing each
  TD error among recently visited states.
- A **model** lets the agent generate simulated experience; **planning** is just learning applied to it.
- **Dyna-Q** interleaves acting, learning, model learning and n planning updates per step — far
  fewer real episodes for the same performance. **Prioritized sweeping** spends planning where values change.

This closes the tabular chapter. Everything so far stored one number per state or state–action
pair. Next we replace the table with a function — and the ideas you now own (TD errors, targets,
bootstrapping, replay of experience) carry over almost unchanged.
