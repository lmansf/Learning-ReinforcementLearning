---
title: Temporal-Difference Learning
chapter: Tabular Methods
summary: Learn a guess from a guess — TD(0), SARSA, Q-learning, Expected SARSA and Double Q-learning, with the cliff as the proving ground.
minutes: 50
---

# Temporal-difference learning

> If one had to identify one idea as central and novel to reinforcement learning, it would
> undoubtedly be temporal-difference learning. — Sutton & Barto

Monte Carlo waits until the end of the episode to find out what a state was worth. Dynamic
programming never waits, but needs the model. **Temporal-difference (TD) learning** takes the
best of both: like MC it learns from raw experience, and like DP it **bootstraps** — it updates
an estimate using other estimates, one step at a time.

## The driving-home example

You leave the office and predict *30 minutes* to get home. Five minutes in, it starts raining;
traffic will be slow, and you now predict *35 more minutes* — a total of 40.

- **Monte Carlo** says: wait until you are home, see that it took 43, *then* correct the guess
  you made at the office.
- **TD** says: you already have new information. Your office estimate (30) and your current
  estimate (5 + 35 = 40) disagree by 10 minutes. Move the office estimate towards 40 **now**.

Every step gives you a slightly better-informed guess about the same quantity. TD learns from
the *difference between successive predictions* — hence the name.

## TD(0) prediction

Recall the two forms of v_π from the DP lesson:

$$
v_π(s) = E_π[ Gₜ | Sₜ = s ] = E_π[ Rₜ₊₁ + γ v_π(Sₜ₊₁) | Sₜ = s ]
$$

MC targets the first form: the sampled return Gₜ. TD targets the second, replacing the unknown
v_π(Sₜ₊₁) with the current estimate V(Sₜ₊₁):

$$
V(Sₜ) ← V(Sₜ) + α [ Rₜ₊₁ + γ V(Sₜ₊₁) − V(Sₜ) ]
$$

The bracket is the **TD error**, the single most important quantity in this course:

$$
δₜ = Rₜ₊₁ + γ V(Sₜ₊₁) − V(Sₜ)
$$

δ > 0 means "things turned out better than I expected"; δ < 0, worse. If Sₜ₊₁ is terminal its
value is 0 by definition, so δ = Rₜ₊₁ − V(Sₜ).

| | target | needs a model? | bootstraps? | when can it update? |
|---|---|---|---|---|
| DP | E[R + γ V(S')] over all s' | yes | yes | any time (sweeps) |
| MC | Gₜ (full sampled return) | no | no | end of episode |
| TD(0) | Rₜ₊₁ + γ V(Sₜ₊₁) (one sample) | no | yes | after every step |

> [!KEY] The TD error
> δ = R + γ V(S') − V(S) compares what you predicted before a step with a better-informed
> prediction made one step later. Nudging V by α·δ is how TD learns — and δ reappears in SARSA,
> Q-learning, actor-critic, GAE and PPO.

### A hand-worked chain

Three states 0 → 1 → 2 → terminal; the only reward is +1 on the final step; γ = 1, α = 0.5,
V starts at 0.

- **Episode 1.** Steps from 0 and 1 have δ = 0 + 0 − 0 = 0: nothing changes. From 2:
  δ = 1 − 0 = 1, so V(2) = 0.5.
- **Episode 2.** From 1: δ = 0 + V(2) − V(1) = 0.5, so V(1) = 0.25. From 2: δ = 1 − 0.5, V(2) = 0.75.

The reward percolates backwards **one state per episode**. Monte Carlo would have set all three
states to 0.5 after the very first episode. Is TD slower, then? Not in general — see below.

```quiz
{"q": "V(S) = 2, V(S') = 5, R = −1, γ = 0.9, α = 0.1. What is V(S) after one TD(0) update?",
 "options": ["2.15", "2.35", "2.6", "1.85"],
 "answer": 0,
 "explain": "δ = −1 + 0.9·5 − 2 = 1.5, so V(S) ← 2 + 0.1·1.5 = 2.15."}
```

### The random walk

Sutton & Barto's Example 6.2: five states A–E in a row, start in C, step left or right with
equal probability. Falling off the right end pays +1, the left end 0. The true values are
1/6, 2/6, …, 5/6 — the probability of exiting right.

Before you write TD code on your own, type along once. The walkthrough trains *Expected SARSA*
(it averages over the next action instead of sampling it) on a small grid with a pit, and shows
every idiom this lesson's four exercises need, with an explanation for each line.

```exercise
00-walkthrough-expected-sarsa
```

Now the same tools, no ghost text:

```exercise
01-td0-random-walk
```

On this task TD reaches a lower error than constant-α MC for every step size tried, and it
gets there faster. Why? TD exploits the **Markov property**: a state's value only depends on
what happens next, and TD's target uses everything already learned about the next state.
MC treats every return as an independent fact and ignores that structure.

> [!NOTE] Bias and variance
> The MC target Gₜ is unbiased but noisy — it depends on *every* random event until the episode
> ends. The TD target R + γV(S') is biased (V(S') is just a guess) but depends on only one random
> transition, so its variance is much lower. Lower variance usually wins in practice. Lesson 7
> turns this into a dial you can set anywhere between the two.

## SARSA: on-policy TD control

For control we switch to action values — the reason is the same as in the MC lesson: acting
greedily on Q needs no model. The TD update for Q, applied to the quintuple
(Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁, Aₜ₊₁), gives the algorithm its name:

$$
Q(S, A) ← Q(S, A) + α [ R + γ Q(S', A') − Q(S, A) ]
$$

A' is the action the agent **will actually take** next, drawn from the same ε-greedy policy
it is following. So SARSA evaluates its own behaviour, exploration included, and improves it at
the same time: GPI with one-step evaluations.

```
choose A from S (ε-greedy on Q)
loop:
    take A, observe R, S'
    choose A' from S' (ε-greedy on Q)
    Q(S,A) += α [R + γ Q(S',A') − Q(S,A)]
    S, A ← S', A'
```

### Cliff walking

`rlenvs.CliffWalking` (Example 6.6) is a 4 × 12 grid. Start bottom-left, goal bottom-right,
and the cells between are a cliff: stepping in costs −100 and sends you back to the start.
Every other step is −1. The shortest path hugs the cliff edge (13 steps, return −13).

Press play and watch SARSA learn. The `C` cells are the cliff.

```widget
{"type": "qlearning", "map": ["............", "............", "............", "SCCCCCCCCCCG"], "algo": "sarsa", "alpha": 0.5, "gamma": 1.0, "epsilon": 0.1}
```

```exercise
02-sarsa-cliff
```

SARSA does not take the shortest path. It walks up and around, two or three rows away from the
edge. That is not a bug: SARSA learns the value of the policy it *follows*, and that policy is
ε-greedy. Next to the cliff, one random step in ten is a disaster, so cells next to the edge
really are worth less **for an ε-greedy agent**.

## Q-learning: off-policy TD control

Change one term — bootstrap from the **best** next action instead of the one you will take:

$$
Q(S, A) ← Q(S, A) + α [ R + γ maxₐ' Q(S', a') − Q(S, A) ]
$$

Now the learned Q approximates q*, the optimal action values, **no matter how the agent
behaves** (as long as every pair keeps being visited). The behaviour policy is ε-greedy; the
target policy is greedy. That is off-policy learning — without importance sampling. Why are no
ratios needed? Because the target for Q(S, A) only involves the next action, and the max takes
the place of the expectation over the target policy; the action A itself is already given.

```widget
{"type": "qlearning", "map": ["............", "............", "............", "SCCCCCCCCCCG"], "algo": "q", "alpha": 0.5, "gamma": 1.0, "epsilon": 0.1}
```

```exercise
03-q-learning-cliff
```

> [!TIP] The cliff paradox
> Q-learning's greedy path is optimal (−13) but its **online** return is worse than SARSA's,
> because it keeps walking the cliff edge while exploring and every so often slips off. Which
> algorithm is "better" depends on what you care about: the policy you end up with, or the
> reward you collect while learning. If ε decays to 0, both converge to the optimal path.

```quiz
{"q": "A robot learns on real hardware where falls are expensive. Exploration will stay on during training. Which update better reflects the cost of its own exploratory slips?",
 "options": ["Q-learning, because it learns q*", "SARSA, because it evaluates the ε-greedy policy it actually follows", "Neither — TD cannot handle risk"],
 "answer": 1,
 "explain": "SARSA's target uses the action really taken, so the danger of random actions near the cliff is priced into Q. Q-learning assumes perfect greedy play from the next step on."}
```

## Expected SARSA

SARSA's target depends on the random draw of A'. Why sample when you know the policy? Expected
SARSA averages over it:

$$
Q(S, A) ← Q(S, A) + α [ R + γ Σₐ' π(a'|S') Q(S', a') − Q(S, A) ]
$$

It moves in the same direction SARSA moves *on average*, with the variance from A' removed, so
it tolerates much larger step sizes (often α = 1 works on deterministic tasks like the cliff).
If π is greedy, the sum is just the max: **Q-learning is the special case of Expected SARSA with
a greedy target policy.** With an ε-greedy π it is on-policy; with any other π it is a
general off-policy method.

```quiz
{"q": "Q(S', ·) = [2, 0, −1, 1], π is ε-greedy with ε = 0.2 over 4 actions. What is Σₐ' π(a'|S') Q(S', a')?",
 "options": ["2.0", "1.7", "0.5", "1.55"],
 "answer": 1,
 "explain": "Greedy action gets 1 − 0.2 + 0.05 = 0.85, the others 0.05: 0.85·2 + 0.05·(0 − 1 + 1) = 1.7."}
```

## Maximisation bias and Double Q-learning

The max in Q-learning hides a trap. Suppose every action in a state has true value 0 but our
estimates are noisy: some come out at +0.3, some at −0.4. The max of the *estimates* is
positive even though the max of the *true values* is 0. Using max of noisy estimates as an
estimate of the max is **biased upwards** — maximisation bias.

The fix is to decouple **selecting** the best action from **evaluating** it. Keep two
independent tables Q₁ and Q₂. On each step flip a coin; if heads:

$$
A* = argmaxₐ Q₁(S', a)        Q₁(S, A) ← Q₁(S, A) + α [ R + γ Q₂(S', A*) − Q₁(S, A) ]
$$

and the mirror image if tails. Q₂'s noise is independent of the noise that made A* look good
in Q₁, so the estimate is unbiased. Behave ε-greedily with respect to Q₁ + Q₂. The same idea,
"one network selects, the other evaluates", becomes **Double DQN** in the deep RL chapter.

```exercise
04-double-q
```

## Pitfalls

> [!WARN] Common TD bugs
> - **Bootstrapping from a terminal state.** If `terminated`, the target is just R. Using
>   Q(S', ·) of a terminal state is wrong unless you guarantee it stays 0.
> - **Truncated is not terminated.** A time limit cuts the episode, but the state still has
>   a future. Keep bootstrapping on `truncated`; only drop the bootstrap on `terminated`.
> - **SARSA's A' mismatch.** Choose A' once, use it in the target, and then *take* it. Re-drawing
>   a fresh action at the top of the loop silently turns SARSA into something else.
> - **argmax ties.** With a zero-initialised table, `np.argmax` always picks action 0. Break ties
>   at random or your agent may never leave the first wall it walks into.
> - **Step size too large for noise.** α = 0.5 is fine on the deterministic cliff, terrible on a
>   noisy task. If curves oscillate, lower α.

## Summary

- **TD(0)** updates V(S) towards R + γV(S') after every step; the gap is the **TD error δ**.
- TD is model-free like MC, bootstraps like DP, works online and on continuing tasks, and is
  usually faster than MC thanks to lower variance.
- **SARSA** (on-policy) bootstraps from the action actually taken; **Q-learning** (off-policy)
  from the greedy one; **Expected SARSA** from the policy's average.
- On the cliff, SARSA learns the safe path, Q-learning the optimal-but-risky one.
- **Double Q-learning** removes maximisation bias by letting one table select and the other evaluate.

Next: why look only *one* step ahead? n-step methods and eligibility traces fill the gap between
TD and MC, and Dyna shows how a learned model can multiply every real step into many imagined ones.
