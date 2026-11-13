---
title: Monte Carlo Methods
chapter: Tabular Methods
summary: Learn values and policies from complete episodes of experience — no model required. Blackjack, ε-soft control and importance sampling.
minutes: 45
---

# Monte Carlo methods

Dynamic programming was a planner's dream: hand it `p(s', r | s, a)` and it computes v_π, then
v*, exactly. But almost nothing interesting comes with that model attached. Nobody gives you the
transition probabilities of a Blackjack shoe, a robot's gearbox or a stock market.

**Monte Carlo (MC)** methods throw the model away. They learn from **sample episodes** — actual
sequences of states, actions and rewards — and they do the simplest thing imaginable with them:

> the value of a state is the average return you have seen after visiting it.

That is it. No Bellman backup, no `p(s'|s,a)`. Just play, write down what happened, average.

## Why averaging works

Recall the definition from lesson 3:

$$
v_π(s) = E_π[ Gₜ | Sₜ = s ]
$$

An expectation is the long-run mean of samples. Every time an episode passes through *s*, the
return Gₜ that followed is one unbiased sample of v_π(s). By the law of large numbers the
average converges to v_π(s), and its standard error shrinks like 1/√n.

Three properties make MC very different from DP:

- **Model-free.** It needs experience, not dynamics.
- **No bootstrapping.** Each estimate is built from *real* returns, never from other estimates.
  If one state's value is wrong, its neighbours are not infected.
- **Episode-by-episode.** You only know Gₜ once the episode is *over*, so MC only works for
  episodic tasks and only learns at the end of each one.

> [!NOTE] Cost independent of the state space
> The work to estimate one state's value does not depend on how many other states there are.
> If you only care about the value of the starting position, you can simulate from there and
> ignore the rest of the world. DP has no such shortcut.

## Monte Carlo prediction

Given a policy π, play episodes, and for each episode walk **backwards** accumulating the
return with the recursion you wrote in lesson 1:

$$
G ← Rₜ₊₁ + γ G        (for t = T−1, T−2, …, 0)
$$

and add G to the list of returns observed for Sₜ.

### First visit or every visit?

A state can appear several times in one episode. Two flavours handle that differently:

| | which returns count | properties |
|---|---|---|
| **first-visit MC** | only the return after the *first* occurrence of s in each episode | each sample is i.i.d.; unbiased; the textbook's default |
| **every-visit MC** | the return after *every* occurrence | samples within an episode are correlated; biased for finite n but still consistent |

A worked example. One episode, γ = 1:

| t | Sₜ | Rₜ₊₁ | Gₜ |
|---|---|---|---|
| 0 | A | 1 | 1 + 0 + 2 = **3** |
| 1 | A | 0 | 0 + 2 = **2** |
| 2 | B | 2 | **2** |

First-visit gives V(A) = 3. Every-visit gives V(A) = (3 + 2)/2 = 2.5. Both give V(B) = 2. With
many episodes both converge to v_π(A); they just get there differently.

```quiz
{"q": "Same episode (A, A, B with rewards 1, 0, 2) but γ = 0.5. What is the FIRST-visit estimate of V(A)?",
 "options": ["3", "1.5", "1.25", "2"],
 "answer": 1,
 "explain": "Walk backwards: G₂ = 2, G₁ = 0 + 0.5·2 = 1, G₀ = 1 + 0.5·1 = 1.5. First-visit uses G₀ = 1.5 (every-visit would average 1.5 and 1 → 1.25)."}
```

### The Blackjack testbed

Our main example is the classic from Sutton & Barto (Example 5.1), bundled as `rlenvs.Blackjack`.
You play against a dealer from an infinite deck. Face cards count 10, an ace counts 1 or 11.

- **state** `(player_sum, dealer_showing, usable_ace)` — e.g. `(15, 10, False)`
- **actions** `0 = stick`, `1 = hit`
- **reward** +1 win, 0 draw, −1 loss, given only at the end; γ = 1

An ace is *usable* if you can count it as 11 without busting — it is a free safety net, which is
why it is part of the state. Writing down the transition probabilities of this game is a
nightmare (what is p(next state | sum 14, hit) when the dealer then plays out its hand?).
Simulating it is ten lines of code. That gap is exactly where Monte Carlo shines.

Consider the naive policy *stick on 20 or 21, otherwise hit*. How good is it? Let Monte Carlo
tell you.

```exercise
01-mc-prediction
```

> [!TIP] Read the surface
> After **Run**, look at the two grids. The top rows (20 and 21) are bright: sticking there wins
> most of the time. Everything below 20 is negative — this policy keeps hitting until it busts.
> The usable-ace surface is noisier: those states are rarer, so they have fewer samples.

## From prediction to control

To *improve* a policy without a model we need **action values**. With v_π alone, choosing the
best action requires a one-step look-ahead through `p(s'|s,a)` — the model we do not have. With
q_π you simply take `argmax_a Q(s, a)`. So model-free control estimates **Q(s, a)**: the average
return after taking *a* in *s* and following π afterwards.

Then we run **generalised policy iteration** (GPI) exactly as in lesson 4, with sampled episodes
doing the evaluation:

$$
π₀ → q_π₀ → π₁ → q_π₁ → π₂ → …       (evaluate by averaging, improve by acting greedily)
$$

We do not wait for evaluation to converge — after every episode we update Q and the next episode
already uses the improved policy. GPI is happy with partial evaluation.

### The exploration problem

There is a catch. If π is deterministic and greedy, then from each state we only ever try one
action, so we only ever get returns for that one action. Q for the others stays at its initial
value forever and we can never discover they are better. Two classic fixes:

1. **Exploring starts.** Begin every episode in a random state with a random first action, so
   every (s, a) pair has a chance to be the start. It works in simulation, but you cannot tell a
   real robot to "start in a random state".
2. **ε-soft policies.** Never become fully greedy. Act **ε-greedily**: with probability ε pick a
   uniformly random action, otherwise the greedy one. Every action keeps probability ≥ ε/|A|.

$$
π(a|s) = 1 − ε + ε/|A|   if a = argmax Q(s, ·),   otherwise ε/|A|
$$

The policy improvement theorem still holds *within the class of ε-soft policies*: ε-greedy with
respect to q_π is at least as good as π. So on-policy MC control converges to the best ε-soft
policy — not quite optimal, but close when ε is small.

```widget
{"type": "epsilon", "values": [0.2, -0.4, 0.6, 0.1], "epsilon": 0.1}
```

```quiz
{"q": "Why does model-free control estimate Q(s, a) rather than V(s)?",
 "options": ["Q is smaller to store", "Choosing a greedy action from V needs the transition model; from Q it is just argmax", "V cannot be estimated by Monte Carlo"],
 "answer": 1,
 "explain": "argmaxₐ Σ p(s',r|s,a)[r + γV(s')] needs p. argmaxₐ Q(s,a) does not — the model is already baked into Q."}
```

### Incremental averages

Storing every return and re-averaging is wasteful. The incremental mean from the bandits lesson
does the same in O(1):

$$
N(s,a) ← N(s,a) + 1        Q(s,a) ← Q(s,a) + (G − Q(s,a)) / N(s,a)
$$

Replace 1/N with a constant step α and you get a recency-weighted average that tracks a changing
target — useful because, in control, the policy (and hence the target) keeps changing.

```exercise
02-mc-control
```

> [!KEY] On-policy MC control in one sentence
> Play an episode ε-greedily with respect to Q, then move each visited Q(s, a) towards the return
> that actually followed it. Repeat. Evaluation and improvement happen at the same time.

The policy your agent finds should look like the famous *basic strategy* charts: always hit on 11
or less, stick on hard 17+, and for hard 13–16, stick when the dealer shows a weak card (2–6) and
hit when it shows a strong one (7–A). Expect a noisy version after 200 000 hands. A dealer showing a 5 or 6 is likely to bust, so you let them.

## Off-policy learning

The ε-greedy agent above learns about the policy it *follows*, exploration noise included. That
is **on-policy** learning. Often we want something different:

- learn the **greedy** policy while behaving exploratorily,
- learn from data produced by someone else (a human, an old version of the agent, a log file),
- evaluate a new policy *before* deploying it.

That is **off-policy** learning. Two policies are now involved:

| | role |
|---|---|
| **target policy** π | the one we want to learn about (often greedy) |
| **behaviour policy** b | the one that actually generated the data (must explore) |

For this to be possible, b must try everything π might do: if π(a|s) > 0 then b(a|s) > 0.
This is the **coverage** assumption.

### Importance sampling

Returns collected under b have the wrong expectation: E_b[G] = v_b, not v_π. The fix is to
re-weight each return by how much more (or less) likely its trajectory is under π than under b.
The environment's transition probabilities appear in both and cancel, leaving only the policies:

$$
ρ = Πₜ π(Aₜ|Sₜ) / b(Aₜ|Sₜ)
$$

ρ is the **importance-sampling ratio**. If π is deterministic and the behaviour took an action π
would never take, ρ = 0 and that episode tells us nothing. If b took π's action with probability
½ at each of three steps, ρ = 2³ = 8 — that rare trajectory is worth eight ordinary ones.

Two ways to average the weighted returns from n episodes:

$$
ordinary:  V(s) = Σᵢ ρᵢ Gᵢ / n          weighted:  V(s) = Σᵢ ρᵢ Gᵢ / Σᵢ ρᵢ
$$

| | bias | variance | intuition |
|---|---|---|---|
| **ordinary IS** | unbiased | can be huge, even infinite | one lucky ρ = 64 episode dominates everything |
| **weighted IS** | biased (→ 0 as n grows) | bounded, much lower | a weighted *average* of real returns, so it never leaves their range |

After one episode with ρ = 4 and G = 2, ordinary IS says 8 — a Blackjack return of 8 is
impossible! Weighted IS says 2, the return actually observed. In practice weighted IS is almost
always preferred.

```quiz
{"q": "Target π: always HIT. Behaviour b: hit or stick 50/50. An episode is hit, hit, stick. What is ρ?",
 "options": ["0.125", "8", "0", "4"],
 "answer": 2,
 "explain": "π never sticks, so π(stick|s) = 0 and the whole product is 0. That trajectory is impossible under π and contributes nothing."}
```

```exercise
03-importance-sampling
```

### Off-policy MC control (in brief)

Put the pieces together: behave with a soft b (e.g. ε-greedy), keep a greedy target π, and update
Q with weighted importance sampling, walking each episode backwards. Because π is greedy,
ρ becomes 0 the moment b takes a non-greedy action, so the algorithm only learns from the
**tails** of episodes after the last exploratory move. That makes it slow when exploratory
actions are common — one motivation for the temporal-difference methods in the next lesson.

## Pitfalls

> [!WARN] Common Monte Carlo bugs
> - **Wrong reward alignment.** The triple is (Sₜ, Aₜ, Rₜ₊₁): the reward that *follows* the action.
>   Storing the next state instead of the current one shifts every return by one step.
> - **Forward instead of backward.** Computing each Gₜ by summing forward is O(T²). Walk backwards once.
> - **Re-seeding every episode.** `env.reset(seed=0)` inside the loop replays the *same* deal forever.
>   Seed once, then let the environment's RNG run.
> - **Greedy too early.** With ε = 0 and Q initialised to zeros, argmax always picks action 0 —
>   some actions will never be tried.
> - **Ordinary IS explosions.** Long episodes multiply many ratios; ρ can be astronomically large.
>   Use weighted IS, or keep episodes short.

## Summary

- MC estimates values by **averaging sample returns**; it needs no model and does not bootstrap.
- **First-visit** and **every-visit** MC both converge to v_π; first-visit is unbiased.
- Control estimates **Q(s, a)** and runs GPI episode by episode, with **ε-soft** policies (or
  exploring starts) to keep exploring.
- **Off-policy** learning evaluates π from data generated by b using the **importance-sampling
  ratio**; **weighted** IS trades a little bias for a big drop in variance.

The weakness of every method here: nothing is learned until the episode ends. Next lesson we fix
that by updating a guess from a guess — **temporal-difference learning**.
