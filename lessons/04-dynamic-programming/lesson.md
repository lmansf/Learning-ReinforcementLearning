---
title: Dynamic Programming
chapter: Tabular Methods
summary: Planning with a perfect model — policy evaluation, policy improvement, policy iteration, value iteration and GPI.
minutes: 50
---

# Dynamic programming

The previous lesson ended with two equations, and with a problem: the Bellman expectation
equation is linear and can be solved directly, but only for a fixed policy, and only at O(n³)
cost. The optimality equation has a max in it and cannot be solved in one step at all.

**Dynamic programming (DP)** is the family of algorithms that solves both equations by
**turning them into update rules** and applying them over and over until nothing changes. DP
needs a perfect model of the environment, the `env.P` you have been using, so it is
*planning*, not learning. It matters for two reasons:

1. On small problems it gives you the **exact** answer, which you will use as ground truth to
   test every learning algorithm in later lessons.
2. Every RL algorithm afterwards (Monte Carlo, TD, Q-learning, DQN, actor-critic) is an
   attempt to do what DP does **without the model** and **without visiting every state**.

## Iterative policy evaluation

Take the Bellman expectation equation and read the "=" as "←":

$$
Vₖ₊₁(s) ← Σₐ π(a|s) Σₛ',ᵣ p(s', r | s, a) [ r + γ Vₖ(s') ]
$$

Start from any V₀ (zeros are fine). Each full pass over the states is a **sweep**. As k → ∞,
Vₖ → v_π. Why it works: with γ < 1 each sweep shrinks the worst-case error by at least a factor of
γ, because the update is a **contraction**. Errors in the future values are discounted before
they feed back.

We cannot sweep forever, so we stop when the largest change in a sweep, Δ, falls below a small
threshold θ.

> [!NOTE] Two copies or one?
> The textbook equation uses two arrays: compute all Vₖ₊₁ from the old Vₖ (a *synchronous*
> sweep). Most implementations use **one array updated in place**: states later in the sweep
> immediately see the new values of earlier states. In-place sweeps converge to the same
> answer, usually faster. All exercises below use in-place sweeps.

Worked example: corridor `S . G`, uniform random policy, +1 for entering G, γ = 0.9, V₀ = 0.
Sweep 1 updates S first: every outcome leads to a state worth 0 and pays 0, so V(S) stays 0. Then
the middle cell: ¼ of the time it enters G for +1, so V(M) = 0.25. Sweep 2: S now sees V(M), so
V(S) = 0.9·(¾·0 + ¼·0.25) ≈ 0.056. Value **spreads backward from the reward one step per sweep**,
and after enough sweeps it settles at the exact answer from the last lesson,
v = [0.439, 0.634, 0].

Watch the same thing on a 4 × 3 world. Each click of *step* is one sweep of policy evaluation
for the uniform random policy:

```widget
{"type": "gridworld", "map": ["...G", ".#.X", "S..."], "gamma": 0.9, "step": -0.04, "slip": 0.2, "mode": "policy-evaluation"}
```

```quiz
{"q": "In iterative policy evaluation, why do the values near the goal settle before the values far from it?",
 "options": ["Those states are updated more often", "Reward information propagates backward one transition per sweep", "θ is smaller near the goal", "The policy is better near the goal"],
 "answer": 1,
 "explain": "Each backup only looks one step ahead, so a state k steps from the reward first 'hears' about it in sweep k. Far states need more sweeps."}
```

Before you write the algorithms on your own, type along once. The walkthrough evaluates a fixed
*always RIGHT* policy sweep by sweep (in place and with two arrays), stops when Δ < θ, and
extracts a greedy policy, with an explanation for each line.

```exercise
00-walkthrough-sweeps
```

Now the same tools, no ghost text:

```exercise
01-policy-evaluation
```

## Policy improvement

Knowing v_π is only useful if it helps us find a better policy. Suppose that in state *s* we
deviate from π just once: take action *a*, then follow π. That is worth exactly q_π(s, a). If
some action has q_π(s, a) > v_π(s), deviating to it in *s* is better than following π there.

> [!KEY] The policy improvement theorem
> If π' is any policy with q_π(s, π'(s)) ≥ v_π(s) in every state, then v_π'(s) ≥ v_π(s) in
> every state. In particular, the **greedy** policy
> π'(s) = argmaxₐ q_π(s, a) = argmaxₐ Σₛ',ᵣ p(s', r | s, a)[ r + γ v_π(s') ]
> is at least as good as π everywhere, and strictly better somewhere unless π is already optimal.

The proof is a short unrolling: since choosing π'(s) once is at least as good as following
π, apply the same argument at the next state, and the next. Each step can only gain. Choosing
better actions *everywhere* therefore never hurts.

And when does improvement stop? When greedy improvement returns the same policy, then
v_π(s) = maxₐ Σ p [r + γ v_π(s')] for all s. That is the **Bellman optimality equation**, so
π is optimal. Improvement either makes strict progress or proves you are done.

```quiz
{"q": "Greedy improvement of the uniform random policy gives π'. What can you say for sure?",
 "options": ["π' is optimal", "v_π'(s) ≥ v_random(s) in every state", "v_π' is better only at the start state", "Nothing until π' is evaluated"],
 "answer": 1,
 "explain": "The policy improvement theorem guarantees π' is at least as good everywhere. Optimal usually needs more rounds of evaluation and improvement."}
```

## Policy iteration

Alternate the two steps:

$$
π₀ →(evaluate) v_π₀ →(improve) π₁ →(evaluate) v_π₁ →(improve) π₂ → … → π*
$$

Each policy is strictly better than the last (or optimal), and a finite MDP has finitely many
deterministic policies, so **policy iteration terminates in a finite number of iterations**.
In practice the number is tiny: often fewer than ten even for large gridworlds, because each
improvement step changes the action in every state at once. Compare that with the
4096 policies you enumerated by brute force in the last lesson.

Two practical details make policy iteration fast and reliable:

- **Warm start**: evaluate each new policy starting from the previous policy's V, not from zeros.
  Consecutive policies have similar values, so evaluation needs only a few sweeps.
- **Ties**: if two actions are equally good, an implementation that re-picks between them
  arbitrarily can flip forever and never report "stable". Keep the old action unless a strictly
  better one exists.

> [!TIP] The 4×3 world is a classic
> `GridWorld(["...G", ".#.X", "S..."], slip=0.2, step_reward=-0.04)` is the world from Russell &
> Norvig's *AI: A Modern Approach*. With γ = 1 your values should match their famous table,
> each shifted by +0.04, because their convention also charges −0.04 for the state you start in
> (for example v*(start) = 0.745 here vs 0.705 in the book).
> Look at the bottom row: the optimal policy takes the long way round rather than walk up past
> the pit, where a 10% slip would cost −1.

```exercise
02-policy-iteration
```

## Value iteration

Policy iteration evaluates each policy *to convergence* before improving it. That seems
wasteful: after a few sweeps the greedy policy usually stops changing. Push the idea to the
extreme and **do just one sweep of evaluation, fused with improvement**:

$$
Vₖ₊₁(s) ← maxₐ Σₛ',ᵣ p(s', r | s, a) [ r + γ Vₖ(s') ]
$$

That is the Bellman optimality equation turned into an update. It converges to v* from any
start (again a γ-contraction), and once V is close to v* you extract the policy with a single
one-step lookahead. The only difference from policy evaluation is **max instead of an average
over π**.

```python
for s in range(env.n_states):
    V[s] = max(sum(p * (r + (0 if done else gamma * V[s2])) for p, s2, r, done in env.P[s][a])
               for a in range(env.n_actions))
```

Step through value iteration below. Notice that the greedy arrows usually become optimal
long before the numbers stop changing. That is why stopping early (or using the policy
before V has fully converged) usually works well.

```widget
{"type": "gridworld", "map": ["....G", ".##.X", "....#", "S...."], "gamma": 0.95, "step": -0.04, "slip": 0.2, "mode": "value-iteration"}
```

```quiz
{"q": "Value iteration on a world with γ = 0.9 has Δ = 1 after the first sweep. Roughly how many more sweeps until Δ < 10⁻⁶?",
 "options": ["About 6", "About 130", "About 10⁶", "It never gets below 10⁻⁶"],
 "answer": 1,
 "explain": "The error shrinks by at least a factor γ per sweep: 0.9ᵏ < 10⁻⁶ needs k > 6 / log₁₀(1/0.9) ≈ 131. Larger γ means slower convergence: γ = 0.99 needs about ten times as many sweeps."}
```

```exercise
03-value-iteration
```

## Generalized policy iteration

Policy iteration and value iteration are two points on one spectrum. Between them sits
*modified policy iteration*: evaluate with k sweeps, then improve. With k = 1 you get value
iteration; with k = ∞, policy iteration. You do not even need full sweeps: **asynchronous DP**
backs up states in any order, as many times as you like (focusing on states the agent actually
visits, say), and still converges as long as no state is ignored forever.

Sutton & Barto call the common pattern **generalized policy iteration (GPI)**: two processes
interact. Evaluation pulls V toward v_π for the current policy; improvement pulls π toward
greedy(V). Each process partially undoes the other (a new policy makes V stale; a new V makes
the policy non-greedy), yet together they converge on the only fixed point where both are
satisfied: v* and π*.

> [!KEY] GPI is the template for all of RL
> Almost every method in this course is GPI with cheaper, noisier ingredients. Monte Carlo and TD
> replace the model-based evaluation with estimates from sampled experience. SARSA and
> Q-learning interleave one-sample evaluation with ε-greedy improvement at every step.
> Actor-critic keeps an explicit critic (evaluation) and actor (improvement).

## The cost of needing a model

DP is efficient in a precise sense. One sweep costs O(|S|²·|A|) in the worst case (much less
when, as in grids, each action has only a few outcomes), and it finds π* in time polynomial in
|S| and |A|, even though there are |A|^|S| deterministic policies. That is exponentially better
than brute force. But it has two hard limits:

1. **It needs p(s', r | s, a).** For backgammon, a robot arm, or a game like Core Runner, nobody
   hands you the transition probabilities. You could estimate them, which is the idea behind
   model-based RL and Dyna-Q later, but usually you can only *sample* the environment by acting.
2. **It sweeps every state.** A 10 × 10 grid is 100 states; a 19 × 19 Go board has about 10¹⁷⁰.
   The number of states grows exponentially with the number of state variables, which Bellman
   called the **curse of dimensionality**.

The next lessons drop both requirements: Monte Carlo and TD methods learn from sampled
experience without any model, and only update the states the agent actually visits.

```quiz
{"q": "Which statement about DP is FALSE?",
 "options": ["Value iteration needs the full transition model", "Policy iteration converges in finitely many iterations on a finite MDP", "DP must evaluate all |A|^|S| policies to find the optimum", "In-place sweeps converge to the same values as two-array sweeps"],
 "answer": 2,
 "explain": "DP is polynomial in |S| and |A|. It never enumerates policies, which was the point of your brute-force exercise: to see what DP avoids."}
```

## Pitfalls

- **Bootstrapping through terminal transitions.** If `terminated` is True the target is r alone.
  Terminal states must stay at value 0.
- **Comparing floats for "policy stable".** Compare *actions* (`np.array_equal` on one-hot
  policies), never values, and keep old actions on ties.
- **Stopping on the wrong quantity.** Δ < θ says the values stopped changing, not that they
  are within θ of v*. With γ = 0.99 the true error can be about 100 times larger than Δ.
- **γ = 1 with a policy that never terminates.** Policy evaluation diverges (values go to −∞
  with step costs). Start policy iteration from a policy that reaches a terminal, such as the
  uniform random one, or use γ < 1.
- **Forgetting walls.** GridWorld numbers walls as states too. They are never reachable and
  their values stay 0. Use `env.walls_mask()` when plotting so they are not confused with real
  zeros.

## What carries forward

| DP idea | Model-free successor |
|---|---|
| Iterative policy evaluation | Monte Carlo prediction, TD(0) |
| Bellman optimality backup | Q-learning, DQN target r + γ maxₐ' Q(s', a') |
| Greedy policy improvement | ε-greedy control, policy-gradient updates |
| Generalized policy iteration | SARSA, Q-learning, actor-critic |
| Exact v* on small worlds | Ground truth for testing every learner you build |
