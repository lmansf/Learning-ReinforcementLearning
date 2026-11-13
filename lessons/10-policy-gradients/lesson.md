---
title: Policy Gradients
chapter: Deep RL
summary: Optimise the policy directly — softmax policies, the log-derivative trick, the policy gradient theorem, REINFORCE and baselines.
minutes: 50
---

# Policy gradients

Everything so far has been **value-based**: learn q̂(s, a), then act greedily (or ε-greedily)
with respect to it. The policy was an afterthought — an argmax.

Policy gradient methods flip this around. Write down a **parameterised policy** π(a | s, θ)
and adjust θ by gradient ascent on the expected return:

$$
J(θ) = E_π[ G₀ ]          θ ← θ + α ∇J(θ)
$$

No argmax, no value table needed (though values will come back as helpers).

## Why optimise the policy directly?

- **Stochastic policies.** Rock-paper-scissors, partially observed mazes, bluffing in poker:
  the best policy can be random. ε-greedy can only be "greedy plus noise"; a softmax policy can
  learn *any* probabilities.
- **Continuous actions.** argmaxₐ q(s, a) over a torque in [−2, 2] is an optimisation problem
  at every step. A policy that outputs a Gaussian over torques just *samples*.
- **Smooth changes.** A tiny change in q̂ can flip an argmax and change the behaviour
  completely. A tiny change in θ changes the action probabilities a tiny bit. This makes
  learning better behaved — and is the foundation of PPO.
- **Sometimes the policy is simpler than the values.** In CartPole "push toward the side the pole
  is falling" is a linear rule; the value function is a complicated curved surface.

## The softmax policy

For discrete actions, compute a **preference** h(s, a, θ) for every action and turn the
preferences into probabilities with a softmax:

$$
π(a | s, θ) = exp(h(s, a, θ)) / Σ_b exp(h(s, b, θ))
$$

With linear preferences h(s, a) = x(s)ᵀ θₐ (one column of θ per action), this is exactly
multinomial logistic regression — the policy is a classifier whose "labels" will come from
rewards. Drag the temperature: high temperature flattens the distribution (explore), low
temperature sharpens it toward the argmax (exploit).

```widget
{"type": "softmax", "prefs": [2.0, 1.0, 0.1, -1.0], "temperature": 1.0}
```

> [!TIP] Stable softmax
> exp(1000) overflows. Since softmax(h) = softmax(h − c) for any constant c, always subtract
> max(h) before exponentiating. Every deep learning library does this internally.

## The log-derivative trick

We want ∇θ E[G]. The expectation is over trajectories whose *probabilities* depend on θ, so we
cannot simply move the gradient inside. The trick: for any probability p(x; θ),

$$
∇θ p(x; θ) = p(x; θ) ∇θ log p(x; θ)
$$

(just the chain rule for log). Applying it to the expected return of a one-step problem:

$$
∇θ E_{a∼π}[ r(a) ] = Σₐ r(a) ∇π(a) = Σₐ π(a) r(a) ∇log π(a) = E_{a∼π}[ r(a) ∇log π(a) ]
$$

The right-hand side is an expectation under the policy — so we can **estimate it by sampling**:
take an action, observe r, and r · ∇log π(a) is an unbiased estimate of the gradient. We never
need to know how r depends on θ, and the environment need not be differentiable.

∇log π(a | s, θ) is called the **score function**. For the linear softmax it is beautifully simple:

$$
∇θ log π(a | s) = x(s) (𝟙ₐ − π(· | s))ᵀ          (one column per action)
$$

Read it as: *raise the preference of the action you took, lower all actions in proportion to
how likely they already were*. Multiply by the return and good actions get pushed up, bad ones down.

### Worked example

Two actions, x(s) = [1], θ = [0, 0] so π = [0.5, 0.5]. We take a = 1 and get G = 4, α = 0.1.

- score: x · (𝟙₁ − π) = [−0.5, +0.5]
- update: θ ← [0, 0] + 0.1 · 4 · [−0.5, 0.5] = [−0.2, 0.2]
- new π = softmax([−0.2, 0.2]) ≈ [0.40, 0.60] — action 1 became more likely.

```quiz
{"q": "With a softmax policy, π(·|s) = [0.7, 0.2, 0.1] and you took action 0. Which column of the score function for this state is NEGATIVE (for x(s) > 0)?",
 "options": ["Column 0 only", "Columns 1 and 2", "All columns", "None"],
 "answer": 1,
 "explain": "The score is x·(𝟙ₐ − π) = x·[0.3, −0.2, −0.1]. The taken action's preference goes up (by 1 − 0.7), every other action's goes down in proportion to its probability."}
```

```exercise
01-softmax-policy
```

## The policy gradient theorem

For full episodes the same trick gives the **policy gradient theorem**: the gradient of J only
involves the score of each action, weighted by how good things went *afterwards*:

$$
∇J(θ) ∝ E_π[ Σₜ Gₜ ∇θ log π(Aₜ | Sₜ, θ) ]
$$

Remarkably the gradient does **not** involve the derivative of the state distribution — the
part that depends on the unknown environment dynamics. That is what makes model-free policy
gradients possible.

Note the use of **Gₜ**, the return *from step t on* ("reward-to-go"), not G₀: an action can't
affect rewards that came before it, so including them only adds noise.

## REINFORCE

Monte Carlo policy gradient (Williams, 1992):

```python
loop:
    play an episode with π_θ: S₀, A₀, R₁, …, S_{T−1}, A_{T−1}, R_T
    for t = 0 … T−1:
        Gₜ = Σₖ γᵏ⁻ᵗ⁻¹ Rₖ          # return-to-go
        θ ← θ + α Gₜ ∇log π(Aₜ | Sₜ, θ)
```

It is unbiased and dead simple. Its problem is **variance**. In CartPole every reward is +1,
so every Gₜ is positive: every action taken gets pushed up, and the only learning signal is
that actions in long episodes get pushed up *more*. The gradient estimate swings wildly from
episode to episode.

## Baselines: subtract what you expected

Replace Gₜ by Gₜ − b(Sₜ) for any baseline b that does not depend on the action:

$$
∇J(θ) ∝ E_π[ Σₜ (Gₜ − b(Sₜ)) ∇log π(Aₜ | Sₜ, θ) ]
$$

This adds **no bias**, because the score has zero mean under the policy:

$$
E_{a∼π}[ b(s) ∇log π(a|s) ] = b(s) Σₐ ∇π(a|s) = b(s) ∇ Σₐ π(a|s) = b(s) ∇1 = 0
$$

but it can reduce variance enormously. The natural choice is b(s) = v̂(s, w), a learned value
function — then Gₜ − v̂(Sₜ) is the **advantage**: was this outcome better or worse than usual
from here? Better-than-expected actions are reinforced, worse-than-expected ones discouraged,
even when all rewards are positive.

Two more variance-reduction tricks used everywhere in practice:

- **Batch several episodes** per update and average the gradient.
- **Normalise the advantages** within a batch to mean 0, std 1. This acts like an adaptive
  baseline and step size, and makes the learning rate far less sensitive to reward scale.

```quiz
{"q": "Why does subtracting a baseline b(s) not bias the policy gradient?",
 "options": ["Because b(s) is usually small", "Because E[∇log π(a|s)] = 0 under the policy, so the b(s) term averages out", "Because the baseline is learned by gradient descent too", "It does bias it, but only a little"],
 "answer": 1,
 "explain": "Σₐ π ∇log π = Σₐ ∇π = ∇Σₐ π = ∇1 = 0. Any action-independent baseline multiplies a zero-mean quantity, so it changes the variance but not the expectation."}
```

```exercise
02-reinforce-baseline
```

> [!KEY] Push up actions that did better than expected
> Policy gradient = Σ (advantage) × ∇log π(action). The score ∇log π says *how* to make an action
> more likely; the advantage says *whether* to. Everything that follows — actor-critic, PPO —
> is a better way to estimate the advantage or a safer way to take the step.

## Debugging policy gradients

- **Check the sign.** We *ascend* J. If returns drop steadily from the start, you are descending.
  In a loss-based framework the loss is −mean(Â · log π).
- **Watch the entropy** H(π) = −Σ π log π. If it collapses to 0 early, the policy became
  deterministic before it found anything good — lower the learning rate or add an entropy bonus.
- **Watch the probabilities.** Print π(·|s) for a few fixed states during training. Slowly
  moving probabilities are healthy; flipping to 0/1 in a handful of updates means steps are too big.
- **Learning rate sensitivity is real.** Without normalisation the gradient scales with the
  return (here up to ~100), so the right α can differ by orders of magnitude between tasks.
- **One seed is an anecdote.** Policy gradient curves are noisy; compare methods over several seeds.

```quiz
{"q": "REINFORCE's entropy drops to almost zero within the first 20 updates and the return is stuck at ~20. What should you try first?",
 "options": ["Increase the learning rate", "Decrease the learning rate and/or add an entropy bonus", "Remove the baseline", "Increase γ to 1"],
 "answer": 1,
 "explain": "The policy collapsed to a deterministic, bad behaviour and can no longer explore. Smaller steps or an entropy bonus keep it stochastic long enough to find better actions."}
```

## If you have PyTorch

With autograd you never write the score function — you write a "loss" whose gradient *is* the
policy gradient:

```python
logits = policy_net(torch.as_tensor(X))                 # (N, n_actions)
logp = torch.log_softmax(logits, -1).gather(1, A[:, None]).squeeze(1)
adv = (G - baseline).detach()                           # no gradient through the advantage
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
loss = -(adv * logp).mean()                             # minus: optimisers minimise
opt.zero_grad(); loss.backward(); opt.step()
```
