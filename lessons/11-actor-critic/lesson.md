---
title: Actor-Critic
chapter: Deep RL
summary: A learned critic turns REINFORCE into an online, low-variance learner — advantages, one-step actor-critic, A2C and GAE(λ).
minutes: 45
---

# Actor-critic methods

REINFORCE has two annoying properties:

- It must wait until the **end of the episode** to compute Gₜ. In a 500-step CartPole episode
  the first action's update has to wait 500 steps; in a continuing task it waits forever.
- Gₜ is a sum of many random rewards, so its **variance** is huge — even with a baseline.

We already know the cure from the tabular chapters: **bootstrap**. Instead of waiting for the
full return, estimate it from one reward plus a learned value of the next state. A policy
gradient method that bootstraps with a learned value function is an **actor-critic**:

- the **actor** π(a | s, θ) chooses actions — trained by policy gradient
- the **critic** v̂(s, w) judges them — trained by TD learning

## From baseline to critic

Last lesson's baseline v̂(s) was only used to *centre* the return: Gₜ − v̂(Sₜ). An actor-critic
also uses it *inside* the target. The one-step return is Rₜ₊₁ + γ v̂(Sₜ₊₁), so the advantage
estimate becomes the familiar **TD error**:

$$
δₜ = Rₜ₊₁ + γ v̂(Sₜ₊₁, w) − v̂(Sₜ, w)
$$

Why is δ an advantage? The **advantage** of an action is how much better it is than the policy's
average from that state:

$$
A_π(s, a) = q_π(s, a) − v_π(s)          and          E[ δₜ | Sₜ = s, Aₜ = a ] = q_π(s, a) − v_π(s)   (if v̂ = v_π)
$$

So with an accurate critic, δₜ is an unbiased sample of the advantage — and it involves a single
reward instead of hundreds. With an *inaccurate* critic it is biased. Trading a little bias for a
lot less variance is the bargain at the heart of actor-critic.

## One-step actor-critic

```python
for each step of each episode:
    a ~ π(·|s, θ);   s2, r, terminated, truncated = env.step(a)
    δ = r + γ (1 − terminated) v̂(s2, w) − v̂(s, w)
    w ← w + α_w δ ∇v̂(s, w)                 # critic: semi-gradient TD(0)
    θ ← θ + α_θ δ ∇log π(a | s, θ)          # actor: policy gradient with δ as the advantage
    s ← s2
```

It is fully **online** — it learns at every step, works for continuing tasks, and needs no
episode memory. Note the two step sizes: the critic usually needs to learn *faster* than the
actor, so the actor's advantages mean something.

```quiz
{"q": "In one-step actor-critic, the critic estimates v̂(s) = 10, v̂(s′) = 12, the reward is 1 and γ = 0.9. The action taken will be made…",
 "options": ["more likely (δ = +1.8)", "less likely (δ = −1.0)", "more likely (δ = +3.0)", "neither — δ = 0"],
 "answer": 0,
 "explain": "δ = 1 + 0.9·12 − 10 = 1.8 > 0: things went better than the critic expected, so the actor increases π(a|s)."}
```

## n-step returns and GAE(λ)

The one-step TD error is low-variance but leans heavily on the critic. The full Monte Carlo
return is unbiased but noisy. In between are **n-step advantages** — the same spectrum you met
with n-step TD:

$$
Â⁽¹⁾ = δₜ          Â⁽²⁾ = δₜ + γδₜ₊₁          Â⁽ⁿ⁾ = Σₖ₌₀ⁿ⁻¹ γᵏ δₜ₊ₖ          Â⁽∞⁾ = Gₜ − v̂(Sₜ)
$$

(The δ's telescope: δₜ + γδₜ₊₁ = Rₜ₊₁ + γRₜ₊₂ + γ²v̂(Sₜ₊₂) − v̂(Sₜ).)

**Generalised Advantage Estimation** (Schulman et al., 2016) does not pick one n — it takes an
exponentially weighted average of all of them, exactly like TD(λ):

$$
Âₜ = Σₖ (γλ)ᵏ δₜ₊ₖ          computed backwards as          Âₜ = δₜ + γλ Âₜ₊₁
$$

- **λ = 0** → Âₜ = δₜ (one-step, low variance, biased by the critic)
- **λ = 1** → Âₜ = Gₜ − v̂(Sₜ) (Monte Carlo, unbiased, high variance)
- **λ ≈ 0.95** is the near-universal default.

The critic is trained to predict the **λ-return** Âₜ + v̂(Sₜ), so the same backward pass gives
both the actor's advantages and the critic's targets.

### Worked example

Rollout of three steps, γ = 0.5, λ = 0.5: rewards [1, 0, 2], critic values [1, 2, 1], and the
state after the rollout is worth 4.

- δ₂ = 2 + 0.5·4 − 1 = **3**
- δ₁ = 0 + 0.5·1 − 2 = **−1.5**
- δ₀ = 1 + 0.5·2 − 1 = **1**
- Â₂ = 3,  Â₁ = −1.5 + 0.25·3 = **−0.75**,  Â₀ = 1 + 0.25·(−0.75) = **0.8125**

### Episode boundaries inside a rollout

A rollout of fixed length usually spans several episodes. If step t ended an episode, nothing
after it belongs to the same trajectory: δₜ must not bootstrap (no γv̂(Sₜ₊₁)) and Âₜ must not
accumulate Âₜ₊₁. Multiply both by (1 − doneₜ).

> [!WARN] Truncation, again
> A time-limit cut is not a real ending — but for GAE it *is* the end of the trajectory in the
> rollout. The standard fix: on truncation, add γ·v̂(s_final) to the last reward, then treat the
> step as done. The provided A2C loop does exactly that.

Before you write code on your own, type along once. The walkthrough works through a hand-made
rollout and shows every idiom the next two exercises need — aligning V(sₜ₊₁), backward sums with
done-masks and a linear critic update — with an explanation for each line.

```exercise
00-walkthrough-returns-and-critic
```

Now the same tools, no ghost text:

```exercise
01-gae
```

## A2C: batched actor-critic

Updating on every single transition is noisy and slow in numpy. **A2C** (Advantage Actor-Critic)
collects a short rollout of n steps (often from several environments in parallel), then does one
batched update:

1. run the policy for n steps (e.g. 32), storing states, actions, rewards, dones
2. compute GAE advantages and λ-return targets with the critic
3. **actor loss** L_π = −mean(Âₜ log π(Aₜ|Sₜ)) — normalise Â within the batch first
4. **critic loss** L_V = ½ mean(v̂(Sₜ) − targetₜ)²
5. optionally an **entropy bonus**: subtract β·H(π) from the loss to keep the policy exploring

Writing the objectives as *losses* lets an off-the-shelf minimiser (Adam) do the work: minimising
−Â log π is ascending the policy gradient.

> [!NOTE] A2C vs A3C
> The original A3C (2016) ran many actor-learners asynchronously on CPU threads. A2C is the
> synchronous version: wait for all environments, update once. It is simpler, deterministic and
> just as good — the "A3" mostly survives in old blog posts.

In this exercise the actor is still a **linear** softmax (enough for CartPole) while the critic
is a small **neural network** — V(s) for CartPole is a curved, symmetric surface that no linear
function of the raw state can represent.

```quiz
{"q": "Why normalise the advantages within each batch before the actor update?",
 "options": ["It is required for the policy gradient theorem to hold", "It keeps the size of the actor's step roughly constant regardless of reward scale and critic accuracy", "It makes the critic learn faster", "It removes the bias of the critic"],
 "answer": 1,
 "explain": "Normalisation is a practical trick: it fixes the scale of the update (and centres it). It doesn't remove bias, and the theorem doesn't need it."}
```

```exercise
02-a2c-cartpole
```

> [!KEY] Actor-critic = policy gradient + learned advantage
> The actor follows Â · ∇log π. The critic supplies Â by bootstrapping, which slashes variance
> and allows online, per-rollout updates. GAE's λ dials between the critic's bias (λ → 0) and
> Monte Carlo's variance (λ → 1).

## Debugging actor-critic

- **Look at the critic first.** Plot the critic loss and a few v̂(s) values. For CartPole with
  γ = 0.99, values should climb toward ~100 as the policy improves. A critic that predicts
  garbage gives the actor garbage advantages.
- **Scale value targets.** Networks like outputs around ±1. Either normalise returns or predict
  V·c for a small c (the `value_scale` in the exercise).
- **Signs.** Adam minimises. Actor loss = −Â·log π; the advantage is a *constant* (no gradient
  flows from the actor loss into the critic).
- **Explained variance** 1 − Var(target − v̂)/Var(target) is a great one-number critic health
  check: near 1 is good, ≤ 0 means the critic is useless.
- **Returns that rise then collapse** often mean the actor's step is too large — the motivation
  for PPO, next.

```quiz
{"q": "Your A2C agent's return goes up, then crashes to ~10 and never recovers. Which change is most likely to help?",
 "options": ["Larger actor learning rate", "Smaller actor learning rate (or a trust-region method like PPO)", "Set λ = 0", "Remove the critic"],
 "answer": 1,
 "explain": "A too-large policy step can move the policy somewhere terrible; the new data it collects is then bad too, and it can't recover. Smaller or constrained steps (PPO) fix this."}
```
