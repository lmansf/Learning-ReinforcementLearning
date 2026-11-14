---
title: Proximal Policy Optimization
chapter: Deep RL
summary: Reuse each batch for several careful updates — probability ratios, the clipped surrogate, value and entropy terms, and why PPO is the default.
minutes: 50
---

# Proximal Policy Optimization

Vanilla policy gradient and A2C have an awkward relationship with their data:

- **One step per batch.** The gradient Â · ∇log π is only valid *at the policy that collected
  the data*. After one update the policy has changed, so the batch is thrown away. Collecting
  experience is the expensive part — wasting it hurts.
- **Step-size roulette.** Too small and learning crawls. Too large and one update moves the
  policy somewhere terrible; the next batch is collected by that terrible policy, so there may
  be no way back. In supervised learning a bad step costs a little loss; in RL it can poison
  all future data.

PPO (Schulman et al., 2017) fixes both with one idea: **take several gradient steps on the
same batch, but never let the new policy drift far from the one that collected it.**

## Trust regions in one paragraph

TRPO, PPO's predecessor, maximised the expected advantage of the new policy *subject to* a
hard limit on how much it may change, measured by KL divergence:

$$
maximise  E[ (π_θ(a|s) / π_old(a|s)) Â ]     subject to     E[ KL(π_old ‖ π_θ) ] ≤ δ
$$

That constraint defines a **trust region** — the neighbourhood where our advantage estimates,
collected under π_old, can still be trusted. TRPO enforces it with conjugate gradients and a
line search: effective, but complicated. PPO gets most of the benefit with a clipped objective
and plain SGD.

## The probability ratio

To reuse data from π_old while optimising π_θ, weight each sample by how much more (or less)
likely the new policy is to take that action — **importance sampling**:

$$
rₜ(θ) = π_θ(Aₜ | Sₜ) / π_old(Aₜ | Sₜ) = exp( log π_θ(Aₜ|Sₜ) − log π_old(Aₜ|Sₜ) )
$$

The **surrogate objective** L(θ) = E[ rₜ(θ) Âₜ ] has exactly the policy gradient as its
gradient at θ = θ_old (where r = 1 and ∇r = ∇log π). Further away it is still a reasonable
estimate — as long as r stays close to 1. Maximising it without limits would happily push
r → ∞ for every positive-advantage action. That is the failure PPO prevents.

## The clipped surrogate

$$
L^CLIP(θ) = E[ min( rₜ Âₜ ,  clip(rₜ, 1 − ε, 1 + ε) Âₜ ) ]          ε ≈ 0.2
$$

Read it case by case:

| advantage | ratio | what min() picks | gradient |
|---|---|---|---|
| Â > 0 (good action) | r ≤ 1 + ε | r·Â | push π(a) up |
| Â > 0 | r > 1 + ε | (1 + ε)·Â — constant | **0**: already up enough |
| Â < 0 (bad action) | r ≥ 1 − ε | r·Â | push π(a) down |
| Â < 0 | r < 1 − ε | (1 − ε)·Â — constant | **0**: already down enough |

The min makes the objective **pessimistic**: clipping only ever removes the incentive to move
*further* in the helpful direction. If an update accidentally moved the wrong way (Â > 0 but
r < 1), the unclipped term is the smaller one and its gradient pulls the policy back.

### Worked example

π_old(a|s) = 0.5, ε = 0.2, Â = +1.

- π_θ(a|s) = 0.55 → r = 1.1 → min(1.1, 1.1) = 1.1, gradient flows
- π_θ(a|s) = 0.75 → r = 1.5 → min(1.5, 1.2) = **1.2**, flat: no more push
- π_θ(a|s) = 0.25 → r = 0.5 → min(0.5, 0.8) = **0.5**, gradient flows (undo the mistake)

```quiz
{"q": "Â = −2, ε = 0.2, and after a few updates r = 0.7 for this sample. What is its contribution to L^CLIP and its gradient?",
 "options": ["−1.4, gradient pushes π(a) further down", "−1.6, zero gradient", "−1.4, zero gradient", "−1.6, gradient pushes π(a) up"],
 "answer": 1,
 "explain": "min(0.7·(−2), 0.8·(−2)) = min(−1.4, −1.6) = −1.6 — the clipped, constant term. The bad action is already 30% less likely, more than the 20% PPO allows credit for, so it stops pushing."}
```

Before you write code on your own, type along once. The walkthrough estimates a new policy's value
from old data and shows every idiom the next two exercises need — log-prob gathering, ratios,
`np.minimum` capping with zero gradients via `np.where`, and shuffled minibatch epochs — with an
explanation for each line.

```exercise
00-walkthrough-ratios-and-minibatches
```

Now the same tools, no ghost text:

```exercise
01-clipped-objective
```

## The full PPO loss

PPO is an actor-critic. With a shared optimiser (or separate ones, as in the exercise) the
total loss to minimise is

$$
L = −L^CLIP(θ)  +  c_v · ½ E[ (v̂(Sₜ) − targetₜ)² ]  −  c_e · E[ H(π_θ(·|Sₜ)) ]
$$

- **Value loss** — the critic regresses onto the GAE λ-returns (Âₜ + v̂_old(Sₜ)).
- **Entropy bonus** — H(π) = −Σ π log π rewards keeping the policy stochastic, delaying
  premature collapse. c_e ≈ 0.01 for discrete actions; often 0 for continuous ones.

## The algorithm

```python
for iteration in range(...):
    # 1. collect N steps (e.g. 512 here; 2048 × 8 envs in big runs) with π_old
    #    store s, a, log π_old(a|s), r, done
    # 2. GAE: advantages Â and value targets  (λ = 0.95, γ = 0.99)
    for epoch in range(K):                    # K ≈ 4–10
        for minibatch in shuffle(batch, size=64):
            normalise Â in the minibatch
            ratio = exp(log π_θ(a|s) − log π_old(a|s))
            actor:  minimise −min(r·Â, clip(r)·Â) − c_e · H
            critic: minimise ½ (v̂(s) − target)²
```

To backprop through a softmax actor you only need one fact from lesson 10:
∂ log π(a|s) / ∂ logits = 𝟙ₐ − π. The clipped loss gives ∂L/∂ log π, and the chain rule does
the rest.

```quiz
{"q": "Why does PPO store log π_old(a|s) at collection time instead of recomputing it later?",
 "options": ["To save computation only", "Because the policy changes during the K epochs — the ratio must compare against the policy that actually collected the data", "Because log-probabilities can't be recomputed for discrete actions", "It's needed for GAE"],
 "answer": 1,
 "explain": "After the first minibatch step π_θ ≠ π_old. The ratio is only meaningful relative to the frozen behaviour policy, so its log-probabilities are recorded when acting."}
```

```exercise
02-ppo-cartpole
```

> [!KEY] Many small, safe steps from one batch
> PPO squeezes several epochs of SGD out of each rollout, and the clipped ratio removes any
> incentive to move the policy more than ~ε from the one that collected the data. Data
> efficiency of off-policy reuse, stability of a trust region, simplicity of plain SGD.

## Why PPO is the default workhorse

- **Robust.** The same hyperparameters (ε = 0.2, λ = 0.95, γ = 0.99, lr ≈ 3e-4, 4–10 epochs)
  work across Atari, robotics simulators and board games with little tuning.
- **Simple.** No replay buffer, no target networks, no second-order optimisation — about 300
  lines in a single file (see CleanRL's `ppo.py`).
- **General.** Discrete or continuous actions (swap the softmax for a Gaussian), recurrent
  policies, many parallel environments.
- **Scales.** OpenAI Five (Dota 2) and the RLHF stage that fine-tunes large language models
  both used PPO.

Its weakness is **sample efficiency**: it is on-policy, so every batch is discarded after K
epochs. When environment steps are expensive (real robots), off-policy methods win.

## Debugging PPO

Log these four numbers every update — they tell you more than the return curve:

- **approx KL** between π_old and π_θ: typically 0.001–0.02. Much larger → lower the learning
  rate or the number of epochs (many implementations stop the epochs early if KL > ~0.015).
  Near zero → the actor is not learning: check the gradient chain.
- **clip fraction**: share of samples with |r − 1| > ε. Around 0.05–0.2 is healthy.
- **entropy**: should decline slowly, not crash to 0 in a few iterations.
- **explained variance** of the critic: should rise toward 1.

And the classic bugs: forgetting to normalise advantages, normalising *returns* used as value
targets inconsistently, a stale `logp_old` (recomputed after updating), mixing up the sign of
the loss, and bootstrapping through `terminated`.

```quiz
{"q": "Your PPO logs show approx KL ≈ 0.3 per update and the return oscillates wildly. What is the best first fix?",
 "options": ["Increase ε to 0.5", "Lower the learning rate and/or the number of epochs", "Remove the entropy bonus", "Use a larger minibatch only"],
 "answer": 1,
 "explain": "KL that large means each update moves the policy far outside the trust region — the clip can't fully stop many epochs of large steps. Smaller lr / fewer epochs (or KL early stopping) restores 'proximal'."}
```

## If you have PyTorch

```python
dist = torch.distributions.Categorical(logits=actor(obs_mb))
logp = dist.log_prob(act_mb)
ratio = torch.exp(logp - logp_old_mb)
adv = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
pg_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv).mean()
v_loss = 0.5 * (critic(obs_mb).squeeze(-1) - ret_mb).pow(2).mean()
loss = pg_loss + 0.5 * v_loss - 0.01 * dist.entropy().mean()
opt.zero_grad(); loss.backward()
nn.utils.clip_grad_norm_(params, 0.5); opt.step()
```

## Where to go next

You have now built, from scratch, the core of modern deep RL. Some directions from here:

- **Continuous control.** Replace the softmax with a Gaussian policy (mean from the network,
  learned log-std) and PPO works on MuJoCo-style robots. The strongest off-policy alternatives
  are **SAC** (Soft Actor-Critic — maximum-entropy RL with twin Q-networks) and **TD3** (DDPG
  plus clipped double-Q and delayed policy updates).
- **Gymnasium** (the maintained successor of OpenAI Gym) — the standard environment API you
  have been using all along: `gymnasium.make("CartPole-v1")` runs your code unchanged.
  Add `gymnasium[mujoco]` or `ale-py` for robotics and Atari.
- **Stable-Baselines3** — reliable PyTorch implementations of PPO, A2C, DQN, SAC, TD3 with a
  three-line API. Use it as a baseline before trusting your own implementation.
- **CleanRL** — single-file, readable implementations of every major algorithm; ideal for
  reading after this course.
- **Beyond:** distributional RL (C51, QR-DQN), Rainbow, model-based RL (MuZero, Dreamer),
  offline RL (CQL, IQL), multi-agent RL, and RL from human feedback (PPO/DPO for language models).
- **Books & courses:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.,
  free online); OpenAI's *Spinning Up in Deep RL*; David Silver's UCL lectures.
