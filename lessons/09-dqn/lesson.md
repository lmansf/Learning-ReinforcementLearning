---
title: Deep Q-Networks
chapter: Deep RL
summary: Q-learning with a neural network — manual backprop, experience replay, target networks, Huber loss and Double DQN.
minutes: 60
---

# Deep Q-Networks

In 2015 a single algorithm learned to play 49 Atari games from raw pixels, many at human
level. Under the hood it was **Q-learning** — the same update you wrote for the cliff — with
a neural network in place of the table, plus two engineering tricks that kept it from blowing
up. This lesson builds that algorithm, **DQN**, from scratch in numpy.

Tile coding needed *you* to design features. A neural network learns its own features: the
hidden layers transform the raw state into a representation where the output layer — a
linear function, just like last lesson — can read off the Q-values.

$$
q̂(s, ·; θ) = W₃ · relu(W₂ · relu(W₁ s + b₁) + b₂) + b₃          one output per action
$$

## A neural network is a function with a gradient

A **multi-layer perceptron** (MLP) alternates linear layers and non-linearities. With a batch
of inputs X (shape B × n_in) one layer computes

$$
H = relu(X W + b)          W: n_in × n_out,   b: n_out
$$

Training needs ∂loss/∂(every weight). **Backpropagation** is just the chain rule applied layer
by layer, from the output backwards. If D = ∂L/∂(output of a linear layer) and X was its input:

$$
∂L/∂W = Xᵀ D          ∂L/∂b = Σ_batch D          ∂L/∂X = D Wᵀ
$$

and passing back through a ReLU multiplies by its derivative, which is 1 where the input was
positive and 0 elsewhere. That is the whole of it — PyTorch's `loss.backward()` does exactly
this bookkeeping for you, for any graph.

> [!TIP] Check every gradient numerically
> For any parameter θᵢ, (L(θᵢ + ε) − L(θᵢ − ε)) / 2ε ≈ ∂L/∂θᵢ. If your analytic gradient
> disagrees by more than ~1e-6 (relative) on a small network in float64, the backward pass
> has a bug. Ten minutes of gradient checking saves days of "why doesn't my agent learn?".

```quiz
{"q": "A linear layer has input X of shape (32, 4) and weights W of shape (4, 64). Upstream gradient D has shape (32, 64). What is the shape of ∂L/∂W = XᵀD?",
 "options": ["(32, 64)", "(4, 64)", "(64, 4)", "(32, 4)"],
 "answer": 1,
 "explain": "Xᵀ is (4, 32), D is (32, 64), so XᵀD is (4, 64) — the same shape as W, as every gradient must be. The batch dimension is summed out by the matrix product."}
```

```exercise
01-numpy-mlp
```

## Why naive deep Q-learning fails

Plug a network into Q-learning and update after every step with the target
r + γ maxₐ′ q̂(s′, a′; θ). It usually diverges or oscillates. Three reasons:

1. **Correlated samples.** Consecutive states are nearly identical. SGD assumes independent
   samples; a stream of similar ones drags the whole network toward the current situation and
   it forgets everything else.
2. **A moving target.** The target uses the same θ we are updating. Each step changes the
   target, which changes the next step — the network chases its own tail.
3. **The deadly triad.** Function approximation + bootstrapping + off-policy learning, all at once.

DQN answers (1) with **experience replay** and (2) with a **target network**.

## Experience replay

Store every transition (s, a, r, s′, terminated) in a big circular buffer — say the last
50 000 steps — and train on **random minibatches** sampled from it.

- Random sampling breaks temporal correlations: a batch mixes early, late, left, right.
- Each transition is reused many times: much better sample efficiency.
- Q-learning is off-policy, so learning from old ε-greedy data is legitimate.

> [!NOTE] Store `terminated`, not `done`
> The buffer must remember whether s′ was a *real* terminal state. A CartPole episode cut at
> 500 steps is truncated: the pole is still up and s′ has a future worth bootstrapping from.

### Huber loss

Early on, TD errors can be huge. With squared loss the gradient grows with the error, so one
bad target produces a giant update. The **Huber loss** is quadratic near zero and linear
beyond δ, so its gradient is simply the error **clipped to [−δ, δ]**:

$$
Huber(x) = ½x²  if |x| ≤ δ,     δ(|x| − ½δ)  otherwise          ∂Huber/∂x = clip(x, −δ, δ)
$$

```exercise
02-replay-buffer
```

## Target networks

Keep a second copy of the network, θ⁻, frozen most of the time. Targets use it:

$$
yᵢ = rᵢ + γ (1 − doneᵢ) · maxₐ′ q̂(s′ᵢ, a′; θ⁻)
$$

Every C steps (e.g. 500), copy θ⁻ ← θ. Between syncs the regression problem is stationary: the
network fits fixed labels, like supervised learning. A common alternative is a "soft" update
θ⁻ ← τθ + (1 − τ)θ⁻ every step with τ ≈ 0.005.

```quiz
{"q": "What problem does the target network mainly solve?",
 "options": ["Correlation between consecutive samples", "Targets that move every time the network is updated", "Overestimation of Q-values", "Lack of exploration"],
 "answer": 1,
 "explain": "Replay fixes correlation; the frozen target network keeps the regression labels still for C steps. (Overestimation is Double DQN's job.)"}
```

## The full algorithm

```python
for t in range(total_steps):
    eps = epsilon_by_step(t)                        # 1.0 → 0.05 over the first 10k steps
    a = random action if rng.random() < eps else argmax q(s; θ)
    s2, r, terminated, truncated, _ = env.step(a)
    buffer.add(s, a, r, s2, terminated)
    if t >= learning_starts:
        S, A, R, S2, D = buffer.sample(64)
        y = R + γ (1 − D) max q(S2; θ⁻)             # targets from the frozen network
        loss = mean Huber(q(S; θ)[A] − y)           # only the taken actions get gradient
        θ ← Adam step on ∇loss
    if t % C == 0: θ⁻ ← θ
```

The **ε schedule** matters more than it looks: the agent starts knowing nothing, so it should
explore almost uniformly, then lean on its estimates as they improve. A linear decay from 1.0 to
0.05 over the first ~20% of training is a robust default. Try it below — at ε = 0.05 how much
probability does a non-greedy action still get?

```widget
{"type": "epsilon", "values": [1.2, 0.4], "epsilon": 0.05}
```

## Double DQN

The max in the target is **biased upward**: if Q-estimates are noisy, maxₐ picks whichever
action's noise happened to be positive. Over many bootstraps the overestimation compounds.

**Double DQN** separates *choosing* the next action from *evaluating* it:

$$
y = r + γ (1 − done) · q̂(s′, argmaxₐ′ q̂(s′, a′; θ), θ⁻)
$$

The online network picks a′; the target network scores it. Their errors are partly
independent, so the bias shrinks. It is a one-line change and almost always helps.

```quiz
{"q": "Q_online(s′) = [3, 1], Q_target(s′) = [2, 5], r = 1, γ = 0.5, not terminal. What is the Double DQN target?",
 "options": ["3.5", "2.0", "2.5", "1.5"],
 "answer": 1,
 "explain": "The online net picks a′ = 0 (value 3). The target net evaluates action 0: 2. y = 1 + 0.5·2 = 2.0. Plain DQN would use max Q_target = 5 → 3.5."}
```

```exercise
03-dqn-cartpole
```

> [!KEY] DQN = Q-learning + a network + stability tricks
> The update is still "move q̂(s, a) toward r + γ max q̂(s′, ·)". Replay decorrelates the data,
> the target network freezes the labels, Huber loss caps the step, and Double DQN reduces
> overestimation. Every trick fights the deadly triad.

## Debugging deep RL

Deep RL fails silently: no exception, just an agent that doesn't improve. Work through this list.

- **Unit-test the pieces** — gradient check the network; test targets and loss on a 3-row batch you computed by hand (that is what the checks do).
- **Overfit a fixed batch.** A few hundred steps on one minibatch should drive the loss near zero. If not, the problem is the network or the loss, not RL.
- **Log more than the return:** the loss, the mean of max Q, and ε. For CartPole with γ = 0.99, Q should rise toward ~1/(1−γ) = 100. Q values of 10⁶ mean divergence; Q stuck at 0 means no learning.
- **Check `terminated` vs `truncated`** — the most common silent bug.
- **Evaluate greedily**, separately from training. Training returns include exploration noise.
- **Expect instability.** DQN on CartPole often reaches 500 and then collapses for a while ("catastrophic forgetting" when the buffer fills with only good states). Keep the **best checkpoint**, as the provided loop does, and judge algorithms over **several seeds**.
- **Change one thing at a time.** Learning rate and target-sync period are the usual suspects.

```quiz
{"q": "Your DQN's average max-Q climbs past 10⁵ on CartPole (γ = 0.99, reward 1 per step). What is the most likely explanation?",
 "options": ["The agent has learned a great policy", "The values are diverging — e.g. bootstrapping through terminal states, no target network, or a too-large learning rate", "ε is too small", "The replay buffer is too big"],
 "answer": 1,
 "explain": "With reward ≤ 1 per step and γ = 0.99, no true value exceeds 1/(1−γ) = 100. Anything far above that is divergence."}
```

## If you have PyTorch

Everything you built by hand maps directly onto PyTorch; autograd replaces `backward()`:

```python
import torch, torch.nn as nn
q = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
q_target = copy.deepcopy(q)
opt = torch.optim.Adam(q.parameters(), lr=1e-3)

S, A, R, S2, D = (torch.as_tensor(x) for x in buffer.sample(64))
with torch.no_grad():                                   # no gradient through the target
    a2 = q(S2).argmax(1, keepdim=True)                  # Double DQN
    y = R + 0.99 * (1 - D) * q_target(S2).gather(1, a2).squeeze(1)
q_sa = q(S).gather(1, A.long().unsqueeze(1)).squeeze(1)
loss = nn.functional.smooth_l1_loss(q_sa, y)            # = Huber with δ = 1
opt.zero_grad(); loss.backward(); opt.step()
```

`torch.no_grad()` is the "semi" in semi-gradient: no gradient flows through the target.
