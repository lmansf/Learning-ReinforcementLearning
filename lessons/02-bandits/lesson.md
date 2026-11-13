---
title: Multi-Armed Bandits
chapter: Foundations
summary: Exploration vs exploitation in the simplest possible RL problem — action values, ε-greedy, optimism, UCB and gradient bandits.
minutes: 45
---

# Multi-armed bandits

Strip reinforcement learning down to its bones: there is **one state**, nothing you do changes
the world, and every action pays out immediately. All that remains is the question that never
goes away — *should I do what has worked so far, or try something I know less about?*

That stripped-down problem is the **k-armed bandit**, named after a row of slot machines
("one-armed bandits"). It is small enough to solve on paper, yet every idea you meet here —
value estimates, incremental updates, ε-greedy exploration, optimism, softmax policies —
reappears, unchanged, in Q-learning, DQN and PPO.

## The k-armed bandit problem

You face k levers. Pulling lever *a* gives a random reward drawn from a distribution you cannot
see. Each arm has a true mean, its **action value**:

$$
q*(a) = E[ Rₜ | Aₜ = a ]
$$

If you knew q* you would always pull `argmax q*(a)` and be done. You do not, so you must
**estimate** it from the rewards you observe, and your estimates are only as good as the data
you choose to collect. Pull an arm rarely and you will know little about it.

Try it yourself. The widget hides the true values; pull arms by hand for a while, then let the
automatic ε-greedy player take over and compare.

```widget
{"type": "bandit", "arms": 10, "seed": 3}
```

> [!NOTE] The testbed in this course
> `rlenvs.KArmedBandit(k=10, seed=s)` draws each q*(a) from N(0, 1), and each pull pays
> N(q*(a), 1). That is Sutton & Barto's **10-armed testbed**. `bandit.optimal_action` tells
> *you* (not the agent) which arm is truly best, so you can measure performance.

## Estimating action values

The natural estimate of q*(a) is the average of the rewards you have received from arm *a*:

$$
Qₜ(a) = (sum of rewards when a was taken) / (number of times a was taken)
$$

By the law of large numbers, Qₜ(a) → q*(a) as the arm is pulled infinitely often. Storing every
reward to recompute that mean would be wasteful. Instead, write the mean after *n* rewards in
terms of the mean after *n − 1*:

$$
Qₙ₊₁ = Qₙ + (1/n) [ Rₙ − Qₙ ]
$$

Worked example: rewards 1, 2, 3, 6 from one arm, starting at Q₁ = 0.

| n | Rₙ | step 1/n | Qₙ₊₁ = Qₙ + (1/n)(Rₙ − Qₙ) |
|---|---|---|---|
| 1 | 1 | 1 | 0 + 1·(1 − 0) = 1 |
| 2 | 2 | 1/2 | 1 + ½·(2 − 1) = 1.5 |
| 3 | 3 | 1/3 | 1.5 + ⅓·(3 − 1.5) = 2 |
| 4 | 6 | 1/4 | 2 + ¼·(6 − 2) = 3 |

Exactly the mean of (1, 2, 3, 6), using constant memory and one subtraction per step.

> [!KEY] The update rule of reinforcement learning
> **NewEstimate ← OldEstimate + StepSize · [Target − OldEstimate]**
>
> `[Target − OldEstimate]` is an **error**. Each update moves the estimate a fraction of the
> way toward the target. TD learning, Q-learning and even neural-network value losses are all
> this line with a cleverer target.

```quiz
{"q": "Q = 4 after 3 pulls of an arm. The 4th pull pays 8. What is the new sample-average estimate?",
 "options": ["6", "5", "4.5", "8"],
 "answer": 1,
 "explain": "Q ← 4 + (1/4)(8 − 4) = 5. Check: the old total was 12, the new total 20, and 20 / 4 = 5."}
```

## Greedy and ε-greedy action selection

The **greedy** agent always takes `argmax Qₜ(a)`. It exploits, but if an unlucky early reward
makes the best arm look bad, the greedy agent may never pull it again: its estimate is never
corrected, because correcting it requires pulling it.

The fix is to explore a little all the time. An **ε-greedy** agent:

- with probability 1 − ε picks a greedy arm (ties broken at random)
- with probability ε picks an arm uniformly at random, **including** the greedy one

So the greedy arm is chosen with probability (1 − ε) + ε/k, and every other arm with ε/k.
Every arm keeps getting pulled forever, so every estimate converges to the truth.

```widget
{"type": "epsilon", "values": [0.2, 1.4, 0.9, -0.3, 1.1], "epsilon": 0.1}
```

```quiz
{"q": "With ε = 0.1 and k = 10 arms, how often does ε-greedy pick the greedy arm (assuming no ties)?",
 "options": ["90%", "91%", "99%", "10%"],
 "answer": 1,
 "explain": "(1 − ε) + ε/k = 0.9 + 0.01 = 0.91. The random branch also lands on the greedy arm 1 time in 10."}
```

> [!WARN] Ties matter more than you think
> `np.argmax` returns the **first** maximal index. At the start every Q is 0, so an argmax-greedy
> agent always tries arm 0 first, and in a deterministic task it might never try anything else.
> Break ties at random: `rng.choice(np.flatnonzero(Q == Q.max()))`.

```exercise
01-epsilon-greedy
```

## Measuring an algorithm: the testbed

A single run on a single bandit is dominated by luck. To compare methods, we average over
**many independent bandit problems** and plot two curves against time:

- **average reward** at step t (what we actually care about)
- **% optimal action** at step t (how often the agent has found the best arm)

Sutton & Barto's classic finding, which you will reproduce: greedy climbs fastest at first,
then flattens at roughly a third optimal; ε = 0.1 explores more, finds the best arm sooner and
ends up near 80% optimal; ε = 0.01 improves slowly but would eventually win, because it wastes
less on random pulls once it knows the answer. You will build this experiment after two more
ideas: constant step sizes and optimism.

## Nonstationary problems and constant step size

The sample average weights the 1st reward and the 1000th equally. That is right when q* never
changes. In most real problems it does drift (opponents adapt, users change), and old rewards
become misleading. Use a **constant** step size α ∈ (0, 1]:

$$
Qₙ₊₁ = Qₙ + α [ Rₙ − Qₙ ]  =  (1 − α)ⁿ Q₁ + Σᵢ α (1 − α)ⁿ⁻ⁱ Rᵢ
$$

This is an **exponential recency-weighted average**: a reward *i* steps old has weight
α(1 − α)ⁱ, which shrinks geometrically. The estimate never fully converges; it keeps
tracking, which is exactly what you want when the target moves.

> [!MATH] When do step sizes guarantee convergence?
> For a stationary problem, a sequence of step sizes αₙ makes Q converge to q* with
> probability 1 if Σ αₙ = ∞ (steps are big enough to overcome any start) and Σ αₙ² < ∞
> (steps shrink fast enough to average out noise). 1/n satisfies both; a constant α fails the
> second condition, which is why it keeps tracking instead of settling.

```quiz
{"q": "With constant α = 0.5, how much weight does the most recent reward get in Q?",
 "options": ["1/n", "0.5", "0.25", "It depends on Q₁"],
 "answer": 1,
 "explain": "The newest reward has weight α = 0.5, the one before α(1 − α) = 0.25, then 0.125, … Recent data dominates."}
```

## Optimistic initial values

Everything so far starts with Q₁(a) = 0. Start instead with something wildly **optimistic**,
say Q₁(a) = 5 on a testbed where rewards are around 0–2. Now even a purely greedy agent
explores: whichever arm it pulls disappoints it, so its estimate drops below the others and
the agent moves on. Every arm gets tried several times before the estimates settle.

This trick is simple and surprisingly strong on stationary problems (it even beats ε = 0.1 on
the testbed), but it is a *one-off* burst of exploration at the start. If the problem changes
later, optimism has long been used up. Use a constant α with it, otherwise sample averages
wash out the initial value after a single pull.

> [!TIP] Optimism reappears later
> Initialising Q-values optimistically is still a standard trick for tabular Q-learning, and
> "optimism in the face of uncertainty" is the idea behind UCB below.

Now put the testbed together and watch greedy, ε-greedy, optimistic-greedy, and the
sample-average vs constant-α race on a drifting bandit, all in one experiment.

```exercise
02-testbed
```

## Upper confidence bound (UCB) action selection

ε-greedy explores *blindly*: a random pull is equally likely to go to an arm you have tried
500 times as to one you have tried twice. It would be smarter to prefer arms that are both
promising **and** uncertain. UCB picks

$$
Aₜ = argmaxₐ [ Qₜ(a) + c · √( ln t / Nₜ(a) ) ]
$$

where Nₜ(a) is how many times *a* has been pulled. The square-root term is an **uncertainty
bonus**:

- each pull of *a* raises Nₜ(a) and shrinks its bonus
- time passing raises ln t and slowly grows the bonus of every arm that is *not* pulled
- arms with Nₜ(a) = 0 count as maximally uncertain and are tried first
- c > 0 controls how bold the exploration is (c = 2 is the textbook setting)

Worked example, c = 1, t = 11: arm A has Q = 1.0 after 10 pulls, arm B has Q = 0.5 after 1 pull.
A scores 1.0 + √(2.40/10) ≈ 1.49, B scores 0.5 + √(2.40/1) ≈ 2.05. UCB pulls B: one sample is
not enough to rule it out.

```quiz
{"q": "Arm a has never been pulled while every other arm has been pulled 50 times. What does UCB do?",
 "options": ["Ignores it until ε fires", "Pulls it — its bonus is infinite", "Pulls it only if its Q is highest", "Pulls it with probability 1/k"],
 "answer": 1,
 "explain": "With N(a) = 0 the bound is treated as +∞, so every arm is tried once before any estimate is trusted."}
```

```exercise
03-ucb
```

> [!NOTE] Why UCB rarely leaves the bandit chapter
> UCB needs counts N(a) for every action, and the ln t schedule assumes a stationary problem.
> With large state spaces or function approximation those counts do not exist, so later
> chapters mostly use ε-greedy or stochastic policies, though "count-based exploration
> bonuses" are a live idea in deep RL.

## Gradient bandits: learning preferences instead of values

A different approach skips value estimates entirely. Keep a numerical **preference** Hₜ(a)
for each arm and act stochastically with the **softmax** (Gibbs/Boltzmann) distribution:

$$
πₜ(a) = exp(Hₜ(a)) / Σ_b exp(Hₜ(b))
$$

Only *differences* between preferences matter: adding 100 to every H leaves π unchanged. Play
with the temperature slider: high temperature flattens the distribution, low temperature makes
it nearly greedy.

```widget
{"type": "softmax", "prefs": [1.0, 2.0, 0.5, 0.0], "temperature": 1.0}
```

After taking Aₜ and receiving Rₜ, nudge the preferences with stochastic gradient ascent on
expected reward:

$$
Hₜ₊₁(Aₜ) = Hₜ(Aₜ) + α (Rₜ − R̄ₜ)(1 − πₜ(Aₜ))
Hₜ₊₁(a)  = Hₜ(a)  − α (Rₜ − R̄ₜ) πₜ(a)         for a ≠ Aₜ
$$

R̄ₜ is the average reward so far, the **baseline**. Better than average → make the chosen arm
more likely (and the others less); worse → the reverse. In code it is a few lines:

```python
pi = np.exp(H - H.max()); pi /= pi.sum()      # subtract max for numerical safety
a = rng.choice(k, p=pi)
r = bandit.pull(a)
baseline += (r - baseline) / t
one_hot = np.eye(k)[a]
H += alpha * (r - baseline) * (one_hot - pi)
```

> [!KEY] You have just seen your first policy gradient
> "Increase the log-probability of actions in proportion to how much better than a baseline
> they turned out" is precisely REINFORCE with a baseline, which leads to actor-critic and PPO.
> The baseline does not change the expected update, but without it the variance is much larger.

```quiz
{"q": "In a gradient bandit, the chosen arm paid exactly the baseline (Rₜ = R̄ₜ). What happens to the preferences?",
 "options": ["The chosen arm's preference rises", "Nothing changes", "All preferences reset to 0", "The chosen arm's preference falls"],
 "answer": 1,
 "explain": "Every update is multiplied by (Rₜ − R̄ₜ) = 0. Only better- or worse-than-average outcomes carry information."}
```

## Pitfalls

- **argmax without random tie-breaking** biases early behaviour toward low indices.
- **Off-by-one in 1/N**: increment N(a) *before* computing the step size, otherwise the first
  update uses a step of 1/0.
- **Optimism with sample averages** is wiped out after one pull: the first reward replaces Q₁.
- **ln 0**: start UCB's time counter at t = 1.
- **One run proves nothing**: compare algorithms over hundreds of seeded runs, as in the
  testbed exercise, before believing a curve.

## What carries forward

| Bandit idea | Where it returns |
|---|---|
| Incremental update toward a target | Monte Carlo, TD(0), SARSA, Q-learning, DQN loss |
| ε-greedy | Every tabular control method, DQN |
| Constant α for nonstationarity | Policy changes make value targets nonstationary in all of RL |
| Optimistic initial values | Tabular Q-learning, exploration bonuses |
| Softmax preferences + baseline | REINFORCE, actor-critic, PPO |

Bandits ignore one thing: actions that change **which situation you are in next**. The next
lesson adds states, and with them the full reinforcement learning problem: the Markov decision
process.
