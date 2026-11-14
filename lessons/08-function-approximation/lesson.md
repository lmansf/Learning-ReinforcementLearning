---
title: Function Approximation
chapter: Deep RL
summary: Replace the table with a parameterised function — features, linear values, semi-gradient TD, tile coding, and the deadly triad.
minutes: 45
---

# Function approximation

Every method so far kept **one number per state** (or per state–action pair) in a table. That
works for a 4×12 cliff. It does not work for:

- **CartPole** — four *continuous* numbers. There is no "state 17"; you will never see the
  exact same state twice.
- **Backgammon** — about 10²⁰ states. **Go** — about 10¹⁷⁰. A table would not fit in the universe.
- **Atari from pixels** — 256^(210·160·3) possible screens.

The real problem is not memory, it is **generalisation**. A table learns nothing about state
*B* from a visit to state *A*, even if they are nearly identical. To learn in large spaces the
agent must assume that *similar states have similar values* — and that is exactly what a
function approximator does.

$$
v̂(s, w) ≈ v_π(s)          q̂(s, a, w) ≈ q_π(s, a)          w ∈ ℝᵈ,  d ≪ |S|
$$

Changing one weight now changes the estimate of *many* states at once. That is the power
(generalisation) and the danger (an update for one state can wreck another).

## Prediction as (almost) supervised learning

If an oracle gave us the true $v_π(s)$, we would just do regression. We want weights that
minimise the **mean squared value error**, weighted by how often each state is visited (μ):

$$
VE(w) = Σₛ μ(s) [ v_π(s) − v̂(s, w) ]²
$$

Stochastic gradient descent on one sample $Sₜ$ gives

$$
w ← w + α [ v_π(Sₜ) − v̂(Sₜ, w) ] ∇v̂(Sₜ, w)
$$

We do not know $v_π(Sₜ)$, so we plug in a **target** $Uₜ$ that estimates it — the same targets
you already know from the tabular chapters:

| method | target Uₜ | unbiased? |
|---|---|---|
| gradient Monte Carlo | Gₜ (the full return) | yes — true SGD, converges to a local minimum of VE |
| semi-gradient TD(0) | Rₜ₊₁ + γ v̂(Sₜ₊₁, w) | no — the target uses the current w |
| semi-gradient n-step TD | Rₜ₊₁ + … + γⁿ⁻¹Rₜ₊ₙ + γⁿ v̂(Sₜ₊ₙ, w) | no |

> [!NOTE] Why "semi"-gradient?
> The TD target contains w too. A *full* gradient would differentiate through it. We
> deliberately don't: we treat the target as a fixed label and only follow the gradient of the
> prediction. It is only "half" of the true gradient — hence *semi*-gradient. It learns faster
> and works well in practice, but it is not SGD on any objective, and that will come back to bite us.

## Linear value functions

The simplest, best-understood approximator is **linear** in a feature vector x(s) ∈ ℝᵈ:

$$
v̂(s, w) = wᵀ x(s) = Σᵢ wᵢ xᵢ(s)          ∇v̂(s, w) = x(s)
$$

The gradient is just the features, so every update has the same shape:

$$
w ← w + α · (target − v̂(Sₜ, w)) · x(Sₜ)
$$

A tabular method is the special case where x(s) is a one-hot vector with one entry per state.
*All* the intelligence lives in the choice of features.

### Worked example

Two features, w = [0, 1], α = 0.1, γ = 0.9. We are in S with x(S) = [1, 0], receive r = 0.5,
and land in S′ with x(S′) = [0, 1].

- prediction v̂(S) = 0·1 + 1·0 = 0
- target r + γ v̂(S′) = 0.5 + 0.9 · 1 = 1.4
- TD error δ = 1.4 − 0 = 1.4
- update w ← [0, 1] + 0.1 · 1.4 · [1, 0] = **[0.14, 1]**

Only the weight of the feature that was *active* in S moved. With overlapping features, several
weights move — and every state that shares them changes value too.

```quiz
{"q": "With a linear v̂(s, w) = wᵀx(s), what is ∇v̂(s, w)?",
 "options": ["w", "x(s)", "wᵀx(s) · x(s)", "It depends on the TD error"],
 "answer": 1,
 "explain": "The derivative of Σᵢ wᵢxᵢ(s) with respect to wᵢ is xᵢ(s). That is why linear methods are so cheap: the gradient is the feature vector itself."}
```

### Gradient MC vs semi-gradient TD

On the 1000-state random walk (jump up to 100 states left or right; −1 off the left edge, +1
off the right) with **state aggregation** — 10 groups of 100 states, one weight per group —
both methods learn a staircase approximation of a nearly straight line.

- **Gradient MC** converges to the *best* staircase (minimum VE), but slowly: its targets are
  noisy ±1 outcomes, so α must be small.
- **Semi-gradient TD(0)** converges to the **TD fixed point**, which is close to — but not
  exactly — the minimum. For linear TD there is a guarantee: VE(w_TD) ≤ 1/(1−γ) · min VE.
  In exchange it learns from every step with far less variance.

Before you write code on your own, type along once. The walkthrough fits a toy function with
bucketed features and shows every idiom the next three exercises need — feature vectors,
`w @ x`, the in-place gradient step, `np.floor`/`np.clip` bucketing and sparse indices — with an
explanation for each line.

```exercise
00-walkthrough-linear-features
```

Now the same tools, no ghost text:

```exercise
01-linear-value-updates
```

## Features: where the knowledge goes

Some classic choices for a continuous state s = (s₁, …, s_k):

- **State aggregation** — chop the space into cells, one-hot per cell. Simple; values jump at cell boundaries.
- **Polynomials** — 1, s₁, s₂, s₁s₂, s₁², … Global: every feature affects every state. Unstable in high degree.
- **Radial basis functions** — xᵢ(s) = exp(−‖s − cᵢ‖² / 2σ²). Smooth, local, but you must place the centres.
- **Tile coding** — several overlapping, *shifted* grids. Binary, sparse, local and blazingly fast.
- **Learned features** — a neural network learns x(s) itself. That is the next lesson.

> [!TIP] Normalise your inputs
> CartPole's velocity is ±2, MountainCar's is ±0.07. Unscaled features with very different ranges
> make a single step size α far too big along one direction and far too small along another.
> Rescale every input to roughly [−1, 1] or [0, 1] before building features.

## Tile coding

Lay a grid ("tiling") over the state space: a state activates exactly **one** tile. Now lay
several more grids, each shifted by a fraction of a tile. A state activates one tile **per
tiling** — with 8 tilings, 8 of the (say) 648 features are 1 and the rest are 0.

```
                       x
tiling 0  |....|....|....|....|       x is in cell 2
tiling 1    |....|....|....|....|     (shifted ⅖ tile)  cell 2
tiling 2      |....|....|....|....|   (shifted ⅘ tile)  cell 1
```

Two states **share** a tile in every tiling where they fall in the same cell. Nearby states
share most of their tiles (generalisation); distant ones share none (discrimination). The
resolution is finer than any single grid — 8 tilings of width w distinguish positions to w/8 —
while generalisation spreads about one tile width.

Two practical rules:

1. **Step size per tiling.** Exactly n features are active, so wᵀx is a sum of n weights. Use
   α = α₀ / n so that one update moves the estimate by about α₀ of the error, whatever n is.
2. **Asymmetric offsets.** Shifting tiling *i* by i/n along *every* dimension lines all tilings
   up on the diagonal. Shifting by (1, 3, 5, …) · i/n per dimension spreads them out.

```quiz
{"q": "A tile coder has 8 tilings. You want each update to correct about half of the TD error. Which per-weight step size?",
 "options": ["α = 0.5", "α = 0.5 / 8", "α = 0.5 · 8", "α = 1 / 8²"],
 "answer": 1,
 "explain": "Eight weights are active and each moves by α·δ, so the estimate moves by 8·α·δ. Setting α = 0.5/8 makes it move by 0.5·δ."}
```

```exercise
02-tile-coder
```

## Control: semi-gradient SARSA

For control we approximate **action values** q̂(s, a, w) and act ε-greedily, exactly as in the
tabular case. The SARSA update becomes

$$
w ← w + α [ Rₜ₊₁ + γ q̂(Sₜ₊₁, Aₜ₊₁, w) − q̂(Sₜ, Aₜ, w) ] ∇q̂(Sₜ, Aₜ, w)
$$

With few discrete actions, the easiest design is **one weight vector per action**: q̂(s, a) =
wₐᵀ x(s). Only the row of the action you took gets updated.

### Mountain Car

An under-powered car sits in a valley. Full throttle is not enough to climb the right hill; it
has to back up the left slope first to build momentum. Reward is −1 per step until the flag,
so the return is minus the number of steps. The state is (position, velocity) — continuous.

Two details make this work with ε = 0:

- **Optimistic initialisation.** All weights start at 0, but every real return is negative.
  Untried actions look better than tried ones, so the greedy agent explores systematically.
- **Truncation is not termination.** Our episodes are cut off after `max_steps`. That cut is
  about *our* patience, not the car: the state still has a future. On truncation, keep
  bootstrapping from q̂(S′, A′). Only a real `terminated` gets target = r.

> [!WARN] The truncation bug
> Treating a time-limit cut as a terminal state teaches the agent that states near the time
> limit are worth 0 — a lie it will happily believe. In Gymnasium-style APIs, bootstrap
> unless `terminated` is True, regardless of `truncated`.

```quiz
{"q": "In MountainCar, why does a purely greedy (ε = 0) agent with zero-initialised weights still explore?",
 "options": ["MountainCar is deterministic, so exploration is not needed", "Every true value is negative, so actions it has not tried yet look better (optimism)", "Tile coding adds random noise", "SARSA always explores"],
 "answer": 1,
 "explain": "Each visit lowers the estimate of the action taken (rewards are −1). Untouched actions keep their optimistic 0 and get picked next. This drives systematic exploration without any randomness."}
```

```exercise
03-mountain-car-sarsa
```

After training, the demo draws −max q̂ over the state space: a spiral-shaped "cost-to-go"
landscape. The low-cost ridge follows the swinging trajectories the car learned.

## The deadly triad

Function approximation does not always converge. Sutton & Barto identify three ingredients
that together can make values **diverge to infinity**, even on tiny problems with linear features:

1. **Function approximation** — updates generalise to other states.
2. **Bootstrapping** — targets built from the current estimates (TD, DP).
3. **Off-policy learning** — training on a distribution of states different from the policy being evaluated (e.g. Q-learning's greedy target with ε-greedy data).

Any two are fine. All three together are dangerous. Baird's counterexample (7 states, linear
features, all rewards zero!) sends the weights to infinity under semi-gradient off-policy TD.

> [!WARN] Deep Q-learning lives inside the deadly triad
> DQN (next lesson) uses a neural network (1), bootstraps (2) and learns off-policy from a replay
> buffer (3). Its tricks — target networks, replay, Huber loss — exist precisely to tame this
> instability. When deep RL blows up, the triad is the first suspect.

> [!KEY] Generalisation is the point — and the risk
> A function approximator lets an update in one state inform similar states. With linear
> features, on-policy semi-gradient TD is stable and converges near the best approximation.
> Add off-policy bootstrapping and that guarantee disappears.

## Debugging function approximation

- **Test on a problem with known values first** (like the random walk): compare v̂ with the truth before trusting a control result.
- **Watch the weight norm.** ‖w‖ growing without bound means divergence: lower α, check the off-policy setup.
- **Scale α by the number of active features** (tile coding) or by feature magnitude (‖x‖²).
- **Bootstrap on truncation**, never on termination.
- **Seed everything** and run more than one seed — a single lucky run proves nothing.
- **Plot something interpretable**: the value surface, the steps per episode, a greedy rollout.

```quiz
{"q": "Which combination makes up the deadly triad?",
 "options": ["Exploration, exploitation and discounting", "Function approximation, bootstrapping and off-policy training", "Large α, small γ and ε-greedy", "Tile coding, Monte Carlo and SARSA"],
 "answer": 1,
 "explain": "Any two are safe; all three together can diverge. Monte Carlo (no bootstrapping) or on-policy linear TD avoid it."}
```
