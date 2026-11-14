---
title: Markov Decision Processes
chapter: Foundations
summary: The formal model behind RL — states, dynamics, policies, value functions and the Bellman equations.
minutes: 45
---

# Markov decision processes

In the bandit lesson, actions had no consequences beyond their immediate reward. Real problems
are not like that: a chess move changes the board, a robot's step changes where it stands, and
a good move now may only pay off twenty moves later. The **Markov decision process (MDP)** is
the mathematical model of that situation, and almost every RL algorithm is defined in its
terms.

This lesson is the most mathematical in the course. Take it slowly: everything you build in
the next several lessons (dynamic programming, Monte Carlo, TD, Q-learning, DQN) is an
approximate way of solving the two equations at the end of it.

## The Markov property

A state is **Markov** if it captures everything about the past that matters for the future:

$$
Pr{ Sₜ₊₁, Rₜ₊₁ | Sₜ, Aₜ }  =  Pr{ Sₜ₊₁, Rₜ₊₁ | S₀, A₀, R₁, …, Sₜ, Aₜ }
$$

Once you know the current state and action, the history adds nothing. A chess position is
Markov (castling rights and the move counter included). A single frame of a video game usually
is **not**: one frame tells you where the ball is, not which way it is moving, which is why
DQN stacks the last four frames.

> [!NOTE] Markov is a property of your state representation
> The same world can be Markov or not depending on what you put in the state. Positions alone
> are not Markov for a bouncing ball; positions and velocities are. Much of practical RL is
> choosing a state that is (close enough to) Markov.

```quiz
{"q": "A robot's state is just its (x, y) position, but it moves with momentum. Is that state Markov?",
 "options": ["Yes, position is all that matters", "No, the next position also depends on velocity, which the state omits", "Only if rewards are deterministic", "Markov only applies to discrete states"],
 "answer": 1,
 "explain": "Two visits to the same (x, y) with different velocities lead to different futures, so the past (which reveals velocity) still matters. Add velocity to the state to restore the Markov property."}
```

## The MDP

A finite MDP is the tuple **(S, A, p, γ)**:

- **S**, a finite set of states, and **A**, a finite set of actions
- **p**, the **dynamics**: the probability of each next state and reward
- **γ ∈ [0, 1]**, the discount factor from the previous lesson

$$
p(s', r | s, a) = Pr{ Sₜ₊₁ = s', Rₜ₊₁ = r | Sₜ = s, Aₜ = a }
$$

Because it is a probability distribution, Σₛ' Σᵣ p(s', r | s, a) = 1 for every (s, a). From the
four-argument p you can derive everything else you might want:

$$
p(s' | s, a) = Σᵣ p(s', r | s, a)          r(s, a) = Σₛ' Σᵣ r · p(s', r | s, a)
$$

In this course's `GridWorld`, p is stored literally. `env.P[s][a]` is the list of outcomes
`(prob, next_state, reward, terminated)`, one entry per possible (s', r):

```python
from rlenvs import GridWorld
env = GridWorld(["...G",
                 ".#.X",
                 "S..."], slip=0.2, step_reward=-0.04)
env.P[env.state_of(1, 2)][0]     # try to go UP from the cell below the goal row
# [(0.8, 2, -0.04, False),  (0.1, 7, -1.0, True),  (0.1, 6, -0.04, False)]
#  went up                  slipped right: the pit   slipped left: bumped the wall, stayed
```

> [!KEY] Everything about the world is in p
> An MDP is fully specified by its dynamics. **Planning** methods (next lesson) are handed p and
> compute good behaviour from it. **Learning** methods (the rest of the course) never see p:
> they only sample from it by acting.

Terminal states such as G and X end the episode. By convention their value is 0: no reward ever
follows them. The `terminated` flag tells you when a transition enters one.

## Policies and value functions

A **policy** π(a | s) is the probability of taking action *a* in state *s*. A deterministic
policy puts all its probability on one action per state. The two value functions from lesson 1,
now in full:

$$
v_π(s) = E_π[ Gₜ | Sₜ = s ]            q_π(s, a) = E_π[ Gₜ | Sₜ = s, Aₜ = a ]
$$

v_π answers "how good is it to be here, if I then follow π?". q_π answers "how good is it to
take *a* here and follow π afterwards?". They are tied together by two one-line identities you
should know by heart:

$$
v_π(s) = Σₐ π(a|s) · q_π(s, a)
q_π(s, a) = Σₛ',ᵣ p(s', r | s, a) · [ r + γ · v_π(s') ]
$$

The first averages over **your** choice (the policy). The second averages over the
**world's** response (the dynamics). The second one is called a **one-step lookahead**: it
turns state values into action values, and acting greedily on those action values turns values
into a policy.

Worked example: corridor `S . G`, step reward −1, entering G pays +10, γ = 0.5, and suppose
V = [2, 5, 0]. From S, RIGHT reaches the middle: q(S, RIGHT) = −1 + 0.5·5 = 1.5. The other three
actions bump the wall and stay: q = −1 + 0.5·2 = 0. From the middle, RIGHT enters the terminal
goal: q = 10, with no γ·V term because nothing follows termination.

```quiz
{"q": "In the corridor above, what is q(middle, LEFT)?",
 "options": ["−1 + 0.5·2 = 0", "−1 + 0.5·5 = 1.5", "10", "−1"],
 "answer": 0,
 "explain": "LEFT from the middle leads to S with reward −1, so q = −1 + γ·V(S) = −1 + 0.5·2 = 0."}
```

Before you write planning code on your own, type along once. The walkthrough writes down a
three-state robot MDP in the same `P[s][a]` format as GridWorld, computes q-values with loops
and with NumPy, and solves a policy's Bellman equation exactly, with an explanation for each line.

```exercise
00-walkthrough-tiny-mdp
```

Now the same tools, no ghost text:

```exercise
01-one-step-lookahead
```

## The Bellman expectation equation

Substitute one identity into the other and you get a recursive equation for v_π alone:

$$
v_π(s) = Σₐ π(a|s) Σₛ',ᵣ p(s', r | s, a) [ r + γ v_π(s') ]
$$

It says: *the value of a state is the expected immediate reward plus the discounted value of
wherever you land*, which is just Gₜ = Rₜ₊₁ + γ Gₜ₊₁ with expectations taken.

> [!NOTE] Backup diagrams, in words
> Picture the v_π equation as a two-level tree. At the root is an open circle, state *s*. From it
> hang one branch per action, each ending in a solid dot (a state–action pair), weighted by
> π(a|s). From each dot hang one branch per outcome (s', r), weighted by p, each ending in an open
> circle s'. Computing v_π(s) "backs up" the values from the leaves to the root: average over the
> outcome branches, then average over the action branches. The q_π diagram is the same tree
> started from a solid dot: outcomes first, then the next action.

### Solving it as a linear system

For a fixed π, there is one Bellman equation per state, and each one is **linear** in the
unknown values. Collect them into vectors. Let r_π(s) be the expected immediate reward under
π and P_π(s, s') the probability of moving from s to a non-terminal s':

$$
v_π = r_π + γ P_π v_π      ⇔      (I − γ P_π) v_π = r_π
$$

That is an ordinary `n × n` linear system, `np.linalg.solve(I − γ P_π, r_π)`. With γ < 1 the
matrix is always invertible; with γ = 1 it is invertible as long as π reaches a terminal state
with probability 1 from everywhere.

Worked example: corridor `S . G`, reward only for entering G (+1), uniform random policy, γ = 0.9.
From S, one action in four moves right; the rest stay. From the middle, RIGHT ends the
episode with +1, LEFT goes back, UP/DOWN stay:

$$
v(S) = 0.9 · ( ¾ v(S) + ¼ v(M) )
v(M) = ¼ · 1 + 0.9 · ( ¼ v(S) + ½ v(M) )
$$

Solving gives v(S) ≈ 0.439 and v(M) ≈ 0.634. Your code will produce exactly these numbers.

```quiz
{"q": "Why does solving v_π with a linear system NOT scale to large problems?",
 "options": ["Bellman equations are nonlinear", "It needs the full model p, and solving costs about n³ for n states", "It only works for deterministic policies", "It only works for γ = 1"],
 "answer": 1,
 "explain": "Direct solution costs O(n³) and needs P_π explicitly. Backgammon has ~10²⁰ states. Iterative methods (next lesson) and then sampling methods (later) get around both problems."}
```

```exercise
02-solve-bellman
```

You can watch values form under a fixed policy in the widget below. It applies the Bellman
expectation equation repeatedly to the uniform random policy, the iterative alternative to the
linear solve that the next lesson develops.

```widget
{"type": "gridworld", "map": ["...G", ".#.X", "S..."], "gamma": 0.9, "step": -0.04, "slip": 0.2, "mode": "policy-evaluation"}
```

## Optimal value functions

Value functions impose an ordering on policies: π ≥ π' if v_π(s) ≥ v_π'(s) **in every
state**. A remarkable fact about finite MDPs is that there is always at least one policy that is
better than or equal to every other in every state at once. That is an **optimal policy** π*,
and all optimal policies share the same value functions:

$$
v*(s) = maxπ v_π(s)            q*(s, a) = maxπ q_π(s, a)
$$

### The Bellman optimality equations

An optimal policy picks the best action, not an average one, so the "average over the policy"
step becomes a max:

$$
v*(s) = maxₐ Σₛ',ᵣ p(s', r | s, a) [ r + γ v*(s') ]
q*(s, a) = Σₛ',ᵣ p(s', r | s, a) [ r + γ maxₐ' q*(s', a') ]
$$

In the backup-diagram picture, the branch over actions at the root is no longer averaged;
it is maxed (drawn as an arc across the action branches).

> [!KEY] Optimal values make optimal behaviour easy
> If you know v*, act greedily with a one-step lookahead: π*(s) = argmaxₐ Σ p [r + γ v*(s')].
> If you know q*, you do not even need the model: π*(s) = argmaxₐ q*(s, a). That is why
> Q-learning learns q* rather than v*.

Greedy with respect to v* is optimal even though it only looks one step ahead, because v*
already accounts for every step beyond it. Long-term consequences have been folded into a
local quantity.

```quiz
{"q": "What makes the Bellman optimality equation harder to solve than the expectation equation?",
 "options": ["It has more unknowns", "The max makes it nonlinear, so one linear solve no longer works", "It requires γ < 1", "It is only defined for deterministic MDPs"],
 "answer": 1,
 "explain": "Same number of unknowns, but max is nonlinear. We need iterative schemes such as value iteration and policy iteration, which is the topic of the next lesson."}
```

### Finding π* the hard way

For a small MDP you can find π* without any clever algorithm: there are |A|^|S| deterministic
policies, and you can evaluate each one exactly. A 3 × 3 grid with 6 decision states has
4⁶ = 4096. The 4 × 3 grid above already has 4⁹ = 262 144, and a 10 × 10 grid has about
4¹⁰⁰ ≈ 10⁶⁰. Do it once anyway: seeing the winner satisfy the
optimality equation and dominate every other policy in **every** state makes the theory
concrete.

```exercise
03-optimal-policy
```

## Pitfalls

- **Bootstrapping through termination.** If a transition ends the episode, its target is just
  r, never r + γ·V(s'). Forgetting this is the most common bug in every algorithm from here to PPO.
- **Mixing up p(s'|s,a) and p(s',r|s,a).** In GridWorld one outcome tuple already pairs a next
  state with its reward; just weight each tuple by its probability.
- **Evaluating with the wrong policy.** v_π depends on π. The values of the random policy
  are not "the values of the grid".
- **γ = 1 with a policy that never terminates** makes I − γP_π singular: the return is
  infinite (or undefined). Use γ < 1 or guarantee termination.

## What carries forward

| Concept | Where it returns |
|---|---|
| The model p, stored in env.P | Dynamic programming (next lesson), Dyna-Q's learned model |
| One-step lookahead | Policy improvement, value iteration, every greedy policy |
| Bellman expectation equation | Policy evaluation, TD(0), SARSA, critics in actor-critic |
| Bellman optimality equation | Value iteration, Q-learning, DQN's target |
| Markov state design | Frame stacking, recurrent policies, feature engineering |

The linear solve worked because the policy was fixed. Next you will solve for the *best*
policy, iteratively, with **dynamic programming**.
