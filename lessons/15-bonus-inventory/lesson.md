---
title: Bonus Capstone — Stock the Shelves
chapter: Bonus
summary: A real-world RL project end to end — retail inventory replenishment, from sales data and a classic planner to an agent judged on a year it has never seen.
minutes: 90
---

# Stock the shelves

Every evening a store manager answers the same question for thousands of products:
**how many do I order from the supplier?** Order too little and customers leave empty-handed.
Order too much and money sits on the shelf, then goes stale. The supplier needs two days, the
weekend is busy, marketing has a promotion next Thursday, and nobody knows tomorrow's demand.

This bonus capstone is a complete, realistic RL project on that problem. No game, no grid:
a simulator built from **sales data**, a **classic operations-research baseline** that you must
beat, an agent, and an honest **out-of-time evaluation** on a year of data it never trained on.
You can run it on the bundled two-year sales file — or on your own export.

## Why this is an RL problem

Replenishment has every ingredient from lesson 01:

- **Sequential decisions** — today's order changes tomorrow's stock, which changes what you
  should order tomorrow.
- **Delayed consequences** — with a two-day lead time, today's order cannot help today's
  customers. The reward for a good decision arrives days later, mixed with the effects of
  every other order.
- **Uncertainty** — demand is noisy, seasonal and promotional. You choose under risk, and the
  outcome of one day says little about whether the decision was good.
- **A clear scalar goal** — profit. Rarely does the reward hypothesis fit this naturally.

The simulator lives in the bundled `inventory` module. One `step` is one day:

1. an order of `env.order_sizes[a]` units is placed (a fixed fee if it is non-zero)
2. deliveries ordered `lead_time` days ago arrive; anything above the shelf capacity is refused
3. customers arrive: sales = min(stock, demand); the rest is **lost**
4. a few percent of the remaining stock **spoils** overnight
5. holding cost is charged on what is left

```python
from inventory import InventoryEnv
env = InventoryEnv()                      # one simulated year of a fresh bakery item
obs, info = env.reset(seed=0)             # obs: int state in [0, env.n_states)
obs, reward, terminated, truncated, info = env.step(4)    # order env.order_sizes[4] units
print(info["sales"], info["lost_sales"], info["spoiled"], info["costs"])
```

## The MDP

| | In the simulator | Why |
|---|---|---|
| **State** | inventory *position* (on hand **+ in transit**), weekday, *promo soon* flag | what you own, and what demand is coming |
| **Action** | order 0, 6, 12, …, 48 units | a small discrete set, like real case packs |
| **Reward** | the day's profit | revenue − cost of goods − waste − fees − holding − stockout penalty |
| **γ** | 0.9 – 0.97 | an order's effects fade within a week or two |

`obs="tabular"` packs the state into one integer (`env.decode_state(s)` unpacks it).
`obs="vector"` gives 14 features (stock levels, weekday one-hot, promo flags, recent demand
level) for function approximation.

```quiz
{"q": "Why does the state use inventory position (on hand + in transit) rather than just the stock on the shelf?",
 "options": ["It is easier to compute", "Two shelves with equal stock but different deliveries on the way need different orders — without the pipeline the state is not Markov", "Because holding cost is charged on in-transit units"],
 "answer": 1,
 "explain": "With a lead time, units already ordered will arrive regardless of today's action. Ignoring them makes the agent order twice for the same gap. Operations research reached the same conclusion decades ago: base-stock policies act on inventory position."}
```

## The bar to beat: the planner

Long before RL, operations research solved simplified versions of this problem exactly.

The **newsvendor** sells a perishable product for one day. If an unsold unit costs *cₒ* (overage)
and a missing unit costs *cᵤ* (lost margin + goodwill), the optimal order is a quantile of the
demand distribution:

$$
q* = F⁻¹( cᵤ / (cᵤ + cₒ) )
$$

With a margin of 4 + a goodwill penalty of 1 against an overage of 6, that ratio is 5/11 ≈ 0.45:
stock slightly *below* median demand.

For many periods with a lead time and a fixed fee per order, the classic answer is an
**(s, S) policy**: *whenever the inventory position drops to s or below, order up to S*. The gap
S − s batches orders so you do not pay the fee every day. Two numbers, tuned on history by grid
search, and it is remarkably hard to beat.

> [!KEY] Beat a tuned baseline, or you have nothing
> An RL agent that is "profitable" means nothing on its own. The question a business asks is
> *how much better than what we do today?* Always tune the classic policy **as well as you can**,
> on the same data, and report the agent relative to it.

```exercise
01-baseline-planner
```

## Reward design: you get what you pay for

The reward is "just profit", yet there are several ways to get it badly wrong.

> [!WARN] The agent that never stocks anything
> Charge the purchase price when goods are ordered, end the episode after 365 days, and forget
> to value the leftover stock: near the end of every episode ordering looks like pure loss, and
> with short training horizons the agent learns that *empty shelves are cheap*. The simulator
> credits leftover stock at cost on the final day (`salvage=True`) to remove that cliff.

**Cash vs accrual.** Paying for goods the day you order them (`accounting="cash"`) and booking a
unit's cost when it is sold or spoils (`accounting="accrual"`, the default) give the same yearly
profit, up to the value of the starting stock. They are very different learning problems: with
cash accounting an order is a big negative reward followed by small positive ones up to a week
later; with accrual, each day's reward reflects mostly that day's decisions. Same objective,
much easier credit assignment.

**Missing costs get exploited.** Set the stockout penalty and spoilage to zero and the agent
happily runs the shelf empty every Sunday. Set holding cost too high and it learns the same thing
for a different reason. Every number in the reward is a business judgement — agree on them with
the people who own the P&L, not by tuning until the curve looks nice.

**Discounting vs quarterly targets.** γ = 0.95 means a profit 14 days out counts half. That is a
statement about *credit assignment*, not about what the business values. If management is paid on
quarterly profit, the true objective is an undiscounted 90-day sum; γ < 1 is a tool to make learning
stable, and you should still **evaluate** on the undiscounted business metric.

```widget
{"type": "discount", "gamma": 0.95, "rewards": [-5, 0, 24, 24, 24, -3, -3, -3, -3, -3, -3, -3, -3, -3]}
```

An order: a fee today, margins when it sells, then holding and spoilage costs if you over-ordered.
Lower γ and the late costs nearly vanish — the agent becomes blind to over-stocking.

```quiz
{"q": "Your agent's test profit is great, but you notice it lets the shelf run empty most Sundays. What should you check first?",
 "options": ["Increase the learning rate", "Whether the stockout penalty and lost margin in the reward reflect what an empty shelf really costs the business", "Use a deeper neural network"],
 "answer": 1,
 "explain": "The agent maximised exactly what you rewarded. If an empty shelf costs only the lost margin of a few units, being out of stock on a quiet day is optimal. If customers who find an empty shelf go elsewhere for good, the reward must say so."}
```

## Learning on top of the planner

Plain tabular Q-learning on the full state can learn this problem — eventually. In a budget of a
few hundred simulated years it does **not** reliably match a well-tuned (s, S). Rewards are noisy
(demand varies ±40 % day to day), most of the state space is rarely visited, and the greedy policy
inherits the noise in every Q-value.

The practical answer is **residual RL**: keep the planner, and let the agent learn *adjustments*.
Here the agent sees only a 14-value context (weekday × promotion coming) and chooses an offset that
shifts both s and S for today. Offset 0 *is* the planner, so an untrained agent is exactly as good
as today's process, and every gain is something the data supports: stock up on Thursday for the
Saturday rush, lean out on Sunday, prepare for a promotion.

> [!TIP] Why industry loves residual policies
> They are safe to deploy (the fallback is the existing process), easy to explain to a planner
> ("on Thursdays the agent orders up to 20 more"), and they need far less data than learning
> from scratch. The same idea appears as residual control in robotics and as "RL on top of a
> heuristic" in pricing and logistics.

```exercise
02-residual-q
```

## Data: simulated, then real

The bundled file `data/sample_sales.csv` holds two years of daily sales for one product, generated
with weekly seasonality, a yearly cycle, a slow upward trend, a pre-Christmas rush and planned
promotions (columns `date,demand,dow,promo`).

`load_demand_csv(path)` needs only `date` and a quantity column; `dow` is recomputed and `promo`
is optional. `InventoryEnv(demand_series=..., promo_series=..., dates=...)` then **replays** that
history day by day, and `jitter_weeks=k` adds a seasonal bootstrap (each day's demand drawn from the
same weekday up to *k* weeks away) so you can train on one historical year without memorising it.

**Bring your own data.** Any daily sales export works. For example, the Kaggle *Store Sales — Time
Series Forecasting* data (Favorita) has `date, store_nbr, family, sales, onpromotion`: keep one store
and one product family, then

```python
import csv
rows = [r for r in csv.DictReader(open("train.csv")) if r["store_nbr"] == "1" and r["family"] == "BREAD/BAKERY"]
with open("my_sales.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["date", "demand", "promo"])
    for r in rows:
        w.writerow([r["date"], r["sales"], int(float(r["onpromotion"]) > 0)])
```

Order sizes, capacity and the search grids scale with the product's mean demand, so the same code
runs on a product that sells 3 a day or 300.

> [!WARN] Never let the future leak into training
> Split time series **by time**: train on the past, test on the future. A random split lets the
> model "see" next week's promotion effects while learning about this week. The same goes for
> every number derived from data — the mean demand that scales order sizes, the tuned (s, S), the
> weekday profile — compute them on the training period only.

The honest caveats of offline evaluation:

- **Sales are not demand.** On days you ran out, the history records what you *sold*, not what
  customers *wanted*. Replaying sales as demand underestimates it exactly when it matters. Real
  projects "uncensor" demand from stockout flags or comparable stores.
- **Replay assumes your actions do not change demand.** Here they do not; in reality, an empty
  shelf trains customers to shop elsewhere and a full display sells more.
- **Sim-to-real gap.** Real lead times vary, suppliers short-ship, staff mis-count. An agent that
  exploits a quirk of the simulator will fail in the store. Stress-test it on perturbed
  simulators (longer lead times, noisier demand) before trusting it.
- **One test year is one sample.** A 3 % uplift on a single year may be luck. Look at several
  products, several stores, and at *where* the gain comes from (fill rate? fewer orders? less waste?).

```quiz
{"q": "You scale the test environment with the mean demand of the TEST year instead of the training year. Why is that a problem?",
 "options": ["It isn't — scaling is not learning", "It leaks information about the future (e.g. the year-on-year growth) into the policy, so the test result is optimistic", "It makes the simulation slower"],
 "answer": 1,
 "explain": "Order sizes and capacity would silently adapt to next year's demand level — something you could never know in advance. Any statistic computed on test data is a leak."}
```

```exercise
03-real-data
```

## Beyond the shelf

The same recipe — a simulator calibrated on data, a strong classic baseline, an agent that improves
on it, and an out-of-time evaluation — carries over to many operational problems:

| Problem | State | Action | Reward | Classic baseline |
|---|---|---|---|---|
| Dynamic pricing | stock left, days to season end, demand signals | price level | revenue (− markdown waste) | fixed markdown calendar |
| Ad budget pacing | spend so far, time of day, conversion rates | bid multiplier | conversions − cost | even pacing |
| HVAC / energy | temperatures, occupancy, electricity price | set-points, battery charge | − (energy cost + comfort penalty) | thermostat schedule |
| Fleet rebalancing | vehicles and requests per zone | move vehicles between zones | served trips − driving cost | rebalance to historical demand |

In all of them the hard parts are the ones you met here: a reward that matches the business,
a state that is Markov enough, a baseline that is genuinely tuned, and an evaluation that never
peeks into the future.

## Debrief

| Step | What you did | Lesson |
|---|---|---|
| 1 | weekday profile, promo lift, tuned (s, S) | understand the data and set the bar |
| 2 | residual Q-learning over (weekday, promo) | learn adjustments, not everything from scratch |
| 3 | train on year 1, evaluate on year 2 | judge on the future, report business metrics |

Ideas to take further:

- Re-run the study with `lead_time=4` or `spoil_rate=0.1` and see which policy degrades more.
- Use `obs="vector"` with linear function approximation or a small DQN, residual or not.
- Run exercise 3 on **ten products** from a public dataset and report the distribution of uplifts.
- Replace ε-greedy exploration with **offline** learning from the planner's own historical
  decisions — the setting most real deployments start from.
