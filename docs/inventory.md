---
title: inventory
---

## InventoryEnv
sig: `InventoryEnv(horizon=365, lead_time=2, order_step=None, n_actions=9, order_sizes=None, capacity=None, price=10.0, unit_cost=6.0, holding_cost=0.10, order_cost=5.0, stockout_penalty=1.0, spoil_rate=0.05, salvage=True, obs="tabular", base_demand=12.0, demand_series=None, promo_series=None, dates=None, start=0, random_start=False, jitter_weeks=0, init_stock=None, ref_demand=None, accounting="accrual", demand_kw=None, dispersion=20.0)`
match: InventoryEnv(
tags: inventory, replenishment, retail, bonus

One product, one store, one decision per day: how many units to order. Orders arrive after `lead_time` days, demand is seasonal and noisy, unmet demand is lost, stock spoils and costs money to hold. Episodes never terminate; they are **truncated** after `horizon` days.

```python
from inventory import InventoryEnv
env = InventoryEnv(horizon=365)
obs, info = env.reset(seed=0)            # obs == 84, info["on_hand"] == 36
obs, reward, terminated, truncated, info = env.step(3)   # order env.order_sizes[3] == 18 units
```

> [!WARN] Defaults scale with demand
> `order_step` (≈ half the mean daily demand), `capacity` (10 × mean) and `init_stock` (3 × mean) are derived from the demand level. Change `base_demand` or the series and the action meanings change too.

## InventoryEnv.step (one day)
sig: `obs, profit, False, truncated, info = env.step(a)`
match: lost_sales, ["sales"], ["costs"], ["spoiled"]
tags: dynamics, reward, profit, info

Each day: (1) place the order (fixed `order_cost` fee if > 0); (2) deliveries due today arrive, anything above `capacity` is refused; (3) sales = min(stock, demand), the rest is lost (stockout penalty); (4) a fraction `spoil_rate` of the leftover spoils; (5) holding cost on what's left. The reward is the day's **profit**.

```python
from inventory import InventoryEnv
env = InventoryEnv(); env.reset(seed=0)
_, r, _, _, info = env.step(3)
sorted(info)[:6]       # ['arrived', 'costs', 'date', 'day', 'demand', 'dow']
info["sales"], info["lost_sales"], info["spoiled"], round(r, 2)   # (2, 0, 2, -12.2)
```

> [!WARN] accrual vs cash
> With the default `accounting="accrual"` a unit's cost is booked when it is sold or wasted, so a good order pays off within days. `"cash"` charges purchases immediately — same yearly total, but much harder credit assignment.

## order_sizes / nearest_action
sig: `env.order_sizes` → int array (the action set) · `env.order_step` · `env.nearest_action(units)` → action index
match: .order_sizes, .order_step, .nearest_action(
tags: actions, order quantity, discretisation

Action `a` orders `env.order_sizes[a]` units — by default `order_step * [0, 1, …, 8]`. `nearest_action(units)` maps a desired quantity to the closest available action (0 for ≤ 0).

```python
from inventory import InventoryEnv
env = InventoryEnv()
env.order_step, env.order_sizes        # (6, array([ 0,  6, 12, 18, 24, 30, 36, 42, 48]))
env.nearest_action(10)                 # 2  (12 units)
env.nearest_action(100)                # 8  (largest order)
```

> [!WARN] Action ≠ units
> `env.step(12)` is an error (only 9 actions). Convert units with `nearest_action`.

## position / on_hand / in_transit
sig: `env.on_hand` · `env.in_transit` · `env.position` (= on hand + in transit) · `env.pipeline`
match: .position, .on_hand, .in_transit, .pipeline, ._transit
tags: stock, inventory position, lead time

`on_hand` is what's on the shelf; `pipeline` lists orders still on their way (index 0 arrives next); `position` is everything you own — the quantity classic policies reason about.

```python
from inventory import InventoryEnv
env = InventoryEnv(); env.reset(seed=0)
env.step(3)
env.on_hand, env.pipeline, env.position     # (32, [0, 18], 50)
```

> [!WARN] Order on position, not shelf
> Ordering based on `on_hand` alone forgets what is already in transit and leads to over-ordering whenever `lead_time > 0`.

## decode_state / state_index
sig: `env.n_states` · `env.state_index()` → int · `env.decode_state(s)` → `(position_bucket, dow, promo_soon)`
match: .decode_state(, .state_index(
tags: tabular state, encoding, Q-table

With `obs="tabular"`, the observation packs (inventory-position bucket in units of `order_step`, weekday 0–6, promo-soon flag) into one int: `(bucket·7 + dow)·2 + promo_soon`.

```python
from inventory import InventoryEnv
import numpy as np
env = InventoryEnv()
obs, _ = env.reset(seed=0)
env.decode_state(obs)                         # (6, 0, 0) — 36 units, Monday, no promo soon
Q = np.zeros((env.n_states, env.n_actions))   # (294, 9)
```

## InventoryEnv.features
sig: `env.features()` → float32 (14,) · `obs="vector"` returns it · `env.obs_dim == 14`
match: .features(, .obs_dim
tags: vector observation, features, function approximation

The vector observation: `[on_hand, in_transit, position]` / capacity, one-hot weekday (7), promo today, promo soon, trailing-7-day demand ratio, fraction of the horizon elapsed.

```python
from inventory import InventoryEnv
env = InventoryEnv(obs="vector")
x, _ = env.reset(seed=0)
x.shape, x[3]          # ((14,), 1.0) — x[3:10] is the weekday one-hot, Monday
```

## InventoryEnv.totals
sig: `env.totals` → dict: profit, demand, sales, lost, spoiled, overflow, ordered, orders
match: .totals
tags: summary, KPIs, fill rate

Running episode totals, reset by `reset`. Fill rate = `sales / demand`.

```python
from inventory import InventoryEnv, s_S_policy, run_policy
env = InventoryEnv()
G, totals, _ = run_policy(env, s_S_policy(env, 20, 40), seed=0)
round(totals["sales"] / totals["demand"], 2)      # fill rate
```

## InventoryEnv replay options
sig: `demand_series=`, `promo_series=`, `dates=`, `start=`, `random_start=`, `jitter_weeks=`, `ref_demand=`
match: demand_series=, ref_demand=, jitter_weeks=, random_start=
tags: real data, replay, backtest, train test split

Pass a historical demand series to replay it day by day instead of synthetic demand. `random_start=True` starts each episode at a random day; `jitter_weeks=k` swaps each day with the same weekday up to k weeks away (noise for training on one path); `ref_demand=` fixes the scaling (order step, capacity) from training data.

```python
from inventory import InventoryEnv, load_demand_csv, sample_csv_path
h = load_demand_csv(sample_csv_path())
train, test = h["demand"][:365], h["demand"][365:]
env_tr = InventoryEnv(demand_series=train, random_start=True, horizon=90, jitter_weeks=2,
                      ref_demand=train.mean())
env_te = InventoryEnv(demand_series=test, horizon=365, ref_demand=train.mean())
```

> [!WARN] Leakage
> Without `ref_demand`, the test env scales its actions by the *test* mean — information you wouldn't have in advance. Scale both from training data.

## order_up_to_policy
sig: `order_up_to_policy(env, S)` → `policy(obs) -> action`
match: order_up_to_policy(
tags: base stock, baseline, heuristic

Base-stock baseline: every day order enough to lift the inventory position to `S` (rounded to an available size). It reads `env` directly, so it works with any obs mode.

```python
from inventory import InventoryEnv, order_up_to_policy, evaluate_policy
env = InventoryEnv()
evaluate_policy(env, lambda e: order_up_to_policy(e, 36))   # mean yearly profit ≈ 10 290
```

## s_S_policy
sig: `s_S_policy(env, s, S)` → `policy(obs) -> action`
match: s_S_policy(
tags: (s, S) policy, baseline, reorder point

The classic (s, S) policy: when the position drops to `s` or below, order up to `S`; otherwise order nothing. The fixed order fee is what makes waiting worthwhile.

```python
from inventory import InventoryEnv, s_S_policy
env = InventoryEnv(); obs, _ = env.reset(seed=0)
pi = s_S_policy(env, 20, 40)
pi(obs)              # 0 — position 36 > 20, no order today
```

> [!WARN] It closes over env
> The returned policy reads `env.position`, so use it with the same env you step.

## run_policy
sig: `run_policy(env, policy, seed=None, record=False)` → `(total_reward, totals_dict, trace)`
match: run_policy(
tags: rollout, evaluate, trace, backtest

Runs one full episode. With `record=True`, `trace` holds per-day lists (`demand`, `sales`, `lost_sales`, `order`, `inventory`, `spoiled`, `position`) and summed `costs` — ready for `rlhud.plot`.

```python
import rlhud
from inventory import InventoryEnv, s_S_policy, run_policy
env = InventoryEnv()
G, totals, trace = run_policy(env, s_S_policy(env, 20, 40), seed=0, record=True)
rlhud.plot(trace["inventory"], label="on hand", title="Stock level")
```

## evaluate_policy
sig: `evaluate_policy(env, make_policy, seeds=range(5))` → float (mean total reward)
match: evaluate_policy(
tags: evaluate, mean profit, compare

Mean total profit over several seeded episodes. Note the second argument is a **factory** `make_policy(env) -> policy`, not a policy.

```python
from inventory import InventoryEnv, s_S_policy, evaluate_policy
env = InventoryEnv()
evaluate_policy(env, lambda e: s_S_policy(e, 20, 40), seeds=range(3))
```

> [!WARN] Pass a factory
> `evaluate_policy(env, my_policy)` calls `my_policy(env)` and then uses the *result* as the policy. Wrap it: `lambda e: my_policy`.

## generate_sales_history
sig: `generate_sales_history(days=730, seed=0, start="2023-01-02", base=12.0, dispersion=20.0, **kw)` → dict
match: generate_sales_history(
tags: synthetic data, demand, history

A realistic synthetic daily demand history: trend, summer peak, pre-Christmas rush, weekday pattern, promotions and over-dispersed noise. Returns `{"dates", "demand", "dow", "promo"}` of equal length.

```python
from inventory import generate_sales_history
h = generate_sales_history(days=10, seed=0)
h["dates"][0], h["demand"][:3], h["dow"][:3]   # ('2023-01-02', array([ 5,  4, 12]), array([0, 1, 2]))
```

## load_demand_csv / sample_csv_path
sig: `load_demand_csv(path, date_col="date", demand_col="demand", promo_col="promo")` → dict · `sample_csv_path()` → str
match: load_demand_csv(, sample_csv_path(
tags: CSV, real data, import

Loads a daily demand series from CSV (only date and demand are required). Rows are sorted, duplicate dates summed, missing days filled with 0, negatives clipped. `sample_csv_path()` is the bundled ~2-year file.

```python
from inventory import load_demand_csv, sample_csv_path
h = load_demand_csv(sample_csv_path())
len(h["dates"]), h["dates"][0]             # (730, '2023-01-02')
# your own export: load_demand_csv("sales.csv", demand_col="unit_sales")
```

> [!WARN] Sales are not demand
> Real exports record *sales*, which are capped by stock. Days with stockouts underestimate true demand.

## save_csv
sig: `save_csv(history, path)` — writes columns date, demand, dow, promo
match: save_csv(
tags: export, CSV, write

Writes a history dict (like the one `generate_sales_history` returns) to CSV, creating folders as needed.

```python
from inventory import generate_sales_history, save_csv, load_demand_csv
save_csv(generate_sales_history(days=30, seed=1), "out/demand.csv")
len(load_demand_csv("out/demand.csv")["demand"])    # 30
```

## demand_mean_profile
sig: `demand_mean_profile(dates, base=12.0, trend=0.08, season=0.20, promo=None, promo_lift=2.5, weekly=None)` → ndarray
match: demand_mean_profile(
tags: expected demand, seasonality, forecast

Expected demand per day (no noise): base × trend × yearly season × holiday bump × weekday × promo lift. Takes `datetime.date` objects. A perfect "forecast" to compare learned policies against.

```python
import datetime as dt
from inventory import demand_mean_profile
days = [dt.date(2023, 1, 2) + dt.timedelta(days=i) for i in range(3)]
demand_mean_profile(days).round(2)        # array([5.27, 6.16, 7.05])  Mon, Tue, Wed
```

## DOW_NAMES / WEEKLY
sig: `DOW_NAMES = ["Mon", …, "Sun"]` · `WEEKLY` → relative demand per weekday (mean 1)
match: DOW_NAMES, WEEKLY
tags: weekday, labels, seasonality

Weekday labels for tables and charts, and the default weekly demand multipliers (quiet Monday, busy Saturday).

```python
from inventory import DOW_NAMES, WEEKLY
DOW_NAMES[5], round(float(WEEKLY[5]), 2)     # ('Sat', 1.82)
```

## InventoryEnv.render
sig: `env.render()` → one-line string
match: env.render()
tags: debug, print, text

A compact text line: day, weekday, on-hand stock, pipeline, promo flag.

```python
from inventory import InventoryEnv
env = InventoryEnv(); env.reset(seed=0); env.step(3)
env.render()     # 'day   1 Mon | on hand   32 | in transit [0, 18] | promo 0'
```
