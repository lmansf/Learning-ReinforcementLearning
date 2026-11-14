import numpy as np
import rlhud
from inventory import InventoryEnv, generate_sales_history, DOW_NAMES


#> A **base-stock** policy: every evening, order enough to lift the inventory position
#> back up to S.
def order_up_to(env, S):
    #> The policy reads the env directly: `env.position` = on hand + in transit.
    #> `env.nearest_action(units)` turns a number of units into an action index
    #> (0 if units ≤ 0). The lambda ignores `obs` but must accept it: every policy is
    #> called as `policy(obs)`.
    return lambda obs: env.nearest_action(S - env.position)


#> Run one simulated year and add up business numbers from the per-day `info` dicts.
def simulate(env, policy, seed):
    obs, info = env.reset(seed=seed)
    #> Running totals: ints for unit counts, a float for money.
    log = {"profit": 0.0, "demand": 0, "sales": 0, "orders": 0}
    stock = []
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        #> The reward IS the day's profit.
        log["profit"] += r
        #> `info` describes ONE day: demand, sales, lost_sales, spoiled, order, inventory…
        log["demand"] += info["demand"]
        log["sales"] += info["sales"]
        #> `int(True) == 1`: count the days on which an order was actually placed.
        log["orders"] += int(info["order"] > 0)
        stock.append(info["inventory"])
        #> This env never terminates — the year ends by truncation. Test both anyway.
        done = terminated or truncated
    return log, np.array(stock)


#> Average every metric over several runs: a dict comprehension over the first run's keys.
def mean_metrics(logs):
    return {k: float(np.mean([log[k] for log in logs])) for k in logs[0]}


#> **Tune S** by brute force: score every candidate on the SAME seeds, keep the best.
def tune_S(env, S_grid, seeds):
    table = {}
    #> Start below any possible profit, so the first candidate always wins.
    best_S, best_profit = None, -np.inf
    for S in S_grid:
        logs = [simulate(env, order_up_to(env, S), seed)[0] for seed in seeds]
        table[S] = mean_metrics(logs)["profit"]
        if table[S] > best_profit:
            best_S, best_profit = S, table[S]
    return best_S, best_profit, table


#> A **time window** of a history dict: slice EVERY column the same way, order preserved.
def window(history, start, stop):
    #> Works for numpy arrays and plain lists alike (e.g. the "dates" column).
    return {k: v[start:stop] for k, v in history.items()}


#> Total demand per weekday, restricted by a boolean mask (e.g. "non-promo days only").
def weekday_totals(demand, dow, mask):
    #> `&` is element-wise AND; the parentheses around `dow == k` are required, because
    #> `&` binds tighter than `==`. Indexing with a boolean array keeps the True entries.
    return np.array([demand[mask & (dow == k)].sum() for k in range(7)])


if __name__ == "__main__":
    #> Two years of synthetic daily sales: a dict of arrays "demand", "dow", "promo", …
    hist = generate_sales_history(730, seed=1)
    #> Split by TIME: statistics for decisions come from the first year only.
    train = window(hist, 0, 365)
    normal = train["promo"] == 0
    totals = weekday_totals(train["demand"], train["dow"], normal)
    env = InventoryEnv()
    best_S, best_profit, table = tune_S(env, range(30, 121, 15), seeds=range(3))
    log, stock = simulate(env, order_up_to(env, best_S), seed=100)
    #> Fill rate = share of demand served. Guard against a zero denominator.
    fill = log["sales"] / max(1, log["demand"])
    #= given
    rlhud.bars(totals, labels=DOW_NAMES, highlight=int(np.argmax(totals)),
               title="Year 1 demand per weekday (non-promo days)")
    rlhud.plot(list(table.values()), x=list(table), label="mean profit", xlabel="S",
               title="Order-up-to S: profit on 3 training years")
    rlhud.plot(stock[:56], label="stock after sales", xlabel="day",
               title=f"S = {best_S} · first 8 weeks of held-out year 100")
    rlhud.stat("Best S", best_S)
    rlhud.stat("Profit · held-out year 100", log["profit"])
    rlhud.stat("Fill rate", fill, good=fill > 0.9)
    rlhud.stat("Orders placed", log["orders"])
    #= end
