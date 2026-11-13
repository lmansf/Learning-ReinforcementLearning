import numpy as np
import rlhud
from inventory import (InventoryEnv, load_demand_csv, sample_csv_path, s_S_policy,
                       evaluate_policy, run_policy, DOW_NAMES)


def weekday_profile(history):
    """Mean demand per weekday divided by the mean over all days, using NON-promo days only.

    history: dict with numpy arrays "demand", "dow" (0 = Mon) and "promo" (1 on promo days).
    Returns a length-7 array that averages ~1.0 (Saturday ≈ 2.0 means twice a normal day)."""
    d, dow, promo = (np.asarray(history[k]) for k in ("demand", "dow", "promo"))
    normal = promo == 0
    overall = d[normal].mean()
    return np.array([d[normal & (dow == k)].mean() / overall for k in range(7)])


def promo_lift(history):
    """How much a promotion multiplies demand, corrected for the weekday mix.

    lift = (total demand on promo days) / (total demand you would EXPECT on those days
    without a promotion), where the expectation for a day is the mean non-promo demand
    of its weekday."""
    d, dow, promo = (np.asarray(history[k]) for k in ("demand", "dow", "promo"))
    normal = promo == 0
    base = np.array([d[normal & (dow == k)].mean() for k in range(7)])
    on = promo == 1
    return float(d[on].sum() / base[dow[on]].sum())


def grid_search_sS(env, s_grid, S_grid, seeds):
    """Try every (s, S) with S > s; score = mean episode profit over `seeds`
    (use inventory.evaluate_policy with inventory.s_S_policy).

    Returns (best_s, best_S, best_profit, table) where table[(s, S)] = mean profit."""
    table = {}
    best = (None, None, -np.inf)
    for s in s_grid:
        for S in S_grid:
            if S <= s:
                continue
            profit = evaluate_policy(env, lambda e: s_S_policy(e, s, S), seeds=seeds)
            table[(s, S)] = profit
            if profit > best[2]:
                best = (s, S, profit)
    return best[0], best[1], best[2], table


TRAIN_SEEDS = range(0, 4)        # demand years we tune on
TEST_SEEDS = range(100, 120)     # demand years we never looked at while tuning
S_GRID_LOW = range(20, 81, 10)   # reorder points s
S_GRID_HIGH = range(30, 121, 10)  # order-up-to levels S


if __name__ == "__main__":
    hist = load_demand_csv(sample_csv_path())
    n = len(hist["demand"])
    rlhud.plot(hist["demand"][:120], label="daily demand", title="Sample sales · first 120 days",
               ylabel="units", xlabel="day")
    rlhud.plot(hist["promo"][:120] * hist["demand"][:120].max(), label="promo days",
               title="Sample sales · first 120 days")
    prof = weekday_profile(hist)
    rlhud.bars(prof, labels=DOW_NAMES, title="Weekday profile (non-promo days)",
               highlight=int(np.argmax(prof)))
    rlhud.stat("Days of history", n)
    rlhud.stat("Mean demand / day", float(hist["demand"].mean()))
    rlhud.stat("Promo lift", promo_lift(hist), unit="×")

    env = InventoryEnv()
    s, S, train_profit, table = grid_search_sS(env, S_GRID_LOW, S_GRID_HIGH, TRAIN_SEEDS)
    heat = [[table.get((lo, hi), None) for hi in S_GRID_HIGH] for lo in S_GRID_LOW]
    rlhud.grid(heat, title="Mean training profit by (s rows, S columns)", fmt=".0f", cmap="heat",
               labels=[[f"{v / 1000:.1f}k" if v is not None else "—" for v in row] for row in heat])
    test_profit = evaluate_policy(env, lambda e: s_S_policy(e, s, S), seeds=TEST_SEEDS)
    G, totals, trace = run_policy(env, s_S_policy(env, s, S), seed=100, record=True)
    rlhud.plot(trace["demand"][:56], label="demand", title="Tuned (s, S) · first 8 weeks of a test year")
    rlhud.plot(trace["inventory"][:56], label="stock after sales", title="Tuned (s, S) · first 8 weeks of a test year")
    rlhud.stat("Tuned (s, S)", f"({s}, {S})")
    rlhud.stat("Profit · train years", train_profit)
    rlhud.stat("Profit · held-out test years", test_profit, good=test_profit >= 12500)
    rlhud.stat("Fill rate (test year 100)", totals["sales"] / totals["demand"])
