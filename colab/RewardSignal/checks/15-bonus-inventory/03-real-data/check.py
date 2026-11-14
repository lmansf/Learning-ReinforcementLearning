import numpy as np
from inventory import InventoryEnv, load_demand_csv, sample_csv_path, s_S_policy, run_policy

time_split = require("time_split")
fill_rate = require("fill_rate")
summarize = require("summarize")
make_envs = require("make_envs")
evaluate = require("evaluate")
run_pipeline = require("run_pipeline")
grid_search_sS = require("grid_search_sS")

# ---- 1. split by time -----------------------------------------------------------------------
toy = {"dates": [f"2024-01-{d:02d}" for d in range(1, 11)], "demand": np.arange(10),
       "dow": np.arange(10) % 7, "promo": np.zeros(10, dtype=int)}
tr, te = time_split(toy, 7)
check(set(tr) == set(toy) and set(te) == set(toy), "time_split keeps every column", f"train keys {sorted(tr)}, test keys {sorted(te)}")
check(list(tr["demand"]) == list(range(7)) and list(te["demand"]) == [7, 8, 9],
      "time_split: first n_train days train, the rest test, order preserved",
      f"train demand {list(tr['demand'])}, test demand {list(te['demand'])}")
check(tr["dates"][-1] < te["dates"][0], "every training date comes before every test date (no peeking)")

# ---- 2. metrics -----------------------------------------------------------------------------
check(approx(fill_rate(90, 100), 0.9), "fill_rate(90 sold, 100 demanded) = 0.9", f"got {fill_rate(90, 100)}")
check(approx(fill_rate([5, 5], [5, 10]), 10 / 15), "fill_rate sums sequences before dividing",
      f"expected 0.6667 (10 of 15), got {fill_rate([5, 5], [5, 10])} — the mean of daily ratios is NOT the fill rate")
check(approx(fill_rate(0, 0), 1.0), "fill_rate with zero demand is 1.0", f"got {fill_rate(0, 0)}")
tot = {"profit": 1234.5, "demand": 200, "sales": 180, "lost": 20, "spoiled": 7, "overflow": 0, "ordered": 190, "orders": 31}
m = summarize(tot)
want = {"profit": 1234.5, "fill_rate": 0.9, "lost_sales": 20, "spoiled": 7, "orders": 31}
check(isinstance(m, dict) and set(m) == set(want) and all(approx(m[k], v) for k, v in want.items()),
      "summarize(totals) gives profit, fill_rate, lost_sales, spoiled, orders", f"expected {want}, got {m}")

# ---- 3. environments ------------------------------------------------------------------------
hist = load_demand_csv(sample_csv_path())
train, test = time_split(hist, 365)
tr_env, te_env = make_envs(train, test)
train_mean = float(np.mean(train["demand"]))
check(tr_env.replay and te_env.replay, "both envs replay the CSV demand (demand_series=...)")
check(tr_env.horizon == 365 and te_env.horizon == len(test["demand"]), "horizons = number of days in each split",
      f"train {tr_env.horizon} (want 365), test {te_env.horizon} (want {len(test['demand'])})")
check(tr_env.jitter_weeks > 0, "train env adds seasonal bootstrap noise (jitter_weeks=2)", f"jitter_weeks={tr_env.jitter_weeks}")
check(te_env.jitter_weeks == 0 and not te_env.random_start, "test env replays history exactly (no jitter, no random start)")
check(approx(te_env.mean_demand, train_mean, 1e-6) and te_env.order_step == tr_env.order_step and te_env.capacity == tr_env.capacity,
      "test env is scaled from TRAINING data (ref_demand = mean training demand)",
      f"test env mean_demand {te_env.mean_demand:.3f} vs training mean {train_mean:.3f}; order_step {te_env.order_step} vs {tr_env.order_step}. "
      "Using the test year's own mean leaks future information into the policy.")
te_env.reset(seed=0)
check(list(te_env.demand[:te_env.horizon]) == list(test["demand"]), "test env demand equals the test year's history day by day")

# ---- 4. the evaluation harness --------------------------------------------------------------
small = InventoryEnv(horizon=40)
got = evaluate(small, lambda e: s_S_policy(e, 40, 70), seeds=[0, 1, 2])
ref = [run_policy(small, s_S_policy(small, 40, 70), seed=s)[1] for s in (0, 1, 2)]
want = {"profit": np.mean([t["profit"] for t in ref]),
        "fill_rate": np.mean([t["sales"] / t["demand"] for t in ref]),
        "lost_sales": np.mean([t["lost"] for t in ref]), "spoiled": np.mean([t["spoiled"] for t in ref]),
        "orders": np.mean([t["orders"] for t in ref])}
check(isinstance(got, dict) and set(got) == set(want) and all(approx(got[k], v, 1e-6) for k, v in want.items()),
      "evaluate averages each metric over the seeds", f"expected {want}, got {got}")

# ---- 5. the out-of-time study ---------------------------------------------------------------
ref_env = InventoryEnv(demand_series=train["demand"], promo_series=train["promo"], dates=train["dates"],
                       horizon=365, jitter_weeks=2)
s_ref, S_ref, _, _ = grid_search_sS(ref_env)
results = [run_pipeline(sample_csv_path(), n_train=365, seed=sd) for sd in (0, 1)]
check(all(tuple(r["s_S"]) == (s_ref, S_ref) for r in results), "the planner is tuned on the TRAINING year only",
      f"expected (s, S) = {(s_ref, S_ref)} from grid_search_sS(train_env); got {[r['s_S'] for r in results]}")
ratio = float(np.mean([r["agent"]["profit"] / r["baseline"]["profit"] for r in results]))
fill = float(np.mean([r["agent"]["fill_rate"] for r in results]))
check(ratio >= 1.0, "trained on year 1, the agent out-earns the planner on unseen year 2 (mean of 2 seeds)",
      f"agent/planner profit ratio {ratio:.3f}")
check(fill >= 0.88, "agent fill rate on the test year ≥ 88 %", f"mean agent fill rate {fill:.3f}")
check(results[0]["baseline"]["fill_rate"] > 0.8, "baseline metrics look sane",
      f"planner fill rate {results[0]['baseline']['fill_rate']:.3f}")
