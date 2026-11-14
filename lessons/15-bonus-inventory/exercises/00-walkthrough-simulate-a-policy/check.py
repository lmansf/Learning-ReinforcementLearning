import numpy as np
from inventory import InventoryEnv

order_up_to = require("order_up_to")
simulate = require("simulate")
mean_metrics = require("mean_metrics")
tune_S = require("tune_S")
window = require("window")
weekday_totals = require("weekday_totals")

env = InventoryEnv(horizon=40)
env.reset(seed=0)
env.on_hand, env.pipeline, env._transit = 20, [0, 6], 6          # position 26
pol = order_up_to(env, 50)
check(pol(None) == env.nearest_action(24), "order_up_to orders S − position (position 26, S = 50 → 24 units)",
      f"expected action {env.nearest_action(24)}, got {pol(None)}")
check(order_up_to(env, 10)(None) == 0, "order_up_to orders nothing when the position is already above S")

log, stock = simulate(env, order_up_to(env, 60), seed=3)
T = env.totals
check(isinstance(log, dict) and set(log) == {"profit", "demand", "sales", "orders"},
      "simulate returns a dict with profit, demand, sales, orders", f"got {log if isinstance(log, dict) else type(log)}")
check(approx(log["profit"], T["profit"], 1e-6) and log["demand"] == T["demand"] and log["sales"] == T["sales"],
      "simulate adds up the reward and the per-day info (matches env.totals)",
      f"yours {log}, env.totals profit={T['profit']:.2f} demand={T['demand']} sales={T['sales']}")
check(log["orders"] == T["orders"], "simulate counts the days with a non-zero order", f"yours {log['orders']}, env says {T['orders']}")
check(np.shape(stock) == (40,), "one stock value per day of the episode", f"got shape {np.shape(stock)}")
log2, _ = simulate(env, order_up_to(env, 60), seed=3)
check(log2 == log, "simulate is reproducible for a fixed seed")

m = mean_metrics([{"a": 1.0, "b": 4}, {"a": 3.0, "b": 0}])
check(m == {"a": 2.0, "b": 2.0}, "mean_metrics averages every key", f"got {m}")

env = InventoryEnv(horizon=60)
best_S, best_profit, table = tune_S(env, [30, 60, 90], seeds=[0, 1])
ref = {S: np.mean([simulate(env, order_up_to(env, S), s)[0]["profit"] for s in (0, 1)]) for S in (30, 60, 90)}
check(set(table) == {30, 60, 90} and all(approx(table[S], ref[S], 1e-6) for S in ref),
      "tune_S scores every candidate on the same seeds", f"expected {ref}, got {table}")
best = max(ref, key=ref.get)
check(best_S == best and approx(best_profit, ref[best], 1e-6), "tune_S returns the best S and its profit",
      f"expected ({best}, {ref[best]:.1f}), got ({best_S}, {best_profit})")

toy = {"dates": ["d0", "d1", "d2", "d3"], "demand": np.array([5, 6, 7, 8])}
w = window(toy, 1, 3)
check(set(w) == {"dates", "demand"} and w["dates"] == ["d1", "d2"] and list(w["demand"]) == [6, 7],
      "window slices every column the same way", f"got {w}")

d = np.array([1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70])
dow = np.arange(14) % 7
promo = np.zeros(14, dtype=int); promo[8] = 1
t = np.asarray(weekday_totals(d, dow, promo == 0))
check(t.tolist() == [11, 2, 33, 44, 55, 66, 77], "weekday_totals sums each weekday inside the mask",
      f"got {t.tolist()} — the promo Tuesday (20) must be left out")
