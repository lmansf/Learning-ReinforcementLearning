import numpy as np
from inventory import InventoryEnv, s_S_policy, evaluate_policy, generate_sales_history, WEEKLY

weekday_profile = require("weekday_profile")
promo_lift = require("promo_lift")
grid_search_sS = require("grid_search_sS")

# ---- 1. weekday profile on two hand-made weeks --------------------------------------------
# Mon..Sun demand, week 2 identical except Saturday is a promo day (must be ignored)
toy = {"demand": np.array([5, 5, 5, 5, 10, 20, 10] * 2), "dow": np.array(list(range(7)) * 2),
       "promo": np.array([0] * 12 + [1, 0])}
toy["demand"][12] = 60          # the promo Saturday sells 60
prof = np.asarray(weekday_profile(toy), dtype=float)
expect = np.array([5, 5, 5, 5, 10, 20, 10]) / np.concatenate([[5, 5, 5, 5, 10, 20, 10], [5, 5, 5, 5, 10, 10]]).mean()
check(prof.shape == (7,), "weekday_profile returns 7 numbers", f"got shape {prof.shape}")
check(prof.shape == (7,) and approx(prof, expect, 1e-6), "weekday_profile matches a hand-computed case",
      f"expected {np.round(expect, 3)}, got {np.round(prof, 3)}. Exclude promo days, and divide by the mean "
      "of ALL non-promo days (not by the mean of the 7 weekday means).")

# ---- 2. promo lift --------------------------------------------------------------------------
lift = promo_lift(toy)
check(approx(lift, 3.0, 1e-6), "promo_lift on the hand-made case",
      f"the promo Saturday sold 60 vs a normal Saturday of 20 → 3.0; got {lift}")
h = generate_sales_history(730, seed=11)
lift = promo_lift(h)
check(2.2 < lift < 2.8, "promo_lift recovers the simulator's ×2.5 promotions from 2 years of data",
      f"got {lift:.2f}")
prof = np.asarray(weekday_profile(h), dtype=float)
check(prof.shape == (7,) and int(np.argmax(prof)) == 5 and prof[5] > 1.5,
      "weekday_profile finds the busy Saturday in synthetic data", f"got {np.round(prof, 2)}")

# ---- 3. grid search -------------------------------------------------------------------------
env = InventoryEnv(horizon=60)
s, S, p, table = grid_search_sS(env, [20, 40], [30, 60], seeds=[0, 1])
ref = {(lo, hi): evaluate_policy(env, lambda e: s_S_policy(e, lo, hi), seeds=[0, 1])
       for lo, hi in [(20, 30), (20, 60), (40, 60)]}
best = max(ref, key=ref.get)
check(set(table) == set(ref), "grid_search_sS tries exactly the pairs with S > s",
      f"expected keys {sorted(ref)}, got {sorted(table)}")
check((s, S) == best and approx(p, ref[best], 1e-6), "grid_search_sS returns the best pair and its mean profit",
      f"expected {best} with {ref[best]:.1f}, got ({s}, {S}) with {p}")

# ---- 4. the tuned planner on held-out years -------------------------------------------------
env = InventoryEnv()
s, S, p, _ = grid_search_sS(env, range(20, 81, 10), range(30, 121, 10), range(0, 4))
test = evaluate_policy(env, lambda e: s_S_policy(e, s, S), seeds=range(100, 120))
check(test >= 13000, "Tuned (s, S) earns ≥ 13,000 per held-out test year",
      f"tuned ({s}, {S}) earned {test:,.0f} on average over test seeds 100–119")
