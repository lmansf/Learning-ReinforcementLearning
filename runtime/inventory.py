"""
inventory — a retail inventory-replenishment simulator for the bonus capstone.

One product, one store, one decision per day: *how many units do we order from the
supplier?* Orders arrive after a lead time, demand is seasonal and noisy, unmet demand
is lost, stock costs money to hold and slowly spoils, and the shelf has a capacity.

    from inventory import InventoryEnv, generate_sales_history, load_demand_csv
    env = InventoryEnv(horizon=365)               # synthetic demand, new path every reset
    obs, info = env.reset(seed=0)                 # obs: int in [0, env.n_states)
    obs, reward, terminated, truncated, info = env.step(3)   # order env.order_sizes[3] units

    hist = load_demand_csv(sample_csv_path())     # ~2 years of daily sales (or YOUR csv)
    env = InventoryEnv(demand_series=hist["demand"], promo_series=hist["promo"],
                       dates=hist["dates"])       # replay real demand day by day

Only the standard library and numpy are used.

Timeline of one day (one call to step(a)):

  1. the order of `order_sizes[a]` units is placed and paid for (unit cost + fixed fee);
     it will arrive `lead_time` days later (lead_time=0 means immediately)
  2. deliveries due today arrive; anything above `capacity` is refused (overflow, wasted)
  3. customers arrive: sales = min(on_hand, demand); the rest is lost
  4. a fraction of the stock left on the shelf spoils overnight (`spoil_rate`)
  5. holding cost is charged on the stock that is left

reward = revenue − purchases − order fees − holding − stockout penalty
(and on the final day, leftover stock is credited back at unit cost: `salvage=True`).
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
import os

import numpy as np

__all__ = [
    "InventoryEnv", "generate_sales_history", "demand_mean_profile", "save_csv",
    "load_demand_csv", "sample_csv_path", "order_up_to_policy", "s_S_policy",
    "run_policy", "evaluate_policy", "DOW_NAMES",
]

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# relative demand by weekday: quiet early week, busy weekend (think fresh bakery item)
WEEKLY = np.array([0.60, 0.70, 0.80, 0.90, 1.40, 2.00, 1.30])
WEEKLY = WEEKLY / WEEKLY.mean()


# =========================================================================== data
def sample_csv_path() -> str:
    """Absolute path of the bundled ~2-year synthetic sales file (repo/data/sample_sales.csv)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "data", "sample_sales.csv"))


def _dates(start: str, days: int):
    d0 = _dt.date.fromisoformat(start)
    return [d0 + _dt.timedelta(days=i) for i in range(days)]


def _promo_schedule(days: int, rng, rate: float = 0.06, length: int = 3):
    """Planned promotions: short blocks of `length` days, roughly `rate` of all days."""
    promo = np.zeros(days, dtype=np.int8)
    p_start = rate / length
    t = 0
    while t < days:
        if rng.random() < p_start:
            promo[t:t + length] = 1
            t += length + 7            # no back-to-back promos
        else:
            t += 1
    return promo


def demand_mean_profile(dates, base: float = 12.0, trend: float = 0.08,
                        season: float = 0.20, promo=None, promo_lift: float = 2.5, weekly=None):
    """Expected demand per day: base × trend × yearly season × holiday bump × weekday × promo."""
    n = len(dates)
    t = np.arange(n)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    dow = np.array([d.weekday() for d in dates])
    yearly = 1.0 + season * np.sin(2 * np.pi * (doy - 80) / 365.25)          # summer peak
    holiday = 1.0 + 0.6 * np.exp(-0.5 * ((doy - 352) / 6.0) ** 2)             # pre-Christmas rush
    wk = WEEKLY if weekly is None else np.asarray(weekly, dtype=float) / np.mean(weekly)
    mu = base * (1.0 + trend * t / 365.0) * yearly * holiday * wk[dow]
    if promo is not None:
        mu = mu * np.where(np.asarray(promo) > 0, promo_lift, 1.0)
    return mu


def _neg_binomial(rng, mu, dispersion: float = 20.0):
    """Over-dispersed counts with mean `mu` (variance mu + mu²/dispersion)."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    p = dispersion / (dispersion + mu)
    return rng.negative_binomial(dispersion, p)


def generate_sales_history(days: int = 730, seed: int = 0, start: str = "2023-01-02",
                           base: float = 12.0, dispersion: float = 20.0, **kw):
    """A realistic synthetic daily demand history for one product.

    Returns a dict of equal-length sequences:
      dates  : list of 'YYYY-MM-DD' strings
      demand : int array (true customer demand — in real data you usually only see *sales*!)
      dow    : int array, 0 = Monday … 6 = Sunday
      promo  : int array, 1 on promotion days
    """
    rng = np.random.default_rng(seed)
    ds = _dates(start, days)
    promo = _promo_schedule(days, rng)
    mu = demand_mean_profile(ds, base=base, promo=promo, **kw)
    demand = _neg_binomial(rng, mu, dispersion).astype(int)
    return {"dates": [d.isoformat() for d in ds], "demand": demand,
            "dow": np.array([d.weekday() for d in ds]), "promo": promo.astype(int)}


def save_csv(history, path: str):
    """Write a history dict to CSV with columns date,demand,dow,promo."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "demand", "dow", "promo"])
        for i, d in enumerate(history["dates"]):
            w.writerow([d, int(history["demand"][i]), int(history["dow"][i]), int(history["promo"][i])])


def load_demand_csv(path: str, date_col: str = "date", demand_col: str = "demand",
                    promo_col: str = "promo"):
    """Load a daily demand series from CSV. Only `date` and `demand` columns are required.

    Works for your own exports too (e.g. one store × one item from a Kaggle store-sales
    dataset): rename the columns, or pass `date_col=` / `demand_col=` (e.g. 'sales',
    'unit_sales'). Rows are sorted by date, duplicate dates are summed, missing days are
    filled with 0 demand, and negative values (returns) are clipped to 0. A promo column
    is optional: any non-zero value counts as a promotion day.
    """
    per_day: dict = {}
    promo_day: dict = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        dc, qc = cols.get(date_col.lower()), cols.get(demand_col.lower())
        pc = cols.get(promo_col.lower())
        if dc is None or qc is None:
            raise ValueError(f"{path}: need columns {date_col!r} and {demand_col!r}, found {reader.fieldnames}")
        for row in reader:
            raw = (row[dc] or "").strip()
            if not raw:
                continue
            d = _dt.date.fromisoformat(raw[:10])
            try:
                q = float(row[qc] or 0)
            except ValueError:
                q = 0.0
            per_day[d] = per_day.get(d, 0.0) + max(0.0, q)
            if pc is not None:
                try:
                    pv = float(row[pc] or 0)
                except ValueError:
                    pv = 1.0 if str(row[pc]).strip().lower() in ("true", "yes", "y") else 0.0
                promo_day[d] = max(promo_day.get(d, 0), int(pv != 0))
    if not per_day:
        raise ValueError(f"{path}: no rows")
    d0, d1 = min(per_day), max(per_day)
    ds = [d0 + _dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    demand = np.array([int(round(per_day.get(d, 0.0))) for d in ds])
    return {"dates": [d.isoformat() for d in ds], "demand": demand,
            "dow": np.array([d.weekday() for d in ds]),
            "promo": np.array([promo_day.get(d, 0) for d in ds], dtype=int)}


# =========================================================================== env
class InventoryEnv:
    """Daily replenishment of one product. Gymnasium-style API.

    Parameters (defaults model a fresh bakery item selling ~12 units/day)
      horizon      : days per episode (truncated afterwards)
      lead_time    : days between ordering and delivery
      order_sizes  : the discrete action set (units). Default 0, step, …, 8·step with
                     step = order_step (default ≈ 0.5 × mean daily demand, at least 1)
      capacity     : max units on the shelf/backroom; extra deliveries are refused
      price, unit_cost, holding_cost (per unit per night), order_cost (fixed fee per order),
      stockout_penalty (goodwill cost per lost sale), spoil_rate (fraction of leftover stock
      that spoils overnight), salvage (credit leftover stock at unit cost on the last day)
      obs          : "tabular" (int state) or "vector" (np.float32 features)

    Demand source
      synthetic (default): every reset draws a fresh year of demand with the same seasonal
        pattern as generate_sales_history (fresh noise and promo calendar). base_demand sets
        the level; random_start=True starts on a random day of the year.
      replay: pass demand_series (and optionally promo_series, dates). Each episode replays
        it from `start` (or a random start if random_start=True). jitter_weeks=k turns on a
        seasonal bootstrap: each day's demand is taken from the same weekday up to k weeks
        earlier/later — useful noise for *training* on a single historical path.

    Tabular state = (inventory-position bucket, weekday, promo-soon flag):
      position = on hand + in transit, bucketed in units of order_step
      promo-soon = a promotion falls in the window an order placed today would serve
    """

    def __init__(self, horizon: int = 365, lead_time: int = 2, order_step: int | None = None,
                 n_actions: int = 9, order_sizes=None, capacity: int | None = None,
                 price: float = 10.0, unit_cost: float = 6.0, holding_cost: float = 0.10,
                 order_cost: float = 5.0, stockout_penalty: float = 1.0,
                 spoil_rate: float = 0.05, salvage: bool = True, obs: str = "tabular",
                 base_demand: float = 12.0, demand_series=None, promo_series=None, dates=None,
                 start: int = 0, random_start: bool = False, jitter_weeks: int = 0,
                 init_stock: int | None = None, ref_demand: float | None = None,
                 accounting: str = "accrual", demand_kw: dict | None = None,
                 dispersion: float = 20.0):
        self.replay = demand_series is not None
        if self.replay:
            self.series = np.asarray(demand_series, dtype=float)
            n = len(self.series)
            self.promo_series = (np.zeros(n, dtype=int) if promo_series is None
                                 else np.asarray(promo_series, dtype=int))
            if dates is None:
                dates = [(_dt.date(2023, 1, 2) + _dt.timedelta(days=i)).isoformat() for i in range(n)]
            self.series_dates = list(dates)
            self.series_dow = np.array([_dt.date.fromisoformat(d[:10]).weekday() for d in self.series_dates])
            mean_demand = float(self.series.mean())
            horizon = min(horizon, n - start) if not random_start else min(horizon, n)
        else:
            mean_demand = base_demand
        if ref_demand is not None:          # scale settings from TRAINING data only
            mean_demand = float(ref_demand)
        self.mean_demand = mean_demand
        self.base_demand = base_demand
        step = order_step or max(1, int(round(0.5 * mean_demand)))
        self.order_step = step
        self.order_sizes = (np.asarray(order_sizes, dtype=int) if order_sizes is not None
                            else step * np.arange(n_actions))
        self.n_actions = len(self.order_sizes)
        self._uniform = bool(np.array_equal(self.order_sizes, step * np.arange(self.n_actions)))
        self.capacity = int(capacity or round(10 * mean_demand))
        self.horizon, self.lead_time = int(horizon), int(lead_time)
        self.price, self.unit_cost = float(price), float(unit_cost)
        self.holding_cost, self.order_cost = float(holding_cost), float(order_cost)
        self.stockout_penalty, self.spoil_rate = float(stockout_penalty), float(spoil_rate)
        self.salvage, self.obs_mode = bool(salvage), obs
        if accounting not in ("accrual", "cash"):
            raise ValueError("accounting must be 'accrual' or 'cash'")
        self.accounting = accounting
        self.demand_kw = dict(demand_kw or {})
        self.dispersion = float(dispersion)
        self.start, self.random_start, self.jitter_weeks = int(start), bool(random_start), int(jitter_weeks)
        self.init_stock = int(round(3 * mean_demand)) if init_stock is None else int(init_stock)
        # tabular state layout
        self.n_pos = int(math.ceil(self.capacity / step)) + 1
        self.n_states = self.n_pos * 7 * 2
        self.obs_dim = 14
        self.rng = np.random.default_rng()
        self.t = 0

    # ------------------------------------------------------------------ episode data
    W = 7   # days of demand history the agent can see (trailing mean -> demand level)

    def _new_episode_data(self):
        H, L, W = self.horizon, self.lead_time, self.W
        n = W + H + L + 2                   # history + episode + look-ahead for the promo flag
        if self.replay:
            N = len(self.series)
            s0 = int(self.rng.integers(0, max(1, N - H + 1))) if self.random_start else self.start
            idx = np.clip(np.arange(s0 - W, s0 - W + n), 0, N - 1)
            dem = self.series[idx]
            if self.jitter_weeks > 0:     # seasonal block bootstrap: same weekday, nearby weeks
                shift = 7 * self.rng.integers(-self.jitter_weeks, self.jitter_weeks + 1, size=n)
                j = idx + shift
                j = np.where((j < 0) | (j >= N), idx, j)
                dem = self.series[j]
            demand = np.round(dem).astype(int)
            promo = self.promo_series[idx]
            dow = self.series_dow[idx]
            self.dates = [self.series_dates[i] for i in idx[W:W + H]]
        else:
            offset = int(self.rng.integers(0, 365)) if self.random_start else 0
            ds = _dates("2022-12-26", 365 + n)[offset:offset + n]      # W days before 2023-01-02
            promo = _promo_schedule(n, self.rng)
            mu = demand_mean_profile(ds, base=self.base_demand, promo=promo, **self.demand_kw)
            demand = _neg_binomial(self.rng, mu, self.dispersion).astype(int)
            dow = np.array([d.weekday() for d in ds])
            self.dates = [d.isoformat() for d in ds[W:W + H]]
        # per-episode arrays are indexed by day t = 0..H (+ look-ahead)
        self.demand, self.promo, self.dow = demand[W:], promo[W:], dow[W:]
        m = len(self.demand)
        # promo_soon[t]: a promotion on day t+L or t+L+1 (the days today's order will serve)
        p = self.promo
        self.promo_soon = np.array([int(p[min(t + L, m - 1)] or p[min(t + L + 1, m - 1)]) for t in range(m)])
        # level[t]: trailing 7-day mean demand (days t-7..t-1) vs the reference level -> 0/1/2
        csum = np.concatenate([[0], np.cumsum(demand)])
        trail = (csum[W:W + m] - csum[0:m]) / W
        ratio = trail / max(1e-9, self.mean_demand)
        self.trail_ratio = ratio
        self.level = np.where(ratio < 0.85, 0, np.where(ratio > 1.15, 2, 1))
        # plain-Python copies: indexing lists is much faster than numpy scalars in step()
        self._d, self._dow = self.demand.tolist(), self.dow.tolist()
        self._soon = self.promo_soon.tolist()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._new_episode_data()
        self.t = 0
        self.on_hand = self.init_stock
        self.pipeline = [0] * self.lead_time     # pipeline[0] arrives next, pipeline[-1] last
        self._transit = 0
        self.totals = {"profit": 0.0, "demand": 0, "sales": 0, "lost": 0, "spoiled": 0,
                       "overflow": 0, "ordered": 0, "orders": 0}
        return self._obs(), self._info()

    # ------------------------------------------------------------------ observation
    @property
    def in_transit(self) -> int:
        return self._transit

    @property
    def position(self) -> int:
        """Inventory position: on hand + in transit (what you 'own')."""
        return self.on_hand + self._transit

    def nearest_action(self, units) -> int:
        """Index of the available order size closest to `units` (0 if units <= 0)."""
        if units <= 0:
            return 0
        sizes = self.order_sizes
        if self._uniform:
            return min(self.n_actions - 1, int(units / self.order_step + 0.5))
        return int(np.argmin(np.abs(sizes - units)))

    def state_index(self) -> int:
        b = (self.on_hand + self._transit) // self.order_step
        if b >= self.n_pos:
            b = self.n_pos - 1
        t = self.t if self.t < len(self._dow) else len(self._dow) - 1
        return (b * 7 + self._dow[t]) * 2 + self._soon[t]

    def decode_state(self, s: int):
        """Inverse of state_index: (position_bucket, dow, promo_soon)."""
        s, promo = divmod(int(s), 2)
        b, dow = divmod(s, 7)
        return b, dow, promo

    def features(self):
        t = min(self.t, len(self.dow) - 1)
        x = np.zeros(self.obs_dim, dtype=np.float32)
        x[0] = self.on_hand / self.capacity
        x[1] = self.in_transit / self.capacity
        x[2] = self.position / self.capacity
        x[3 + self.dow[t]] = 1.0
        x[10] = self.promo[t]
        x[11] = self.promo_soon[t]
        x[12] = self.trail_ratio[t]
        x[13] = self.t / max(1, self.horizon)
        return x

    def _obs(self):
        return self.state_index() if self.obs_mode == "tabular" else self.features()

    def _info(self, **extra):
        t = min(self.t, len(self.dow) - 1)
        info = {"day": self.t, "date": self.dates[min(self.t, len(self.dates) - 1)],
                "dow": int(self.dow[t]), "promo": int(self.promo[t]), "on_hand": self.on_hand,
                "in_transit": self.in_transit, "position": self.position}
        info.update(extra)
        return info

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action}")
        q = int(self.order_sizes[a])
        t = self.t
        if t >= self.horizon:
            raise RuntimeError("episode is over: call reset()")
        purchase = q * self.unit_cost
        fee = self.order_cost if q > 0 else 0.0
        # 1-2: place order, receive deliveries due today
        if self.lead_time == 0:
            arriving = q
        else:
            self.pipeline.append(q)
            arriving = self.pipeline.pop(0)
            self._transit += q - arriving
        stock = self.on_hand + arriving
        overflow = max(0, stock - self.capacity)
        stock -= overflow
        # 3: customers
        d = self._d[t]
        sales = d if d < stock else stock
        lost = d - sales
        stock -= sales
        # 4: overnight spoilage
        spoiled = int(self.rng.binomial(stock, self.spoil_rate)) if stock > 0 and self.spoil_rate > 0 else 0
        stock -= spoiled
        # 5: holding
        holding = stock * self.holding_cost
        revenue = sales * self.price
        penalty = lost * self.stockout_penalty
        salvage = 0.0
        truncated = self.t + 1 >= self.horizon
        if self.accounting == "cash":
            # money in minus money out today; leftover stock credited at cost on the last day
            if truncated and self.salvage:
                salvage = (stock + self._transit) * self.unit_cost
            reward = revenue - purchase - fee - holding - penalty + salvage
        else:
            # accrual ("cost of goods sold"): a unit's cost is booked when it is sold or wasted.
            # Same total profit (up to the value of the starting stock) as cash accounting,
            # but the reward for a good order shows up much sooner.
            cogs = (sales + spoiled + overflow) * self.unit_cost
            if truncated and not self.salvage:
                cogs += (stock + self._transit) * self.unit_cost
            reward = revenue - cogs - fee - holding - penalty
        self.on_hand = stock
        self.t += 1
        T = self.totals
        T["profit"] += reward; T["demand"] += d; T["sales"] += sales; T["lost"] += lost
        T["spoiled"] += spoiled; T["overflow"] += overflow; T["ordered"] += q; T["orders"] += int(q > 0)
        info = self._info(order=q, arrived=arriving, demand=d, sales=sales, lost_sales=lost,
                          spoiled=spoiled, overflow=overflow, inventory=stock,
                          costs={"revenue": revenue, "cogs": sales * self.unit_cost,
                                 "purchase": purchase, "order_fee": fee,
                                 "holding": holding, "stockout": penalty,
                                 "waste": (spoiled + overflow) * self.unit_cost, "salvage": salvage})
        return self._obs(), float(reward), False, truncated, info

    def render(self) -> str:
        t = max(0, self.t - 1)
        return (f"day {self.t:3d} {DOW_NAMES[self.dow[t]]} | on hand {self.on_hand:4d} | "
                f"in transit {self.pipeline} | promo {int(self.promo[t])}")


# =========================================================================== baselines
def order_up_to_policy(env: InventoryEnv, S: int):
    """Base-stock policy: every day, order enough to lift the inventory position to S
    (rounded to the nearest available order size). Works with any obs mode — it reads env."""
    def policy(obs):
        return env.nearest_action(S - env.position)
    return policy


def s_S_policy(env: InventoryEnv, s: int, S: int):
    """(s, S) policy: when the inventory position drops to s or below, order up to S;
    otherwise order nothing. The fixed order fee is what makes s < S worthwhile."""
    def policy(obs):
        pos = env.position
        return 0 if pos > s else env.nearest_action(S - pos)
    return policy


def run_policy(env: InventoryEnv, policy, seed=None, record: bool = False):
    """Run one full episode. Returns (total_reward, totals_dict, trace) where trace is a
    dict of per-day lists (only filled when record=True)."""
    obs, info = env.reset(seed=seed)
    G = 0.0
    keys = ("demand", "sales", "lost_sales", "order", "inventory", "spoiled", "position")
    trace = {k: [] for k in keys}
    trace["costs"] = {}
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        G += r
        done = term or trunc
        if record:
            for k in keys:
                trace[k].append(info[k])
            for k, v in info["costs"].items():
                trace["costs"][k] = trace["costs"].get(k, 0.0) + v
    return G, dict(env.totals), trace


def evaluate_policy(env: InventoryEnv, make_policy, seeds=range(5)):
    """Mean total profit of `make_policy(env)` over several seeded episodes."""
    out = []
    for s in seeds:
        G, _, _ = run_policy(env, make_policy(env), seed=s)
        out.append(G)
    return float(np.mean(out))


if __name__ == "__main__":
    # regenerate the bundled sample file
    h = generate_sales_history(days=730, seed=2024)
    save_csv(h, sample_csv_path())
    print("wrote", sample_csv_path(), "mean demand", h["demand"].mean())
