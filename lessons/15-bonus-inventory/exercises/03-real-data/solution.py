import numpy as np
import rlhud
from inventory import (InventoryEnv, load_demand_csv, sample_csv_path, s_S_policy,
                       evaluate_policy, run_policy, DOW_NAMES)

# Point this at your own export (columns date,demand[,promo]) to run the whole study on it.
CSV_PATH = sample_csv_path()


# ============================================================ your work: the evaluation harness
def time_split(history, n_train):
    """Split a history dict by TIME: the first n_train days train, everything after tests.
    Every key ("dates", "demand", "dow", "promo") is sliced the same way. Never shuffle!"""
    train = {k: v[:n_train] for k, v in history.items()}
    test = {k: v[n_train:] for k, v in history.items()}
    return train, test


def fill_rate(sales, demand):
    """Fraction of demand that was served: total sales / total demand (1.0 if no demand).
    Accepts numbers or sequences."""
    total = float(np.sum(demand))
    return 1.0 if total == 0 else float(np.sum(sales)) / total


def summarize(totals):
    """Business metrics from env.totals (as returned by run_policy):
    {"profit", "fill_rate", "lost_sales", "spoiled", "orders"}."""
    return {"profit": float(totals["profit"]),
            "fill_rate": fill_rate(totals["sales"], totals["demand"]),
            "lost_sales": float(totals["lost"]),
            "spoiled": float(totals["spoiled"]),
            "orders": float(totals["orders"])}


def make_envs(train, test):
    """Build (train_env, test_env) from the two halves of the history.

    train_env: replays the training days with a seasonal bootstrap (jitter_weeks=2) so every
               episode is a slightly different but realistic year; horizon = len(train days).
    test_env : replays the test days exactly as they happened (no jitter, no random start),
               horizon = len(test days). Its scale settings (order sizes, capacity) must come
               from the TRAINING data: pass ref_demand = mean training demand."""
    train_env = InventoryEnv(demand_series=train["demand"], promo_series=train["promo"],
                             dates=train["dates"], horizon=len(train["demand"]), jitter_weeks=2)
    test_env = InventoryEnv(demand_series=test["demand"], promo_series=test["promo"],
                            dates=test["dates"], horizon=len(test["demand"]),
                            ref_demand=float(np.mean(train["demand"])))
    return train_env, test_env


def evaluate(env, make_policy, seeds=range(5)):
    """Run make_policy(env) for one episode per seed (inventory.run_policy) and return the
    mean of each summarize() metric over the seeds."""
    rows = [summarize(run_policy(env, make_policy(env), seed=s)[1]) for s in seeds]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def run_pipeline(csv_path=CSV_PATH, n_train=365, seed=0):
    """The full study. Returns {"s_S": (s, S), "baseline": metrics, "agent": metrics}.

    1. load the CSV, time_split it, make_envs
    2. tune the planner on the TRAIN env: grid_search_sS(train_env, S_GRID_LOW, S_GRID_HIGH, TUNE_SEEDS)
    3. train the residual agent on the TRAIN env: train_residual_q(train_env, s, S, seed=seed)
    4. evaluate both on the TEST env with evaluate(...) and the default seeds"""
    hist = load_demand_csv(csv_path)
    train, test = time_split(hist, n_train)
    train_env, test_env = make_envs(train, test)
    s, S, _, _ = grid_search_sS(train_env, S_GRID_LOW, S_GRID_HIGH, TUNE_SEEDS)
    Q, _ = train_residual_q(train_env, s, S, seed=seed)
    return {"s_S": (s, S),
            "baseline": evaluate(test_env, lambda e: s_S_policy(e, s, S)),
            "agent": evaluate(test_env, lambda e: residual_policy(e, Q, s, S)),
            "Q": Q}


# ============================================================ from exercises 1 and 2 (done)
TUNE_SEEDS = range(4)
OFFSETS = np.array([-20, -10, 0, 10, 20, 30, 40])
N_CONTEXTS = 14


def _grid(env):
    step = env.order_step       # scale the search grid to this product's order size
    return range(4 * step, 15 * step, 2 * step), range(5 * step, 21 * step, 2 * step)


def grid_search_sS(env, s_grid=None, S_grid=None, seeds=TUNE_SEEDS):
    if s_grid is None or S_grid is None:
        s_grid, S_grid = _grid(env)
    table, best = {}, (None, None, -np.inf)
    for s in s_grid:
        for S in S_grid:
            if S > s:
                table[(s, S)] = p = evaluate_policy(env, lambda e: s_S_policy(e, s, S), seeds=seeds)
                if p > best[2]:
                    best = (s, S, p)
    return best[0], best[1], best[2], table


S_GRID_LOW = S_GRID_HIGH = None     # None = scale the grid from the env's order size


def context(env, obs):
    _, dow, promo_soon = env.decode_state(obs)
    return dow * 2 + promo_soon


def adjusted_action(env, s, S, offset):
    pos = env.position
    return 0 if pos > s + offset else env.nearest_action(S + offset - pos)


def train_residual_q(env, s, S, offsets=OFFSETS, episodes=500, alpha=0.05, gamma=0.95,
                     eps_start=0.5, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.asarray(offsets) * env.order_step / 6     # offsets were tuned for 6-unit steps
    Q = np.zeros((N_CONTEXTS, len(offsets)))
    Q[:, int(np.flatnonzero(offsets == 0)[0])] = 1.0
    returns = []
    for ep in range(episodes):
        eps = max(0.0, eps_start * (1 - ep / (0.7 * episodes)))
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        c, G, done = context(env, obs), 0.0, False
        while not done:
            k = int(rng.integers(len(offsets))) if rng.random() < eps else int(np.argmax(Q[c]))
            obs, r, terminated, truncated, _ = env.step(adjusted_action(env, s, S, offsets[k]))
            c2 = context(env, obs)
            Q[c, k] += alpha * ((r if terminated else r + gamma * Q[c2].max()) - Q[c, k])
            c, G, done = c2, G + r, terminated or truncated
        returns.append(G)
    return Q, returns


def residual_policy(env, Q, s, S, offsets=OFFSETS):
    offsets = np.asarray(offsets) * env.order_step / 6
    return lambda obs: adjusted_action(env, s, S, offsets[int(np.argmax(Q[context(env, obs)]))])


if __name__ == "__main__":
    hist = load_demand_csv(CSV_PATH)
    n_train = 365
    rlhud.plot(hist["demand"], label="daily demand", title="Full history · train | test split at day 365",
               smooth=7, xlabel="day")
    res = run_pipeline(CSV_PATH, n_train=n_train)
    s, S = res["s_S"]
    rlhud.stat("Tuned on year 1: (s, S)", f"({s}, {S})")
    rows = []
    for k in ("profit", "fill_rate", "lost_sales", "spoiled", "orders"):
        b, a = res["baseline"][k], res["agent"][k]
        rows.append([k, round(b, 3), round(a, 3), f"{100 * (a / b - 1):+.1f} %" if b else "—"])
    rlhud.table(rows, headers=["test year (unseen)", "planner (s, S)", "residual Q", "change"],
                title="Out-of-time evaluation")
    up = res["agent"]["profit"] / res["baseline"]["profit"] - 1
    rlhud.stat("Profit uplift on the test year", 100 * up, unit=" %", good=up > 0)
    rlhud.stat("Agent fill rate", res["agent"]["fill_rate"], good=res["agent"]["fill_rate"] >= 0.9)

    # what does the agent do differently? learned (s, S) shift per context
    Q = res["Q"]
    shift = [float(OFFSETS[np.argmax(Q[c])]) for c in range(N_CONTEXTS)]
    rlhud.bars(shift, labels=[f"{DOW_NAMES[c // 2]}{'+P' if c % 2 else ''}" for c in range(N_CONTEXTS)],
               title="Learned shift of (s, S) per context (+P = promo soon)")
    if up <= 0:
        rlhud.text("On this data the agent did not beat the planner out of time. That is a real, "
                   "useful result: check the train/test demand shift and the size of the uplift "
                   "on the training year before trusting any RL policy.", title="Honest result", tone="warn")
