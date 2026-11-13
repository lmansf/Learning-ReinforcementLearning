import numpy as np
import rlhud
from inventory import InventoryEnv, s_S_policy, run_policy, evaluate_policy, DOW_NAMES

# The planner's tuned policy from exercise 1 — RL learns *adjustments* to it.
BASE_s, BASE_S = 50, 70
OFFSETS = np.array([-20, -10, 0, 10, 20, 30, 40])   # shift both s and S by this many units
N_CONTEXTS = 14                                      # 7 weekdays × promo-soon flag


def context(env, obs):
    """Map the env's tabular observation to a small context index in [0, 14):
    dow * 2 + promo_soon. Use env.decode_state(obs) -> (position_bucket, dow, promo_soon).
    The position is deliberately dropped: the (s, S) rule already reacts to it."""
    # TODO
    raise NotImplementedError


def adjusted_action(env, s, S, offset):
    """The env action of an (s + offset, S + offset) policy in the env's current state:
    order nothing while env.position > s + offset, otherwise order up to S + offset
    (env.nearest_action(units) converts a number of units into an action index)."""
    # TODO
    raise NotImplementedError


def train_residual_q(env, s=BASE_s, S=BASE_S, offsets=OFFSETS, episodes=500, alpha=0.05,
                     gamma=0.95, eps_start=0.5, seed=0):
    """Tabular Q-learning over (context, offset). Returns (Q, returns).

    - Q has shape (N_CONTEXTS, len(offsets)); start at 0 but give the zero-offset column a
      small head start (+1.0) so that, before learning anything, the agent IS the planner.
    - ε decays linearly from eps_start to 0 over the first 70% of the episodes.
    - Each episode: env.reset(seed=<a fresh int from rng>), then step until truncated."""
    rng = np.random.default_rng(seed)
    offsets = np.asarray(offsets)
    Q = np.zeros((N_CONTEXTS, len(offsets)))
    # TODO: head start for the zero-offset column
    returns = []
    for ep in range(episodes):
        # TODO: ε schedule, reset with a fresh seed, run one episode:
        #   k = ε-greedy offset index for the current context
        #   step with adjusted_action(env, s, S, offsets[k])
        #   Q-learning update on Q[c, k]; accumulate the episode's profit
        raise NotImplementedError
    return Q, returns


def residual_policy(env, Q, s=BASE_s, S=BASE_S, offsets=OFFSETS):
    """Greedy policy obs -> action: pick the best offset for the context, apply it."""
    offsets = np.asarray(offsets)
    # TODO: return a function obs -> action
    raise NotImplementedError


def train_plain_q(env, episodes=300, alpha=0.1, gamma=0.9, seed=0):
    """For comparison: Q-learning from scratch on the full tabular state (position too)."""
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, env.n_actions), 400.0)        # optimistic start
    returns = []
    for ep in range(episodes):
        eps = max(0.02, 1 - ep / (0.7 * episodes))
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        G, done = 0.0, False
        while not done:
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, terminated, truncated, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s, G, done = s2, G + r, terminated or truncated
        returns.append(G)
    return Q, returns


TEST_SEEDS = range(100, 120)

if __name__ == "__main__":
    env = InventoryEnv()
    Q, returns = train_residual_q(env, seed=0)
    Qp, returns_plain = train_plain_q(env, seed=0)
    rlhud.plot(returns, label="residual Q (offsets on (s, S))", title="Training profit per simulated year", smooth=20)
    rlhud.plot(returns_plain, label="plain Q from scratch", title="Training profit per simulated year", smooth=20)

    base = evaluate_policy(env, lambda e: s_S_policy(e, BASE_s, BASE_S), seeds=TEST_SEEDS)
    agent = evaluate_policy(env, lambda e: residual_policy(e, Q), seeds=TEST_SEEDS)
    plain = evaluate_policy(env, lambda e: (lambda o: int(np.argmax(Qp[o]))), seeds=TEST_SEEDS)
    rlhud.stat("Planner (s, S) · test profit", base)
    rlhud.stat("Plain Q · test profit", plain, good=plain > base)
    rlhud.stat("Residual Q · test profit", agent, good=agent > base)
    rlhud.stat("Uplift vs planner", 100 * (agent / base - 1), unit=" %", good=agent > base)

    best = [int(OFFSETS[np.argmax(Q[c])]) for c in range(N_CONTEXTS)]
    rlhud.bars(best, labels=[f"{DOW_NAMES[c // 2]}{'+P' if c % 2 else ''}" for c in range(N_CONTEXTS)],
               title="Learned shift of (s, S) per context (+P = promo soon)", ylabel="units")

    _, tb, trb = run_policy(env, s_S_policy(env, BASE_s, BASE_S), seed=100, record=True)
    _, ta, tra = run_policy(env, residual_policy(env, Q), seed=100, record=True)
    days = slice(140, 196)
    title = "Test year 100 · 8 weeks: demand vs stock after sales"
    rlhud.plot(tra["demand"][days], label="demand", title=title, xlabel="day")
    rlhud.plot(trb["inventory"][days], label="planner stock", title=title)
    rlhud.plot(tra["inventory"][days], label="agent stock", title=title)

    rows = []
    for key in ("revenue", "cogs", "waste", "order_fee", "holding", "stockout"):
        rows.append([key, round(trb["costs"][key]), round(tra["costs"][key])])
    rows += [["lost sales (units)", tb["lost"], ta["lost"]],
             ["spoiled (units)", tb["spoiled"], ta["spoiled"]],
             ["orders placed", tb["orders"], ta["orders"]],
             ["fill rate", round(tb["sales"] / tb["demand"], 3), round(ta["sales"] / ta["demand"], 3)],
             ["profit", round(tb["profit"]), round(ta["profit"])]]
    rlhud.table(rows, headers=["test year 100", "planner (s, S)", "residual Q"], title="Cost breakdown")
