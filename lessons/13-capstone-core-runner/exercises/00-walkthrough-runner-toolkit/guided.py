import numpy as np
import rlhud
from corerunner import CoreRunner


#> A **linear schedule**: `start` at t = 0, `end` from t = n on, a straight line between.
#> Use it for ε (or a learning rate) against episodes, steps or budget used.
def linear(t, n, start, end):
    #> `min(…, 1.0)` freezes the value once t ≥ n; `max(1, n)` avoids dividing by zero.
    frac = min(t / max(1, n), 1.0)
    return start + frac * (end - start)


#> argmax with **random tie-breaking**. On an all-zero table plain `np.argmax` always
#> returns 0 — in Core Runner that is "UP" every time, which explores nothing.
def argmax_random(row, rng):
    #> `np.flatnonzero(mask)` → the indices where mask is True: here every tied maximum.
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best))


#> Play one episode. Returns the return, the replay frames, the final info and a map of
#> visits per cell. (`corerunner.run_episode` is similar — now you know what is inside.)
def play(env, policy, seed=None):
    #> With the default `obs="tabular"` the observation IS the state index: an int in
    #> `[0, env.n_states)`, ready to pick a table row like `Q[s]`.
    s, info = env.reset(seed=seed)
    #> `env.snapshot()` is a JSON-friendly picture of the board. Record the start, too.
    frames = [env.snapshot()]
    #> One counter per grid cell, shaped like the map: `(rows, cols)`.
    visits = np.zeros((env.h, env.w), dtype=int)
    G = 0.0
    while True:
        a = policy(s)
        s, r, terminated, truncated, info = env.step(a)
        G += r
        frames.append(env.snapshot())
        #> `env.pos` is a `(row, col)` TUPLE, and a tuple indexes ONE cell of a 2-D array.
        visits[env.pos] += 1
        #> `terminated`: core run won or lost. `truncated`: the battery ran out.
        if terminated or truncated:
            break
    return G, frames, info, visits


#> **Evaluation**: mean return and success rate of a policy over a fixed list of seeds.
def evaluate(env, policy, seeds):
    returns, escapes = [], 0
    for seed in seeds:
        G, _, info, _ = play(env, policy, seed)
        returns.append(G)
        #> `info["event"]` says HOW it ended: "escaped", "vent", "caught", "battery", …
        #> `True` counts as 1 when added to an int.
        escapes += info["event"] == "escaped"
    return float(np.mean(returns)), escapes / len(seeds)


if __name__ == "__main__":
    env = CoreRunner(level=1)
    rng = np.random.default_rng(0)
    #> The table shape every tabular agent uses: one row per state, one column per action.
    N = np.zeros((env.n_states, env.n_actions))

    #> A count-based explorer: random with probability ε, else the LEAST-tried action.
    #> The closure reads `eps` and `N` from the enclosing scope at call time.
    def curious(s):
        if rng.random() < eps:
            a = int(rng.integers(env.n_actions))
        else:
            #> Negate the counts: the argmax of −N[s] is the least-tried action.
            a = argmax_random(-N[s], rng)
        #> Write into the table with a `[row, col]` index pair.
        N[s, a] += 1
        return a
    #= given

    returns, heat = [], np.zeros((env.h, env.w))
    #= end
    for ep in range(60):
        #> ε falls from 1.0 to 0.1 over the first 40 episodes, then stays there.
        eps = linear(ep, 40, 1.0, 0.1)
        G, frames, info, visits = play(env, curious)
        #= given
        returns.append(G)
        heat += visits
    rlhud.plot(returns, label="count-based explorer", title="Return / episode", smooth=10)
    rlhud.grid(heat, title="Visits per cell over 60 episodes", fmt=".0f", cmap="heat")
    rlhud.stat("States ever tried", int((N.sum(axis=1) > 0).sum()))
    rlhud.stat("Table shape", str(N.shape))
    mean, rate = evaluate(env, lambda s: argmax_random(-N[s], rng), seeds=range(5))
    rlhud.stat("Least-tried policy: mean return", mean)
    rlhud.replay(frames, title=f"Last episode · {info['event']}")
    #= end
