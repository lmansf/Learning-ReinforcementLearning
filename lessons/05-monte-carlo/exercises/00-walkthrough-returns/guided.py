#> `defaultdict(list)` is a dict that creates an empty list the first time you touch a
#> missing key, so `returns[s].append(G)` just works.
from collections import defaultdict

import numpy as np
import rlhud
from rlenvs import GridWorld


#> A random policy that also takes the generator: `rng.integers(4)` is 0, 1, 2 or 3.
def wander(obs, rng):
    return int(rng.integers(4))


#> Play one episode and keep **everything**: a list of `(state, action, reward)`
#> triples, where the reward is the one that *followed* the action:
#> `[(S0, A0, R1), (S1, A1, R2), …]`.
def record_episode(env, policy, rng, seed=None):
    obs, _ = env.reset(seed=seed)
    episode = []
    while True:
        action = policy(obs, rng)
        #> Keep `obs` until the triple is stored: it is the state the action was taken in.
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode.append((obs, action, reward))
        if terminated or truncated:
            return episode
        obs = next_obs


#> The return from **every** time step at once, by walking the episode **backwards**:
#> `G_t = R_{t+1} + γ·G_{t+1}`. One pass, no nested sums.
def returns_to_go(episode, gamma):
    #> `[0.0] * n` makes a list of n zeros that we fill from the end.
    out = [0.0] * len(episode)
    G = 0.0
    #> `reversed(range(n))` counts n-1, n-2, …, 0.
    for t in reversed(range(len(episode))):
        #> Tuple unpacking with `_` for the parts we don't need.
        _, _, r = episode[t]
        G = r + gamma * G
        out[t] = G
    #> Check: `out[0]` equals the forward sum
    #> `sum(gamma**t * r for t, (_, _, r) in enumerate(episode))`.
    return out


#> **First-visit** bookkeeping: append the return of the first time each key appears.
#> With `by_action=True` the key is the pair `(s, a)` instead of `s` — tuples are
#> hashable, so they make perfect dict keys (that is how you'll index Q).
def add_first_visits(episode, gamma, returns, by_action=False):
    #> A `set` answers "seen it before?" in constant time.
    seen = set()
    #> `zip` walks the triples and their returns side by side.
    for (s, a, _), G in zip(episode, returns_to_go(episode, gamma)):
        key = (s, a) if by_action else s
        if key in seen:
            continue
        seen.add(key)
        returns[key].append(G)


#> Average each list. A **dict comprehension** builds `{key: value}` in one line;
#> `float(...)` turns NumPy's scalar into a plain number.
def averages(returns):
    return {key: float(np.mean(gs)) for key, gs in returns.items()}


if __name__ == "__main__":
    env = GridWorld(["..G", ".#X", "S.."], slip=0.1, step_reward=-0.04)
    rng = np.random.default_rng(0)
    #> Two separate tables: one keyed by state, one by (state, action).
    returns_v, returns_q = defaultdict(list), defaultdict(list)
    #> Seed only the **first** reset; afterwards the env's own RNG keeps going.
    for i in range(3000):
        episode = record_episode(env, wander, rng, seed=0 if i == 0 else None)
        add_first_visits(episode, 0.95, returns_v)
        add_first_visits(episode, 0.95, returns_q, by_action=True)
    V = averages(returns_v)
    Q = averages(returns_q)
    #> `dict.get(key, default)` avoids a KeyError for states that were never visited.
    values = [V.get(s, 0.0) for s in range(env.n_states)]
    #= given
    rlhud.grid(env.to_grid(values), title="First-visit MC: v of the random policy",
               walls=env.walls_mask(), labels=env.labels())
    rlhud.plot(returns_v[env.start], label="G from start", smooth=100,
               title="Sampled returns from the start state", xlabel="episode")
    names = ["UP", "RIGHT", "DOWN", "LEFT"]
    rlhud.bars([Q.get((env.start, a), 0.0) for a in range(4)], labels=names,
               title="Q(start, a) from the same episodes")
    print(f"{len(V)} states and {len(Q)} (state, action) pairs seen")
    #= end
