import numpy as np
import rlhud
from rlenvs import GridWorld


#> **Offline n-step targets** for a finished episode. `rewards[t]` is the reward that
#> followed step t (R_{t+1}) and `values[t]` is the current V(S_t). For each t: up to n
#> rewards, then bootstrap from V **only if the window ends before the episode does**.
def nstep_targets(rewards, values, n, gamma):
    T = len(rewards)
    targets = np.zeros(T)
    for t in range(T):
        #> `min` clips the window at the end of the episode — the index gotcha of n-step.
        end = min(t + n, T)
        #> A slice `[t:end]` takes elements t … end-1 (end itself is excluded).
        window = rewards[t:end]
        #> `enumerate` pairs each reward with its offset i, giving the γ⁰, γ¹, … weights.
        G = sum(gamma ** i * r for i, r in enumerate(window))
        #> Bootstrap only when the window stopped early; past the end there is no V.
        if end < T:
            G += gamma ** (end - t) * values[end]
        targets[t] = G
    return targets


#> An **eligibility trace** over a sequence of visited states: each visit bumps the
#> state's trace by 1, and every step all traces fade by γλ. Returns one row per step.
def trace_history(visits, n_states, gamma, lam):
    e = np.zeros(n_states)
    rows = []
    for s in visits:
        e[s] += 1.0
        #> `.copy()` matters: appending `e` itself would store the same array every time,
        #> and the in-place `*=` below would change all rows at once.
        rows.append(e.copy())
        #> A vector operation: every entry of `e` is scaled in one go, no loop needed.
        e *= gamma * lam
    return np.array(rows)


#> A **model** for planning: a dict that remembers the last outcome of each `(s, a)`.
class Memory:
    def __init__(self, seed=0):
        self.model = {}
        self.rng = np.random.default_rng(seed)

    #> Tuple key in, tuple value out. Storing again simply overwrites: in a deterministic
    #> world, the last outcome is *the* outcome.
    def store(self, s, a, r, s2, done):
        self.model[(s, a)] = (r, s2, done)

    #> Pick a remembered pair **uniformly**: `list(dict)` lists its keys, and
    #> `rng.integers(len(keys))` a random index. (`rng.choice` on a list of tuples would
    #> turn it into a 2-D array: indexing a plain list keeps the tuples intact.)
    def sample(self):
        keys = list(self.model)
        s, a = keys[self.rng.integers(len(keys))]
        r, s2, done = self.model[(s, a)]
        return s, a, r, s2, done


if __name__ == "__main__":
    rewards, values = [0.0, 0.0, 1.0, 0.0, 2.0], [0.5, 0.5, 0.5, 0.5, 0.5]
    for n in [1, 2, 5]:
        G = nstep_targets(rewards, values, n, 0.9)
        rlhud.plot(G, label=f"n = {n}", title="n-step targets along one episode",
                   xlabel="t")
    #> Visits to states 0, 1, 1, 2, 4: watch the trace of state 1 pile up, then fade.
    E = trace_history([0, 1, 1, 2, 4], 5, 1.0, 0.8)
    rlhud.grid(E, title="Trace e(s) per step (rows = time)", cmap="heat", fmt=".2f")
    #= given
    env, mem = GridWorld(["S..", ".#.", "..G"]), Memory(seed=0)
    rng = np.random.default_rng(0)
    for episode in range(20):
        s, _ = env.reset()
        while True:
            a = int(rng.integers(4))
            s2, r, terminated, truncated, _ = env.step(a)
            mem.store(s, a, r, s2, terminated)
            if terminated or truncated:
                break
            s = s2
    picks = [mem.sample()[:2] for _ in range(2000)]
    rlhud.stat("Remembered (s, a) pairs", len(mem.model))
    rlhud.stat("Distinct pairs sampled", len(set(picks)))
    #= end
