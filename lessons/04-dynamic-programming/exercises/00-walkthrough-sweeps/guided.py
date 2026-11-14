import numpy as np
import rlhud
from rlenvs import GridWorld


#> The value of taking action `a` in `s`, then continuing with values `V`:
#> `Σ prob · (r + γ·V(s'))` over the outcomes in `env.P[s][a]`.
def backup(env, V, s, a, gamma):
    total = 0.0
    #> Each outcome is a tuple `(prob, next_state, reward, terminated)`: unpack it.
    for prob, s2, r, done in env.P[s][a]:
        #> The conditional expression `x if cond else y` drops γ·V(s') after termination.
        total += prob * (r + (0.0 if done else gamma * V[s2]))
    return total


#> One **sweep** for a *deterministic* policy (`actions[s]` = the action in s).
#> `in_place=True` writes every new value straight into V, so later states already see it.
def sweep(env, V, actions, gamma, in_place=True):
    #> The aliasing gotcha: `out = V` is the **same** array (in place); `V.copy()` is a
    #> second array, so every backup reads only the old values (the "two-array" version).
    out = V if in_place else V.copy()
    #> Δ = the largest change in this sweep. Track it state by state: after an in-place
    #> write the old value is gone, so you cannot compute it afterwards.
    delta = 0.0
    for s in range(env.n_states):
        #> Read from `V`: in place it *is* `out` (fresh values), otherwise the old copy.
        v_new = backup(env, V, s, actions[s], gamma)
        delta = max(delta, abs(v_new - out[s]))
        out[s] = v_new
    return out, delta


#> Repeat sweeps until Δ < θ. Return the values and the list of Δs (its length is the
#> number of sweeps), handy for a convergence plot.
def evaluate_until(env, actions, gamma, theta=1e-8, in_place=True, max_sweeps=10_000):
    #> A fresh float array we own, so in-place writes never touch anyone else's data.
    V = np.zeros(env.n_states)
    deltas = []
    #> A bounded `for` loop with `break` instead of `while True`: a bug can't hang it.
    for _ in range(max_sweeps):
        V, delta = sweep(env, V, actions, gamma, in_place)
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas


#> Policy extraction: the best action in every state given V, as a **one-hot** array.
def greedy_one_hot(env, V, gamma):
    #> A nested list comprehension builds the whole (n_states, n_actions) q table.
    q = np.array([[backup(env, V, s, a, gamma) for a in range(env.n_actions)]
                  for s in range(env.n_states)])
    #> `np.argmax(q, axis=1)` picks the best column of every row (ties → lowest index).
    #> Indexing the identity matrix with it, `np.eye(k)[actions]`, gives one-hot rows.
    return np.eye(env.n_actions)[np.argmax(q, axis=1)]


if __name__ == "__main__":
    env = GridWorld(["...G", ".#.X", "S..."], slip=0.1, step_reward=-0.04)
    gamma = 0.9
    #> Policy: always RIGHT (action 1). `np.full(n, 1)` makes an int array of ones.
    actions = np.full(env.n_states, 1)
    V = np.zeros(env.n_states)
    for k in [1, 2, 3]:
        V, _ = sweep(env, V, actions, gamma)
        #> `id=` keeps three separate panels; `walls` and `labels` draw the map.
        rlhud.grid(env.to_grid(V), title=f"Always RIGHT · {k} sweep(s)", id=f"sw-{k}",
                   walls=env.walls_mask(), labels=env.labels())
    V, deltas = evaluate_until(env, actions, gamma)
    better = greedy_one_hot(env, V, gamma)
    #> `policy_grid` turns per-state actions (or one-hot rows) into arrows for the grid.
    rlhud.grid(env.to_grid(V), title="v_π of always RIGHT + its greedy improvement",
               policy=env.policy_grid(better), walls=env.walls_mask(),
               labels=env.labels())
    #> `np.array_equal` is True only if shape and every entry match: a clean
    #> "did the policy change?" test.
    changed = not np.array_equal(better, np.eye(env.n_actions)[actions])
    print("Greedy improvement changed the policy:", changed)
    #= given
    _, two = evaluate_until(env, actions, gamma, in_place=False)
    rlhud.plot(np.log10(np.maximum(deltas, 1e-12)), label="in place", xlabel="sweep",
               title="log₁₀ Δ per sweep")
    rlhud.plot(np.log10(np.maximum(two, 1e-12)), label="two arrays",
               title="log₁₀ Δ per sweep")
    rlhud.table([["in place", len(deltas)], ["two arrays", len(two)]],
                headers=["version", "sweeps to Δ < 1e-8"], title="In place vs two arrays")
    #= end
