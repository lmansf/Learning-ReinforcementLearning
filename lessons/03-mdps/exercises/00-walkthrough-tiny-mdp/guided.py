#> `itertools.product` enumerates combinations; we use it to list every policy at the end.
import itertools

import numpy as np
import rlhud
from rlenvs import GridWorld


#> A hand-written MDP: a robot with a **low** (0) or **high** (1) battery, and a terminal
#> **dead** state (2). Actions: 0 = wait (recharge), 1 = search (earn reward, drain).
#> Making it a class with `n_states`, `n_actions` and `P` gives it the same interface as
#> `GridWorld`, so the same code works on both.
class RobotMDP:
    #> Press Enter after this line and the model `P` fills in. `P[s][a]` is a **list of
    #> outcomes** `(prob, next_state, reward, terminated)`, exactly like `env.P` in
    #> `rlenvs`; the probabilities in one list add up to 1.
    n_states, n_actions = 3, 2
    #= given
    P = {
        0: {0: [(1.0, 1, 0.0, False)],
            1: [(0.7, 0, 1.0, False), (0.3, 2, -3.0, True)]},
        1: {0: [(1.0, 1, 0.0, False)],
            1: [(0.6, 1, 2.0, False), (0.4, 0, 2.0, False)]},
        2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
    }
    #= end


#> Flatten the dict-of-lists into two arrays: expected reward `R[s, a]` and the
#> transition tensor `T[s, a, s']`. Once the model is in arrays, NumPy does the maths.
def model_arrays(env):
    n, k = env.n_states, env.n_actions
    #> `np.zeros(shape)` with a **tuple** of sizes: 2-D for R, 3-D for T.
    R = np.zeros((n, k))
    T = np.zeros((n, k, n))
    for s in range(n):
        for a in range(k):
            #> The loop of every planning exercise: unpack each outcome tuple.
            for prob, s2, r, done in env.P[s][a]:
                #> `+=`, never `=`: several outcomes contribute to the same entry.
                R[s, a] += prob * r
                #> A terminating transition has **no future**: it pays its reward but adds
                #> nothing to T, so γ·V(s') never gets added for it.
                if not done:
                    T[s, a, s2] += prob
    return R, T


#> All q-values in one line: `q(s, a) = R[s, a] + γ Σ_s' T[s, a, s'] V(s')`.
#> `T @ V` contracts the **last** axis of T with V, giving shape (n, k). The same thing
#> spelled out with indices is `np.einsum("ijk,k->ij", T, V)`.
def q_values(R, T, V, gamma):
    return R + gamma * T @ V


#> Exact v_π of a **deterministic** policy (`actions[s]` = the action taken in s), by
#> solving the linear system `(I − γ T_π) v = r_π`.
def evaluate(R, T, actions, gamma):
    #> `np.arange(n)` → [0, 1, …, n-1]. Indexing with two arrays picks one entry per pair:
    #> `R[states, actions]` is `[R[0, actions[0]], R[1, actions[1]], …]`.
    states = np.arange(len(actions))
    r_pi = R[states, actions]
    #> The same fancy index on T picks one (n,) row per state → an (n, n) matrix.
    T_pi = T[states, actions]
    #> `np.eye(n)` is the identity matrix. `np.linalg.solve(A, b)` returns x with A x = b;
    #> it is faster and more accurate than computing the inverse.
    return np.linalg.solve(np.eye(len(actions)) - gamma * T_pi, r_pi)


if __name__ == "__main__":
    env, gamma = RobotMDP(), 0.9
    R, T = model_arrays(env)
    rows = []
    #> `product(range(2), repeat=2)` yields (0, 0), (0, 1), (1, 0), (1, 1): one action for
    #> each of the two non-terminal states. (The dead state's action does not matter.)
    for choice in itertools.product(range(env.n_actions), repeat=2):
        #> All zeros, then write the chosen actions into states 0 and 1 in one go.
        actions = np.zeros(env.n_states, dtype=int)
        actions[[0, 1]] = choice
        V = evaluate(R, T, actions, gamma)
        rows.append([str(choice), *np.round(V[:2], 3).tolist()])
    #= given
    rlhud.table(rows, headers=["(a_low, a_high)", "v(low)", "v(high)"],
                title="Every deterministic policy (0 = wait, 1 = search)")
    V = evaluate(R, T, np.array([0, 1, 0]), gamma)
    Q = q_values(R, T, V, gamma)
    rlhud.bars(Q[0], labels=["wait", "search"], title="q(low, a): wait low, search high")
    rlhud.bars(Q[1], labels=["wait", "search"], title="q(high, a) for the same policy")
    #= end
    #> `np.argmax(Q, axis=1)` is the argmax of **each row**: the greedy action per state.
    print("greedy w.r.t. this policy's q:", np.argmax(Q, axis=1)[:2])
    #> Nothing above was specific to the robot: any env with `P` works. A slippery
    #> GridWorld has 9 states × 4 actions, and `np.full(n, 1)` means "always RIGHT".
    world = GridWorld(["..G", ".#X", "S.."], slip=0.2, step_reward=-0.04)
    Rg, Tg = model_arrays(world)
    Vg = evaluate(Rg, Tg, np.full(world.n_states, 1), gamma)
    #> `to_grid` reshapes the flat vector to rows × cols; `walls_mask` and `labels` draw
    #> the map. The same call shows your results in the next exercises.
    rlhud.grid(world.to_grid(Vg), title="GridWorld: v_π of always RIGHT",
               walls=world.walls_mask(), labels=world.labels())
