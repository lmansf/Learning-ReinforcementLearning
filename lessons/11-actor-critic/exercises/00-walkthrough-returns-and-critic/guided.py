import numpy as np
import rlhud


#> A rollout of T steps stores `values[t]` = V(sₜ). Many formulas need V(sₜ₊₁) lined up
#> with step t: shift left by one and append the value of the state AFTER the rollout.
def shifted_values(values, last_value):
    #> `np.append` returns a NEW array (it never modifies `values`); the length stays T.
    return np.append(np.asarray(values, dtype=float)[1:], last_value)


#> Discounted returns for a rollout that may hold SEVERAL episodes:
#> Gₜ = rₜ + γ (1 − doneₜ) Gₜ₊₁, starting from G_T = last_value (the critic's guess).
def rollout_returns(rewards, dones, last_value, gamma):
    rewards = np.asarray(rewards, dtype=float)
    #> Floats, so `1.0 - dones` is exactly 0.0 or 1.0 (bools work too, but be explicit).
    dones = np.asarray(dones, dtype=float)
    out = np.zeros(len(rewards))
    #> The running sum starts from the bootstrap value of the state after the rollout.
    G = float(last_value)
    #> Backwards: each return is built from the NEXT one, so walk t = T−1, …, 1, 0.
    for t in reversed(range(len(rewards))):
        #> `(1 − done)` cuts the chain: when an episode ended at step t, nothing from the
        #> next episode may leak back into it.
        G = rewards[t] + gamma * (1.0 - dones[t]) * G
        out[t] = G
    return out


#> A **linear critic** V(s) = x(s)ᵀw, fitted by gradient descent on ½·mean((V − target)²).
def critic_step(w, X, targets, lr):
    #> One prediction per row of X `(N, d)` → shape `(N,)`.
    v = X @ w
    #> Convention: err = prediction − target, so ∂(½ err²)/∂v = +err.
    err = v - targets
    #> `X.T @ err` sums xᵢ·errᵢ over the rows; divide by N for the mean gradient.
    #> `-=` because we MINIMISE a loss — and in place, so the caller's `w` changes.
    w -= lr * X.T @ err / len(X)
    return float(0.5 * np.mean(err ** 2))


#> Networks output a COLUMN `(N, 1)`. Subtracting a flat `(N,)` target from it broadcasts
#> to an `(N, N)` matrix — no error, just silently wrong numbers.
def column_error(v_col, targets):
    #> `reshape(-1, 1)` turns `(N,)` into `(N, 1)`, so the subtraction is row by row.
    return v_col - np.asarray(targets, dtype=float).reshape(-1, 1)


if __name__ == "__main__":
    #> A hand-made rollout of 8 steps: reward 1 per step, an episode ends at step 2.
    rewards = np.ones(8)
    dones = np.zeros(8)
    dones[2] = 1.0
    G = rollout_returns(rewards, dones, last_value=5.0, gamma=0.9)
    #= given
    rlhud.bars(G, labels=[f"t{t}" for t in range(8)], highlight=2,
               title="Returns (episode ends at t = 2, then bootstrap from 5.0)")
    print("V(s_t+1) lined up:", shifted_values(np.arange(8.0), 10.0))
    #= end
    #> Critic data: a corridor of 6 cells walked left to right, many times (6 episodes).
    pos = np.tile(np.arange(6), 6)
    ends = (pos == 5).astype(float)
    targets = rollout_returns(np.ones(len(pos)), ends, last_value=0.0, gamma=0.9)
    #> Features x(s) = [position / 5, 1]: `np.c_` stacks columns side by side → `(36, 2)`.
    X = np.c_[pos / 5.0, np.ones(len(pos))]
    w = np.zeros(2)
    losses = [critic_step(w, X, targets, lr=0.5) for _ in range(300)]
    #= given
    rlhud.plot(losses, label="critic loss", title="Linear critic: ½·MSE", xlabel="step")
    title = "Corridor values: targets vs linear critic"
    rlhud.plot(targets[:6], label="return Gₜ", title=title, xlabel="cell")
    rlhud.plot(X[:6] @ w, label="xᵀw", title=title)
    print("column error shape:", column_error(np.zeros((4, 1)), np.ones(4)).shape)
    #= end
