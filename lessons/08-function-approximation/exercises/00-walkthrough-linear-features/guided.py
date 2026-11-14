#> NumPy for arrays and maths, `rlhud` for the VISUALS tab.
import numpy as np
import rlhud


#> **Bucketing**: which of `n` equal bins on `[low, high]` holds the value `x`?
def bucket(x, low, high, n):
    #> Rescale so that one bin has width 1: `low` → 0 and `high` → n. Because it is plain
    #> arithmetic, it works on a single number *and* on an array of coordinates.
    scaled = (np.asarray(x, dtype=float) - low) / (high - low) * n
    #> `np.floor` rounds DOWN (−0.5 → −1; plain `int()` would truncate to 0).
    #> `.astype(int)` turns the float result into integer indices.
    idx = np.floor(scaled).astype(int)
    #> `x == high` lands in bin `n` and points below `low` in bin −1 — both out of range.
    #> `np.clip` pins every index into `0 … n−1`.
    return np.clip(idx, 0, n - 1)


#> **State aggregation** as features: a one-hot vector x(s) with a 1 in the bin of s.
def one_hot_features(s, low, high, n):
    #> A fresh float vector on every call — reusing one array for two states is a
    #> classic bug.
    x = np.zeros(n)
    x[bucket(s, low, high, n)] = 1.0
    return x


#> The linear prediction v̂ = wᵀx. For two 1-D arrays `w @ x` is the dot product.
def predict(w, x):
    #> `float(...)` turns the 0-d NumPy result into a plain Python number.
    return float(w @ x)


#> One SGD step on ½(y − wᵀx)². Its gradient w.r.t. w is −(y − wᵀx)·x, so we step
#> along +x.
def sgd_step(w, x, y, alpha):
    #> Compute the error BEFORE changing w.
    err = y - predict(w, x)
    #> `+=` updates the caller's array **in place**; `w = w + …` would create a new array
    #> that the caller never sees. Only weights whose feature xᵢ ≠ 0 move.
    w += alpha * err * x
    return err


#> Binary features don't need the full vector: keep the indices of the 1s instead.
#> `W` has one weight ROW per output, shape `(n_outputs, n_features)`.
def sparse_predict(W, idx):
    #> `W[:, idx]` keeps the active columns → `(n_outputs, len(idx))`. Summing each row
    #> (`axis=1`) gives every output's wᵀx at once.
    return W[:, idx].sum(axis=1)


#> The matching update for ONE output row `k`: every active weight moves by α·err.
def sparse_step(W, idx, k, y, alpha):
    err = y - W[k, idx].sum()
    #> Fancy-index assignment writes straight into W. (Safe because `idx` has no repeats.)
    W[k, idx] += alpha * err
    return err


#> A cell of a 2-D grid as ONE integer: bucket each coordinate, then flatten row-major.
def grid_index(point, low, high, n):
    #> `bucket` is vectorised: 2 coordinates in, 2 bin indices out (`low`/`high` are arrays).
    cell = bucket(point, np.asarray(low, float), np.asarray(high, float), n)
    #> `np.ravel_multi_index((r, c), (n, n))` = r·n + c. It wants a TUPLE of indices.
    return int(np.ravel_multi_index(tuple(cell), (n, n)))


#> Everything below runs on RUN, but not when CHECK imports your functions.
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    #> 10 weights, one per bucket of [0, 1]: a staircase approximation of a sine.
    w = np.zeros(10)
    for _ in range(4000):
        s = rng.random()
        #> A noisy target — like a sampled return, it is right only on average.
        y = np.sin(2 * np.pi * s) + 0.2 * rng.normal()
        sgd_step(w, one_hot_features(s, 0.0, 1.0, 10), y, alpha=0.05)
    #= given
    xs = np.linspace(0.0, 1.0, 101)
    title = "Linear fit on 10 aggregated states"
    rlhud.plot(np.sin(2 * np.pi * xs), x=xs, label="target sin(2πs)", title=title)
    rlhud.plot([predict(w, one_hot_features(s, 0.0, 1.0, 10)) for s in xs], x=xs,
               label="wᵀx(s)", title=title)
    rlhud.bars(w, labels=[f"b{i}" for i in range(10)], title="One weight per bucket")
    #= end
    #> Two outputs (sin and cos) that share the same sparse features.
    W = np.zeros((2, 10))
    for _ in range(4000):
        s = rng.random()
        #> An index ARRAY (here of length 1) works exactly like a list of active features.
        idx = np.array([bucket(s, 0.0, 1.0, 10)])
        sparse_step(W, idx, 0, np.sin(2 * np.pi * s), 0.05)
        sparse_step(W, idx, 1, np.cos(2 * np.pi * s), 0.05)
    #= given
    print("both outputs at s = 0.3:", np.round(sparse_predict(W, np.array([3])), 3))
    cells = [[grid_index([x, y], [0, 0], [1, 1], 4) for y in np.linspace(0.05, 0.95, 4)]
             for x in np.linspace(0.05, 0.95, 4)]
    rlhud.grid(cells, title="grid_index on a 4×4 grid (row-major numbering)", fmt=".0f")
    #= end
