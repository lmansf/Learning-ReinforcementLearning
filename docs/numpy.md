---
title: NumPy
---

## np.array
sig: `np.array(object, dtype=None)` → ndarray (always a new copy)
match: np.array(
tags: create, list to array, convert

Builds a new array from a list, tuple or another array. The dtype is inferred: all ints → `int64`, any float → `float64`.

```python
np.array([1, 2, 3])           # array([1, 2, 3])        dtype int64
np.array([1, 2, 3.0])         # array([1., 2., 3.])     dtype float64
np.array([[1, 2], [3, 4]]).shape   # (2, 2)
```

> [!WARN] Integer arrays truncate
> `Q = np.array([0, 0, 0]); Q[0] += 0.5` leaves `Q[0] == 0` — the array is integer. Start value tables with `np.zeros(n)` (float) or pass `dtype=float`.

## np.asarray
sig: `np.asarray(a, dtype=None)` → ndarray (no copy if `a` already is one)
match: np.asarray(
tags: convert, no copy, accept lists

Like `np.array`, but returns the input unchanged when it is already an ndarray of the right dtype. Use it at the top of a function so callers may pass lists *or* arrays.

```python
def mean_return(rs):
    rs = np.asarray(rs, dtype=float)
    return rs.mean()
mean_return([1, 2, 3])       # 2.0
```

> [!WARN] It may alias
> `b = np.asarray(a)` is the *same object* when `a` is a float array, so writing to `b` changes `a`. Use `np.array(a)` or `a.copy()` when you need an independent copy.

## np.zeros
sig: `np.zeros(shape, dtype=float)` → ndarray
match: np.zeros(
tags: create, Q-table, initialise

A float array of zeros. The usual way to create value tables: `V = np.zeros(n_states)`, `Q = np.zeros((n_states, n_actions))`.

```python
Q = np.zeros((3, 2))
Q.shape                 # (3, 2)
Q[1, 0] = 0.5           # row = state, column = action
np.zeros(4, dtype=np.int64)   # array([0, 0, 0, 0])  — e.g. visit counts
```

> [!WARN] Shape is ONE argument
> `np.zeros(3, 2)` is an error (the 2nd positional argument is `dtype`). Pass a tuple: `np.zeros((3, 2))`.

## np.ones
sig: `np.ones(shape, dtype=float)` → ndarray
match: np.ones(
tags: create, bias column

Array of ones — handy for a bias feature or a uniform policy before normalising.

```python
np.ones(3)              # array([1., 1., 1.])
np.ones((2, 4)) / 4     # uniform policy over 4 actions in 2 states
```

## np.full
sig: `np.full(shape, fill_value, dtype=None)` → ndarray
match: np.full(
tags: create, optimistic init, constant

Array filled with one value. Great for **optimistic initial values** (`np.full(k, 5.0)`) or ε-soft policies.

```python
np.full(4, 5.0)            # array([5., 5., 5., 5.])
np.full((2, 3), -np.inf)   # "nothing seen yet" placeholder
```

> [!WARN] dtype follows the fill value
> `np.full(3, 0)` is an *integer* array. Write `np.full(3, 0.0)` for floats.

## np.zeros_like
sig: `np.zeros_like(a, dtype=None)` → ndarray with `a`'s shape and dtype
match: np.zeros_like(, np.ones_like(, np.full_like(
tags: create, same shape, gradient buffers

A fresh array shaped like `a`. Typical for gradient / moment buffers: `m = [np.zeros_like(p) for p in params]`. Siblings: `np.ones_like`, `np.full_like(a, v)`.

```python
W = np.ones((2, 3))
np.zeros_like(W).shape       # (2, 3)
```

> [!WARN] Inherits the dtype
> `np.zeros_like(np.arange(3))` is an int array, so fractional updates are silently truncated. Use `np.zeros_like(x, dtype=float)` when `x` holds ints.

## np.arange
sig: `np.arange(stop)` · `np.arange(start, stop, step=1)` → ndarray
match: np.arange(
tags: range, sequence, indices

Evenly spaced values in `[start, stop)` — like Python's `range`, but returns an array you can do maths on.

```python
np.arange(4)            # array([0, 1, 2, 3])
0.9 ** np.arange(4)     # array([1.   , 0.9  , 0.81 , 0.729])  — powers of γ in one line
```

> [!WARN] Floats
> With a float step the end point is fuzzy (`np.arange(0, 1, 0.1)` may or may not include values near 1). Use `np.linspace` for float grids.

## np.linspace
sig: `np.linspace(start, stop, num=50)` → ndarray
match: np.linspace(
tags: grid, evenly spaced

`num` evenly spaced points from `start` to `stop`, **inclusive** of both ends.

```python
np.linspace(0, 1, 5)    # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

## np.logspace
sig: `np.logspace(start, stop, num=50, base=10.0)` → ndarray
match: np.logspace(
tags: log grid, learning rates, sweep

Points spaced evenly on a log scale: from `base**start` to `base**stop`. Good for sweeping step sizes or picking log-spaced episode indices to plot.

```python
np.logspace(-3, 0, 4)      # array([0.001, 0.01 , 0.1  , 1.   ])
np.unique(np.logspace(0, 3, 10).astype(int))   # array([   1,    2,    4,   10,   21,   46,  100,  215,  464, 1000])
```

> [!WARN] Exponents, not values
> `np.logspace(1, 1000)` means 10¹ … 10¹⁰⁰⁰ (overflow!). Pass exponents: `np.logspace(0, np.log10(1000), n)`.

## np.eye
sig: `np.eye(N, M=None, k=0, dtype=float)` → ndarray (identity matrix)
match: np.eye(
tags: identity, one-hot, linear system

Identity matrix. Two RL uses: `I - γ P` when solving the Bellman equation, and **one-hot vectors** by indexing a row.

```python
np.eye(3)[1]               # array([0., 1., 0.])   one-hot for action 1
np.eye(4)[[0, 2]]          # two one-hot rows, shape (2, 4)
A = np.eye(2) - 0.9 * np.array([[0, 1], [1, 0]])
```

## np.append
sig: `np.append(arr, values, axis=None)` → new ndarray
match: np.append(
tags: concatenate, bias feature, bootstrap value

Returns a **new** array with `values` added at the end (flattened when `axis=None`). Handy for adding a bias feature or the bootstrap value after the last step.

```python
np.append(np.array([0.2, 0.5]), 1.0)     # array([0.2, 0.5, 1. ])
values = np.array([1., 2., 3.])
np.append(values[1:], 9.0)               # array([2., 3., 9.])  — V(sₜ₊₁) for every t
```

> [!WARN] Not in place, and slow in loops
> Unlike `list.append`, `np.append` does not modify `arr` — assign the result. Growing an array this way inside a loop is O(n²); collect a Python list and call `np.array(lst)` once.

## np.concatenate
sig: `np.concatenate((a1, a2, ...), axis=0)` → ndarray
match: np.concatenate(
tags: join, stack, features

Joins arrays along an **existing** axis. All other dimensions must match.

```python
np.concatenate([np.zeros(2), np.ones(3)])          # array([0., 0., 1., 1., 1.])
np.concatenate([np.zeros((2, 3)), np.ones((1, 3))]).shape   # (3, 3)
```

> [!WARN] Pass a sequence
> `np.concatenate(a, b)` treats `b` as the axis. Wrap the arrays in a list or tuple.

## np.stack
sig: `np.stack(arrays, axis=0)` → ndarray (adds a NEW axis)
match: np.stack(, np.vstack(, np.hstack(
tags: batch, stack, join

Stacks same-shaped arrays along a **new** axis — e.g. a list of observations into a `(batch, obs_dim)` matrix. `np.vstack` / `np.hstack` join row-wise / column-wise.

```python
obs = [np.array([1., 2.]), np.array([3., 4.])]
np.stack(obs).shape          # (2, 2)
np.stack(obs, axis=1).shape  # (2, 2), but columns are the observations
```

## np.c_
sig: `np.c_[a, b, ...]` → arrays joined as columns
match: np.c_[
tags: columns, bias, design matrix

Index-trick that glues 1-D arrays (as columns) or 2-D arrays side by side. A quick way to append a bias column to a feature matrix.

```python
X = np.array([[1., 2.], [3., 4.]])
np.c_[X, np.ones(2)]       # array([[1., 2., 1.], [3., 4., 1.]])
np.c_[[1, 2], [3, 4]]      # array([[1, 3], [2, 4]])
```

## np.atleast_2d
sig: `np.atleast_2d(*arys)` → ndarray with ndim ≥ 2
match: np.atleast_2d(, np.atleast_1d(
tags: batch dimension, single observation

Turns a single observation of shape `(d,)` into a batch of one, `(1, d)`; leaves 2-D arrays alone. Lets one function handle both a single obs and a batch.

```python
x = np.array([0.1, 0.2, 0.3, 0.4])
np.atleast_2d(x).shape             # (1, 4)
V = lambda o: net.forward(np.atleast_2d(o))[:, 0]   # works for one obs or many
```

## .shape / .ndim / .size
sig: `a.shape` → tuple · `a.ndim` → int · `a.size` → int · `np.shape(x)`
match: .shape, np.shape(, .ndim
tags: dimensions, debug, length

`shape` is the size of each axis, `ndim` the number of axes, `size` the total element count. Printing `x.shape` is the #1 debugging tool for neural-net code.

```python
Q = np.zeros((48, 4))
Q.shape, Q.ndim, Q.size      # ((48, 4), 2, 192)
Q[0].shape                   # (4,)
np.shape([[1, 2, 3]])        # (1, 3) — works on lists too
```

> [!WARN] (n,) is not (n, 1)
> A vector `(n,)` minus a column `(n, 1)` broadcasts to an `(n, n)` matrix — no error, just wrong numbers. When a loss looks strange, check both shapes; fix with `v[:, 0]` or `t[:, None]`.

## .reshape
sig: `a.reshape(new_shape)` → view · `a.ravel()` / `a.flatten()` → 1-D
match: .reshape(, .ravel(, .flatten(
tags: shape, grid, flatten

Re-arranges the same data into another shape (row-major). Use `-1` for "whatever fits". `ravel()` flattens (a view when possible); `flatten()` always copies.

```python
V = np.arange(6)
V.reshape(2, 3)        # array([[0, 1, 2], [3, 4, 5]])  — state s at row s//3, col s%3
V.reshape(-1, 1).shape # (6, 1)
np.zeros((5, 5, 5)).ravel().shape   # (125,)
```

> [!WARN] Reshape is not transpose
> `reshape(3, 2)` of a `(2, 3)` array re-reads the numbers in order; it does not swap rows and columns. Use `.T` for that.

## .T (transpose)
sig: `a.T` → view with axes reversed
match: .T @, .T), .T]
tags: transpose, backprop, matrix

Swaps rows and columns (reverses all axes). In backprop: `dW = x.T @ dout` and `dx = dout @ W.T`.

```python
x = np.ones((32, 4)); d = np.ones((32, 2))
(x.T @ d).shape      # (4, 2)  — same shape as W
np.arange(3).T.shape # (3,)    — .T does nothing to a 1-D array
```

> [!WARN] 1-D vectors
> `.T` cannot turn `(n,)` into a column. Use `v[:, None]` or `v.reshape(-1, 1)`.

## None / np.newaxis
sig: `a[:, None]` · `a[None, :]` · `a[..., None]`  (None is np.newaxis)
match: [None, None], [:, None], np.newaxis
tags: add axis, column vector, broadcasting

Indexing with `None` inserts a length-1 axis. It is how you turn a vector into a column or row so that broadcasting lines up.

```python
v = np.array([1., 2., 3.])
v[:, None].shape        # (3, 1)
v[None, :].shape        # (1, 3)
v[:, None] * np.array([10., 20.])   # (3, 2) outer-product-style table
```

## Broadcasting
sig: shapes are compared from the right; sizes must be equal or 1
match: keepdims=True, [None, :], [:, None]
tags: shapes, vectorise, rows, columns

Arithmetic between different shapes "stretches" axes of size 1. That is how `Q - Q.max(axis=1, keepdims=True)` subtracts each row's max from that row.

```python
Q = np.array([[1., 3.], [2., 0.]])
Q - Q.max(axis=1, keepdims=True)    # array([[-2.,  0.], [ 0., -2.]])
Q * np.array([10., 1.])             # scales column 0 by 10
```

> [!WARN] Silent shape surprises
> `(N,) - (N, 1)` becomes `(N, N)`. Broadcasting never complains when the result is merely *wrong*; assert shapes in vectorised code.

## Axis and keepdims
sig: `a.sum(axis=None, keepdims=False)` (same for mean, max, min, argmax, …)
match: axis=, keepdims=
tags: reduce, rows, columns

`axis=0` collapses rows (one result per column); `axis=1` collapses columns (one per row); `axis=-1` is the last axis. `keepdims=True` keeps the reduced axis as size 1 so the result broadcasts back.

```python
Q = np.array([[1., 5.], [4., 2.]])
Q.max(axis=1)                    # array([5., 4.])    best value per state
Q.argmax(axis=1)                 # array([1, 0])      greedy action per state
Q.sum(axis=1, keepdims=True)     # array([[6.], [6.]]) shape (2, 1)
```

> [!WARN] Which axis?
> For a Q-table `(states, actions)`, "per state" is `axis=1`. Getting it wrong gives an array of the wrong length — check `.shape`.

## dtype and .astype
sig: `a.dtype` · `a.astype(dtype)` → converted copy
match: .astype(, dtype=
tags: int, float, convert, cast

Every array has one dtype. `.astype` converts (copying). Common: booleans to float masks, floats to int indices.

```python
d = np.array([True, False]).astype(float)   # array([1., 0.])   done mask
np.array([1.7, -0.2]).astype(int)          # array([1, 0])   truncates toward 0
np.zeros(3, dtype=np.int64).dtype          # dtype('int64')
```

> [!WARN] astype(int) truncates
> It does not round: `2.9 → 2`, `-0.5 → 0`. Use `np.floor(x).astype(int)` for bins (tile coding) or `np.round(x).astype(int)` for nearest.

## np.float32
sig: `np.float32` · `np.float64` (the default float)
match: np.float32, float32
tags: precision, observations, buffers

32-bit floats use half the memory; environments (`CartPole`, `CoreRunner(obs="vector")`) return `float32` observations, and replay buffers often store `float32`. NumPy's default is `float64`.

```python
x = np.zeros(3, dtype=np.float32)
(x + 1.0).dtype                 # dtype('float32')
(x + np.zeros(3)).dtype         # dtype('float64')  — mixing upcasts
np.float32(0.1) == 0.1          # False (different rounding)
```

> [!WARN] Exact comparisons
> Never compare floats with `==` across precisions. Use `np.allclose` / `approx`. And don't store sums of many tiny rewards in float32.

## np.int64 / np.integer
sig: `np.int64` (default int) · `np.integer` (abstract base of all NumPy int types)
match: np.int64, np.integer
tags: type check, isinstance, actions

NumPy integers are **not** Python `int`s. To accept both in a type check use `isinstance(a, (int, np.integer))`.

```python
a = np.argmax([0.1, 0.9])
type(a)                                   # <class 'numpy.int64'>
isinstance(a, int)                        # False
isinstance(a, (int, np.integer))          # True
int(a)                                    # 1  — plain Python int
```

## np.ndarray
sig: `np.ndarray` — the array type
match: np.ndarray
tags: type check, isinstance

The class of every NumPy array — use it in `isinstance` checks. Create arrays with `np.array` / `np.zeros`, never by calling `np.ndarray(...)` directly.

```python
isinstance(np.zeros(3), np.ndarray)    # True
isinstance([0, 0, 0], np.ndarray)      # False — a list
```

## Converting to Python scalars
sig: `int(x)` · `float(x)` · `x.item()` · `a.tolist()`
match: int(np., float(np., .item(
tags: scalar, json, action, int

Wrap NumPy scalars with `int()` / `float()` before returning actions or logging. Environments index lists with actions, and JSON can't hold `np.float32`.

```python
a = int(np.argmax([0.2, 0.8]))     # 1  (Python int)
float(np.mean([1, 2]))             # 1.5
np.array([[3.5]]).item()           # 3.5
```

> [!WARN] float() of an array
> `float(np.array([1., 2.]))` raises; only size-1 arrays convert. Index first: `float(v[0])`.

## .tolist
sig: `a.tolist()` → nested Python lists of Python scalars
match: .tolist(
tags: list, json, convert

Converts an array into plain Python lists — useful for fast scalar indexing in hot loops, JSON, or printing.

```python
np.array([[1, 2], [3, 4]]).tolist()   # [[1, 2], [3, 4]]
np.float32(0.5).tolist()              # 0.5 (Python float)
```

## .copy
sig: `a.copy()` → independent array · `np.copy(a)`
match: .copy(), np.copy(
tags: aliasing, snapshot, view

Makes a real copy. Needed when you keep an old version of a table (`V_old = V.copy()`) or snapshot weights for a target network.

```python
V = np.zeros(3)
alias, snap = V, V.copy()
V[0] = 1.0
alias[0], snap[0]       # (1.0, 0.0)
```

> [!WARN] `=` never copies
> `V_old = V` gives a second name for the *same* array — your convergence test `np.max(np.abs(V - V_old))` will always be 0. Slices (`a[1:]`) are views too.

## Slicing and views
sig: `a[start:stop:step]` · `a[:, j]` · `a[i]` · `a[-1]`
match: [:, 0], [1:-1], [1:], [:-1], [::-1], [-1]
tags: index, row, column, view

Basic slices return **views** that share memory with the original. `a[i]` is a row; `a[:, j]` a column; negative indices count from the end.

```python
a = np.arange(6).reshape(2, 3)
a[1]          # array([3, 4, 5])
a[:, 0]       # array([0, 3])
a[0, -1]      # 2
v = np.arange(5); v[1:-1]   # array([1, 2, 3])  — drop terminal ends
v[::-1]       # array([4, 3, 2, 1, 0])
```

> [!WARN] Q[s][a] vs Q[s, a]
> Both read the same value, but `Q[s, a]` is one indexing operation and also works with arrays of indices. And remember: writing into a slice writes into the original.

## Fancy indexing
sig: `Q[np.arange(n), a]` · `a[[i, j, k]]` · `a[idx]` with an int array
match: [np.arange(, [idx], [mb]
tags: gather, batch, taken action, minibatch

Indexing with integer arrays *gathers* elements (and returns a copy). The classic RL use: pick each row's taken action from a batch of Q-values.

```python
q = np.array([[1., 2.], [3., 4.], [5., 6.]])
a = np.array([1, 0, 1])
q[np.arange(3), a]         # array([2., 3., 6.])   Q(sᵢ, aᵢ)
q[[0, 2]]                  # rows 0 and 2
```

> [!WARN] q[:, a] is different
> `q[:, a]` gives a `(3, 3)` table (every row × every listed column). Pair the row index with `np.arange(n)`. Also, `x[idx] += 1` with repeated indices adds only once — use `np.add.at(x, idx, 1)`.

## Boolean masks
sig: `a[a > 0]` · `mask = (x > 0)` · `&`, `|`, `~`
match:  > 0), ) & (, ) | (
tags: filter, mask, ReLU, condition

A comparison gives a boolean array; use it to select, to zero things out (`d * (h > 0)` = ReLU derivative), or combine with `&` / `|` / `~`.

```python
x = np.array([-1., 2., -3., 4.])
x > 0                    # array([False,  True, False,  True])
x[x > 0]                 # array([2., 4.])
x * (x > 0)              # array([-0.,  2., -0.,  4.])   ReLU
((x > 0) & (x < 3)).sum()   # 1
```

> [!WARN] Use & and | with parentheses
> `x > 0 and x < 3` raises "truth value of an array is ambiguous", and `x > 0 & x < 3` parses wrongly. Write `(x > 0) & (x < 3)`.

## Matrix multiply @
sig: `A @ B` (np.matmul) — `(n, k) @ (k, m) → (n, m)`
match: @ W, @ w, @ self, @ np, @ d, @ x, @ theta, .T @
tags: matmul, linear layer, features, dot

The matrix product. A linear layer is `x @ W + b`; linear value functions are `X @ w`; a vector `@` vector is a dot product.

```python
x = np.ones((32, 4)); W = np.zeros((4, 2)); b = np.zeros(2)
(x @ W + b).shape          # (32, 2)
np.array([1., 2.]) @ np.array([3., 4.])   # 11.0
```

> [!WARN] * is element-wise
> `A * B` multiplies element by element (with broadcasting). Use `@` for matrix products; inner dimensions must agree.

## np.dot
sig: `np.dot(a, b)` → dot product (1-D) or matrix product (2-D)
match: np.dot(, .dot(
tags: inner product, linear value, features

Inner product of two vectors — e.g. a linear value estimate `v̂(s, w) = w · x(s)`. For 2-D arrays it is a matrix product (prefer `@`).

```python
w = np.array([0.5, -1.0]); x = np.array([2.0, 1.0])
np.dot(w, x)         # 0.0
float(w @ x)         # same
```

## np.outer
sig: `np.outer(a, b)` → matrix with `M[i, j] = a[i] * b[j]`
match: np.outer(
tags: gradient, eligibility, linear policy

Outer product of two vectors. Shows up in linear policy-gradient updates: `∇θ = outer(x, onehot(a) - π)`.

```python
np.outer([1., 2.], [10., 0., -1.])
# array([[ 10.,   0.,  -1.],
#        [ 20.,   0.,  -2.]])
```

## np.linalg.solve
sig: `np.linalg.solve(A, b)` → x with `A @ x = b`
match: np.linalg.solve(, linalg.solve(
tags: Bellman, exact policy evaluation, linear system

Solves a linear system exactly. Policy evaluation in closed form: `v = solve(I - γ P_π, r_π)`.

```python
P = np.array([[0.5, 0.5], [0.0, 1.0]]); r = np.array([1.0, 0.0]); g = 0.9
v = np.linalg.solve(np.eye(2) - g * P, r)
v            # array([1.81818182, 0.        ])
```

> [!WARN] Don't invert
> `np.linalg.inv(A) @ b` is slower and less accurate. And with γ = 1 and no terminal states `I - P` is singular → `LinAlgError`.

## np.sum
sig: `np.sum(a, axis=None, keepdims=False)` · `a.sum(...)`
match: np.sum(, .sum(
tags: total, reduce, normalise

Adds elements (all of them, or along an axis). Normalise probabilities with `p / p.sum()`.

```python
p = np.array([1., 3.])
p / p.sum()              # array([0.25, 0.75])
np.ones((2, 3)).sum(axis=0)   # array([2., 2., 2.])
```

> [!WARN] Builtin sum on 2-D arrays
> Python's `sum(A)` adds the *rows* and returns a vector; `A.sum()` is the grand total. Prefer the NumPy version.

## np.mean
sig: `np.mean(a, axis=None)` · `a.mean(axis=None)`
match: np.mean(, .mean(
tags: average, smooth, evaluate

Arithmetic mean. Used for average returns, MSE losses (`np.mean((pred - y) ** 2)`) and smoothing curves.

```python
np.mean([1, 2, 3, 6])                 # 3.0
np.ones((4, 2)).mean(axis=0)          # array([1., 1.])
np.mean(np.array(returns)[-100:])     # mean of the last 100 episodes
```

> [!WARN] Empty input
> `np.mean([])` returns `nan` with a RuntimeWarning. Guard against zero-length slices.

## np.std
sig: `np.std(a, axis=None, ddof=0)` · `a.std()` · `np.var(a)`
match: np.std(, .std(, np.var(, .var(
tags: spread, normalise, variance, standard error

Standard deviation (population, `ddof=0`, by default). Used to normalise advantages and to report error bars.

```python
x = np.array([1., 2., 3., 4.])
x.std()                         # 1.118033988749895
(x - x.mean()) / (x.std() + 1e-8)   # zero mean, unit std
```

> [!WARN] Divide by zero
> A constant batch has `std() == 0`. Always add a small epsilon when dividing by it.

## np.max / np.min
sig: `np.max(a, axis=None)` · `a.max()` · `np.min(a)` · `a.min()`
match: np.max(, .max(, np.min(, .min(
tags: best value, greedy, bootstrap

Largest / smallest element. `Q[s2].max()` is the Q-learning bootstrap; `np.max(np.abs(V - V_old))` is the classic convergence test.

```python
Q = np.array([[1., 5.], [4., 2.]])
Q[1].max()            # 4.0
Q.max(axis=1)         # array([5., 4.])  V*(s) = maxₐ Q(s,a)
```

> [!WARN] np.max vs np.maximum
> `np.max(a, b)` does NOT compare two arrays — the second argument is `axis`. Element-wise max of two arrays is `np.maximum(a, b)`.

## np.argmax
sig: `np.argmax(a, axis=None)` → int (or array of ints along an axis)
match: np.argmax(, .argmax(
tags: greedy, best action, index of max

Index of the **largest** value. The workhorse of greedy action selection: `a = int(np.argmax(Q[s]))`.

```python
np.argmax([0.1, 0.7, 0.7])     # 1  — ties go to the FIRST maximum
Q = np.array([[0., 2.], [3., 1.]])
Q.argmax(axis=1)               # array([1, 0])  greedy action for every state at once
```

> [!WARN] Ties
> Ties always resolve to the lowest index. With an all-zero Q-table that means "always action 0", which can stop exploration cold. Break ties randomly:
> `best = np.flatnonzero(q == q.max()); a = int(rng.choice(best))`

## np.argmin
sig: `np.argmin(a, axis=None)` → int
match: np.argmin(, .argmin(
tags: closest, smallest, index of min

Index of the smallest value — e.g. the nearest grid point or order size: `np.argmin(np.abs(sizes - target))`.

```python
sizes = np.array([0, 6, 12, 18])
int(np.argmin(np.abs(sizes - 10)))   # 2  (12 is closest)
```

> [!WARN] Ties
> Like `argmax`, ties go to the first index.

## np.cumsum
sig: `np.cumsum(a, axis=None)` · `a.cumsum()`
match: np.cumsum(, .cumsum(
tags: running total, sampling, CDF

Running totals. Two uses: cumulative reward curves, and turning probabilities into a CDF for sampling.

```python
np.cumsum([1, 2, 3])          # array([1, 3, 6])
p = np.array([0.2, 0.5, 0.3])
np.cumsum(p)                  # array([0.2, 0.7, 1. ])
```

> [!WARN] Returns-to-go are NOT a cumsum
> Discounted returns run backwards with γ: `G = r + γ·G`. A reversed cumsum only works for γ = 1.

## np.diff
sig: `np.diff(a, n=1)` → `a[1:] - a[:-1]`
match: np.diff(
tags: differences, monotonic, increments

Differences between neighbours; one element shorter than the input. `np.all(np.diff(x) >= 0)` checks that `x` never decreases.

```python
np.diff([1, 4, 9, 16])      # array([3, 5, 7])
```

## np.all / np.any
sig: `np.all(a, axis=None)` · `np.any(a, axis=None)` · `a.all()` · `a.any()`
match: np.all(, np.any(, .all(, .any(
tags: test, condition, convergence

Whether every / at least one element is truthy. Useful in convergence checks and assertions.

```python
x = np.array([0.1, -0.2, 0.3])
np.all(x > -1)        # True
np.any(x > 0.25)      # True
```

> [!WARN] Not `if array:`
> `if x > 0:` on an array with more than one element raises *ambiguous truth value*. Say which you mean with `.any()` or `.all()`.

## np.exp
sig: `np.exp(x)` → eˣ element-wise
match: np.exp(
tags: softmax, probability, ratio

Exponential. Used in softmax policies and PPO probability ratios `np.exp(logp_new - logp_old)`.

```python
np.exp([0.0, 1.0])             # array([1.        , 2.71828183])
z = np.array([1000., 1001.])
np.exp(z - z.max())            # array([0.36787944, 1.        ])  safe
```

> [!WARN] Overflow
> `np.exp(1000)` is `inf` (with a RuntimeWarning), and `inf / inf` is `nan`. Subtract the max before exponentiating.

## np.log
sig: `np.log(x)` (natural log) · `np.log10(x)` · `np.log2(x)`
match: np.log(, np.log10(, np.log2(
tags: log probability, UCB, entropy

Element-wise logarithm. Shows up in UCB (`c * sqrt(log t / N)`), log-probabilities and entropy, and log-scale loss plots.

```python
np.log(np.e)        # 1.0
np.log10(1000)      # 3.0
np.log(0.0)         # -inf  (RuntimeWarning)
```

> [!WARN] log(0)
> A probability of exactly 0 gives `-inf`, and `0 * -inf` is `nan` in an entropy sum. Clip first: `np.log(np.clip(p, 1e-12, 1.0))`.

## np.sqrt
sig: `np.sqrt(x)` → element-wise square root
match: np.sqrt(
tags: UCB, RMSE, Adam, initialisation

Square root — in UCB bonuses, RMS errors (`np.sqrt(np.mean(err**2))`), Adam's denominator and He initialisation `sqrt(2 / n_in)`.

```python
np.sqrt([4.0, 9.0])                      # array([2., 3.])
c, t, N = 2.0, 100, np.array([1, 10])
c * np.sqrt(np.log(t) / N)               # array([4.29193205, 1.35722808])
```

> [!WARN] Untried actions
> `N = 0` gives division by zero → `inf`. In UCB that's actually what you want (try untried arms first) — but handle it explicitly so you don't get `nan` from `0/0`.

## np.abs
sig: `np.abs(x)` → element-wise absolute value
match: np.abs(
tags: error, convergence, distance

Absolute value. `np.max(np.abs(V_new - V))` is the standard "has it converged?" measure (Δ in value iteration).

```python
np.abs([-2.0, 0.5])                       # array([2. , 0.5])
delta = np.max(np.abs(np.array([1., 2.]) - np.array([1.1, 1.5])))   # 0.5
```

## np.maximum / np.minimum
sig: `np.maximum(a, b)` · `np.minimum(a, b)` → element-wise
match: np.maximum(, np.minimum(
tags: ReLU, clip, PPO, elementwise max

Element-wise max/min of two arrays (or array and scalar). `np.maximum(h, 0)` is ReLU; PPO takes `np.minimum(ratio * adv, clipped * adv)`.

```python
np.maximum([-1., 2.], 0.0)            # array([0., 2.])
np.minimum([1., 5.], [3., 4.])        # array([1., 4.])
```

> [!WARN] Not the same as max()
> `np.max` reduces one array to a single number; `np.maximum` compares two arrays element by element.

## np.clip
sig: `np.clip(a, a_min, a_max)` → values limited to `[a_min, a_max]`
match: np.clip(
tags: bound, PPO, Huber, gradient clipping

Limits values to a range. Used for PPO's `clip(ratio, 1-ε, 1+ε)`, the Huber gradient `clip(err, -δ, δ)`, keeping indices in bounds, and taming log(0).

```python
np.clip([-2., 0.5, 3.], -1.0, 1.0)    # array([-1. ,  0.5,  1. ])
np.clip(7, 0, 4)                      # 4
```

## np.where
sig: `np.where(cond, x, y)` → x where cond else y · `np.where(cond)` → indices
match: np.where(
tags: conditional, select, mask

Vectorised if/else: picks from `x` where the condition holds, from `y` elsewhere.

```python
err = np.array([-2., 0.3, 1.5])
np.where(np.abs(err) <= 1.0, 0.5 * err**2, np.abs(err) - 0.5)   # Huber: array([1.5  , 0.045, 1.   ])
np.where(np.array([0, 3, 0]) > 0)       # (array([1]),)  — a tuple!
```

> [!WARN] One-argument form returns a tuple
> `np.where(mask)` gives a tuple of index arrays. For a flat list of indices use `np.flatnonzero(mask)`.

## np.round
sig: `np.round(a, decimals=0)` → rounded floats
match: np.round(
tags: display, rounding, report

Rounds to `decimals` places — mostly for display in tables and stat tiles. The result stays a float array.

```python
np.round(3.14159, 2)              # 3.14
np.round([0.5, 1.5, 2.5])         # array([0., 2., 2.])  — banker's rounding
```

> [!WARN] Halves round to even
> Both NumPy and Python's `round` round `.5` to the nearest *even* number. Don't rely on rounding for bin indices; use `np.floor`.

## np.floor
sig: `np.floor(x)` · `np.ceil(x)` → floats
match: np.floor(, np.ceil(
tags: bins, tile coding, discretise

Rounds down (or up) to whole numbers. The core of discretisation and tile coding: `idx = np.floor((x - low) / width).astype(int)`.

```python
np.floor([1.7, -0.2])             # array([ 1., -1.])
np.floor(np.array([0.95]) * 4).astype(int)   # array([3])
```

> [!WARN] Floor vs int()
> `int(-0.2)` is `0` but `np.floor(-0.2)` is `-1.0`. For negatives, only `floor` gives consistent bins.

## np.tanh
sig: `np.tanh(x)` → element-wise hyperbolic tangent in (−1, 1)
match: np.tanh(
tags: activation, neural network

Smooth activation for small policy/value networks. Its derivative is `1 - tanh(x)**2`, so the backward pass can reuse the forward output.

```python
h = np.tanh(np.array([-10., 0., 0.5]))   # array([-1.        ,  0.        ,  0.46211716])
1.0 - h ** 2                             # derivative at those points
```

## np.sin / np.cos / np.pi
sig: `np.sin(x)` · `np.cos(x)` (radians) · `np.pi`
match: np.sin(, np.cos(, np.pi
tags: trig, toy targets, seasonality

Element-wise trigonometry in **radians**; `np.pi` is π. Used for toy regression targets (`np.sin(3 * x)`) and seasonal demand patterns.

```python
np.sin(np.pi / 2)               # 1.0
np.cos(np.array([0., np.pi]))   # array([ 1., -1.])
```

> [!WARN] Degrees
> `np.sin(90)` is not 1. Convert with `np.deg2rad(90)`.

## np.inf
sig: `np.inf` · `-np.inf` — floating-point infinity
match: np.inf
tags: infinity, initial best, mask

Positive infinity. Use `-np.inf` as the "nothing yet" value for a running best, or to mask illegal actions before an argmax.

```python
best = -np.inf
best = max(best, 3.2)            # 3.2
q = np.array([1., 5., 2.]); q[1] = -np.inf
int(np.argmax(q))                # 2
```

> [!WARN] inf arithmetic
> `np.inf - np.inf` and `0 * np.inf` are `nan`. Keep infinities out of value tables that you update arithmetically.

## np.isfinite / np.isinf / np.isnan
sig: `np.isfinite(x)` · `np.isinf(x)` · `np.isnan(x)` → bool (array)
match: np.isfinite(, np.isinf(, np.isnan(
tags: nan, divergence, check

Detect divergence: a `nan` or `inf` anywhere in your weights means training blew up (step size too large).

```python
x = np.array([1.0, np.inf, np.nan])
np.isfinite(x)          # array([ True, False, False])
np.isnan(x).any()       # True
```

> [!WARN] nan != nan
> `x == np.nan` is always False. Use `np.isnan(x)`.

## np.errstate
sig: `with np.errstate(all="ignore"):` (also divide=, over=, invalid=)
match: np.errstate(
tags: warnings, overflow, silence

Context manager that silences (or raises on) floating-point warnings inside a block — e.g. a parameter sweep where some settings are *expected* to diverge.

```python
with np.errstate(all="ignore"):
    x = np.array([1.0]) / 0.0      # array([inf]), no RuntimeWarning
```

> [!WARN] Silenced, not fixed
> The `inf`/`nan` is still there — check with `np.isfinite` afterwards.

## np.flatnonzero
sig: `np.flatnonzero(a)` → 1-D array of indices where `a` is non-zero / True
match: np.flatnonzero(
tags: ties, indices, argmax ties, where

Indices of the True entries of a (flattened) array. The standard way to find **all** maximisers so you can break ties randomly.

```python
q = np.array([0.0, 0.7, 0.7])
np.flatnonzero(q == q.max())                 # array([1, 2])
rng = np.random.default_rng(0)
int(rng.choice(np.flatnonzero(q == q.max())))   # 1 or 2, at random
```

> [!WARN] Float equality
> After many updates two "equal" values may differ by 1e-16. If exact ties matter, use `np.isclose(q, q.max())`.

## np.searchsorted
sig: `np.searchsorted(a, v, side="left")` → insertion index into sorted `a`
match: np.searchsorted(
tags: sample, CDF, categorical, inverse transform

Finds where `v` would be inserted to keep `a` sorted. With a CDF from `np.cumsum(p)` and a uniform `u`, it samples a categorical action (fast inverse-transform sampling).

```python
p = np.array([0.2, 0.5, 0.3])
cdf = np.cumsum(p)                                 # [0.2, 0.7, 1.0]
int(np.searchsorted(cdf, 0.65, side="right"))     # 1
a = min(int(np.searchsorted(cdf, rng.random(), side="right")), len(p) - 1)
```

> [!WARN] Round-off at the end
> `cdf[-1]` can be 0.9999999; a `u` above it returns `len(p)`. Clamp with `min(..., len(p) - 1)`.

## np.unique
sig: `np.unique(a, return_counts=False)` → sorted unique values
match: np.unique(
tags: distinct, counts, dedupe

Sorted distinct values; `return_counts=True` also returns how often each occurs.

```python
np.unique([3, 1, 3, 2])                         # array([1, 2, 3])
vals, counts = np.unique([0, 1, 1, 1], return_counts=True)   # (array([0, 1]), array([1, 3]))
```

## np.histogram
sig: `np.histogram(a, bins=10, range=None)` → (counts, bin_edges)
match: np.histogram(
tags: distribution, counts, bins

Counts values per bin. Returns **two** arrays: counts (length `bins`) and edges (length `bins + 1`).

```python
counts, edges = np.histogram([0.1, 0.2, 0.9], bins=2, range=(0, 1))
counts        # array([2, 1])
edges         # array([0. , 0.5, 1. ])
```

## np.ravel_multi_index
sig: `np.ravel_multi_index(multi_index, dims)` → flat index · inverse `np.unravel_index(i, dims)`
match: np.ravel_multi_index(, np.unravel_index(
tags: flatten, tile coding, state index, row-major

Converts (row, col, …) coordinates into a single row-major index — and back. Exactly `r * cols + c` for 2-D.

```python
np.ravel_multi_index((2, 1), (3, 5))       # 11  (= 2*5 + 1)
coords = np.array([[0, 1], [2, 4]])        # two points (r, c)
np.ravel_multi_index(coords.T, (3, 5))     # array([ 1, 14])
np.unravel_index(11, (3, 5))               # (2, 1)
```

> [!WARN] Tuple of coordinate arrays
> The first argument is one array *per dimension*. For an `(n_points, dims)` array pass its transpose `coords.T`.

## np.corrcoef
sig: `np.corrcoef(x, y)` → 2×2 correlation matrix
match: np.corrcoef(
tags: correlation, replay, statistics

Pearson correlation. The off-diagonal `[0, 1]` entry is the correlation between `x` and `y`.

```python
x = np.array([1., 2., 3., 4.])
np.corrcoef(x, 2 * x)[0, 1]          # 1.0
np.corrcoef(x[:-1], x[1:])[0, 1]     # lag-1 autocorrelation
```

## np.allclose
sig: `np.allclose(a, b, rtol=1e-05, atol=1e-08)` → bool · `np.isclose` element-wise
match: np.allclose(, np.isclose(
tags: compare, test, tolerance

Compares floats with a tolerance — the right way to test numerical results.

```python
np.allclose([0.1 + 0.2], [0.3])       # True
0.1 + 0.2 == 0.3                       # False
np.isclose([1.0, 2.0], [1.0, 2.1])     # array([ True, False])
```

## np.array_equal
sig: `np.array_equal(a, b)` → bool (same shape and all elements equal)
match: np.array_equal(
tags: compare, policy stable, convergence

Exact equality of two arrays. Policy iteration stops when `np.array_equal(new_policy, policy)`.

```python
np.array_equal([1, 2], [1, 2])     # True
np.array_equal([1, 2], [1, 2, 3])  # False
```

> [!WARN] Not for floats
> For value functions use `np.allclose` or a max-abs-difference threshold; exact float equality almost never happens.

## np.save / np.load
sig: `np.save(file, arr)` · `np.load(file)` → array (or NpzFile for .npz)
match: np.save(, np.load(
tags: save weights, checkpoint, file

Stores one array in binary `.npy` format and loads it back exactly (dtype and shape preserved).

```python
np.save("q_table.npy", np.zeros((48, 4)))
Q = np.load("q_table.npy")
Q.shape            # (48, 4)
```

> [!WARN] Extension
> `np.save` appends `.npy` if the name lacks it — load from the same full name.

## np.savez
sig: `np.savez(file, *arrays, **named_arrays)` → `.npz` archive
match: np.savez(
tags: save weights, multiple arrays, checkpoint

Saves several arrays in one file. Positional arrays are named `arr_0, arr_1, …`; keyword arrays keep their names.

```python
np.savez("net.npz", np.ones(2), np.zeros(3), gamma=np.array(0.99))
data = np.load("net.npz")
data["arr_0"], data["gamma"]       # (array([1., 1.]), array(0.99))
data.files                         # ['arr_0', 'arr_1', 'gamma']
```

## np.random.default_rng
sig: `np.random.default_rng(seed=None)` → Generator
match: np.random.default_rng(, default_rng(
tags: random, seed, reproducible, generator

Creates a random-number **Generator**. Pass a seed for reproducible runs; pass the `rng` object into your functions instead of using global randomness.

```python
rng = np.random.default_rng(0)
rng.random()            # 0.6369616873214543
rng2 = np.random.default_rng(0)
rng2.random()           # 0.6369616873214543  — same seed, same stream
```

> [!WARN] Re-creating it in a loop
> `np.random.default_rng(0)` inside your episode loop restarts the same sequence every episode — the agent sees identical "random" numbers. Create it once and pass it around.

## np.random.seed (legacy)
sig: `np.random.seed(seed)` — seeds the global legacy generator
match: np.random.seed(, np.random.rand(, np.random.randint(
tags: legacy, global seed

The old global API (`np.random.rand`, `np.random.randint`, …). It still works, but code here uses `rng = np.random.default_rng(seed)` so that each component's randomness is independent and explicit.

```python
np.random.seed(0); np.random.rand()     # 0.5488135039273248
rng = np.random.default_rng(0); rng.random()   # preferred
```

> [!WARN] Different streams
> Seeding the global generator does not affect a `default_rng` Generator (or Python's `random` module), and vice versa.

## rng.random
sig: `rng.random(size=None)` → float(s) uniform in [0, 1)
match: rng.random(, .random(
tags: uniform, epsilon greedy, coin flip

Uniform floats in `[0, 1)`. The ε-greedy coin flip is `if rng.random() < epsilon:`.

```python
rng = np.random.default_rng(0)
rng.random()                 # 0.6369616873214543
rng.random(3).shape          # (3,)
rng.random((2, 4)).shape     # (2, 4)
```

## rng.integers
sig: `rng.integers(low, high=None, size=None, endpoint=False)` → int(s) in [low, high)
match: rng.integers(, .integers(
tags: random action, random index, uniform int

Random integers. With one argument it draws from `[0, low)`, so `rng.integers(n_actions)` is a uniformly random action.

```python
rng = np.random.default_rng(0)
int(rng.integers(4))                   # 3   (0..3)
rng.integers(1, 7, size=5)             # five dice rolls, values 1..6
rng.integers(0, 100, size=32)          # minibatch indices
```

> [!WARN] high is exclusive
> `rng.integers(1, 6)` never returns 6 (unlike Python's `random.randint(1, 6)`). Pass `endpoint=True` or use `high + 1`. It returns `np.int64` — wrap in `int()` for actions.

## rng.normal
sig: `rng.normal(loc=0.0, scale=1.0, size=None)` → Gaussian sample(s)
match: rng.normal(, .normal(
tags: gaussian, noise, bandit, initialisation

Normal (Gaussian) samples with mean `loc` and standard deviation `scale`. Bandit rewards `N(q*, 1)`, weight initialisation, noise.

```python
rng = np.random.default_rng(0)
rng.normal(0.0, 1.0, size=3)     # array([ 0.12573022, -0.13210486,  0.64042265])
W = rng.normal(0, np.sqrt(2 / 4), size=(4, 64))   # He init
```

> [!WARN] scale is the std
> `scale` is σ, not σ². For variance 4 pass `scale=2`.

## rng.uniform
sig: `rng.uniform(low=0.0, high=1.0, size=None)` → floats in [low, high)
match: rng.uniform(, .uniform(
tags: uniform, initial state, random points

Uniform floats in a range — random start states, random test points.

```python
rng = np.random.default_rng(0)
rng.uniform(-0.05, 0.05, size=4).shape     # (4,)  CartPole-style start
rng.uniform([-1.2, -0.07], [0.6, 0.07])    # one random (pos, vel) — bounds broadcast
```

## rng.choice
sig: `rng.choice(a, size=None, replace=True, p=None)` → element(s) of `a`
match: rng.choice(, .choice(
tags: sample, tie break, categorical, policy

Picks elements from an array (or from `range(a)` when `a` is an int), optionally with probabilities `p`. Use it for random tie-breaking and sampling from a stochastic policy.

```python
rng = np.random.default_rng(0)
int(rng.choice([2, 5, 7]))                 # one of 2, 5, 7
int(rng.choice(3, p=[0.1, 0.8, 0.1]))      # usually 1
rng.choice(10, size=4, replace=False)      # 4 distinct indices
```

> [!WARN] p must sum to 1
> `p` that sums to 0.9999 raises `ValueError: probabilities do not sum to 1`. Renormalise: `p = p / p.sum()`. In hot loops `rng.choice` is slow; `searchsorted(cumsum(p), rng.random())` is faster.

## rng.permutation / rng.shuffle
sig: `rng.permutation(x)` → shuffled copy (or of `range(x)`) · `rng.shuffle(x)` → None, in place
match: .permutation(, .shuffle(
tags: minibatch, epochs, shuffle

`permutation(N)` gives the indices `0..N-1` in random order — the standard way to iterate minibatches over an epoch (PPO). `shuffle` permutes an array in place and returns `None`.

```python
rng = np.random.default_rng(0)
order = rng.permutation(8)
for mb in np.array_split(order, 4):   # 4 minibatches of 2
    pass
```

> [!WARN] shuffle returns None
> `idx = rng.shuffle(idx)` sets `idx` to `None`. Either shuffle in place or use `permutation`.

## rng.poisson / rng.binomial
sig: `rng.poisson(lam=1.0, size=None)` · `rng.binomial(n, p, size=None)` → ints
match: .poisson(, .binomial(, .negative_binomial(
tags: counts, demand, spoilage

Count distributions: Poisson for arrivals/demand with mean `lam`; binomial for "how many of `n` items fail with probability `p`" (spoilage in the inventory env).

```python
rng = np.random.default_rng(0)
rng.poisson(3.0, size=5).mean()       # ≈ 3
int(rng.binomial(20, 0.05))           # spoiled units out of 20
```
