---
title: Python
---

## Tuple unpacking
sig: `a, b = pair` · `obs, r, term, trunc, info = env.step(a)` · `x, _ = f()`
match: = env.step(, = env.reset(, , _ =
tags: unpack, multiple return, step

A function that returns several values returns a tuple; unpack it straight into names. Use `_` for values you don't need.

```python
def step():
    return 3, -1.0, False, False, {}
s2, r, terminated, truncated, info = step()
s, _ = (0, {})                 # ignore info
a, b = 1, 2; a, b = b, a       # swap: a == 2, b == 1
```

> [!WARN] Count must match
> `s2, r, done, info = env.step(a)` (the old 4-value Gym API) raises `ValueError: too many values to unpack`. The step API here returns **five** values.

## while True / break
sig: `while True: ... if cond: break`
match: while True:, break
tags: loop, episode, until done

Loop until something inside the body says stop. The idiomatic episode loop when you don't know how many steps it will take.

```python
s, t = 0, 0
while True:
    s, t = s + 1, t + 1
    if s >= 3:          # e.g. terminated or truncated
        break
t                       # 3
```

> [!WARN] Forgetting the break
> If your exit condition never becomes true (e.g. you test `terminated` but the episode can only end by `truncated`), the loop runs forever and the check times out. Test `terminated or truncated`.

## for … in range
sig: `range(stop)` · `range(start, stop, step)`
match: in range(
tags: loop, episodes, count

Counts from `start` up to (not including) `stop`. `for ep in range(n_episodes):` is the outer training loop.

```python
list(range(3))            # [0, 1, 2]
list(range(2, 10, 3))     # [2, 5, 8]
list(range(3, 0, -1))     # [3, 2, 1]
```

> [!WARN] Off by one
> `range(n)` ends at `n - 1`. For "the last k items", slicing (`xs[-k:]`) is usually clearer than index arithmetic.

## enumerate
sig: `enumerate(iterable, start=0)` → (index, item) pairs
match: enumerate(
tags: loop, index, counter

Loop over items *and* their position without managing a counter yourself.

```python
for t, r in enumerate([1.0, 0.0, 2.0]):
    print(t, r)          # 0 1.0 / 1 0.0 / 2 2.0
list(enumerate("ab", start=1))   # [(1, 'a'), (2, 'b')]
```

## zip
sig: `zip(a, b, ...)` → tuples of parallel items (stops at the shortest)
match: zip(
tags: parallel, pairs, loop

Walks several sequences in lockstep: states with rewards, parameters with gradients, layer sizes with their successors.

```python
sizes = [4, 64, 2]
list(zip(sizes[:-1], sizes[1:]))     # [(4, 64), (64, 2)]  (n_in, n_out) per layer
for p, g in zip([1.0, 2.0], [0.1, 0.2]):
    print(p - g)                     # 0.9 / 1.8
```

> [!WARN] Silent truncation
> If the lists have different lengths, `zip` quietly stops at the shorter one. Check `len(states) == len(rewards)` when building episode data.

## reversed
sig: `reversed(seq)` → iterator from the end
match: reversed(
tags: backwards, returns, backprop

Iterates a sequence back to front. The return `G = r + γG` and backprop both run backwards in time / layers.

```python
G, gamma, rewards = 0.0, 0.9, [1.0, 0.0, 2.0]
for r in reversed(rewards):
    G = r + gamma * G
round(G, 3)               # 2.62
list(reversed(range(3)))  # [2, 1, 0]
```

## List comprehensions
sig: `[expr for x in xs if cond]` · `{k: v for ...}` · `(expr for ...)`
match: ) for , ] for , for _ in
tags: list, build, compact loop

Build a list (or dict / set) in one expression. `_` is the conventional name for a loop variable you don't use.

```python
[x * x for x in range(4)]                # [0, 1, 4, 9]
[0.0 for _ in range(3)]                  # [0.0, 0.0, 0.0]
[r for r in [1, -2, 3] if r > 0]         # [1, 3]
{a: 0.0 for a in "LR"}                   # {'L': 0.0, 'R': 0.0}
```

> [!WARN] Nested lists by multiplication
> `[[0] * 3] * 2` creates two references to the *same* inner list; writing `g[0][0] = 1` changes both rows. Use `[[0] * 3 for _ in range(2)]`.

## Dictionaries
sig: `d[k]` · `d[k] = v` · `k in d` · `d.items()` · `d.keys()` · `d.values()`
match: .items(), .values(), .keys()
tags: dict, table, map

Maps keys to values. Good for tables over non-integer states (like Blackjack's `(sum, dealer, ace)` tuples).

```python
V = {}
V[(13, 2, False)] = 0.5
(13, 2, False) in V              # True
for k, v in V.items():
    print(k, v)                  # (13, 2, False) 0.5
```

> [!WARN] Lists can't be keys
> Keys must be hashable: tuples work, lists and NumPy arrays don't. Convert an observation with `tuple(obs)` first.

## dict.get / setdefault
sig: `d.get(key, default=None)` · `d.setdefault(key, default)` → value
match: .get(, setdefault(
tags: default, missing key, lazy init

`get` reads with a fallback (no KeyError). `setdefault` inserts the default if missing **and** returns the stored value — perfect for lazily creating table rows.

```python
N = {}
N[3] = N.get(3, 0) + 1             # N == {3: 1}
Q = {}
Q.setdefault("s0", [0.0, 0.0])[1] += 0.5
Q                                   # {'s0': [0.0, 0.5]}
```

> [!WARN] get does not store
> `Q.get(s, np.zeros(2))[a] += 1` updates a temporary array that is thrown away. Use `setdefault` (or `defaultdict`) when you want to write.

## collections.defaultdict
sig: `defaultdict(factory)` — missing keys are created with `factory()`
match: defaultdict(
tags: default, table, lazy init

A dict that creates missing entries automatically. `defaultdict(lambda: np.zeros(n_actions))` is a Q-table over any hashable state.

```python
from collections import defaultdict
import numpy as np
Q = defaultdict(lambda: np.zeros(2))
Q[(13, 2, False)][1] += 0.5
len(Q)                       # 1
returns = defaultdict(list); returns["s"].append(1.0)
```

> [!WARN] Reading creates
> Just *looking up* `Q[s]` inserts a new zero row. Use `s in Q` to test membership without creating it.

## collections.Counter
sig: `Counter(iterable)` → dict of counts · `.most_common(n)`
match: Counter(
tags: count, frequency, histogram

Counts how often each item appears.

```python
from collections import Counter
c = Counter([1, 0, 1, 1])
c[1], c[2]              # (3, 0)  — missing keys count as 0
c.most_common(1)        # [(1, 3)]
```

## collections.deque
sig: `deque(iterable=(), maxlen=None)` · `.append` · `.popleft()`
match: deque(
tags: queue, BFS, sliding window, buffer

A double-ended queue with O(1) appends and pops at both ends. `maxlen=` makes a sliding window that drops old items; `popleft` makes a FIFO for breadth-first search.

```python
from collections import deque
win = deque(maxlen=3)
for r in [1, 2, 3, 4]:
    win.append(r)
list(win)                 # [2, 3, 4]
q = deque([0]); q.popleft()   # 0
```

> [!WARN] Random access is slow
> Indexing `d[i]` in the middle is O(n). For a replay buffer you sample from, use preallocated NumPy arrays instead.

## list.append / pop / insert
sig: `lst.append(x)` · `lst.pop(i=-1)` · `lst.insert(i, x)` · `len(lst)`
match: .append(, .pop(, .insert(
tags: list, collect, trajectory

Lists are the natural way to collect an episode as you go; convert to an array once at the end.

```python
rewards = []
rewards.append(1.0); rewards.append(-1.0)
rewards.pop(0)           # 1.0 — removes and returns the first item
rewards                  # [-1.0]
```

> [!WARN] append returns None
> `rewards = rewards.append(r)` sets `rewards` to `None`. Just call `rewards.append(r)`.

## f-strings and format specs
sig: `f"{value:spec}"` — `.2f`, `.3g`, `>6`, `,`, `%`, `!r`
match: f"
tags: format, print, string

Embed values in strings with `{}`; after a colon, a format spec controls the look.

```python
G, eps = 12.34567, 0.1
f"G = {G:.2f}"            # 'G = 12.35'
f"{eps:.0%} explore"      # '10% explore'
f"{1234567:,}"            # '1,234,567'
f"[{7:>3}]"               # '[  7]'
f"{'s'!r}"                # "'s'"
```

> [!WARN] NumPy arrays and specs
> `f"{np.array([1.0]):.2f}"` raises `TypeError`. Format a scalar (`float(x)`) or use `np.round(arr, 2)`.

## lambda
sig: `lambda args: expression`
match: lambda 
tags: anonymous function, policy, key

A tiny one-expression function — handy for policies passed to a runner, `key=` functions and `defaultdict` factories.

```python
greedy = lambda s: 1
greedy(5)                                # 1
sorted([3, -1, 2], key=lambda x: abs(x)) # [-1, 2, 3]
```

> [!WARN] Late binding in loops
> `fs = [lambda: i for i in range(3)]` — every function returns 2, because `i` is looked up when called. Bind it: `lambda i=i: i`.

## Closures and policy factories
sig: `def make_policy(Q): def policy(s): ...; return policy`
match: def policy(, return policy, def make_
tags: closure, factory, policy, function returning function

A function defined inside another one remembers the outer variables. That's how you build a `policy(obs) -> action` around a Q-table or an environment.

```python
def make_greedy(Q):
    def policy(s):
        return max(range(len(Q[s])), key=lambda a: Q[s][a])
    return policy
pi = make_greedy({0: [0.1, 0.9]})
pi(0)                    # 1
```

> [!WARN] It sees later changes
> The closure keeps a reference, not a copy: if you keep training `Q`, the policy changes too. Pass `Q.copy()` to freeze it.

## Classes, __init__ and self
sig: `class Agent: def __init__(self, ...): self.x = ...`
match: class , def __init__(, self.
tags: class, object, state, method

A class bundles state (attributes on `self`) with methods. `__init__` runs when you create an instance; every method receives the instance as `self`.

```python
class Counter1:
    def __init__(self, start=0):
        self.n = start
    def add(self, k=1):
        self.n += k
        return self.n
c = Counter1(); c.add(); c.add(2)    # 3
```

> [!WARN] Forgetting self
> Inside a method, `n += 1` is a local variable; you need `self.n += 1`. And a method needs `self` as its first parameter, or calling it raises `TypeError: takes 0 positional arguments but 1 was given`.

## @property
sig: `@property def name(self): return ...`
match: @property
tags: attribute, computed, getter

Makes a method look like an attribute — computed on every access, no parentheses.

```python
class Shelf:
    def __init__(self): self.on_hand, self.transit = 5, 3
    @property
    def position(self):
        return self.on_hand + self.transit
Shelf().position         # 8   (no parentheses)
```

## Dunder methods (__len__, __repr__)
sig: `def __len__(self)` · `def __repr__(self)`
match: def __len__, def __repr__, __len__
tags: len, special method, protocol

Special "double-underscore" methods hook your class into built-ins: `len(buf)` calls `buf.__len__()`, printing calls `__repr__`.

```python
class Buf:
    def __init__(self): self.size = 0
    def __len__(self): return self.size
len(Buf())               # 0
```

> [!WARN] Must return an int
> `__len__` returning a NumPy int or float raises `TypeError` in some contexts; return `int(...)`.

## Default arguments
sig: `def f(x, gamma=0.99, seed=None)`
match: seed=None, =None)
tags: optional, keyword, defaults

Parameters with defaults are optional; callers can pass them by name in any order. `None` is the usual "not given" marker.

```python
def run(episodes=100, alpha=0.1, seed=None):
    return episodes, alpha, seed
run(alpha=0.5)                 # (100, 0.5, None)
```

> [!WARN] Mutable defaults are shared
> `def add(x, buf=[])` reuses the **same** list across calls, so data leaks between runs. Use `buf=None` and `if buf is None: buf = []` inside. Same for `np.zeros(...)` defaults.

## *args and **kwargs
sig: `def f(*args, **kwargs)` · `f(*seq)` · `f(**d)`
match: *args, **kw, (*self., (*
tags: unpacking, variadic, keyword arguments

In a definition, `*args` collects extra positional arguments and `**kwargs` extra keyword arguments. In a call, `*seq` spreads a list into positional arguments and `**d` a dict into keywords.

```python
def total(*xs): return sum(xs)
total(1, 2, 3)                 # 6
params = [1, 2]; total(*params)   # 3
cfg = {"sep": "-"}; print("a", "b", **cfg)   # a-b
```

## random module
sig: `random.random()` · `random.randrange(n)` · `random.randint(a, b)` · `random.choice(seq)` · `random.seed(s)` · `random.Random(seed)`
match: import random, random.seed(, random.randrange(, random.randint(, random.gauss(, random.uniform(
tags: stdlib random, seed, uniform

Python's built-in random numbers (separate from NumPy's). `random.Random(seed)` gives an independent generator — `GridWorld`, `Blackjack` and `CoreRunner` use one internally.

```python
import random
random.seed(0)
random.random()          # 0.8444218515250481
random.randrange(4)      # 0..3
random.randint(1, 6)     # 1..6 (INCLUSIVE)
r = random.Random(1); r.choice("LR")
```

> [!WARN] Inclusive vs exclusive
> `random.randint(a, b)` includes `b`; `random.randrange(a, b)` and NumPy's `rng.integers(a, b)` exclude it. Seeding `random` does not seed NumPy, and vice versa.

## math module
sig: `math.sqrt` · `math.log` · `math.exp` · `math.inf` · `math.isclose` · `math.ceil`
match: import math, math.
tags: scalar maths, stdlib

Scalar maths on plain floats — slightly faster than NumPy for single numbers, but doesn't work on arrays.

```python
import math
math.sqrt(2 * math.log(100))      # 3.0348542587702925
math.ceil(7 / 2), math.inf > 1e308  # (4, True)
```

> [!WARN] Arrays
> `math.exp(np.array([1., 2.]))` raises `TypeError: only length-1 arrays can be converted`. Use `np.exp` for arrays.

## if __name__ == "__main__":
sig: `if __name__ == "__main__": demo()`
match: __name__
tags: main guard, demo, check

Code under this guard runs only when the file is executed directly. In this app, **Run** executes your file as `__main__` (the demo runs and plots), while **Check** executes it with `__name__ == "__check__"`, so the demo is skipped and only your functions are tested.

```python
def solve():
    return 42

if __name__ == "__main__":
    print("demo:", solve())    # runs on RUN, skipped on CHECK
```

> [!WARN] Put work in functions
> Anything the checker needs (functions, classes, constants) must be defined *outside* the guard. Code you put only under `__main__` never runs during Check.

## raise NotImplementedError
sig: `raise NotImplementedError("explain what to do")`
match: NotImplementedError
tags: TODO, starter, placeholder

Starter files mark unfinished parts this way, so the program runs but checks fail with a clear message. Replace the `raise` line with your implementation.

```python
def epsilon_greedy(q, eps, rng):
    # TODO: your code here
    raise NotImplementedError("pick an action")
```

> [!WARN] Delete the raise
> A `raise` left *after* your code still fires; remove it once you've implemented the function.

## assert
sig: `assert condition, "message"`
match: assert 
tags: debug, sanity check, shape

Stops with `AssertionError` when the condition is false — a cheap way to catch shape bugs early.

```python
import numpy as np
q = np.zeros((32, 2))
assert q.shape == (32, 2), f"bad shape {q.shape}"
```

> [!WARN] Not a test framework here
> Exercise checks use the injected `check(...)` helper, not `assert`. Asserts in your own code are fine for debugging.

## Slicing and negative indices
sig: `xs[start:stop:step]` · `xs[-1]` · `xs[-k:]` · `xs[::-1]`
match: [-1], [-100:], [::-1], [1:], [:-1]
tags: slice, last, tail, reverse

Slices take `[start, stop)` with an optional step; negative numbers count from the end.

```python
xs = [10, 20, 30, 40, 50]
xs[-1], xs[-2:], xs[:2]     # (50, [40, 50], [10, 20])
xs[::-1]                    # [50, 40, 30, 20, 10]
xs[1:-1]                    # [20, 30, 40]
```

> [!WARN] Lists copy, arrays don't
> A list slice is a new list; a NumPy slice is a *view* that shares memory with the original.

## Integer vs float division
sig: `a / b` → float · `a // b` → floor · `a % b` → remainder · `divmod(a, b)`
match: //, divmod(, % 
tags: division, grid index, row col

`/` always gives a float; `//` rounds *down* to an integer; `%` is the remainder. Grid states use them: `row, col = s // cols, s % cols`.

```python
7 / 2, 7 // 2, 7 % 2       # (3.5, 3, 1)
divmod(13, 4)              # (3, 1)   state 13 in a 4-wide grid → row 3, col 1
-7 // 2                    # -4  (floor, not truncation)
```

> [!WARN] Averages
> `total // n` silently drops the fractional part. Use `/` for means and step sizes like `1 / N[a]`.

## min / max with key
sig: `max(iterable, key=f)` · `min(a, b)` · `max(a, b, ...)`
match: key=, min(, max(
tags: argmax, best, clamp

Built-in `max`/`min` work on any iterable; `key=` picks the item with the largest `key(item)` — a pure-Python argmax. With several arguments they compare those.

```python
q = {"L": 0.2, "R": 0.7}
max(q, key=q.get)              # 'R'
min(range(3), key=lambda i: [5, 1, 3][i])   # 1
max(0.05, 1.0 - 0.001 * 2000)  # 0.05  — ε decay with a floor
```

> [!WARN] Ties and NumPy arrays
> Ties return the *first* maximiser. On a 2-D NumPy array, builtin `max` compares rows and fails; use `arr.max()`.

## sorted
sig: `sorted(iterable, key=None, reverse=False)` → new list
match: sorted(
tags: sort, order, ranking

Returns a new sorted list (the original is untouched); `list.sort()` sorts in place.

```python
sorted([3, 1, 2])                          # [1, 2, 3]
sorted({"a": 2, "b": 5}.items(), key=lambda kv: -kv[1])   # [('b', 5), ('a', 2)]
```

## isinstance
sig: `isinstance(obj, type_or_tuple)` → bool
match: isinstance(
tags: type check, int, ndarray

Tests an object's type; pass a tuple to accept several.

```python
import numpy as np
isinstance(3, (int, np.integer))         # True
isinstance(np.int64(3), int)             # False — NumPy ints are not Python ints
isinstance(np.zeros(2), np.ndarray)      # True
```

> [!WARN] bool is an int
> `isinstance(True, int)` is `True`.

## itertools.product
sig: `itertools.product(*iterables, repeat=1)` → all combinations
match: itertools.product(, product(
tags: grid search, sweep, combinations

Cartesian product — every combination of the inputs. Ideal for hyper-parameter sweeps or enumerating all deterministic policies.

```python
import itertools
list(itertools.product([0.1, 0.5], [0.9, 0.99]))
# [(0.1, 0.9), (0.1, 0.99), (0.5, 0.9), (0.5, 0.99)]
len(list(itertools.product(range(2), repeat=3)))   # 8
```

## time.time
sig: `time.time()` → seconds since the epoch (float)
match: time.time(
tags: timing, benchmark, speed

Wall-clock time; subtract two readings to time a piece of code.

```python
import time
t0 = time.time()
sum(range(100_000))
elapsed = time.time() - t0      # e.g. 0.002 (seconds)
```

## Bit masks
sig: `x << i` · `x >> i` · `x & 1` · `x | m`
match: << , >> , mask |=
tags: bitmask, state encoding, collected

Pack a list of booleans into one integer: bit *i* is set when item *i* is true. Used to encode which cores have been collected in a single state id.

```python
collected = [True, False, True]
mask = 0
for i, got in enumerate(collected):
    mask |= int(got) << i
mask                                   # 5  (binary 101)
[bool(mask >> i & 1) for i in range(3)]   # [True, False, True]
```

> [!WARN] Precedence
> `mask >> i & 1` works because `>>` binds tighter than `&`, but `1 << i + 1` means `1 << (i + 1)`. Add parentheses when unsure.

## Number literals and None checks
sig: `50_000` · `1e-3` · `x is None` · `float("inf")`
match: _000, is None, is not None
tags: literal, none, constants

Underscores make big numbers readable (`50_000 == 50000`); `1e-3` is 0.001. Compare with `None` using `is`.

```python
capacity = 50_000
lr = 3e-4
seed = None
seed is None              # True
float("inf") > 1e308      # True
```

> [!WARN] `if seed:` is not `if seed is not None:`
> `seed = 0` is falsy, so `if seed:` skips a perfectly valid seed.
