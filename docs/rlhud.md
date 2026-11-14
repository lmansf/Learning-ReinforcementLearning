---
title: rlhud
---

## rlhud.plot
sig: `rlhud.plot(y, label="series", title="Chart", x=None, smooth=0, id=None, ylabel="", xlabel="")`
match: rlhud.plot(
tags: chart, line, learning curve, returns

Draws a line series in the VISUALS tab. Series with the same `title` (or `id`) share one chart; calling again with the same `label` **replaces** that series. `smooth=k` overlays a k-step moving average.

```python
import rlhud
returns = [float(i % 7) for i in range(200)]
rlhud.plot(returns, label="Q-learning", title="Return per episode", smooth=20)
rlhud.plot([3.0] * 200, label="optimal", title="Return per episode")   # same chart
```

> [!WARN] One chart per title
> Two calls with *different* titles make two charts. To compare methods, keep the title fixed and vary `label`. Outside the app it prints `[plot] title / label: n=… last-avg=…`.

## rlhud.chart
sig: `c = rlhud.chart(title="Live", smooth=0, id=None, ylabel="", xlabel="", every=0.15)`
match: rlhud.chart(
tags: live chart, streaming, training curve

A chart you append to *while* training. Points are buffered and sent at most every `every` seconds, so calling `add` every episode is cheap.

```python
import rlhud
live = rlhud.chart("Training return", smooth=20, xlabel="episode")
for ep in range(100):
    G = ep * 0.1
    live.add("agent", G)
live.flush()                 # send whatever is still buffered
```

## chart.add
sig: `c.add(label, value)`
match: live.add(, chart.add(, .add("
tags: append, live chart, series

Appends one value to the series `label` of a live chart. Use several labels for several lines on the same chart.

```python
import rlhud
c = rlhud.chart("Loss")
c.add("train", 0.9); c.add("eval", 1.1)
c.flush()
```

> [!WARN] Plain floats
> Pass a number, not an array: `c.add("loss", float(loss))`.

## chart.flush
sig: `c.flush()`
match: .flush()
tags: send, buffer, live chart

Sends the buffered points immediately. Call it once after the training loop so the final points appear (it also runs when the chart object is garbage-collected, but don't rely on that).

```python
import rlhud
c = rlhud.chart("Return")
for i in range(10): c.add("agent", i)
c.flush()
```

## rlhud.bars
sig: `rlhud.bars(values, labels=None, title="Bars", id=None, highlight=None, ylabel="")`
match: rlhud.bars(
tags: bar chart, bandit arms, Q values

Bar chart of a few numbers — e.g. estimated value per bandit arm. `highlight=i` colours bar `i` (say, the true best arm). Labels default to `0, 1, 2, …`.

```python
import rlhud, numpy as np
Q = np.array([0.2, 1.3, 0.7])
rlhud.bars(Q, labels=["a0", "a1", "a2"], title="Q estimates", highlight=int(np.argmax(Q)))
```

> [!WARN] highlight is an int
> Pass a Python int index (`int(np.argmax(...))`), not the label string.

## rlhud.grid
sig: `rlhud.grid(values, title="Grid", policy=None, walls=None, labels=None, id=None, fmt=".2f", cmap="signal")`
match: rlhud.grid(
tags: heatmap, value function, policy arrows, gridworld

Heat-map of a 2-D array (rows × cols). Optional overlays, all 2-D with the same shape:
- `policy` — action per cell drawn as arrows: `0=↑ 1=→ 2=↓ 3=← 4=•`; `None`/`-1` for no arrow
- `walls` — booleans, True cells drawn as walls
- `labels` — strings replacing the number (e.g. `"S"`, `"G"`)

`cmap="signal"` is diverging (good for ±values), `"heat"` sequential.

```python
import rlhud
from rlenvs import GridWorld
env = GridWorld()
V = [0.1 * s for s in range(env.n_states)]
rlhud.grid(env.to_grid(V), title="V", policy=env.policy_grid([1] * env.n_states),
           walls=env.walls_mask(), labels=env.labels())
```

> [!WARN] 2-D, not a vector
> Passing the flat `V` of length `n_states` draws a single row. Reshape with `env.to_grid(V)`; give `policy` as `env.policy_grid(pi)`.

## rlhud.stat
sig: `rlhud.stat(name, value, unit="", good=None)`
match: rlhud.stat(
tags: number, tile, metric, KPI

A big stat tile with one number. The latest value for a given `name` wins, so you can update it as training progresses. `good=True/False` colours it.

```python
import rlhud
rlhud.stat("Mean return (last 100)", 12.34)
rlhud.stat("Success rate", 87.5, unit="%", good=True)
```

> [!WARN] Scalars only
> Pass a number or short string; wrap NumPy values with `float(...)` so the tile shows a clean number.

## rlhud.table
sig: `rlhud.table(rows, headers=None, title="Table", id=None)`
match: rlhud.table(
tags: table, comparison, sweep results

A small table — rows are lists of cells. Format numbers yourself for tidy columns.

```python
import rlhud
rows = [[0.01, f"{1.23:.2f}"], [0.1, f"{1.41:.2f}"]]
rlhud.table(rows, headers=["ε", "mean reward"], title="ε sweep")
```

> [!WARN] Same title replaces
> A second `table` with the same title replaces the first. Use distinct titles (or `id=`) for separate tables.

## rlhud.text
sig: `rlhud.text(message, title="Note", tone="info")` — tone: `info` | `good` | `warn` | `bad`
match: rlhud.text(
tags: message, card, note, result

A highlighted message card — summarise a result or explain what the learner should notice.

```python
import rlhud
rlhud.text("Value iteration converged in 12 sweeps", title="Done", tone="good")
```

## rlhud.progress
sig: `rlhud.progress(i, n, label="Training")`
match: rlhud.progress(
tags: progress bar, status

A progress bar in the HUD status line; `i` is the 0-based iteration. Throttled internally, so call it every iteration.

```python
import rlhud
n = 500
for ep in range(n):
    rlhud.progress(ep, n, "Episodes")
```

## rlhud.replay
sig: `rlhud.replay(frames, title="Replay", id=None, fps=6)`
match: rlhud.replay(
tags: animation, Core Runner, episode video

Animates a Core Runner episode in the VISUALS tab. `frames` is a list of `env.snapshot()` dicts — record one after `reset` and one after every `step`, or let `run_episode(..., record=True)` do it.

```python
import rlhud
from corerunner import CoreRunner, run_episode
env = CoreRunner(level=1)
total, frames, info = run_episode(env, lambda obs: 4, seed=0, record=True)   # always WAIT
rlhud.replay(frames, title="Waiting drone", fps=8)
```

> [!WARN] Snapshots, not observations
> `frames` must be `env.snapshot()` dicts, not `obs` values.

## rlhud.in_app
sig: `rlhud.in_app()` → bool
match: rlhud.in_app(
tags: environment, terminal fallback, portable

True when the code runs inside the Reward Signal app. Outside the app every `rlhud` call prints a short text summary instead (`[stat] name: value`, `[plot] …`), so your script stays portable — no need to guard calls yourself.

```python
import rlhud
if not rlhud.in_app():
    print("running in a plain terminal; visuals become text")
```
