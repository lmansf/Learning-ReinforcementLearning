# Authoring lessons for Reward Signal

Lessons are plain files on disk — no build step. The server scans `lessons/` on every
page load, so you can edit a lesson and simply refresh the browser.

```
lessons/
  02-bandits/
    lesson.md                     # theory, quizzes, widgets, exercise slots
    exercises/
      01-epsilon-greedy/
        meta.json                 # title, prompt, hints, xp, timeout
        starter.py                # what the learner starts with
        solution.py               # reference solution (hidden unless enabled in Settings)
        check.py                  # automated checks (optional, but strongly preferred)
```

Lesson and exercise directories sort lexically, so prefix them with numbers.

Run `python tools/validate.py 02-bandits` after every change. It proves that the
solution passes every check, the starter does not, and every quiz/widget block parses.

---

## lesson.md

Front matter (simple `key: value` lines):

```
---
title: Multi-Armed Bandits
chapter: Foundations
summary: Exploration vs exploitation in the simplest possible RL problem.
minutes: 30
---
```

Chapters used by the curriculum (keep these exact strings so the nav groups correctly):
`Orientation`, `Foundations`, `Tabular Methods`, `Deep RL`, `Capstone`, `Bonus`.

### Markdown supported

- `#`, `##`, `###` headings (`##` headings become entries in the lesson's section rail)
- paragraphs, **bold**, *italic*, `inline code`, [links](https://example.com)
- `-` bullet lists and `1.` numbered lists (one level of nesting with 2-space indent)
- tables with `| a | b |` rows and a `|---|---|` separator
- `---` horizontal rule
- fenced code blocks: ` ```python ` (highlighted, with *Copy* and *Send to editor* buttons)
- callouts:
  ```
  > [!NOTE] Optional title
  > Body text, can span lines.
  ```
  kinds: `NOTE`, `TIP`, `WARN`, `KEY` (key idea — use for the one thing to remember), `MATH`
- equations: a `$$ ... $$` block on its own lines renders as an equation panel; inline `$...$`
  renders as math-styled text. **There is no LaTeX engine** (the app is fully offline):
  write math in Unicode — `V(s) = Σₐ π(a|s) Σₛ' p(s'|s,a) [r + γ V(s')]`, `←`, `≈`, `∈`, `α`,
  `γ`, `ε`, `θ`, `∇`, `ₜ`, `ₜ₊₁`, `²`, `≤`, `argmax`. Keep equations short and readable.

### Interactive blocks

**Quiz** — multiple choice, instant feedback, awards XP:

````
```quiz
{"q": "Which action does a greedy agent pick?",
 "options": ["A random one", "The one with highest estimated value", "The least tried one"],
 "answer": 1,
 "explain": "Greedy = exploit current estimates."}
```
````

**Exercise slot** — places the exercise card at this point in the lesson
(the id is the exercise directory name). Unreferenced exercises appear at the end.

````
```exercise
01-epsilon-greedy
```
````

**Widget** — built-in, in-browser interactive demos (JSON options):

| type | options | what it shows |
|---|---|---|
| `bandit` | `arms` (int), `seed` | pull arms by hand or auto-run ε-greedy; estimates vs hidden truth; reward & %-optimal charts |
| `discount` | `gamma`, `rewards` (list) | slider for γ, bar chart of γᵏ·rₖ and the resulting return |
| `gridworld` | `map` (list of strings, legend `S . # G X`), `gamma`, `step`, `slip`, `mode` = `"value-iteration"` or `"policy-evaluation"` | step through DP sweeps; values as heat map, greedy policy arrows |
| `qlearning` | `map`, `algo` = `"q"` or `"sarsa"`, `alpha`, `gamma`, `epsilon`, `step` | live tabular learning, agent animation, Q arrows, return chart; map may use `C` for cliff cells (−100, back to start) |
| `softmax` | `prefs` (list), `temperature` | action preferences → softmax probabilities with a temperature slider |
| `epsilon` | `values` (list), `epsilon` | how ε-greedy spreads probability over actions |
| `corerunner` | `level` (1–4) | the capstone game, playable with the keyboard |

````
```widget
{"type": "gridworld", "map": ["...G", ".#.X", "S..."], "gamma": 0.9, "mode": "value-iteration"}
```
````

---

## Exercises

### meta.json

```json
{
  "title": "ε-greedy action selection",
  "prompt": "Implement `epsilon_greedy(Q, epsilon, rng)` ... (Markdown allowed)",
  "hints": ["First hint, gentle.", "Second hint, more specific.", "Third hint nearly gives it away."],
  "xp": 100,
  "timeout": 60
}
```

`timeout` is seconds for the *check* run. Keep solutions comfortably under 60 % of it on
an ordinary laptop. Most exercises should check in < 10 s; deep-RL ones may take up to ~90 s.

### starter.py / solution.py

- Put the functions/classes to implement at the top, with docstrings and `# TODO` markers.
  Unimplemented parts should `raise NotImplementedError` (or return a placeholder) so the
  starter *runs* but fails checks.
- Put a demo under `if __name__ == "__main__":` that trains/plots with `rlhud`.
  **Run** executes the file as `__main__` (the learner sees the demo).
  **Check** executes it with `__name__ == "__check__"` (demo skipped), then runs check.py.
- Only rely on: the standard library, `numpy`, and the bundled modules
  `rlhud`, `rlenvs`, `corerunner`. PyTorch may be shown as an *optional* extra in lesson text,
  never required by a check.
- Keep code idiomatic, commented where it teaches something, and short.

### check.py

Runs in the same namespace as the learner's code, with these helpers injected:

| helper | purpose |
|---|---|
| `check(cond, label, detail="")` | record one pass/fail line. `label` says what is being tested; `detail` explains a failure in a way that helps the learner fix it |
| `approx(a, b, tol=1e-6)` | numeric/array closeness |
| `require("name")` | fetch a function/class the learner must define (friendly error if missing) |
| `seed_all(s)` | seed `random` and `numpy.random` |

Write checks that test *behaviour*, with small hand-computed cases first and a
statistical/performance test last. Make failure messages actionable, e.g.
`check(abs(v - 0.9) < 1e-6, "V(start) after one sweep", f"expected 0.9, got {v:.4f}")`.
Use fixed seeds so checks are deterministic. Guard each check so one failure doesn't stop the
rest when practical (a NotImplementedError will be reported nicely by the harness).

### Type-along walkthroughs (`guided.py`)

Each coding lesson opens with a **type-along**: the learner types real code over faint ghost text,
with a note for each line, then later exercises use the same tools with no help ("I do, we do, you do").

An exercise directory with a `guided.py` (and usually no `starter.py`/`solution.py`) is a type-along.
Name it `00-walkthrough-<slug>` so it sorts first, and place it in `lesson.md` before the first solo
exercise. Syntax inside `guided.py`:

```python
#> A note explaining the NEXT code line. Consecutive #> lines join into one note. Markdown allowed.
import numpy as np
#= given
# lines between `#= given` and `#= end` are provided: they fill in automatically
# when the learner presses Enter on the line before them (use for long, unteachable boilerplate)
#= end
```

Rules of thumb:
- 20–45 typed lines. Teach the *idioms the solo exercises need* (NumPy calls, loop shapes, update rules),
  on a **different but related** problem — never the solo exercise's own answer.
- Annotate nearly every non-trivial line: say *why*, name the gotcha, point at the DOCS entry.
- 4-space indentation, no tabs, no trailing whitespace. Keep lines short (≤ 90 chars).
- `check.py` verifies the typed code behaves correctly (small, fast, deterministic). `meta.json`: xp 50–100,
  `"hints": []`. The solution shown/validated is `guided.py` minus the `#>`/`#=` lines.

---

## API reference (`docs/*.md`) — the DOCS tab

The workbench's DOCS tab lists every reference entry whose `match:` strings appear in the current
exercise's code, and `F1` / `Ctrl+I` in the editor looks up the word under the cursor. Entries live in
`docs/<topic>.md` (front matter `title:` = topic name):

```
## np.argmax
sig: np.argmax(a, axis=None) → int
match: np.argmax(, .argmax(
tags: greedy, best action

What it does in one or two sentences, a short example, and a `> [!WARN]` for the classic mistake.
```

`match:` defaults to the title. Keep entries short and practical — they are looked up mid-exercise.

---

## rlhud — visuals from Python

```python
import rlhud
rlhud.plot(ys, label="Q-learning", title="Return per episode", smooth=20)
c = rlhud.chart("Live return", smooth=20); c.add("agent", G); c.flush()
rlhud.bars(Q, labels=["a0", "a1"], title="Q estimates", highlight=best)
rlhud.grid(V2d, title="V", policy=arrows2d, walls=mask, labels=labels)
rlhud.stat("Mean return", 12.3)
rlhud.table(rows, headers=["ε", "mean"])
rlhud.text("Converged in 12 sweeps", title="Done", tone="good")
rlhud.progress(i, n, "Training")
rlhud.replay(frames, title="Agent run")   # Core Runner snapshots
```

## rlenvs — bundled environments

`KArmedBandit`, `GridWorld` (with full model `env.P[s][a] -> [(p, s', r, done)]`),
`CliffWalking`, `Blackjack`, `CartPole`, `MountainCar`. See `runtime/rlenvs.py` docstrings.
