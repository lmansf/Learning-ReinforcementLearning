# REWARD // SIGNAL

A local, hands-on reinforcement-learning training ground that runs in your browser. Read
the theory and try interactive demos on the left; write, run and auto-check real Python
on your own machine on the right. Everything sits inside a game-style HUD with XP, ranks,
a command palette and a settings menu.

## Quick start

You need Python 3.10+ and NumPy.

```bash
pip install numpy
```

```bash
python server.py
```

Your browser opens at <http://127.0.0.1:8765>. Use `--port 9000` to pick another port,
or `--no-browser` to skip opening a browser.

The server uses only the standard library. Your code runs with the **same Python interpreter
that started the server**, so anything you `pip install` (torch, gymnasium,
stable-baselines3…) is available to your exercises.

## Google Drive / Colab edition

The whole course also ships as **Google Colab notebooks that live in your Google Drive**, no
local install needed: theory, quizzes, the interactive demos, auto-checked exercises with hints
and solutions, XP and ranks, and both capstones.

1. Download [`colab/RewardSignal.zip`](colab/RewardSignal.zip) (or the
   [`colab/RewardSignal/`](colab/RewardSignal) folder) and put it in **My Drive** so you have
   `My Drive/RewardSignal/` (*New → Folder upload*, or upload the zip and extract it there).
2. In Drive, open **`00 · START HERE.ipynb`** with Google Colaboratory and run its first cell
   (it mounts your Drive and finds the folder).
3. Work through `00 · Boot Sequence.ipynb` … `15 · Bonus Capstone — Stock the Shelves.ipynb`.

How it maps to the app:

| Desktop app | Colab edition |
|---|---|
| Run | run the exercise cell; `rlhud` charts, grids, stats and Core Runner replays render below it (live charts update in place) |
| Check | the **✅ Check** cell: re-runs the code you last ran in the exercise cell with `__name__ == "__check__"`, then the same `check.py` with the same helpers |
| Hints | expandable 💡 cells that open one after another; a failed check also shows the next hint |
| Solutions | hidden 🔒 form cells (double-click to reveal). The Settings cell can lock them (`after pass` / `never`); `python tools/build_colab.py --no-solutions` builds an edition without them |
| Widgets, quizzes | the same JS demos as the app (bundled from `app/js`), quizzes record answers and XP |
| XP / progress | `progress.json` in the Drive folder, shared by all notebooks; the dashboard in START HERE shows rank, XP, completion and the course map |
| `workspace/` | `RewardSignal/workspace/` in Drive (e.g. the trained weights of Core Runner mission 4) |

Everything is offline after upload: the HUD's JS and CSS live in `RewardSignal/runtime/assets/`
(Google Fonts are optional). The notebooks also run in a local Jupyter: open them from the folder.

**Maintainers:** lessons stay the single source of truth. After editing `lessons/` or `app/js`, run

```bash
python tools/build_colab.py                    # regenerates colab/RewardSignal/ and colab/RewardSignal.zip
pip install nbformat nbclient ipykernel
OMP_NUM_THREADS=1 python tools/validate_colab.py   # executes every notebook with the solutions; all checks must pass
```

## Curriculum

| # | Mission | Chapter |
|---|---|---|
| 00 | Boot Sequence: how the app works | Orientation |
| 01 | The RL Problem: agents, rewards, returns, discounting | Foundations |
| 02 | Multi-Armed Bandits: ε-greedy, optimistic init, UCB, gradient bandits | Foundations |
| 03 | Markov Decision Processes: Bellman equations, v/q, optimality | Foundations |
| 04 | Dynamic Programming: policy evaluation, policy and value iteration | Tabular Methods |
| 05 | Monte Carlo: prediction, control, importance sampling (Blackjack) | Tabular Methods |
| 06 | Temporal-Difference: TD(0), SARSA, Q-learning, Expected SARSA, Double Q | Tabular Methods |
| 07 | n-step & Planning: n-step TD, TD(λ), Dyna-Q | Tabular Methods |
| 08 | Function Approximation: tile coding, semi-gradient SARSA (Mountain Car) | Deep RL |
| 09 | Deep Q-Networks: NumPy MLP + backprop, replay buffer, target nets (CartPole) | Deep RL |
| 10 | Policy Gradients: softmax policies, REINFORCE with baseline | Deep RL |
| 11 | Actor-Critic: advantages, GAE | Deep RL |
| 12 | PPO: clipped surrogate objective | Deep RL |
| 13 | **Capstone: Core Runner**, four missions in an original salvage game | Capstone |
| 14 | Beyond: Gymnasium, Stable-Baselines3, where to go next | Capstone |
| 15 | **Bonus capstone: inventory control** with simulated or your own sales data | Bonus |

All algorithms are implemented from scratch in NumPy. PyTorch is optional.

## The HUD

- **Run** (`Ctrl+Enter`) executes your file. `print()` output goes to *Console*; `rlhud`
  charts, heat-map grids, stat tiles and game replays go to *Visuals*.
- **Check** (`Ctrl+Shift+Enter`) runs the exercise's automated tests and awards XP.
- **Hints** unlock progressively. **Solutions** are *hidden / on request / after pass / always*.
  Change this under Settings → Learning (`Esc`).
- `Ctrl+K` opens the command palette. `[` / `]` move to the previous / next lesson. `\` toggles the mission list.
  `Ctrl+S` exports the current file to `workspace/<lesson>/`.
- Progress and code autosave in your browser. Back them up from Settings → Data.

## Project layout

```
server.py            local server (stdlib only)
app/                 HUD frontend (vanilla JS, no build step)
runtime/             modules importable from exercises:
  rlhud.py           visuals API
  rlenvs.py          bandit, GridWorld, CliffWalking, Blackjack, CartPole, MountainCar
  corerunner.py      the capstone game
  runner.py          execution / checking harness
  rlcolab.py         Colab / Jupyter edition: setup, quizzes, widgets, checks, XP, dashboard
lessons/             Markdown lessons + exercises (see AUTHORING.md)
tools/validate.py    proves every solution passes and every starter does not
tools/build_colab.py builds the Google Drive / Colab edition into colab/
tools/validate_colab.py  executes the generated notebooks and checks every exercise passes
colab/               the generated Colab edition (RewardSignal/ folder + RewardSignal.zip)
workspace/           working directory for your runs and exported files
```

## Security

The server binds to `127.0.0.1` only, rejects foreign `Host` headers and requires a
per-launch token on every API call, so other websites cannot make it run code. It does run
the code *you* write, with your user's permissions, which is the point of the app.
