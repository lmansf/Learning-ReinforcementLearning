---
title: Boot Sequence
chapter: Orientation
summary: How this training ground works — reading panes, the workbench, checks, and your HUD.
minutes: 8
---

# Welcome, operator

This is a hands-on course in **reinforcement learning (RL)** — the branch of machine learning
where an *agent* learns to act by trial and error, guided only by a numeric **reward signal**.
No labelled answers. No teacher showing the right move. Just consequences.

By the end you will have written, from scratch and in plain Python + NumPy:

- bandit algorithms that balance exploration and exploitation
- dynamic programming planners that solve a known world exactly
- Monte Carlo, SARSA and Q-learning agents that learn from experience
- Dyna-Q, which learns a model and *imagines* extra experience
- linear function approximation, a Deep Q-Network, REINFORCE, actor-critic and PPO
- an agent that beats **CORE RUNNER**, the capstone salvage game

## How the HUD works

The screen is split in two. **This pane** is the briefing: theory, interactive widgets and
quizzes. The **workbench** on the right is where you write and run real Python on your own machine.

| Control | What it does |
|---|---|
| **Run** `Ctrl+Enter` | Runs your file like `python file.py` — demos, plots, prints |
| **Check** `Ctrl+Shift+Enter` | Runs the automated checks for the exercise and awards XP |
| **Hints / Solution** tabs | Hints unlock one at a time. Solutions follow your *Settings* |
| `Ctrl+K` | Command palette — jump to any lesson |
| `Esc` | Settings (solutions, hints, display, execution timeout…) |
| `[` / `]` | Previous / next lesson (when not typing) |

Your code and progress save automatically in this browser. **Export** writes the current
file to `workspace/<lesson>/` so you can open it in your own editor too.

> [!KEY] The loop you will see everywhere
> The agent observes a state, chooses an action, the environment answers with a reward and a
> new state. Repeat. Every algorithm in this course is a different answer to one question:
> *how should the agent change its behaviour given the rewards it has seen?*

## Your first transmission

Visuals in the workbench come from a tiny helper module called `rlhud`. Anything you
`print()` lands in **Console**; anything sent with `rlhud` appears in **Visuals**.

```python
import rlhud
rlhud.plot([1, 4, 9, 16], label="squares", title="Hello HUD")
rlhud.stat("Operator status", "online")
```

```quiz
{"q": "What does the Check button do that Run does not?",
 "options": ["Runs your code faster", "Runs automated tests against your code and awards XP", "Uploads your code to the cloud"],
 "answer": 1,
 "explain": "Check runs the exercise's test file against the functions you wrote. Everything runs locally — nothing leaves your machine."}
```

```exercise
01-first-signal
```
