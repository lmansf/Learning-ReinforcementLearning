"""
rlenvs — small, dependency-light classic environments for the lessons.

All environments follow the Gymnasium step API:

    obs, info = env.reset(seed=None)
    obs, reward, terminated, truncated, info = env.step(action)

Contents
    KArmedBandit   k-armed Gaussian bandit (stationary or drifting)
    GridWorld      tabular grid MDP from an ASCII map, with full model env.P for DP
    CliffWalking   the classic 4x12 cliff (Sutton & Barto, Example 6.6)
    Blackjack      simplified casino blackjack (S&B Example 5.1)
    CartPole       pole balancing (classic control physics), numpy
    MountainCar    under-powered car in a valley, numpy

Gymnasium is NOT required. If you have it installed, the same algorithms work on
gymnasium.make("CartPole-v1") etc. with no changes.
"""
from __future__ import annotations

import math
import random

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# =====================================================================
class KArmedBandit:
    """k-armed bandit. Arm a pays N(q*[a], 1).

    bandit = KArmedBandit(k=10, seed=0)
    r = bandit.pull(3)
    bandit.optimal_action  -> index of best arm (changes over time if nonstationary)
    """

    def __init__(self, k: int = 10, seed: int | None = None, nonstationary: bool = False,
                 drift: float = 0.01, mean: float = 0.0):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.nonstationary = nonstationary
        self.drift = drift
        self.q_star = (np.full(k, mean) if nonstationary else self.rng.normal(mean, 1.0, size=k)).astype(float)
        self.n_actions = k

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.q_star))

    def pull(self, a: int) -> float:
        r = float(self.rng.normal(self.q_star[a], 1.0))
        if self.nonstationary:
            self.q_star += self.rng.normal(0, self.drift, size=self.k)
        return r

    # gym-ish alias: a bandit is a one-state MDP
    def reset(self, seed=None):
        return 0, {}

    def step(self, a):
        return 0, self.pull(a), False, False, {}


# =====================================================================
class GridWorld:
    """Tabular grid MDP built from an ASCII map.

    Legend:  S start   . floor   # wall   G goal (terminal)   X pit (terminal)

        env = GridWorld(["...G",
                         ".#.X",
                         "S..."], slip=0.1, step_reward=-0.04)

    Rewards: entering G -> rewards['G'] (default +1), entering X -> rewards['X'] (default -1),
             any other move -> step_reward. Walls/borders: you stay put.
    slip: probability the move goes to one of the two perpendicular directions instead
          (split equally) — makes the MDP stochastic.

    Full model for dynamic programming (FrozenLake-style):
        env.P[s][a] -> list of (prob, next_state, reward, terminated)
    States are numbered row-major: s = r * cols + c (walls included, but never reachable).
    Helpers: env.to_grid(vector) -> 2-D array for rlhud.grid, env.walls_mask(), env.labels(),
             env.terminal_states, env.pos_of(s), env.state_of(r, c)
    """

    def __init__(self, layout=None, slip: float = 0.0, step_reward: float = 0.0,
                 rewards: dict | None = None, max_steps: int = 200, seed: int | None = None):
        self.layout = list(layout or ["...G", ".#.X", "S..."])
        self.rows, self.cols = len(self.layout), len(self.layout[0])
        self.slip = slip
        self.step_reward = step_reward
        self.rewards = {"G": 1.0, "X": -1.0}
        self.rewards.update(rewards or {})
        self.max_steps = max_steps
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.rng = random.Random(seed)
        self.start = None
        for r, line in enumerate(self.layout):
            for c, ch in enumerate(line):
                if ch == "S":
                    self.start = self.state_of(r, c)
        if self.start is None:
            self.start = 0
        self.terminal_states = {self.state_of(r, c) for r, line in enumerate(self.layout)
                                for c, ch in enumerate(line) if ch in "GX"}
        self.wall_states = {self.state_of(r, c) for r, line in enumerate(self.layout)
                            for c, ch in enumerate(line) if ch == "#"}
        self.P = self._build_model()
        self.s = self.start
        self.t = 0

    # ---- indexing
    def state_of(self, r, c):
        return r * self.cols + c

    def pos_of(self, s):
        return divmod(s, self.cols)

    def cell(self, s):
        r, c = self.pos_of(s)
        return self.layout[r][c]

    def _move(self, s, a):
        r, c = self.pos_of(s)
        dr, dc = _DELTAS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.layout[nr][nc] != "#":
            return self.state_of(nr, nc)
        return s

    def _build_model(self):
        P = {}
        for s in range(self.n_states):
            P[s] = {}
            for a in range(4):
                if s in self.terminal_states or s in self.wall_states:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                outcomes = {}
                perp = [(a + 1) % 4, (a + 3) % 4]
                for prob, aa in [(1 - self.slip, a)] + [(self.slip / 2, p) for p in perp]:
                    if prob <= 0:
                        continue
                    ns = self._move(s, aa)
                    ch = self.cell(ns)
                    rwd = self.rewards.get(ch, self.step_reward) if ch in "GX" else self.step_reward
                    done = ns in self.terminal_states
                    key = (ns, rwd, done)
                    outcomes[key] = outcomes.get(key, 0.0) + prob
                P[s][a] = [(p, ns, rwd, d) for (ns, rwd, d), p in outcomes.items()]
        return P

    # ---- gym API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.s = self.start
        self.t = 0
        return self.s, {}

    def step(self, a):
        outcomes = self.P[self.s][int(a)]
        x, acc = self.rng.random(), 0.0
        choice = outcomes[-1]
        for o in outcomes:
            acc += o[0]
            if x < acc:
                choice = o
                break
        _, ns, r, done = choice
        self.s = ns
        self.t += 1
        truncated = (not done) and self.t >= self.max_steps
        return ns, r, done, truncated, {}

    # ---- display helpers
    def to_grid(self, v):
        v = np.asarray(v, dtype=float).reshape(self.rows, self.cols).copy()
        for s in self.wall_states:
            v[self.pos_of(s)] = 0.0
        return v

    def walls_mask(self):
        return [[ch == "#" for ch in line] for line in self.layout]

    def labels(self):
        return [[ch if ch in "SGX" else "" for ch in line] for line in self.layout]

    def policy_grid(self, policy):
        """policy: array of actions per state (or n_states x n_actions probs) -> 2-D arrow grid."""
        p = np.asarray(policy)
        if p.ndim == 2:
            p = p.argmax(axis=1)
        g = p.reshape(self.rows, self.cols).tolist()
        for s in self.terminal_states | self.wall_states:
            r, c = self.pos_of(s)
            g[r][c] = None
        return g


# =====================================================================
class CliffWalking:
    """4x12 cliff. Start bottom-left, goal bottom-right, cliff in between.
    Every step -1; stepping into the cliff gives -100 and teleports you back to start.
    States 0..47 (row-major), actions UP RIGHT DOWN LEFT."""

    rows, cols = 4, 12
    n_states, n_actions = 48, 4

    def __init__(self, max_steps: int = 500):
        self.start = self.state_of(3, 0)
        self.goal = self.state_of(3, 11)
        self.cliff = {self.state_of(3, c) for c in range(1, 11)}
        self.max_steps = max_steps
        self.s, self.t = self.start, 0

    def state_of(self, r, c):
        return r * self.cols + c

    def pos_of(self, s):
        return divmod(s, self.cols)

    def reset(self, seed=None):
        self.s, self.t = self.start, 0
        return self.s, {}

    def step(self, a):
        r, c = self.pos_of(self.s)
        dr, dc = _DELTAS[int(a)]
        r, c = min(max(r + dr, 0), 3), min(max(c + dc, 0), 11)
        ns = self.state_of(r, c)
        self.t += 1
        if ns in self.cliff:
            self.s = self.start
            return self.s, -100.0, False, self.t >= self.max_steps, {"fell": True}
        self.s = ns
        done = ns == self.goal
        return ns, -1.0, done, (not done) and self.t >= self.max_steps, {}

    def to_grid(self, v):
        return np.asarray(v, dtype=float).reshape(4, 12)

    def labels(self):
        lab = [["" for _ in range(12)] for _ in range(4)]
        lab[3][0], lab[3][11] = "S", "G"
        for c in range(1, 11):
            lab[3][c] = "≈"
        return lab


# =====================================================================
class Blackjack:
    """Simplified blackjack vs dealer, infinite deck.
    obs = (player_sum 4..21, dealer_showing 1..10, usable_ace bool)
    actions: 0 = stick, 1 = hit.  reward +1 win, 0 draw, -1 loss (only at the end)."""

    n_actions = 2

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _card(self):
        return min(self.rng.randint(1, 13), 10)

    @staticmethod
    def _value(hand):
        s = sum(hand)
        usable = 1 in hand and s + 10 <= 21
        return (s + 10 if usable else s), usable

    def _obs(self):
        v, u = self._value(self.player)
        return (v, self.dealer[0], u)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.player = [self._card(), self._card()]
        self.dealer = [self._card(), self._card()]
        return self._obs(), {}

    def step(self, a):
        if a == 1:
            self.player.append(self._card())
            if self._value(self.player)[0] > 21:
                return self._obs(), -1.0, True, False, {}
            return self._obs(), 0.0, False, False, {}
        while self._value(self.dealer)[0] < 17:
            self.dealer.append(self._card())
        p, d = self._value(self.player)[0], self._value(self.dealer)[0]
        r = 1.0 if d > 21 or p > d else (0.0 if p == d else -1.0)
        return self._obs(), r, True, False, {}


# =====================================================================
class CartPole:
    """Balance a pole on a cart. obs = [x, x_dot, theta, theta_dot]; actions 0=push left 1=push right.
    +1 reward per step; terminates if |theta| > 12 deg or |x| > 2.4; truncates at max_steps."""

    n_actions = 2
    obs_size = 4

    def __init__(self, max_steps: int = 500, seed=None):
        self.gravity, self.masscart, self.masspole = 9.8, 1.0, 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag, self.tau = 10.0, 0.02
        self.theta_threshold = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costh, sinth = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sinth) / self.total_mass
        thetaacc = (self.gravity * sinth - costh * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costh ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costh / self.total_mass
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.t += 1
        terminated = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        truncated = (not terminated) and self.t >= self.max_steps
        return self.state.astype(np.float32), 1.0, terminated, truncated, {}


# =====================================================================
class MountainCar:
    """Drive an under-powered car up the right hill. obs = [position, velocity];
    actions 0=push left, 1=no push, 2=push right. -1 per step; done at position >= 0.5."""

    n_actions = 3
    obs_size = 2
    low = np.array([-1.2, -0.07])
    high = np.array([0.6, 0.07])

    def __init__(self, max_steps: int = 200, seed=None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.array([self.rng.uniform(-0.6, -0.4), 0.0])
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        pos, vel = self.state
        vel += (int(action) - 1) * 0.001 + math.cos(3 * pos) * (-0.0025)
        vel = float(np.clip(vel, -0.07, 0.07))
        pos += vel
        pos = float(np.clip(pos, -1.2, 0.6))
        if pos <= -1.2 and vel < 0:
            vel = 0.0
        self.state = np.array([pos, vel])
        self.t += 1
        terminated = pos >= 0.5
        truncated = (not terminated) and self.t >= self.max_steps
        return self.state.astype(np.float32), -1.0, terminated, truncated, {}
