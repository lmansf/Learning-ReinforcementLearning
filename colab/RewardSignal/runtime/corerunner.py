"""
CORE RUNNER — the Reward Signal capstone environment.

You pilot a salvage drone through a derelict station. Collect every energy
core, then reach the exit hatch before the battery dies. Vents and patrol
sentries end the run.

API (Gymnasium-style, no dependency on gymnasium):

    from corerunner import CoreRunner
    env = CoreRunner(level=1)                 # 1..4  (4 = procedural)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

Actions: 0=UP 1=RIGHT 2=DOWN 3=LEFT 4=WAIT

Observation modes (choose with obs="..."):
    "tabular" (default)  int in [0, env.n_states)  — position x cores-collected x sentry-phase
    "vector"             np.float32 array of hand-made features (env.obs_size)
    "grid"               np.float32 array (6, H, W) of one-hot channels

Rewards:
    step            -0.1
    bump a wall     -0.5   (you stay put)
    collect core    +10
    exit (unlocked) +20    episode ends; exit is LOCKED until all cores are collected
    vent / sentry   -20    episode ends
    battery empty          episode truncated

Rendering: env.snapshot() returns a JSON-able dict; rlhud.replay(frames) animates them.
"""
from __future__ import annotations

import random as _random
from collections import deque

try:
    import numpy as _np
except Exception:  # numpy is optional for tabular play
    _np = None

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

R_STEP = -0.1
R_BUMP = -0.5
R_CORE = 10.0
R_EXIT = 20.0
R_DEATH = -20.0

# Legend: S start, . floor, # wall, C core, X vent, E exit
LEVELS = {
    1: {
        "name": "Boot Sequence",
        "brief": "One core, one exit. Learn to walk.",
        "map": [
            "S...#",
            ".##..",
            "...#C",
            "#.#..",
            "E....",
        ],
        "sentries": [],
        "battery": 40,
        "par": 28.0,      # return of a strong policy (used for star ratings)
    },
    2: {
        "name": "Vent Maze",
        "brief": "Three cores between scalding vents. Plan the route.",
        "map": [
            "S..X..C",
            ".#.X.#.",
            ".#...#.",
            "C#X#...",
            "...#X#.",
            ".X.....",
            "..C#X.E",
        ],
        "sentries": [],
        "battery": 70,
        "par": 46.0,
    },
    3: {
        "name": "Patrol Deck",
        "brief": "Sentries sweep the corridors. Timing is everything — WAIT is an action.",
        "map": [
            "S...#..C",
            ".##.#.#.",
            "........",
            "C#.##.#.",
            ".#......",
            ".#.#X#..",
            "...#C..E",
        ],
        "sentries": [
            [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]],
            [[4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7]],
        ],
        "battery": 80,
        "par": 44.0,
    },
    4: {
        "name": "Deep Salvage",
        "brief": "A new procedurally generated wreck every episode. Generalise.",
        "map": None,
        "size": (6, 6),
        "n_cores": 2,
        "n_vents": 2,
        "sentries": [],
        "battery": 50,
        "par": 36.0,
    },
}


def _lcm(a, b):
    from math import gcd
    return a * b // gcd(a, b) if a and b else max(a, b)


def _sentry_pos(path, t):
    """Ping-pong along path; position after t moves."""
    n = len(path)
    if n == 1:
        return tuple(path[0])
    period = 2 * (n - 1)
    k = t % period
    idx = k if k < n else period - k
    return tuple(path[idx])


def _bfs(walls, h, w, start, blocked=()):
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DELTAS[:4]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and (nr, nc) not in walls and (nr, nc) not in blocked:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def generate_layout(seed, size=(6, 6), n_cores=2, n_vents=2):
    """Procedural wreck: random walls/vents with every core and the exit reachable."""
    rng = _random.Random(seed)
    h, w = size
    for _ in range(1000):
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        start, exit_ = cells[0], cells[1]
        cores = cells[2:2 + n_cores]
        vents = set(cells[2 + n_cores:2 + n_cores + n_vents])
        walls = set(c for c in cells[2 + n_cores + n_vents:] if rng.random() < 0.18)
        if abs(start[0] - exit_[0]) + abs(start[1] - exit_[1]) < 4:
            continue
        reach = _bfs(walls, h, w, start, blocked=vents)
        if exit_ in reach and all(c in reach for c in cores):
            rows = []
            for r in range(h):
                row = ""
                for c in range(w):
                    p = (r, c)
                    row += ("S" if p == start else "E" if p == exit_ else "C" if p in cores
                            else "X" if p in vents else "#" if p in walls else ".")
                rows.append(row)
            return rows
    raise RuntimeError("could not generate a layout")


class CoreRunner:
    ACTIONS = ACTIONS
    n_actions = 5

    def __init__(self, level: int = 1, obs: str = "tabular", seed: int | None = None,
                 battery: int | None = None, layout: list[str] | None = None):
        if level not in LEVELS and layout is None:
            raise ValueError(f"level must be one of {list(LEVELS)}")
        self.level = level
        self.cfg = LEVELS.get(level, LEVELS[1])
        self.obs_mode = obs
        self._fixed_layout = layout
        self._battery_cfg = battery
        self._rng = _random.Random(seed)
        self._layout_seed = seed
        self._load(layout or self.cfg["map"] or generate_layout(
            self._rng.randrange(1 << 30), self.cfg["size"], self.cfg["n_cores"], self.cfg["n_vents"]))
        self.t = 0
        self.done = True

    # ------------------------------------------------------------ layout
    def _load(self, rows):
        self.rows = list(rows)
        self.h, self.w = len(rows), len(rows[0])
        self.walls, self.vents, cores = set(), set(), []
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                p = (r, c)
                if ch == "#": self.walls.add(p)
                elif ch == "X": self.vents.add(p)
                elif ch == "C": cores.append(p)
                elif ch == "S": self.start = p
                elif ch == "E": self.exit = p
        self.core_list = cores
        self.sentry_paths = [[tuple(p) for p in path] for path in self.cfg.get("sentries", [])] if self._fixed_layout is None else []
        period = 1
        for path in self.sentry_paths:
            period = _lcm(period, max(1, 2 * (len(path) - 1)))
        self.period = period
        self.max_battery = self._battery_cfg or self.cfg["battery"]
        n_cells = self.h * self.w
        self.n_states = n_cells * (2 ** len(cores)) * self.period
        self.obs_size = 5 * 5 * 5 + 8

    # ------------------------------------------------------------ core API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = _random.Random(seed)
        if self.cfg.get("map") is None and self._fixed_layout is None:
            self._load(generate_layout(self._rng.randrange(1 << 30), self.cfg["size"],
                                       self.cfg["n_cores"], self.cfg["n_vents"]))
        self.pos = self.start
        self.collected = [False] * len(self.core_list)
        self.t = 0
        self.battery = self.max_battery
        self.score = 0.0
        self.done = False
        self.last_event = "start"
        return self._obs(), self._info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is over — call reset() first.")
        action = int(action)
        if not 0 <= action < 5:
            raise ValueError(f"action must be 0..4, got {action}")
        reward = R_STEP
        event = ACTIONS[action].lower()
        dr, dc = DELTAS[action]
        old = self.pos
        nr, nc = old[0] + dr, old[1] + dc
        if not (0 <= nr < self.h and 0 <= nc < self.w) or (nr, nc) in self.walls:
            reward += R_BUMP
            event = "bump"
            nr, nc = old
        self.pos = (nr, nc)
        old_s = self.sentries()
        self.t += 1
        self.battery -= 1
        new_s = self.sentries()
        terminated = False

        hit = self.pos in new_s or any(self.pos == o and old == n for o, n in zip(old_s, new_s))
        if hit:
            reward += R_DEATH; terminated = True; event = "caught"
        elif self.pos in self.vents:
            reward += R_DEATH; terminated = True; event = "vent"
        else:
            for i, c in enumerate(self.core_list):
                if c == self.pos and not self.collected[i]:
                    self.collected[i] = True
                    reward += R_CORE
                    event = "core"
            if self.pos == self.exit and all(self.collected):
                reward += R_EXIT; terminated = True; event = "escaped"
        truncated = (not terminated) and self.battery <= 0
        if truncated:
            event = "battery"
        self.score += reward
        self.done = terminated or truncated
        self.last_event = event
        return self._obs(), reward, terminated, truncated, self._info()

    def sentries(self, t: int | None = None):
        t = self.t if t is None else t
        return [_sentry_pos(p, t) for p in self.sentry_paths]

    @property
    def exit_open(self):
        return all(self.collected)

    def _info(self):
        return {"event": self.last_event, "cores": sum(self.collected), "cores_total": len(self.core_list),
                "battery": self.battery, "score": self.score, "t": self.t}

    # ------------------------------------------------------------ observations
    def state_id(self):
        r, c = self.pos
        mask = 0
        for i, got in enumerate(self.collected):
            mask |= int(got) << i
        return ((r * self.w + c) * (2 ** len(self.core_list)) + mask) * self.period + (self.t % self.period)

    def _obs(self):
        if self.obs_mode == "tabular":
            return self.state_id()
        if self.obs_mode == "vector":
            return self.vector()
        if self.obs_mode == "grid":
            return self.grid()
        raise ValueError(f"unknown obs mode {self.obs_mode!r}")

    def vector(self):
        """Egocentric 5x5 view (wall, vent, core, sentry, exit channels) + 8 globals."""
        if _np is None:
            raise RuntimeError("numpy is required for vector observations")
        v = _np.zeros((5, 5, 5), dtype=_np.float32)
        sent = set(self.sentries()) | set(self.sentries(self.t + 1))
        cores = {c for c, g in zip(self.core_list, self.collected) if not g}
        r0, c0 = self.pos
        for i in range(5):
            for j in range(5):
                r, c = r0 + i - 2, c0 + j - 2
                p = (r, c)
                if not (0 <= r < self.h and 0 <= c < self.w) or p in self.walls:
                    v[0, i, j] = 1
                if p in self.vents: v[1, i, j] = 1
                if p in cores: v[2, i, j] = 1
                if p in sent: v[3, i, j] = 1
                if p == self.exit: v[4, i, j] = 1
        # direction to the current target (nearest core, or exit when unlocked)
        if cores:
            tr, tc = min(cores, key=lambda q: abs(q[0] - r0) + abs(q[1] - c0))
        else:
            tr, tc = self.exit
        scale = float(max(self.h, self.w))
        g = _np.array([
            (tr - r0) / scale, (tc - c0) / scale,
            float(tr < r0), float(tc > c0), float(tr > r0), float(tc < c0),
            float(self.exit_open), self.battery / self.max_battery,
        ], dtype=_np.float32)
        return _np.concatenate([v.ravel(), g])

    def grid(self):
        if _np is None:
            raise RuntimeError("numpy is required for grid observations")
        g = _np.zeros((6, self.h, self.w), dtype=_np.float32)
        for p in self.walls: g[0][p] = 1
        for p in self.vents: g[1][p] = 1
        for c, got in zip(self.core_list, self.collected):
            if not got: g[2][c] = 1
        for p in self.sentries(): g[3][p] = 1
        g[4][self.exit] = 1 if self.exit_open else 0.5
        g[5][self.pos] = 1
        return g

    # ------------------------------------------------------------ helpers
    def snapshot(self):
        """JSON-friendly frame for rlhud.replay()."""
        return {
            "w": self.w, "h": self.h, "level": self.level,
            "walls": sorted(self.walls), "vents": sorted(self.vents),
            "cores": [c for c, g in zip(self.core_list, self.collected) if not g],
            "cores_total": len(self.core_list), "exit": self.exit, "exit_open": self.exit_open,
            "sentries": self.sentries(), "agent": self.pos, "t": self.t,
            "battery": self.battery, "max_battery": self.max_battery,
            "score": round(self.score, 2), "event": self.last_event,
        }

    def render(self):
        """ASCII render for terminals."""
        s = set(self.sentries())
        out = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                p = (r, c)
                if p == self.pos: ch = "@"
                elif p in s: ch = "!"
                elif p in self.walls: ch = "#"
                elif p in self.vents: ch = "X"
                elif p in self.core_list and not self.collected[self.core_list.index(p)]: ch = "C"
                elif p == self.exit: ch = "E" if self.exit_open else "e"
                else: ch = "."
                row += ch
            out.append(row)
        out.append(f"t={self.t} battery={self.battery} score={self.score:.1f} event={self.last_event}")
        return "\n".join(out)

    def __repr__(self):
        return f"CoreRunner(level={self.level}, {self.h}x{self.w}, obs={self.obs_mode!r})"


def run_episode(env, policy, seed=None, record=False, max_steps=10_000):
    """Roll out policy(obs) -> action. Returns (total_reward, frames, info)."""
    obs, info = env.reset(seed=seed)
    frames = [env.snapshot()] if record else []
    total = 0.0
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        if record:
            frames.append(env.snapshot())
        if term or trunc:
            break
    return total, frames, info


def export_levels():
    """Level data for the browser 'play it yourself' mode."""
    out = {}
    for k, cfg in LEVELS.items():
        d = {kk: vv for kk, vv in cfg.items()}
        out[str(k)] = d
    return out
