"""
rlhud — send visuals from your Python code to the Reward Signal HUD.

Everything here is optional sugar: plain print() always works. When your code
runs inside the app, these calls render charts, grids, stat tiles and game
replays in the VISUALS tab. When you run the same file from a normal terminal,
they quietly print a short text summary instead, so your scripts stay portable.
Inside a Jupyter / Google Colab notebook they render as rich HUD-style outputs
(via the rlcolab module of the Colab edition).

    import rlhud
    rlhud.plot(returns, label="Q-learning", title="Episode return")
    rlhud.grid(V, title="State values", policy=greedy_actions)
    rlhud.stat("Best return", 42.0)

Live charts (update while training):

    chart = rlhud.chart("Training return", smooth=20)
    for ep in range(500):
        ...
        chart.add("agent", G)        # streamed in small batches
    chart.flush()
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

_MARK = "\x1eRLHUD"
_IN_APP = os.environ.get("RLHUD_APP") == "1"


def _detect_notebook() -> bool:
    """True inside a Jupyter / Colab kernel (set RLHUD_NOTEBOOK=0 to force text output)."""
    if _IN_APP or os.environ.get("RLHUD_NOTEBOOK") == "0":
        return False
    ipy = sys.modules.get("IPython")
    if ipy is None:
        return False
    try:
        ip = ipy.get_ipython()
    except Exception:
        return False
    return ip is not None and getattr(ip, "kernel", None) is not None


_IN_NB = _detect_notebook()

__all__ = [
    "plot", "chart", "bars", "grid", "stat", "replay", "table",
    "progress", "text", "in_app", "in_notebook",
]


def in_app() -> bool:
    """True when running inside the Reward Signal app."""
    return _IN_APP


def in_notebook() -> bool:
    """True when visuals render as rich notebook outputs (Colab / Jupyter, via rlcolab)."""
    return _IN_NB


def _clean(x):
    """Convert numpy scalars/arrays and other oddities into JSON-safe values."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if hasattr(x, "tolist"):  # numpy arrays / scalars
        return _clean(x.tolist())
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return str(x)


def _emit(kind: str, **payload):
    if _IN_NB:
        payload["type"] = kind
        try:
            import rlcolab
            return rlcolab.hud_event(_clean(payload))
        except Exception as e:  # never let a visual break the learner's run
            sys.stderr.write(f"[rlhud] notebook render failed ({type(e).__name__}: {e}); falling back to text\n")
            return False
    if not _IN_APP:
        return False
    payload["type"] = kind
    line = _MARK + json.dumps(_clean(payload), separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return True


# ---------------------------------------------------------------- charts
def plot(y, label: str = "series", title: str = "Chart", x=None,
         smooth: int = 0, id: str | None = None, ylabel: str = "", xlabel: str = ""):
    """Draw (or replace) a line series on a chart.

    y       : sequence of numbers
    label   : legend name of this series (same label replaces the old series)
    title   : chart title; series with the same title/id share one chart
    smooth  : moving-average window shown as a bold line over the raw data
    """
    y = list(_clean(list(y)))
    if not _emit("plot", id=id or title, title=title, label=label, y=y,
                 x=_clean(list(x)) if x is not None else None,
                 smooth=int(smooth), ylabel=ylabel, xlabel=xlabel):
        if y:
            vals = [v for v in y if v is not None]
            tail = vals[-min(len(vals), max(1, smooth or 10)):]
            print(f"[plot] {title} / {label}: n={len(vals)} last-avg={sum(tail)/len(tail):.3f}")


class chart:
    """A live chart you can append to during training."""

    def __init__(self, title: str = "Live", smooth: int = 0, id: str | None = None,
                 ylabel: str = "", xlabel: str = "", every: float = 0.15):
        self.id = id or title
        self.title = title
        self.smooth = smooth
        self.ylabel, self.xlabel = ylabel, xlabel
        self.every = every
        self._buf: dict[str, list] = {}
        self._last = 0.0
        self._count: dict[str, int] = {}
        _emit("chart", id=self.id, title=title, smooth=smooth, ylabel=ylabel, xlabel=xlabel)

    def add(self, label: str, value):
        self._buf.setdefault(label, []).append(value)
        self._count[label] = self._count.get(label, 0) + 1
        now = time.time()
        if now - self._last >= self.every:
            self.flush()

    def flush(self):
        if self._buf:
            if not _emit("append", id=self.id, series=self._buf):
                pass
            self._buf = {}
        self._last = time.time()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


def bars(values, labels=None, title: str = "Bars", id: str | None = None,
         highlight: int | None = None, ylabel: str = ""):
    """Bar chart, e.g. estimated value of each bandit arm."""
    values = _clean(list(values))
    labels = [str(l) for l in (labels if labels is not None else range(len(values)))]
    if not _emit("bars", id=id or title, title=title, values=values, labels=labels,
                 highlight=highlight, ylabel=ylabel):
        print(f"[bars] {title}: " + ", ".join(f"{l}={v:.3g}" for l, v in zip(labels, values)))


# ---------------------------------------------------------------- grids
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "•"}


def grid(values, title: str = "Grid", policy=None, walls=None, labels=None,
         id: str | None = None, fmt: str = ".2f", cmap: str = "signal"):
    """Heat-map of a 2-D array (rows x cols).

    policy : optional 2-D array of action indices (0=up 1=right 2=down 3=left 4=stay)
             drawn as arrows; use -1/None for no arrow.
    walls  : optional 2-D boolean array; True cells are drawn as walls.
    labels : optional 2-D array of strings overriding cell text (e.g. 'S', 'G').
    cmap   : 'signal' (diverging, good for +/- values) or 'heat' (sequential).
    """
    vals = _clean(values)
    if vals and not isinstance(vals[0], list):
        vals = [vals]
    if not _emit("grid", id=id or title, title=title, values=vals,
                 policy=_clean(policy), walls=_clean(walls), labels=_clean(labels),
                 fmt=fmt, cmap=cmap):
        print(f"[grid] {title}")
        for r, row in enumerate(vals):
            cells = []
            for c, v in enumerate(row):
                if walls is not None and _clean(walls)[r][c]:
                    cells.append("  ### ")
                    continue
                s = f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)
                if policy is not None:
                    p = _clean(policy)[r][c]
                    s += ARROWS.get(p, " ") if p is not None else " "
                cells.append(s.rjust(7))
            print(" ".join(cells))


# ---------------------------------------------------------------- misc
def stat(name: str, value, unit: str = "", good: bool | None = None):
    """A HUD stat tile (latest value wins for a given name)."""
    if not _emit("stat", name=name, value=value, unit=unit, good=good):
        v = f"{value:.4g}" if isinstance(value, float) else value
        print(f"[stat] {name}: {v}{unit}")


def table(rows, headers=None, title: str = "Table", id: str | None = None):
    rows = _clean([list(r) for r in rows])
    if not _emit("table", id=id or title, title=title, rows=rows,
                 headers=list(headers) if headers else None):
        print(f"[table] {title}")
        if headers:
            print(" | ".join(map(str, headers)))
        for r in rows:
            print(" | ".join(map(str, r)))


def text(message: str, title: str = "Note", tone: str = "info"):
    """A highlighted message card. tone: info | good | warn | bad"""
    if not _emit("text", title=title, message=str(message), tone=tone):
        print(f"[{title}] {message}")


_last_progress = [0.0]


def progress(i: int, n: int, label: str = "Training"):
    """Progress bar in the HUD status line. Cheap to call every iteration."""
    now = time.time()
    if i + 1 >= n or now - _last_progress[0] > 0.1:
        _last_progress[0] = now
        if not _emit("progress", i=int(i) + 1, n=int(n), label=label):
            if i + 1 >= n:
                print(f"[progress] {label}: done ({n})")


def replay(frames, title: str = "Replay", id: str | None = None, fps: int = 6):
    """Animate a Core Runner episode: a list of env.snapshot() dicts."""
    frames = _clean(list(frames))
    if not _emit("replay", id=id or title, title=title, frames=frames, fps=fps):
        if frames:
            last = frames[-1]
            print(f"[replay] {title}: {len(frames)} frames, final score {last.get('score')}")
