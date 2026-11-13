"""
Execution harness used by the Reward Signal server.

    python runner.py run   <user_file>
    python runner.py check <user_file> <check_file>

In `run` mode the user's file executes as __main__ (exactly like `python file.py`).
In `check` mode the user's file executes with __name__ == "__check__" (so code under
`if __name__ == "__main__":` is skipped), then the check file runs in the same
namespace with a few helpers injected:

    check(condition, label, detail="")   record a pass/fail line
    approx(a, b, tol=1e-6)               numeric closeness (works on arrays)
    require(name)                         fetch a name the learner must define
    seed_all(s)                           seed `random` and numpy
"""
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MARK = "\x1eRLHUD"


def emit(kind, **payload):
    payload["type"] = kind
    sys.stdout.write(MARK + json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


class _Missing(Exception):
    pass


def _user_frames_only(tb, user_file):
    """Format a traceback, hiding harness frames so errors point at the learner's code."""
    frames = [f for f in traceback.extract_tb(tb) if os.path.abspath(f.filename) != os.path.abspath(__file__)]
    return "".join(traceback.format_list(frames))


def run_user(path, name):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    ns = {"__name__": name, "__file__": path, "__builtins__": __builtins__}
    code = compile(src, "your_code.py", "exec")
    exec(code, ns)
    return ns


def main():
    mode = sys.argv[1]
    user_file = sys.argv[2]
    t0 = time.time()

    if mode == "run":
        try:
            run_user(user_file, "__main__")
        except SystemExit:
            raise
        except BaseException as e:  # noqa
            sys.stderr.write("Traceback (most recent call last):\n")
            sys.stderr.write(_user_frames_only(e.__traceback__, user_file))
            sys.stderr.write(f"{type(e).__name__}: {e}\n")
            sys.exit(1)
        return

    # ---------------------------------------------------------------- check mode
    check_file = sys.argv[3]
    results = {"passed": 0, "total": 0}

    def check(cond, label, detail=""):
        ok = bool(cond)
        results["total"] += 1
        results["passed"] += ok
        emit("check", label=str(label), ok=ok, detail=str(detail) if not ok or detail else "")
        return ok

    def approx(a, b, tol=1e-6):
        try:
            import numpy as np
            return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0))
        except Exception:
            return abs(float(a) - float(b)) <= tol

    try:
        ns = run_user(user_file, "__check__")
    except BaseException as e:  # noqa
        tb = _user_frames_only(e.__traceback__, user_file)
        emit("check", label="Your code runs without errors", ok=False,
             detail=f"{tb}{type(e).__name__}: {e}")
        emit("result", passed=0, total=1, elapsed=time.time() - t0)
        return

    def require(name):
        if name not in ns:
            raise _Missing(f"`{name}` is not defined. Did you rename or delete it?")
        return ns[name]

    def seed_all(s=0):
        random.seed(s)
        try:
            import numpy as np
            np.random.seed(s)
        except Exception:
            pass

    cns = dict(ns)
    cns.update(check=check, approx=approx, require=require, seed_all=seed_all, __name__="__checker__")
    with open(check_file, encoding="utf-8") as f:
        csrc = f.read()
    try:
        exec(compile(csrc, "check.py", "exec"), cns)
    except _Missing as e:
        check(False, "Required name present", str(e))
    except NotImplementedError as e:
        check(False, "Implementation complete", f"Something still raises NotImplementedError: {e}".rstrip(": "))
    except BaseException as e:  # noqa
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        # keep only the tail — the part that mentions the learner's code
        check(False, "Checks ran to completion", "".join(tb[-6:]))
    emit("result", passed=results["passed"], total=results["total"], elapsed=time.time() - t0)


if __name__ == "__main__":
    main()
