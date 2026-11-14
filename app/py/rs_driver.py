"""
On-device execution driver for the Reward Signal mobile apps.

Runs inside Pyodide (CPython compiled to WebAssembly) in a Web Worker and gives
learner code the same semantics as server.py's /api/run on the desktop:

  * runtime/runner.py is used unchanged (run / check modes, rlhud markers),
  * RLHUD_APP=1, the working directory is the persistent workspace folder,
  * every run starts from a clean slate: modules imported by the previous run are
    dropped and the random generators are reseeded, like a fresh process.

The worker streams sys.stdout / sys.stderr line by line and turns RLHUD marker lines
into visual events, exactly as server.py does with the subprocess pipes.
"""
import os
import random
import sys
import traceback

RS = "/home/pyodide/rs"
RUNTIME = RS + "/runtime"
WORKSPACE = "/home/pyodide/workspace"
_BASELINE = set()


def setup():
    """Called once after the interpreter boots. Returns an environment report."""
    os.environ.update(RLHUD_APP="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if RUNTIME not in sys.path:
        sys.path.insert(0, RUNTIME)
    os.makedirs(WORKSPACE, exist_ok=True)
    os.chdir(WORKSPACE)
    import numpy  # noqa: F401  (pre-imported: it is the slow part of a cold start)
    _BASELINE.clear()
    _BASELINE.update(sys.modules)
    return report()


def report():
    import platform

    def ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "yes")
        except Exception:
            return None

    try:
        import pyodide
        pv = pyodide.__version__
    except Exception:
        pv = "?"
    return {
        "python": platform.python_version(),
        "executable": f"Pyodide {pv} (WebAssembly, on this device)",
        "platform": f"{sys.platform} / wasm32",
        "pyodide": pv,
        "packages": {"numpy": ver("numpy"), "torch": None, "gymnasium": None, "matplotlib": None},
        "workspace": WORKSPACE,
    }


def _fresh_state():
    # forget modules imported by the previous run (the learner's own imports and the
    # bundled runtime modules), so edits and global state never leak between runs
    for name in list(sys.modules):
        if name not in _BASELINE:
            del sys.modules[name]
    random.seed()
    try:
        import numpy as np
        np.random.seed(None)
    except Exception:
        pass
    os.makedirs(WORKSPACE, exist_ok=True)
    os.chdir(WORKSPACE)


def run(mode, code, check_src=None):
    """Execute like `python runner.py <mode> <file> [check]`. Returns the exit code."""
    _fresh_state()
    user = "/tmp/rs_user.py"
    with open(user, "w", encoding="utf-8") as f:
        f.write(code)
    argv = ["runner.py", mode, user]
    if check_src is not None:
        chk = "/tmp/rs_check.py"
        with open(chk, "w", encoding="utf-8") as f:
            f.write(check_src)
        argv.append(chk)
    sys.argv = argv
    rc = 0
    try:
        import runner
        runner.main()
    except SystemExit as e:
        if e.code is None:
            rc = 0
        elif isinstance(e.code, int):
            rc = e.code
        else:
            sys.stderr.write(str(e.code) + "\n")
            rc = 1
    except BaseException:  # noqa
        traceback.print_exc()
        rc = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
    return rc
