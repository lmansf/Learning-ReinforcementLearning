#!/usr/bin/env python3
"""
REWARD SIGNAL — local reinforcement-learning training ground.

    python server.py            # then open http://127.0.0.1:8765
    python server.py --port 9000 --no-browser

Standard library only. Your code runs with the same Python interpreter that
started this server, so anything you `pip install` (numpy, torch, gymnasium…)
is available to your exercises.

Security: the server binds to 127.0.0.1, rejects foreign Host headers (DNS
rebinding) and requires a per-launch token on every API call, so other web
pages cannot ask it to execute code.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import platform
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "app")
LESSONS = os.path.join(ROOT, "lessons")
RUNTIME = os.path.join(ROOT, "runtime")
WORKSPACE = os.path.join(ROOT, "workspace")
TOKEN = secrets.token_urlsafe(24)
MARK = b"\x1eRLHUD"

RUNS: dict[str, subprocess.Popen] = {}
RUNS_LOCK = threading.Lock()

sys.path.insert(0, RUNTIME)


# ------------------------------------------------------------------ lessons
def parse_front_matter(text: str):
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip()
                    if v.isdigit():
                        v = int(v)
                    meta[k.strip()] = v
            text = text[end + 4:].lstrip("\n")
    return meta, text


def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


def load_lessons():
    lessons = []
    if not os.path.isdir(LESSONS):
        return lessons
    for name in sorted(os.listdir(LESSONS)):
        ldir = os.path.join(LESSONS, name)
        md_path = os.path.join(ldir, "lesson.md")
        if not os.path.isfile(md_path):
            continue
        meta, body = parse_front_matter(_read(md_path))
        exercises = []
        edir = os.path.join(ldir, "exercises")
        if os.path.isdir(edir):
            for ename in sorted(os.listdir(edir)):
                ep = os.path.join(edir, ename)
                if not os.path.isdir(ep):
                    continue
                try:
                    emeta = json.loads(_read(os.path.join(ep, "meta.json"), "{}"))
                except json.JSONDecodeError as e:
                    emeta = {"title": f"{ename} (bad meta.json: {e})"}
                exercises.append({
                    "id": ename,
                    "title": emeta.get("title", ename),
                    "prompt": emeta.get("prompt", ""),
                    "hints": emeta.get("hints", []),
                    "xp": emeta.get("xp", 100),
                    "timeout": emeta.get("timeout", 60),
                    "kind": emeta.get("kind", "code"),
                    "starter": _read(os.path.join(ep, "starter.py")),
                    "solution": _read(os.path.join(ep, "solution.py")),
                    "has_check": os.path.isfile(os.path.join(ep, "check.py")),
                })
        lessons.append({
            "id": name,
            "title": meta.get("title", name),
            "chapter": meta.get("chapter", "Lessons"),
            "summary": meta.get("summary", ""),
            "minutes": meta.get("minutes", 15),
            "icon": meta.get("icon", ""),
            "body": body,
            "exercises": exercises,
        })
    return lessons


def env_report():
    def has(mod):
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    def ver(mod):
        if not has(mod):
            return None
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "yes")
        except Exception:
            return "broken"

    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(terse=True),
        "packages": {m: ver(m) for m in ("numpy", "torch", "gymnasium", "matplotlib")},
        "workspace": WORKSPACE,
    }


SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


# ------------------------------------------------------------------ handler
class Handler(BaseHTTPRequestHandler):
    server_version = "RewardSignal/1.0"

    def log_message(self, fmt, *args):  # quiet: only log API errors
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    # ---- guards
    def _host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _auth_ok(self):
        return secrets.compare_digest(self.headers.get("X-RS-Token", ""), TOKEN)

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- GET
    def do_GET(self):
        if not self._host_ok():
            return self.send_error(HTTPStatus.FORBIDDEN, "bad host")
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            if not self._auth_ok():
                return self._json({"error": "unauthorized"}, 401)
            if url.path == "/api/lessons":
                return self._json({"lessons": load_lessons()})
            if url.path == "/api/env":
                return self._json(env_report())
            if url.path == "/api/levels":
                import importlib
                import corerunner
                importlib.reload(corerunner)
                q = parse_qs(url.query)
                if "seed" in q:
                    cfg = corerunner.LEVELS[4]
                    rows = corerunner.generate_layout(int(q["seed"][0]), cfg["size"], cfg["n_cores"], cfg["n_vents"])
                    return self._json({"map": rows})
                return self._json({"levels": corerunner.export_levels()})
            return self._json({"error": "not found"}, 404)
        return self._static(url.path)

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        full = os.path.normpath(os.path.join(APP, path.lstrip("/")))
        if not full.startswith(APP) or not os.path.isfile(full):
            return self.send_error(HTTPStatus.NOT_FOUND)
        with open(full, "rb") as f:
            data = f.read()
        if full.endswith("index.html"):
            data = data.replace(b"__RS_TOKEN__", TOKEN.encode())
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if full.endswith(".js"):
            ctype = "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ---- POST
    def do_POST(self):
        if not self._host_ok() or not self._auth_ok():
            return self._json({"error": "unauthorized"}, 401)
        try:
            body = self._body()
        except Exception as e:
            return self._json({"error": str(e)}, 400)
        path = urlparse(self.path).path
        if path == "/api/run":
            return self._run(body)
        if path == "/api/stop":
            return self._stop(body)
        if path == "/api/save":
            return self._save(body)
        return self._json({"error": "not found"}, 404)

    def _save(self, body):
        lesson, ex = body.get("lesson", ""), body.get("exercise", "")
        if not (SAFE_ID.match(lesson) and SAFE_ID.match(ex)):
            return self._json({"error": "bad id"}, 400)
        d = os.path.join(WORKSPACE, lesson)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{ex}.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(body.get("code", ""))
        return self._json({"ok": True, "path": p})

    def _stop(self, body):
        with RUNS_LOCK:
            proc = RUNS.get(body.get("id", ""))
        if proc and proc.poll() is None:
            _kill(proc)
            return self._json({"ok": True})
        return self._json({"ok": False})

    def _run(self, body):
        code = body.get("code", "")
        mode = body.get("mode", "run")
        timeout = max(5, min(int(body.get("timeout", 120)), 1800))
        check_path = None
        if mode == "check":
            lesson, ex = body.get("lesson", ""), body.get("exercise", "")
            if not (SAFE_ID.match(lesson) and SAFE_ID.match(ex)):
                return self._json({"error": "bad id"}, 400)
            check_path = os.path.join(LESSONS, lesson, "exercises", ex, "check.py")
            if not os.path.isfile(check_path):
                return self._json({"error": "this exercise has no checks"}, 404)

        os.makedirs(WORKSPACE, exist_ok=True)
        fd, user_path = tempfile.mkstemp(prefix="rs_", suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [sys.executable, "-u", os.path.join(RUNTIME, "runner.py"), mode, user_path]
        if check_path:
            cmd.append(check_path)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["RLHUD_APP"] = "1"
        env["MPLBACKEND"] = "Agg"
        # Small RL networks run much faster on one BLAS thread (no thread-pool thrash).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
        env["PYTHONPATH"] = RUNTIME + os.pathsep + env.get("PYTHONPATH", "")
        popen_kw = {}
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        else:
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        proc = subprocess.Popen(cmd, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kw)
        run_id = uuid.uuid4().hex[:12]
        with RUNS_LOCK:
            RUNS[run_id] = proc

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        lock = threading.Lock()
        alive = [True]

        def send(obj):
            if not alive[0]:
                return
            try:
                with lock:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                alive[0] = False
                _kill(proc)  # browser went away: stop burning CPU

        def pump(stream, tag):
            for raw in iter(stream.readline, b""):
                if raw.startswith(MARK):
                    try:
                        send({"t": "ev", "d": json.loads(raw[len(MARK):])})
                        continue
                    except json.JSONDecodeError:
                        pass
                send({"t": tag, "d": raw.decode("utf-8", "replace")})
            stream.close()

        send({"t": "start", "id": run_id, "mode": mode})
        t0 = time.time()
        threads = [threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True),
                   threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True)]
        for th in threads:
            th.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            proc.wait()
        for th in threads:
            th.join(timeout=2)
        with RUNS_LOCK:
            RUNS.pop(run_id, None)
        try:
            os.unlink(user_path)
        except OSError:
            pass
        rc = proc.returncode
        send({"t": "exit", "code": rc, "elapsed": round(time.time() - t0, 3),
              "timeout": timed_out, "killed": rc is not None and rc < 0 and not timed_out})


def _kill(proc):
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser(description="Reward Signal — local RL training ground")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    os.makedirs(WORKSPACE, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  REWARD // SIGNAL   reinforcement learning lab │")
    print("  └──────────────────────────────────────────────┘")
    print(f"  serving   {url}")
    print(f"  python    {sys.executable}  ({platform.python_version()})")
    print(f"  workspace {WORKSPACE}")
    print("  Ctrl+C to quit\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down")
        with RUNS_LOCK:
            for p in RUNS.values():
                _kill(p)


if __name__ == "__main__":
    main()
