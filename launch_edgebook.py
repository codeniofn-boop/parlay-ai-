#!/usr/bin/env python3
"""
launch_edgebook.py
==================

Double-click launcher for the EdgeBook AI dashboard (no Terminal needed).
Works from IDLE (press F5), from Python Launcher, or from a Terminal.

What it does, step by step, with every message shown in this window:

1. Checks that Streamlit really imports (a half-finished install is treated
   as missing and repaired).
2. If needed, installs Streamlit and pandas for the current user. This is a
   one-time download of roughly 100 MB and can take several minutes; the
   progress lines are streamed here so you can see it is working.
3. Starts the dashboard server and opens your browser at
   http://localhost:8501 once the server says it is ready.
4. Keeps running until you close this window or press Ctrl+C.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")
PORT = 8501
URL = f"http://localhost:{PORT}"


def say(message: str) -> None:
    """Print immediately (IDLE buffers output unless we flush)."""
    print(message, flush=True)


def run_streamed(args: list, label: str) -> int:
    """Run a command and echo every line it prints, so nothing happens silently."""
    say(f"--- {label} ---")
    try:
        proc = subprocess.Popen(args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as exc:
        say(f"Could not start {args[0]}: {exc}")
        return 1
    assert proc.stdout is not None
    for line in proc.stdout:
        say(line.rstrip())
    return proc.wait()


def streamlit_works() -> bool:
    """True only if the interpreter running this file can actually import Streamlit."""
    result = subprocess.run([sys.executable, "-c", "import streamlit, pandas; print(streamlit.__version__)"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        say(f"Streamlit {result.stdout.strip()} is installed.")
        return True
    detail = (result.stderr or result.stdout).strip().splitlines()
    say("Streamlit is not usable yet" + (f": {detail[-1]}" if detail else "."))
    return False


def ensure_streamlit() -> bool:
    if streamlit_works():
        return True
    say("")
    say("Installing Streamlit now. This is a one-time download of about 100 MB.")
    say("It can take 2 to 10 minutes depending on your connection. Please leave this window open.")
    say("")
    code = run_streamed([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "streamlit>=1.32", "pandas>=1.5"],
                        "pip install")
    if code != 0:
        say("")
        say(f"pip finished with exit code {code}. Read the lines above for the reason.")
        say("If it mentions SSL or a network problem, check your internet connection and run this file again.")
        return False
    say("")
    return streamlit_works()


def open_browser_when_ready(stop: threading.Event) -> None:
    """Poll the server's health endpoint, then open the browser exactly once."""
    import urllib.request

    deadline = time.time() + 90
    while not stop.is_set() and time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2) as resp:
                if resp.status == 200:
                    say(f"Server is ready. Opening {URL} in your browser.")
                    webbrowser.open(URL)
                    return
        except Exception:
            pass
        time.sleep(1)


def main() -> int:
    say("EdgeBook AI launcher")
    say(f"Python: {sys.executable}")
    say(f"Folder: {HERE}")
    if not os.path.isfile(APP):
        say("app.py was not found next to this launcher. Keep all the pipeline files in one folder.")
        return 1
    if not ensure_streamlit():
        return 1

    say("")
    say(f"Starting EdgeBook AI. Your browser will open at {URL} when the server is ready.")
    say("Leave this window open while you use the dashboard. Close it (or press Ctrl+C) to stop.")
    say("")
    stop = threading.Event()
    threading.Thread(target=open_browser_when_ready, args=(stop,), daemon=True).start()
    try:
        code = run_streamed([sys.executable, "-m", "streamlit", "run", APP, "--server.port", str(PORT),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], "streamlit")
    except KeyboardInterrupt:
        say("Stopped.")
        code = 0
    finally:
        stop.set()
    if code != 0:
        say("")
        say(f"Streamlit exited with code {code}. The lines above explain why.")
        say("If it says 'Address already in use', the dashboard is already running: open " + URL)
    return code


if __name__ == "__main__":
    sys.exit(main())
