#!/usr/bin/env python3
"""
launch_edgebook.py
==================

Double-click launcher for the EdgeBook AI dashboard (no Terminal needed).

1. Checks that Streamlit is installed; installs it for the current user on
   first run (one-time, needs an internet connection).
2. Starts ``streamlit run app.py`` from this folder, which opens the dashboard
   in your default browser at http://localhost:8501.

Close the window that appears (or press Ctrl+C in it) to stop the server.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_streamlit() -> bool:
    if importlib.util.find_spec("streamlit") is not None:
        return True
    print("Streamlit is not installed yet. Installing it now (one time)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "streamlit>=1.32", "pandas>=1.5"])
    except subprocess.CalledProcessError as exc:
        print(f"Could not install Streamlit automatically (exit code {exc.returncode}).")
        print("Open Terminal and run:  python3 -m pip install --user streamlit")
        return False
    return importlib.util.find_spec("streamlit") is not None


def main() -> int:
    if not ensure_streamlit():
        return 1
    app = os.path.join(HERE, "app.py")
    if not os.path.isfile(app):
        print(f"app.py was not found next to this launcher ({HERE}).")
        return 1
    print("Starting EdgeBook AI at http://localhost:8501 ... (close this window to stop)")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", app, "--server.headless", "false"], cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
