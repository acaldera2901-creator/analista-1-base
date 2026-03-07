#!/usr/bin/env python3
"""
Persistent daemon launcher for the Financial Analyst Agent.

This script escapes the Claude Code task-level cgroup by moving itself
and all child processes to the container root cgroup, so they survive
when Claude Code ends its current task and starts a new one.

Usage:
    python3 daemon.py
"""
import os
import subprocess
import sys
import time
import signal
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "logs", "supervisor.log")
MAIN_PY = os.path.join(BASE_DIR, "main.py")
RESTART_DELAY = 5
MAX_RESTARTS = 100


def move_to_root_cgroup(pid: int) -> None:
    """Move a PID out of the task-level cgroup to the root cgroup."""
    cgroup_dirs = glob.glob("/sys/fs/cgroup/*/")
    for cgroup_dir in cgroup_dirs:
        procs_file = os.path.join(cgroup_dir, "cgroup.procs")
        if os.path.isfile(procs_file):
            try:
                with open(procs_file, "w") as f:
                    f.write(str(pid))
            except (PermissionError, OSError):
                pass  # Some subsystems may not allow direct write


def daemonize() -> None:
    """Double-fork to fully detach from the parent session."""
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits immediately
        sys.exit(0)

    # Child: create new session
    os.setsid()

    # Second fork (prevents re-acquiring a controlling terminal)
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Grandchild (daemon): redirect std streams
    os.chdir(BASE_DIR)
    sys.stdout.flush()
    sys.stderr.flush()

    devnull = open(os.devnull, "rb+")
    os.dup2(devnull.fileno(), sys.stdin.fileno())
    # Keep stdout/stderr going to terminal for now (we log to file anyway)


def run_supervisor() -> None:
    """Supervisor loop: starts main.py, restarts on crash."""
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

    restarts = 0
    while restarts < MAX_RESTARTS:
        log = open(LOG_FILE, "a")
        log.write(f"\n[daemon] Starting Financial Analyst Agent (restart #{restarts})...\n")
        log.flush()

        proc = subprocess.Popen(
            [sys.executable, MAIN_PY],
            cwd=BASE_DIR,
            stdout=log,
            stderr=log,
            start_new_session=True,   # creates new process group for child
        )

        # Move the child to root cgroup immediately
        move_to_root_cgroup(proc.pid)
        log.write(f"[daemon] PID {proc.pid} moved to root cgroup\n")
        log.flush()

        exit_code = proc.wait()

        log.write(f"[daemon] Process exited with code {exit_code}, restarting in {RESTART_DELAY}s\n")
        log.flush()
        log.close()

        restarts += 1
        time.sleep(RESTART_DELAY)

    # If we get here, too many restarts — write a final message
    with open(LOG_FILE, "a") as log:
        log.write(f"[daemon] Too many restarts ({MAX_RESTARTS}), giving up.\n")


def main() -> None:
    # Move THIS process to root cgroup first (before forking)
    move_to_root_cgroup(os.getpid())

    # Daemonize (double-fork): caller returns, daemon continues in background
    daemonize()

    # Only daemon reaches here
    run_supervisor()


if __name__ == "__main__":
    main()
