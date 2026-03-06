#!/bin/bash
# Supervisor script — auto-restarts the agent on crash.
# Usage: bash run.sh &

cd "$(dirname "$0")"

RESTART_DELAY=5
MAX_RESTARTS=10
restarts=0

echo "[supervisor] Starting Financial Analyst Agent..."

while true; do
    python3 main.py
    EXIT_CODE=$?

    restarts=$((restarts + 1))
    echo "[supervisor] Agent exited with code $EXIT_CODE (restart #$restarts)"

    if [ $restarts -ge $MAX_RESTARTS ]; then
        echo "[supervisor] Too many restarts ($MAX_RESTARTS). Giving up."
        exit 1
    fi

    echo "[supervisor] Restarting in ${RESTART_DELAY}s..."
    sleep $RESTART_DELAY
done
