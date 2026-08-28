#!/bin/bash
set -e
echo "[wrapper] start.sh begin at $(date)"
echo "[wrapper] pwd=$(pwd)"
echo "[wrapper] python3=$(which python3) version=$(python3 --version)"
echo "[wrapper] ls /app:"
ls -la /app/
echo "[wrapper] ls /app/models:"
ls -la /app/models/
echo "[wrapper] starting handler.py..."
exec python3 -u /app/handler.py
