#!/bin/bash
# Script to start local server with proper library paths for SHAP
# Uses Python 3.10 environment to match AWS Lambda (scikit-learn 1.0.2)

cd "$(dirname "$0")/.."

# Try Python 3.10 environment first (matches AWS Lambda), fallback to venv_local
if [ -d "venv_py310" ]; then
    echo "[INFO] Using Python 3.10 environment (matches AWS Lambda)"
    source venv_py310/bin/activate
elif [ -d "venv_local" ]; then
    echo "[INFO] Using venv_local (Python 3.11 - may have compatibility issues)"
    source venv_local/bin/activate
else
    echo "[ERROR] No virtual environment found. Please create venv_py310 or venv_local"
    exit 1
fi

# Set library path for llvmlite (needed for SHAP on macOS)
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:/usr/lib:$DYLD_LIBRARY_PATH

# Check if port 8000 is already in use and kill the process
PORT=8000
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "[INFO] Port $PORT is in use by process $PID. Killing it..."
    kill -9 $PID 2>/dev/null
    sleep 1
    echo "[OK] Port $PORT is now free"
fi

# Start server
python serve.py

