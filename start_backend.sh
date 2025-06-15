#!/bin/bash

echo "Starting MONET Dermatology Analyzer Backend..."

cd app

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Start FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload 