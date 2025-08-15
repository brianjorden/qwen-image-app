#!/bin/bash
# Test runner for qwen-image-app

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Running installation..."
    ./scripts/install.sh
    source .venv/bin/activate
fi

# Run the test suite
echo "🚀 Running Qwen-Image test suite..."
python tests/run_tests.py