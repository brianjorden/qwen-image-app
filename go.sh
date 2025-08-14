#!/bin/bash
# Launch the qwen-image-app GUI
set -e

export HF_HUB_OFFLINE=1

echo ""

# Activate virtual environment if it exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Installing..."
    echo ""
    ./scripts/install.sh
fi

source .venv/bin/activate

    # Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "Configuration not found at config.yaml"
    echo "Please create it with your model paths."
    exit 1
fi

# Launch the GUI
echo "Starting qwen-image-app GUI..."
echo "Press Ctrl+C to stop"
echo ""

python -m app.app "$@"
