#!/bin/bash
# Test reward shaping implementation

echo "Testing Reward Shaping Implementation..."
echo "========================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Found virtual environment"
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

echo ""
echo "Running reward shaping tests..."
echo "========================================="
echo ""

python test_reward_shaping.py

echo ""
echo "========================================="
echo "Tests complete!"
