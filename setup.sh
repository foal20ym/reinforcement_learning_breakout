#!/bin/bash

# Setup script for DQN Breakout project

echo "🎮 Setting up DQN Breakout project..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing Atari ROMs..."
pip install "gymnasium[accept-rom-license]"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the setup, run:"
echo "  python main.py --random"
echo ""
echo "To start training, run:"
echo "  python main.py --train"
