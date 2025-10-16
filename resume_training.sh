#!/bin/bash

# Quick resume training script
# Usage: ./resume_training.sh [--fast]

echo "🎮 DQN Breakout Training Resume Tool"
echo ""

# Check if --fast flag is provided
FAST_MODE=false
if [ "$1" == "--fast" ]; then
    FAST_MODE=true
    echo "⚡ Fast mode enabled!"
    echo ""
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls checkpoints/dqn_breakout_episode_*.pt 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoints found!"
    echo "Starting fresh training..."
    python main.py --train
    exit 0
fi

# Extract episode number
EPISODE=$(echo $LATEST_CHECKPOINT | grep -oP 'episode_\K\d+')

echo "📊 Found checkpoint: $LATEST_CHECKPOINT"
echo "📈 Episode: $EPISODE"
echo ""

# Switch to fast config if requested
if [ "$FAST_MODE" = true ]; then
    echo "Switching to fast configuration..."
    sed -i.bak 's/config_breakout/config_fast/g' train.py
    echo "✓ Fast config enabled (backup saved as train.py.bak)"
    echo ""
    echo "Fast config changes:"
    echo "  • Batch size: 32 → 64"
    echo "  • Min replay: 10k → 5k"
    echo "  • Epsilon decay: 1M → 750k steps"
    echo "  • Learning rate: 0.00025 → 0.0003"
    echo ""
fi

# Check GPU availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "🎮 GPU detected: $GPU_INFO"
else
    echo "💻 No GPU detected - training on CPU (will be slower)"
fi

echo ""
echo "▶️  Resuming training from episode $EPISODE..."
echo "   (Press Ctrl+C to stop)"
echo ""
sleep 2

# Resume training
python main.py --resume latest

# Restore config if changed
if [ "$FAST_MODE" = true ] && [ -f "train.py.bak" ]; then
    echo ""
    read -p "Restore original config? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mv train.py.bak train.py
        echo "✓ Original config restored"
    fi
fi

echo ""
echo "Training session completed!"
