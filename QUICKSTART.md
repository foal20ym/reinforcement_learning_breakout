# 🚀 Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install "gymnasium[accept-rom-license]"
   ```

2. **Verify installation:**
   ```bash
   python main.py --random
   ```
   You should see random gameplay in a window.

## Training Your Agent

Start training with default settings:
```bash
python main.py --train
```

Training will:
- Run for up to 10,000 episodes (configurable in `utils/config_breakout.py`)
- Save checkpoints every 100 episodes
- Display progress every 10 episodes
- Generate plots showing reward progression

**Expected output:**
```
Using device: cuda
Creating environment: ALE/Breakout-v5
Number of actions: 4
Episode 10 | Reward: 1.00 | Avg(100): 0.80 | Length: 312 | Epsilon: 0.997
Episode 20 | Reward: 2.00 | Avg(100): 1.15 | Length: 405 | Epsilon: 0.994
...
```

## Evaluating Your Agent

After training (or using a saved checkpoint):
```bash
python main.py --evaluate --checkpoint checkpoints/dqn_breakout_final.pt
```

## Adjusting Hyperparameters

Edit `utils/config_breakout.py`:
```python
# For faster training (less stable):
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_FREQ = 500

# For more exploration:
EPS_END = 0.2
EPS_DECAY_STEPS = 2000000

# For faster learning:
LEARNING_RATE = 0.0005
```

## Tips

1. **Training takes time**: Expect several hours on CPU, 1-2 hours on GPU
2. **Start small**: Test with `MAX_EPISODES = 100` first
3. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
4. **Check logs**: View training progress in `logs/training_plot.png`

## Common Issues

**No GPU detected?**
- PyTorch will automatically use CPU (slower)
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**ROM not found?**
- Run: `pip install "gymnasium[accept-rom-license]"`

**Out of memory?**
- Reduce `BATCH_SIZE` or `REPLAY_BUFFER_SIZE` in config

## Next Steps

- Implement Double DQN
- Add Dueling DQN architecture
- Try Prioritized Experience Replay
- Experiment with different hyperparameters
- Test on other Atari games (change `ENV_NAME` in config)
