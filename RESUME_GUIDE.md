# 🚀 Resuming Training & Speed Optimization Guide

## Resuming Training from Checkpoint

### Quick Resume (Recommended)

Simply run the training command - it will auto-detect your latest checkpoint:

```bash
python main.py --train
# Or
python train.py
```

You'll see:
```
Found existing checkpoint: checkpoints/dqn_breakout_episode_7200.pt
Resume from latest checkpoint? [Y/n]:
```

Press **Enter** or type **y** to resume!

### Resume from Specific Checkpoint

```bash
python main.py --resume checkpoints/dqn_breakout_episode_7200.pt
# Or
python train.py --resume checkpoints/dqn_breakout_episode_7200.pt
```

### Resume from Latest (No Prompt)

```bash
python main.py --resume latest
# Or
python train.py --resume latest
```

### Start Fresh (Ignore Checkpoints)

```bash
python main.py --fresh
# Or
python train.py --fresh
```

## What Gets Restored?

When you resume from a checkpoint, the following are restored:

✅ **Network weights** (policy and target networks)
✅ **Optimizer state** (Adam momentum, learning rates)
✅ **Training step count** (for proper epsilon decay)
✅ **Epsilon value** (exploration rate)
✅ **Training statistics** (rewards, losses, plots)
✅ **Episode counter** (continues from where you left off)

Your agent will continue exactly where it stopped!

## Speed Optimization Guide

### 1. Use Fast Config (Recommended)

Create a fast training script or modify your imports:

```python
# In train.py, change:
from utils import config_breakout as config
# To:
from utils import config_fast as config
```

**What this changes:**
- Larger batch size (32→64): Better GPU utilization
- Start learning sooner (10k→5k steps): Less waiting
- Faster exploration decay (1M→750k steps): Exploit sooner
- Slightly higher learning rate (0.00025→0.0003): Learn faster

**Expected speedup:** ~30-40% faster

### 2. Optimize Your Environment

Check your GPU usage:

```bash
# Monitor GPU in another terminal
watch -n 1 nvidia-smi
```

If GPU utilization is low (<50%), try:

```python
# In utils/config_breakout.py or config_fast.py
BATCH_SIZE = 128  # Even larger batches
```

### 3. Use Multiple CPU Cores

If you have a powerful CPU, you can process frames faster:

```bash
# Set environment variable before training
export OMP_NUM_THREADS=8
python main.py --train
```

### 4. Reduce Checkpoint Frequency

```python
# In config
SAVE_FREQ = 200  # Save every 200 episodes instead of 100
LOG_FREQ = 20    # Log less frequently
```

This reduces I/O overhead during training.

### 5. Compile Your Model (PyTorch 2.0+)

If you have PyTorch 2.0+, add this to `core/agent.py`:

```python
# After creating policy_net
self.policy_net = torch.compile(self.policy_net)
self.target_net = torch.compile(self.target_net)
```

**Expected speedup:** 10-20% on compatible hardware

### 6. Profile Your Training

Find bottlenecks:

```bash
# Run with profiling
python -m cProfile -o profile.stats train.py --resume latest

# Analyze results
python -m pstats profile.stats
> sort cumulative
> stats 20
```

## Speed Comparison

| Configuration | Episodes/Hour (GPU) | Episodes/Hour (CPU) |
| ------------- | ------------------- | ------------------- |
| Default       | ~40-50              | ~5-10               |
| Fast Config   | ~55-70              | ~7-12               |
| + Compiled    | ~60-80              | ~7-12               |
| + Optimized   | ~70-90              | ~8-15               |

*Estimated on RTX 3060 / Ryzen 5600X*

## Training Tips

### Monitor Your Progress

```bash
# Check latest checkpoint
ls -lht checkpoints/ | head

# View training plot
eog logs/training_plot.png  # or use your image viewer
```

### Adjust on the Fly

If training seems too slow or unstable:

1. **Stop training** (Ctrl+C)
2. **Edit config** (`utils/config_breakout.py` or `config_fast.py`)
3. **Resume training** (`python main.py --train`)

The new config will apply from the resumed episode!

### Best Practices

✅ **Do:**
- Resume from latest checkpoint after interruptions
- Use fast config if you have good GPU
- Monitor GPU utilization
- Check training plots regularly

❌ **Don't:**
- Change network architecture mid-training
- Switch between configs frequently
- Use batch sizes that OOM your GPU
- Train on CPU if you have GPU available

## Performance Without Hurting Agent

These optimizations are **safe** and won't hurt performance:

1. ✅ Larger batch size (32→64 or 128)
2. ✅ Model compilation (PyTorch 2.0+)
3. ✅ Reduce checkpoint frequency
4. ✅ Multi-threading for environment

These are **slightly aggressive** but still good:

1. ⚠️ Start learning sooner (10k→5k)
2. ⚠️ Faster epsilon decay (1M→750k)
3. ⚠️ Higher learning rate (0.00025→0.0003)

These might **hurt performance**:

1. ❌ Very high learning rate (>0.001)
2. ❌ Very small batch size (<16)
3. ❌ Skip target updates (frequency <500)
4. ❌ Start learning too early (<1000 steps)

## Quick Commands Reference

```bash
# Resume from latest (auto-detect)
python main.py --train

# Resume from specific checkpoint
python main.py --resume checkpoints/dqn_breakout_episode_7200.pt

# Resume from latest (no prompt)
python main.py --resume latest

# Start completely fresh
python main.py --fresh

# Evaluate current best
python main.py --evaluate --checkpoint checkpoints/dqn_breakout_episode_7200.pt
```

## Troubleshooting

**Q: Training doesn't resume, starts from episode 1**
A: Make sure you're using the updated `train.py` with resume support

**Q: "Checkpoint not found" error**
A: Check that the file exists: `ls checkpoints/`

**Q: GPU out of memory after changing batch size**
A: Reduce `BATCH_SIZE` in config, or reduce `REPLAY_BUFFER_SIZE`

**Q: Training is slower after resume**
A: Normal - epsilon is lower, so less random exploration (more computation)

**Q: Can I switch between configs?**
A: Yes! Just change the import in `train.py` and resume

---

You're currently at **episode 7200** - great progress! 🎉

Resume with: `python main.py --train` or `python main.py --resume latest`
