# 🎛️ Hyperparameter Tuning Guide

## Current Configuration

The default configuration in `utils/config_breakout.py` is based on the original DQN paper:

```python
LEARNING_RATE = 0.00025
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 1000000
TRAIN_FREQ = 4
```

## Suggested Configurations

### 🚀 Fast Testing (Quick Experimentation)
Use this for rapid prototyping and debugging:

```python
MAX_EPISODES = 100              # Just test basic functionality
MIN_REPLAY_SIZE = 1000          # Start learning sooner
REPLAY_BUFFER_SIZE = 10000      # Smaller memory footprint
EPS_DECAY_STEPS = 50000         # Faster exploration decay
SAVE_FREQ = 10                  # Save more frequently
```

### ⚡ Fast Learning (Trade Stability for Speed)
More aggressive learning, less stable:

```python
LEARNING_RATE = 0.0005          # Higher learning rate
BATCH_SIZE = 64                 # Larger batches
MIN_REPLAY_SIZE = 5000          # Start learning earlier
TARGET_UPDATE_FREQ = 500        # Update target more often
EPS_DECAY_STEPS = 500000        # Faster exploration decay
```

### 🎯 Stable Learning (Recommended for Final Training)
Default is already good, but can tune:

```python
LEARNING_RATE = 0.00025         # Standard
BATCH_SIZE = 32                 # Standard
MIN_REPLAY_SIZE = 10000         # Build diverse replay
TARGET_UPDATE_FREQ = 1000       # Standard
EPS_DECAY_STEPS = 1000000       # Gradual exploration
```

### 🏆 Maximum Performance (Long Training)
For best results with more training time:

```python
LEARNING_RATE = 0.0001          # Lower for fine-tuning
BATCH_SIZE = 64                 # Larger, more stable gradients
REPLAY_BUFFER_SIZE = 200000     # More diverse experiences
MIN_REPLAY_SIZE = 20000         # Better initial diversity
TARGET_UPDATE_FREQ = 2000       # More stable targets
EPS_END = 0.05                  # Less exploration at end
EPS_DECAY_STEPS = 2000000       # Very gradual decay
```

## Parameter Effects

### Learning Rate
- **Higher (0.001+)**: Faster learning, but may be unstable
- **Standard (0.00025)**: Balanced, from original paper
- **Lower (0.0001)**: Slower but more stable convergence

### Batch Size
- **Small (16-32)**: Less stable, faster iteration
- **Medium (32-64)**: Good balance
- **Large (64-128)**: More stable gradients, slower

### Replay Buffer Size
- **Small (10k-50k)**: Less memory, less diverse
- **Medium (100k)**: Standard, good balance
- **Large (200k+)**: More diverse, needs more memory

### Epsilon Decay Steps
- **Fast (100k-500k)**: Quick shift to exploitation
- **Standard (1M)**: Original paper recommendation
- **Slow (2M+)**: More exploration, better for complex games

### Target Update Frequency
- **Frequent (500)**: Target tracks policy closely, less stable
- **Standard (1000)**: Good balance
- **Rare (2000+)**: More stable targets, slower adaptation

## Problem-Specific Tuning

### Agent Not Learning (Reward Not Increasing)
Try:
```python
LEARNING_RATE = 0.0005          # Increase
MIN_REPLAY_SIZE = 5000          # Decrease (start learning sooner)
EPS_DECAY_STEPS = 500000        # Decrease (explore less)
```

### Training Unstable (Reward Fluctuating Wildly)
Try:
```python
LEARNING_RATE = 0.0001          # Decrease
BATCH_SIZE = 64                 # Increase
TARGET_UPDATE_FREQ = 2000       # Increase
```

### Agent Plateaus Early
Try:
```python
EPS_END = 0.05                  # More exploration
REPLAY_BUFFER_SIZE = 200000     # More diversity
MIN_REPLAY_SIZE = 20000         # Better initial replay
```

### Out of Memory
Try:
```python
BATCH_SIZE = 16                 # Decrease
REPLAY_BUFFER_SIZE = 50000      # Decrease
```

### Training Too Slow
Try:
```python
BATCH_SIZE = 64                 # Increase (GPU)
TRAIN_FREQ = 8                  # Train less often
MIN_REPLAY_SIZE = 5000          # Start sooner
```

## Advanced Configurations

### Double DQN (Future Extension)
When implementing Double DQN, keep same hyperparameters initially.

### Dueling DQN (Future Extension)
May benefit from:
```python
LEARNING_RATE = 0.0001          # Slightly lower
```

### Prioritized Replay (Future Extension)
Add:
```python
PRIORITY_ALPHA = 0.6            # How much prioritization
PRIORITY_BETA_START = 0.4       # Importance sampling correction
PRIORITY_BETA_FRAMES = 100000   # Anneal to 1.0
```

## Monitoring and Adjustment

### During Training, Watch For:

1. **Average Reward Curve**
   - Should generally increase over time
   - Some fluctuation is normal
   - Plateaus suggest need for tuning

2. **Loss Values**
   - Should decrease initially
   - May fluctuate as agent explores
   - Very high loss suggests learning rate too high

3. **Episode Length**
   - Should increase as agent learns
   - Breakout: starts ~300, can reach 1000+

4. **Epsilon Value**
   - Should decay smoothly
   - Check if agent explores enough initially

## Experimentation Tips

1. **Start with defaults**: They work reasonably well
2. **Change one thing at a time**: Isolate effects
3. **Keep notes**: Track what worked and what didn't
4. **Use checkpoints**: Can always resume from good states
5. **Plot everything**: Visualize to understand behavior
6. **Be patient**: Deep RL takes time to converge

## Quick Reference

| Want to...          | Adjust...          | Direction |
| ------------------- | ------------------ | --------- |
| Learn faster        | LEARNING_RATE      | ↑         |
| More stability      | LEARNING_RATE      | ↓         |
| Smoother gradients  | BATCH_SIZE         | ↑         |
| Less memory         | REPLAY_BUFFER_SIZE | ↓         |
| More exploration    | EPS_DECAY_STEPS    | ↑         |
| Exploit sooner      | EPS_DECAY_STEPS    | ↓         |
| More stable targets | TARGET_UPDATE_FREQ | ↑         |

Remember: Deep RL is as much art as science. Experimentation is key!
