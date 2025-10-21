# Breakout DQN with Reward Shaping

Deep Q-Network implementation for Atari Breakout with reward shaping experiments.

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

**Requirements:**

- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU support)
- Gymnasium with Atari environments

### 2. Basic Training

```bash
# Train with default settings (baseline DQN)
python breakout_dqn.py

# Auto-resume from checkpoint if available
python breakout_dqn.py
```

**Checkpoints saved to:** `models/`

### 3. Testing Trained Model

```bash
# Edit breakout_dqn.py and uncomment the test line:
# breakout_dqn.test(10, "models/CNN_breakout.pt")
```

---

## 🧪 Running Experiments

Compare different reward shaping strategies:

```bash
# Run all experiments (quick test - 100 episodes each)
python experiments/experiment_runner.py --config all --episodes 100

# Run specific experiment
python experiments/experiment_runner.py --config baseline --episodes 500

# GPU-optimized long run (5000 episodes)
python experiments/experiment_runner.py --config all_combined --episodes 5000 --gpu

# List available configurations
python experiments/experiment_runner.py --list
```

**Results saved to:** `experiments/results/`

### Available Configurations

- `baseline` - No reward shaping
- `paddle_hit_only` - Bonus for hitting paddle
- `center_position_only` - Bonus for center positioning
- `side_angle_only` - Bonus for side angles
- `block_multiplier_only` - Scaled block rewards
- `ball_loss_penalty_only` - Penalty for losing ball
- `survival_bonus_only` - Bonus for staying alive
- `all_combined` - All bonuses combined
- `bonuses_only` - All bonuses except penalty

---

## 🔧 Hyperparameter Tuning

Random search for optimal hyperparameters:

```bash
# Run random search (10 trials, 100 episodes each)
python hyperparameter_search/random_search.py

# Results saved to: hyperparameter_search/results/
```

**Searches:**

- Learning rate (3e-5 to 5e-4)
- Epsilon decay (0.985 to 0.998)
- Mini-batch size (32, 64, 128)

**Output:**

- `random_search_results.json` - All trial results
- `best_config.json` - Top configuration

---

## ⚡ Quick Commands Summary

```bash
# Basic training
python breakout_dqn.py

# Run all experiments (quick)
python experiments/experiment_runner.py --config all --episodes 100

# Hyperparameter search
python hyperparameter_search/random_search.py

# List experiment configs
python experiments/experiment_runner.py --list
```

---
