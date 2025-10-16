# 🎮 Deep Q-Network (DQN) for Atari Breakout

A minimal but complete implementation of Deep Q-Network (DQN) for playing Atari Breakout using PyTorch and Gymnasium.

## 📋 Overview

This project implements a DQN agent that learns to play Atari Breakout through reinforcement learning. The implementation includes:

- **Convolutional Neural Network** for Q-value estimation
- **Experience Replay Buffer** for stable training
- **Target Network** with periodic updates
- **Epsilon-greedy exploration** with linear decay
- **Frame preprocessing** (grayscale, resizing, stacking)
- **Reward clipping** for stable learning
- **Training visualization** and checkpointing

## 🏗️ Project Structure

```
reinforcement_learning_breakout/
├── core/
│   ├── agent.py              # DQN Agent with training logic
│   ├── model.py              # CNN architecture for Q-network
│   └── replay_buffer.py      # Experience replay memory
├── environment/
│   └── preprocessing.py      # Environment wrappers (frame stacking, etc.)
├── utils/
│   ├── config_breakout.py    # Hyperparameters
│   └── logger.py             # Training metrics logging
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs and plots
├── main.py                   # Main entry point
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── requirements.txt          # Python dependencies
```

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ROMs (Required for ALE)

```bash
# Download and install Atari ROMs
pip install "gymnasium[accept-rom-license]"
```

Alternatively, you can manually install ROMs:
```bash
ale-import-roms /path/to/roms
```

## 🎯 Usage

### Training

Train a new DQN agent:
```bash
python main.py --train
```

This will:
- Train the agent for up to 10,000 episodes
- Save checkpoints every 100 episodes to `checkpoints/`
- Log training metrics every 10 episodes
- Generate training plots in `logs/`

### Evaluation

Evaluate a trained agent:
```bash
python main.py --evaluate --checkpoint checkpoints/dqn_breakout_final.pt --episodes 10
```

### Random Play (Testing)

Test the environment setup with random actions:
```bash
python main.py --random
```

## ⚙️ Configuration

Edit `utils/config_breakout.py` to adjust hyperparameters:

```python
# Key hyperparameters
LEARNING_RATE = 0.00025        # Learning rate
GAMMA = 0.99                   # Discount factor
BATCH_SIZE = 32                # Training batch size
REPLAY_BUFFER_SIZE = 100000    # Replay buffer capacity
MIN_REPLAY_SIZE = 10000        # Start training after this many steps
TARGET_UPDATE_FREQ = 1000      # Target network update frequency
EPS_START = 1.0                # Initial exploration rate
EPS_END = 0.1                  # Final exploration rate
EPS_DECAY_STEPS = 1000000      # Epsilon decay over N steps
```

## 🧠 Network Architecture

The DQN uses a convolutional neural network:

```
Input: (4, 84, 84) - 4 stacked grayscale frames

Conv1: 32 filters, 8x8 kernel, stride 4 → ReLU
Conv2: 64 filters, 4x4 kernel, stride 2 → ReLU
Conv3: 64 filters, 3x3 kernel, stride 1 → ReLU

Flatten: 64 * 7 * 7 = 3136

FC1: 3136 → 512 → ReLU
FC2: 512 → n_actions

Output: Q-values for each action
```

## 📊 Training Details

### Preprocessing
- **Grayscale conversion**: RGB → Grayscale
- **Frame resize**: 210×160 → 84×84
- **Frame stacking**: Stack 4 consecutive frames
- **Frame skipping**: Repeat action for 4 frames, max pool last 2
- **Reward clipping**: Clip rewards to [-1, 1]

### Exploration Strategy
- Epsilon-greedy with linear decay
- Start: ε = 1.0 (100% random)
- End: ε = 0.1 (10% random)
- Decay: Over 1 million steps

### Training Process
1. Fill replay buffer with random experiences
2. Sample mini-batches and compute TD-error
3. Update Q-network using gradient descent
4. Periodically update target network
5. Decay exploration rate
6. Save checkpoints and log metrics

## 📈 Expected Results

With proper training:
- **Initial episodes**: Score ~0-2 (random exploration)
- **After ~500 episodes**: Score ~5-10 (learning basic mechanics)
- **After ~2000 episodes**: Score ~20-40 (competent play)
- **After ~5000 episodes**: Score ~100+ (strong performance)

Note: Training times vary based on hardware (GPU recommended).

## 🔧 Troubleshooting

### Import Errors
```bash
# Install missing packages
pip install gymnasium[atari] ale-py torch numpy matplotlib opencv-python
```

### ROM Not Found
```bash
# Accept ROM license and install
pip install "gymnasium[accept-rom-license]"
```

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config
- Reduce `REPLAY_BUFFER_SIZE`
- Use CPU training (automatic fallback)

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📝 Assignment Notes

This implementation is designed as a minimal starting point for a Reinforcement Learning course project. Possible extensions:

- **Double DQN**: Reduce overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more frequently
- **Rainbow DQN**: Combine multiple improvements
- **Hyperparameter tuning**: Experiment with learning rates, network sizes, etc.

## 📄 License

This project is for educational purposes.

## 👤 Author

Created for Dr. Teddy Lazebnik's Reinforcement Learning course.
