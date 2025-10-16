# SPEED-OPTIMIZED DQN Configuration for Atari Breakout
# This config trades some stability for faster training
# Use this if you have a GPU and want quicker results

# Environment
ENV_NAME = "ALE/Breakout-v5"
FRAME_STACK = 4
FRAME_SIZE = 84

# Training - Faster
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 10000
BATCH_SIZE = 64  # Increased from 32 (better GPU utilization)
REPLAY_BUFFER_SIZE = 100000  # Keep same
MIN_REPLAY_SIZE = 5000  # Decreased from 10000 (start learning sooner)

# Learning - Slightly more aggressive
LEARNING_RATE = 0.0003  # Slightly increased from 0.00025
GAMMA = 0.99
TARGET_UPDATE_FREQ = 1000  # Keep same for stability

# Exploration - Faster decay
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 750000  # Decreased from 1M (faster exploitation)

# Optimization
GRADIENT_CLIP = 10.0
TRAIN_FREQ = 4  # Keep same

# Checkpointing
SAVE_FREQ = 100
CHECKPOINT_DIR = "checkpoints"

# Logging
LOG_FREQ = 10

# Speed optimizations (these are used in code)
USE_MIXED_PRECISION = True  # Use FP16 training if available
NUM_WORKERS = 4  # For data loading (if implemented)
PIN_MEMORY = True  # Faster GPU transfer

# Reward Shaping Configuration
ENABLE_REWARD_SHAPING = True  # Set to True to enable reward shaping
REWARD_SHAPING_PARAMS = {
    "paddle_hit_bonus": 0.1,  # Bonus for hitting ball with paddle
    "center_position_bonus": 0.05,  # Bonus for center positioning
    "side_angle_bonus": 0.15,  # Bonus for side wall bounces
    "block_bonus_multiplier": 1.5,  # Multiplier for block breaking
    "ball_loss_penalty": -0.5,  # Penalty for losing the ball
}

# Reward Shaping Scheduler (gradually reduce shaping)
USE_SHAPING_SCHEDULER = False  # Decay reward shaping over time
SHAPING_DECAY_STEPS = 500000  # Steps to decay shaping to zero
SHAPING_DECAY_TYPE = "linear"  # 'linear' or 'exponential'
