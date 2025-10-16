# Action space (discrete)
ACTIONS = [0, 1, 2, 3]  # 0: noop, 1: left engine, 2: main engine, 3: right engine

# Generic RL / Q-Learning style parameters
DISCOUNT_FACTOR = 0.99  # Gamma
MAX_EPISODES = 5000  # Total training episodes
MAX_STEPS_PER_EPISODE = 1000  # Environment step cap per episode

# Epsilon-greedy schedule
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # Applied per episode (epsilon *= EPS_DECAY until floor)

# Optimizer / learning
LEARNING_RATE = 1e-4
LR_DECAY_FACTOR = 0.995  # Optional: decay LR over time
LR_DECAY_FREQUENCY = 100  # Apply LR decay every N episodes

# DQN-specific parameters
REPLAY_MEMORY_SIZE = 100_000
MINI_BATCH_SIZE = 64
NETWORK_SYNC_RATE = 1_000
UPDATE_EVERY = 4  # Learn every 4 steps
GRADIENT_CLIP = 10.0  # Clip gradients to prevent exploding gradients

# Network architecture
HIDDEN_SIZES = (256, 256, 128)

# UI / Rendering
SHOW_ANIMATION = True

# Training
CNN = True
# If double DQN should be used instead of normal DQN, Note: DQN still has to be true
USE_CNN = True
USE_NEURAL_NET = False

# Reward shaping
REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    "paddle_hit_bonus": 0.1,  # Bonus for hitting ball with paddle
    "center_position_bonus": 0.05,  # Bonus for center positioning
    "side_angle_bonus": 0.15,  # Bonus for side wall bounces
    "block_bonus_multiplier": 1.5,  # Multiplier for block breaking
    "ball_loss_penalty": -0.8,  # Stronger death penalty
    "survival_bonus": 0.001,  # Small bonus per step alive
}
