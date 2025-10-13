# Action space (discrete)
ACTIONS = [0, 1, 2, 3]  # 0: noop, 1: left engine, 2: main engine, 3: right engine

# Generic RL / Q-Learning style parameters
DISCOUNT_FACTOR = 0.99  # Gamma
MAX_EPISODES = 5000  # Total training episodes
MAX_STEPS_PER_EPISODE = 1000  # Environment step cap per episode

# Epsilon-greedy schedule
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995  # Applied per episode (epsilon *= EPS_DECAY until floor)

# Optimizer / learning
LEARNING_RATE = 1e-4
LR_DECAY_FACTOR = 0.995  # Optional: decay LR over time
LR_DECAY_FREQUENCY = 100  # Apply LR decay every N episodes

# DQN-specific parameters
REPLAY_BUFFER_SIZE = 100_000
MINIBATCH_SIZE = 64
TARGET_NETWORK_TAU = 5e-3
UPDATE_EVERY = 4  # Learn every 4 steps
GRADIENT_CLIP = 10.0  # Clip gradients to prevent exploding gradients

# Network architecture
HIDDEN_SIZES = (256, 256, 128)

# UI / Rendering
SHOW_ANIMATION = True

# Training
DQN = True
# If double DQN should be used instead of normal DQN, Note: DQN still has to be true
DDQN = True
PPO = False
MANUAL_PLAY = False

# Continues toggle, discretization grid for DQN
CONTINUOUS = True
CONTINUOUS_ACTION_GRID = [-1.0, 0.0, 1.0]  # grid values per dimension

# Reward shaping
REWARD_SHAPING = True
FUEL_PENALTY = 0.02
SHAPING_WEIGHTS = {
    "x": 2.0,  # lateral position is critical
    "y": 0.3,  # Slightly altitude matters
    "vx": 0.5,  # horizontal velocity control important
    "vy": 0.5,  # vertical velocity control important
    "angle": 1.0,  # staying upright is crucial
    "ang_v": 0.2,  # angular velocity control
    "legs": 2.0,  # strong reward for leg contact
}
