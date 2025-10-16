"""
Experiment configurations for testing different reward shaping strategies.
"""

# Base configuration (no shaping)
BASE_CONFIG = {
    "name": "baseline",
    "description": "No reward shaping - original environment rewards only",
    "paddle_hit_bonus": 0.0,
    "center_position_bonus": 0.0,
    "side_angle_bonus": 0.0,
    "block_bonus_multiplier": 1.0,
    "ball_loss_penalty": 0.0,
    "survival_bonus": 0.0,
}

# Individual reward shaping experiments
EXPERIMENT_CONFIGS = {
    "baseline": BASE_CONFIG,
    "paddle_hit_only": {
        "name": "paddle_hit_only",
        "description": "Only paddle hit bonus",
        "paddle_hit_bonus": 0.1,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.0,
    },
    "center_position_only": {
        "name": "center_position_only",
        "description": "Only center positioning bonus",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.05,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.0,
    },
    "side_angle_only": {
        "name": "side_angle_only",
        "description": "Only side angle bonus",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.15,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.0,
    },
    "block_multiplier_only": {
        "name": "block_multiplier_only",
        "description": "Only block bonus multiplier",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.5,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.0,
    },
    "ball_loss_penalty_only": {
        "name": "ball_loss_penalty_only",
        "description": "Only ball loss penalty",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": -0.8,
        "survival_bonus": 0.0,
    },
    "survival_bonus_only": {
        "name": "survival_bonus_only",
        "description": "Only survival bonus",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.001,
    },
    "all_combined": {
        "name": "all_combined",
        "description": "All reward shaping combined",
        "paddle_hit_bonus": 0.1,
        "center_position_bonus": 0.05,
        "side_angle_bonus": 0.15,
        "block_bonus_multiplier": 1.5,
        "ball_loss_penalty": -0.8,
        "survival_bonus": 0.001,
    },
    "penalties_only": {
        "name": "penalties_only",
        "description": "Only penalties (no positive shaping)",
        "paddle_hit_bonus": 0.0,
        "center_position_bonus": 0.0,
        "side_angle_bonus": 0.0,
        "block_bonus_multiplier": 1.0,
        "ball_loss_penalty": -0.8,
        "survival_bonus": 0.0,
    },
    "bonuses_only": {
        "name": "bonuses_only",
        "description": "Only positive bonuses (no penalties)",
        "paddle_hit_bonus": 0.1,
        "center_position_bonus": 0.05,
        "side_angle_bonus": 0.15,
        "block_bonus_multiplier": 1.5,
        "ball_loss_penalty": 0.0,
        "survival_bonus": 0.001,
    },
}

# Episode configurations
QUICK_TEST_EPISODES = 100
MEDIUM_TEST_EPISODES = 500
FULL_TEST_EPISODES = 2000
GPU_OPTIMIZED_EPISODES = 5000  # For GPU runs

# Checkpoint settings
CHECKPOINT_FREQUENCY = 100  # Save checkpoint every N episodes
KEEP_BEST_N_CHECKPOINTS = 1  # Keep only top N checkpoints per experiment
