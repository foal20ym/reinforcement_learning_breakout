# Reward Shaping Configuration Presets
# Copy desired preset to config_fast.py

# ============================================================================
# PRESET 1: DISABLED (Default - Original DQN)
# ============================================================================
# Use this to train with standard DQN without any reward shaping
ENABLE_REWARD_SHAPING = False


# ============================================================================
# PRESET 2: CONSERVATIVE (Recommended for Episode 7200+)
# ============================================================================
# Subtle bonuses that won't disrupt learned policies
# Good for refining strategies in later training stages
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.05,       # Small bonus for paddle hits
    'center_position_bonus': 0.02,  # Minimal center bonus
    'side_angle_bonus': 0.1,        # Encourage side angles
    'block_bonus_multiplier': 1.2,  # Slight emphasis on blocks
    'ball_loss_penalty': -0.3,      # Moderate penalty
}
"""


# ============================================================================
# PRESET 3: BALANCED (Good for Mid-Training)
# ============================================================================
# Default parameters - balanced between guidance and true rewards
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.1,        # Standard bonuses
    'center_position_bonus': 0.05,
    'side_angle_bonus': 0.15,
    'block_bonus_multiplier': 1.5,
    'ball_loss_penalty': -0.5,
}
"""


# ============================================================================
# PRESET 4: AGGRESSIVE (For Fast Initial Learning)
# ============================================================================
# Strong bonuses for rapid learning from scratch
# May lead to dependence on shaped rewards
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.2,        # Strong paddle hit bonus
    'center_position_bonus': 0.1,   # Encourage center positioning
    'side_angle_bonus': 0.3,        # Really reward side angles
    'block_bonus_multiplier': 2.0,  # Double block rewards
    'ball_loss_penalty': -1.0,      # Harsh penalty for ball loss
}
"""


# ============================================================================
# PRESET 5: PADDLE FOCUS (Emphasize Ball Control)
# ============================================================================
# Focus on keeping ball in play, minimal other bonuses
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.3,        # High paddle hit reward
    'center_position_bonus': 0.0,   # Disabled
    'side_angle_bonus': 0.0,        # Disabled
    'block_bonus_multiplier': 1.0,  # No modification
    'ball_loss_penalty': -0.8,      # Strong penalty
}
"""


# ============================================================================
# PRESET 6: STRATEGIC PLAY (Emphasize Angles and Positioning)
# ============================================================================
# Reward strategic positioning and angle play
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.0,        # Disabled - assume basic control
    'center_position_bonus': 0.08,  # Encourage good positioning
    'side_angle_bonus': 0.25,       # Heavily reward angles
    'block_bonus_multiplier': 1.3,  # Slight block emphasis
    'ball_loss_penalty': -0.4,      # Moderate penalty
}
"""


# ============================================================================
# PRESET 7: WITH SCHEDULER (Gradual Reduction)
# ============================================================================
# Start with shaping, gradually reduce to pure environment rewards
# Good for long training runs starting from scratch
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.15,
    'center_position_bonus': 0.05,
    'side_angle_bonus': 0.2,
    'block_bonus_multiplier': 1.8,
    'ball_loss_penalty': -0.6,
}

# Enable scheduler to decay shaping over time
USE_SHAPING_SCHEDULER = True
SHAPING_DECAY_STEPS = 500000      # Decay over 500k steps (~5000 episodes)
SHAPING_DECAY_TYPE = 'linear'     # or 'exponential'
"""


# ============================================================================
# PRESET 8: MINIMAL (Just Ball Control)
# ============================================================================
# Absolute minimum shaping - only penalize ball loss
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.0,
    'center_position_bonus': 0.0,
    'side_angle_bonus': 0.0,
    'block_bonus_multiplier': 1.0,
    'ball_loss_penalty': -0.5,      # Only this is active
}
"""


# ============================================================================
# CUSTOM PRESET TEMPLATE
# ============================================================================
# Create your own configuration
"""
ENABLE_REWARD_SHAPING = True
REWARD_SHAPING_PARAMS = {
    'paddle_hit_bonus': 0.0,        # 0.0 to 0.5 recommended
    'center_position_bonus': 0.0,   # 0.0 to 0.1 recommended
    'side_angle_bonus': 0.0,        # 0.0 to 0.3 recommended
    'block_bonus_multiplier': 1.0,  # 1.0 to 2.0 recommended
    'ball_loss_penalty': 0.0,       # -1.0 to 0.0 recommended
}

# Optional: Enable scheduler
USE_SHAPING_SCHEDULER = False
SHAPING_DECAY_STEPS = 500000
SHAPING_DECAY_TYPE = 'linear'
"""


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
1. Choose a preset above that matches your training stage
2. Uncomment the preset (remove triple quotes)
3. Copy the configuration to your config_fast.py file
4. Resume training: python train.py --resume latest

RECOMMENDATIONS BY TRAINING STAGE:

Early Training (Episodes 0-2000):
  • PRESET 4 (Aggressive) - Fast initial learning
  • PRESET 7 (With Scheduler) - If training from scratch

Mid Training (Episodes 2000-5000):
  • PRESET 3 (Balanced) - Standard shaping
  • PRESET 5 (Paddle Focus) - If struggling with ball control

Late Training (Episodes 5000-10000):
  • PRESET 2 (Conservative) - Subtle refinement
  • PRESET 1 (Disabled) - Let agent use learned policy

Your Current Situation (Episode 7200):
  • PRESET 2 (Conservative) - Safe refinement
  • PRESET 1 (Disabled) - Keep what's working
  
  Either choice is good! Your agent has learned the basics,
  so strong shaping isn't needed. Conservative shaping may
  help refine strategies, but it's optional.

TESTING PRESETS:
  After changing configuration, test with:
  ./test_reward_shaping.sh
  
  Then resume training and monitor:
  logs/training_plot.png
"""
