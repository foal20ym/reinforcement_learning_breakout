import gymnasium as gym
import numpy as np
from collections import deque
import torch


class BreakoutRewardShaping(gym.Wrapper):
    """
    Reward shaping wrapper specifically designed for Atari Breakout.

    Adds shaped rewards for:
    1. Paddle hits - Bonus for successfully hitting the ball with the paddle
    2. Center positioning - Bonus for staying near center after firing
    3. Side angles - Bonus for getting ball to bounce off side walls
    4. Block clearing - Extra bonus for breaking blocks
    5. Ball loss - Penalty for losing the ball
    """

    def __init__(
        self,
        env,
        paddle_hit_bonus=0.1,
        center_position_bonus=0.05,
        side_angle_bonus=0.15,
        block_bonus_multiplier=1.5,
        ball_loss_penalty=-0.5,
        enable_shaping=True,
    ):
        """
        Args:
            env: The environment to wrap
            paddle_hit_bonus: Reward for hitting ball with paddle
            center_position_bonus: Reward for center positioning
            side_angle_bonus: Reward for side wall bounces
            block_bonus_multiplier: Multiplier for block breaking
            ball_loss_penalty: Penalty when ball is lost
            enable_shaping: Toggle to enable/disable shaping
        """
        super().__init__(env)
        self.paddle_hit_bonus = paddle_hit_bonus
        self.center_position_bonus = center_position_bonus
        self.side_angle_bonus = side_angle_bonus
        self.block_bonus_multiplier = block_bonus_multiplier
        self.ball_loss_penalty = ball_loss_penalty
        self.enable_shaping = enable_shaping

        # State tracking
        self.prev_lives = None
        self.prev_score = 0
        self.ball_in_play = False
        self.frame_buffer = deque(maxlen=2)
        self.step_count = 0

        # Statistics tracking
        self.total_shaped_reward = 0.0
        self.shaping_stats = {
            "paddle_hits": 0,
            "center_bonuses": 0,
            "side_bounces": 0,
            "blocks_broken": 0,
            "balls_lost": 0,
        }

    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)

        # Get initial lives
        self.prev_lives = self.env.unwrapped.ale.lives()
        self.prev_score = 0
        self.ball_in_play = False
        self.frame_buffer.clear()
        self.frame_buffer.append(obs)
        self.step_count = 0

        return obs, info

    def step(self, action):
        """Take a step and apply reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if not self.enable_shaping:
            return obs, reward, terminated, truncated, info

        # Track frame for analysis
        self.frame_buffer.append(obs)

        # Get current game state
        current_lives = self.env.unwrapped.ale.lives()

        # Initialize shaped reward
        shaped_reward = reward
        original_reward = reward

        # 1. BLOCK CLEARING BONUS
        if reward > 0:
            block_bonus = reward * (self.block_bonus_multiplier - 1.0)
            shaped_reward += block_bonus
            self.shaping_stats["blocks_broken"] += 1
            info["block_broken"] = True

        # 2. BALL LOSS PENALTY
        if self.prev_lives is not None and current_lives < self.prev_lives:
            shaped_reward += self.ball_loss_penalty
            self.ball_in_play = False
            self.shaping_stats["balls_lost"] += 1
            info["ball_lost"] = True

        # 3. PADDLE HIT DETECTION
        if self._detect_paddle_hit():
            shaped_reward += self.paddle_hit_bonus
            self.shaping_stats["paddle_hits"] += 1
            info["paddle_hit"] = True

        # 4. CENTER POSITION BONUS
        if self.ball_in_play and self.step_count > 30:
            center_bonus = self._calculate_center_bonus(action)
            if center_bonus > 0:
                shaped_reward += center_bonus
                self.shaping_stats["center_bonuses"] += 1
                info["center_bonus"] = True

        # 5. SIDE ANGLE BONUS
        if self._detect_side_bounce():
            shaped_reward += self.side_angle_bonus
            self.shaping_stats["side_bounces"] += 1
            info["side_bounce"] = True

        # Update tracking
        self.prev_lives = current_lives
        self.total_shaped_reward += shaped_reward - original_reward

        # Detect ball in play
        if action == 1 or reward != 0:
            self.ball_in_play = True

        # Add shaping info to info dict
        info["original_reward"] = original_reward
        info["shaped_reward"] = shaped_reward
        info["shaping_bonus"] = shaped_reward - original_reward

        return obs, shaped_reward, terminated, truncated, info

    def _detect_paddle_hit(self):
        """Detect paddle hits by analyzing frame differences."""
        if len(self.frame_buffer) < 2:
            return False

        prev_frame = self.frame_buffer[0]
        curr_frame = self.frame_buffer[1]

        # Handle different frame formats
        if isinstance(prev_frame, torch.Tensor):
            prev_frame = prev_frame.numpy()
        if isinstance(curr_frame, torch.Tensor):
            curr_frame = curr_frame.numpy()

        # If frame stacking, take latest frame
        if len(prev_frame.shape) == 3 and prev_frame.shape[0] > 1:
            prev_frame = prev_frame[-1]
            curr_frame = curr_frame[-1]
        elif len(prev_frame.shape) == 4:  # Batch dimension
            prev_frame = prev_frame[0, -1]
            curr_frame = curr_frame[0, -1]

        # Focus on paddle region (bottom 20%)
        h = prev_frame.shape[-2] if len(prev_frame.shape) >= 2 else 84
        paddle_start = int(h * 0.8)

        prev_paddle = prev_frame[..., paddle_start:, :]
        curr_paddle = curr_frame[..., paddle_start:, :]

        # Calculate difference
        diff = np.abs(curr_paddle.astype(float) - prev_paddle.astype(float))
        change_ratio = np.sum(diff > 30) / diff.size

        return change_ratio > 0.02 and self.ball_in_play

    def _calculate_center_bonus(self, action):
        """Calculate bonus for center positioning."""
        # Reward staying still when positioned well
        if action == 0 and self.step_count % 10 == 0:
            return self.center_position_bonus * 0.5
        return 0.0

    def _detect_side_bounce(self):
        """Detect side wall bounces."""
        if len(self.frame_buffer) < 2:
            return False

        prev_frame = self.frame_buffer[0]
        curr_frame = self.frame_buffer[1]

        # Handle different formats
        if isinstance(prev_frame, torch.Tensor):
            prev_frame = prev_frame.numpy()
        if isinstance(curr_frame, torch.Tensor):
            curr_frame = curr_frame.numpy()

        # If frame stacking, take latest
        if len(prev_frame.shape) == 3 and prev_frame.shape[0] > 1:
            prev_frame = prev_frame[-1]
            curr_frame = curr_frame[-1]
        elif len(prev_frame.shape) == 4:
            prev_frame = prev_frame[0, -1]
            curr_frame = curr_frame[0, -1]

        # Get width
        w = prev_frame.shape[-1] if len(prev_frame.shape) >= 2 else 84

        # Check side regions (10% from each edge)
        edge_width = int(w * 0.1)
        left_prev = prev_frame[..., :, :edge_width]
        left_curr = curr_frame[..., :, :edge_width]
        right_prev = prev_frame[..., :, -edge_width:]
        right_curr = curr_frame[..., :, -edge_width:]

        # Calculate changes
        left_diff = np.abs(left_curr.astype(float) - left_prev.astype(float))
        right_diff = np.abs(right_curr.astype(float) - right_prev.astype(float))

        left_change = np.sum(left_diff > 30) / left_diff.size
        right_change = np.sum(right_diff > 30) / right_diff.size

        return (left_change > 0.05 or right_change > 0.05) and self.ball_in_play

    def get_shaping_stats(self):
        """Return statistics about reward shaping."""
        return {
            **self.shaping_stats,
            "total_shaped_reward": self.total_shaped_reward,
        }


class RewardShapingScheduler:
    """
    Gradually reduce reward shaping over training.

    This helps transition from shaped to true rewards.
    """

    def __init__(self, initial_scale=1.0, final_scale=0.0, decay_steps=500000, decay_type="linear"):
        """
        Args:
            initial_scale: Starting scale (1.0 = full shaping)
            final_scale: Final scale (0.0 = no shaping)
            decay_steps: Steps over which to decay
            decay_type: 'linear' or 'exponential'
        """
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0

    def get_scale(self, step=None):
        """Get current shaping scale."""
        if step is not None:
            self.current_step = step

        if self.current_step >= self.decay_steps:
            return self.final_scale

        progress = self.current_step / self.decay_steps

        if self.decay_type == "linear":
            scale = self.initial_scale - progress * (self.initial_scale - self.final_scale)
        elif self.decay_type == "exponential":
            scale = self.final_scale + (self.initial_scale - self.final_scale) * np.exp(-5 * progress)
        else:
            scale = self.initial_scale

        return scale

    def step(self):
        """Increment step counter."""
        self.current_step += 1
        return self.get_scale()

    def apply_to_wrapper(self, wrapper):
        """Apply current scale to wrapper bonuses."""
        scale = self.get_scale()

        # Store original values if not stored
        if not hasattr(self, "_original_values"):
            self._original_values = {
                "paddle_hit_bonus": wrapper.paddle_hit_bonus,
                "center_position_bonus": wrapper.center_position_bonus,
                "side_angle_bonus": wrapper.side_angle_bonus,
                "ball_loss_penalty": wrapper.ball_loss_penalty,
            }

        # Scale bonuses
        wrapper.paddle_hit_bonus = self._original_values["paddle_hit_bonus"] * scale
        wrapper.center_position_bonus = self._original_values["center_position_bonus"] * scale
        wrapper.side_angle_bonus = self._original_values["side_angle_bonus"] * scale
        wrapper.ball_loss_penalty = self._original_values["ball_loss_penalty"] * scale
