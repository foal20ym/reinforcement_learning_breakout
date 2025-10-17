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
        survival_bonus=0.001,
        enable_shaping=True,
    ):
        super().__init__(env)
        self.paddle_hit_bonus = paddle_hit_bonus
        self.center_position_bonus = center_position_bonus
        self.side_angle_bonus = side_angle_bonus
        self.block_bonus_multiplier = block_bonus_multiplier
        self.ball_loss_penalty = ball_loss_penalty
        self.enable_shaping = enable_shaping
        self.survival_bonus = survival_bonus

        # State tracking
        self.prev_lives = None
        self.prev_score = 0
        self.ball_in_play = False
        self.step_count = 0

        # RAM-based position tracking (use Python int to avoid overflow)
        self.last_paddle_x = None
        self.last_ball_x = None
        self.last_ball_y = None

        # Track recent positions for better detection
        self.ball_y_history = deque(maxlen=3)  # Track last 3 Y positions
        self.ball_x_history = deque(maxlen=3)  # Track last 3 X positions

        # Statistics tracking - PER EPISODE
        self.episode_stats = self._create_empty_stats()

        # TOTAL statistics (across all episodes)
        self.total_shaped_reward = 0.0
        self.total_stats = self._create_empty_stats()

    def _create_empty_stats(self):
        """Create a fresh statistics dictionary."""
        return {
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
        self.step_count = 0

        # Reset position tracking
        self.last_paddle_x = None
        self.last_ball_x = None
        self.last_ball_y = None
        self.ball_y_history.clear()
        self.ball_x_history.clear()

        # Reset EPISODE statistics (not total)
        self.episode_stats = self._create_empty_stats()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if not self.enable_shaping:
            info["original_reward"] = reward
            return obs, reward, terminated, truncated, info

        # Read current game state from RAM (convert to Python int)
        try:
            paddle_x = int(self.env.unwrapped.ale.getRAM()[72])
            ball_x = int(self.env.unwrapped.ale.getRAM()[99])
            ball_y = int(self.env.unwrapped.ale.getRAM()[101])

            # Store in info
            info["paddle_x"] = paddle_x
            info["ball_x"] = ball_x
            info["ball_y"] = ball_y

        except Exception:
            paddle_x = ball_x = ball_y = None

        # Get current game state
        current_lives = self.env.unwrapped.ale.lives()

        # Initialize shaped reward
        shaped_reward = reward
        original_reward = reward
        info["original_reward"] = original_reward

        # Add survival bonus if ball is in play
        if self.ball_in_play and not terminated:
            shaped_reward += self.survival_bonus

        # 1. BLOCK CLEARING BONUS
        if reward > 0:
            block_bonus = reward * (self.block_bonus_multiplier - 1.0)
            shaped_reward += block_bonus
            self.episode_stats["blocks_broken"] += int(reward)
            self.total_stats["blocks_broken"] += int(reward)
            info["block_broken"] = True

        # 2. BALL LOSS PENALTY
        if self.prev_lives is not None and current_lives < self.prev_lives:
            shaped_reward += self.ball_loss_penalty
            self.ball_in_play = False
            self.episode_stats["balls_lost"] += 1
            self.total_stats["balls_lost"] += 1
            info["ball_lost"] = True

            # Clear history on life loss
            self.ball_y_history.clear()
            self.ball_x_history.clear()
            self.last_paddle_x = None
            self.last_ball_x = None
            self.last_ball_y = None

        # RAM-based detection (only if we have valid positions)
        if paddle_x is not None and ball_x is not None and ball_y is not None:

            # Add to history
            self.ball_y_history.append(ball_y)
            self.ball_x_history.append(ball_x)

            # 3. PADDLE HIT DETECTION (Improved)
            if len(self.ball_y_history) >= 2 and self.ball_in_play:
                prev_y = self.ball_y_history[-2]
                curr_y = ball_y

                # Ball is bouncing at paddle height
                # Paddle is roughly at Y=189, ball bounces around Y=185-195
                PADDLE_Y_MIN = 180
                PADDLE_Y_MAX = 195

                # Check if ball just bounced at paddle level
                if prev_y < PADDLE_Y_MAX and curr_y >= PADDLE_Y_MIN:
                    # Ball was above paddle and is now at/below paddle level
                    # Check if it's near paddle X position
                    if abs(ball_x - paddle_x) < 25:  # Within paddle width + margin
                        shaped_reward += self.paddle_hit_bonus
                        self.episode_stats["paddle_hits"] += 1
                        self.total_stats["paddle_hits"] += 1
                        info["paddle_hit"] = True

                # Alternative detection: Ball was going down and is now going up near paddle
                elif len(self.ball_y_history) >= 3:
                    y_vel_prev = self.ball_y_history[-2] - self.ball_y_history[-3]
                    y_vel_curr = ball_y - self.ball_y_history[-2]

                    # Ball changed from moving down to moving up
                    if y_vel_prev > 0 and y_vel_curr < 0:
                        # Check if at paddle height
                        if PADDLE_Y_MIN <= curr_y <= PADDLE_Y_MAX:
                            # Check if near paddle X
                            if abs(ball_x - paddle_x) < 25:
                                shaped_reward += self.paddle_hit_bonus
                                self.episode_stats["paddle_hits"] += 1
                                self.total_stats["paddle_hits"] += 1
                                info["paddle_hit"] = True

            # 4. SIDE BOUNCE DETECTION (Improved)
            if len(self.ball_x_history) >= 3 and self.ball_in_play:
                # Calculate X velocity
                x_vel_prev = self.ball_x_history[-2] - self.ball_x_history[-3]
                x_vel_curr = ball_x - self.ball_x_history[-2]

                # Velocity changed direction (bounce)
                velocity_changed = (x_vel_prev < 0 and x_vel_curr > 0) or (x_vel_prev > 0 and x_vel_curr < 0)

                # Check if ball is near walls
                LEFT_WALL = 8
                RIGHT_WALL = 152
                near_wall = ball_x <= LEFT_WALL or ball_x >= RIGHT_WALL

                if velocity_changed and near_wall and abs(x_vel_prev) > 0:
                    shaped_reward += self.side_angle_bonus
                    self.episode_stats["side_bounces"] += 1
                    self.total_stats["side_bounces"] += 1
                    info["side_bounce"] = True

            # 5. CENTER POSITION BONUS
            if self.ball_in_play and self.step_count > 30:
                center_bonus = self._calculate_center_position_bonus(paddle_x, ball_x, ball_y)
                if center_bonus > 0:
                    shaped_reward += center_bonus
                    self.episode_stats["center_bonuses"] += 1
                    self.total_stats["center_bonuses"] += 1
                    info["center_bonus"] = True

            # Update tracking for next step
            self.last_paddle_x = paddle_x
            self.last_ball_x = ball_x
            self.last_ball_y = ball_y

        # Update other tracking
        self.prev_lives = current_lives
        self.total_shaped_reward += shaped_reward - original_reward

        # Detect ball in play
        if action == 1 or reward != 0:
            self.ball_in_play = True

        # Add shaping info to info dict
        info["shaped_reward"] = shaped_reward
        info["shaping_bonus"] = shaped_reward - original_reward

        return obs, shaped_reward, terminated, truncated, info

    def _calculate_center_position_bonus(self, paddle_x, ball_x, ball_y):
        """Reward keeping paddle centered under the ball."""
        try:
            # Only reward when ball is above paddle
            if ball_y < 180:
                distance = abs(paddle_x - ball_x)
                # Reward inverse of distance (closer = better)
                if distance < 50:
                    normalized_distance = float(distance) / 50.0
                    return self.center_position_bonus * (1.0 - normalized_distance)
            return 0.0
        except Exception:
            return 0.0

    def get_shaping_stats(self):
        """Return statistics about reward shaping for current episode."""
        return {
            **self.episode_stats,
            "total_shaped_reward": self.total_shaped_reward,
        }

    def get_total_stats(self):
        """Return total statistics across all episodes."""
        return {
            **self.total_stats,
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
