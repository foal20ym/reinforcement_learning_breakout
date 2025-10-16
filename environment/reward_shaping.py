import gymnasium as gym
import numpy as np
from collections import deque


class BreakoutRewardShaping(gym.Wrapper):
    """
    Reward shaping wrapper specifically designed for Atari Breakout.

    Adds shaped rewards for:
    1. Paddle hits - Bonus for successfully hitting the ball with the paddle
    2. Center positioning - Bonus for staying near center after firing
    3. Side angles - Bonus for getting ball to bounce off side walls
    4. Block clearing - Extra bonus for breaking blocks
    5. Ball loss - Penalty for losing the ball

    These shaped rewards help the agent learn faster by providing more
    immediate feedback for good behaviors.
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
            block_bonus_multiplier: Multiplier for block breaking (applied to original reward)
            ball_loss_penalty: Penalty when ball is lost
            enable_shaping: Toggle to enable/disable shaping (for ablation studies)
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
        self.frame_buffer = deque(maxlen=2)  # For detecting paddle hits
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)

        # Get initial lives from the unwrapped environment
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
            # Return original reward if shaping is disabled
            return obs, reward, terminated, truncated, info

        # Track frame for analysis
        self.frame_buffer.append(obs)

        # Get current game state
        current_lives = self.env.unwrapped.ale.lives()
        current_score = self.env.unwrapped.ale.getEpisodeFrameNumber()

        # Initialize shaped reward with original reward
        shaped_reward = reward

        # 1. BLOCK CLEARING BONUS
        # If we got positive reward, it means we broke a block
        if reward > 0:
            # Apply multiplier to make block breaking more rewarding
            shaped_reward = reward * self.block_bonus_multiplier
            info["block_broken"] = True

        # 2. BALL LOSS PENALTY
        # Detect if we lost a life (ball fell off screen)
        if self.prev_lives is not None and current_lives < self.prev_lives:
            shaped_reward += self.ball_loss_penalty
            self.ball_in_play = False
            info["ball_lost"] = True

        # 3. PADDLE HIT DETECTION
        # When ball is in play and moving, reward paddle contact
        if self._detect_paddle_hit():
            shaped_reward += self.paddle_hit_bonus
            info["paddle_hit"] = True

        # 4. CENTER POSITION BONUS (after firing)
        # Encourage staying near center for better ball coverage
        if self.ball_in_play and self.step_count > 30:  # After initial phase
            center_bonus = self._calculate_center_bonus(action)
            shaped_reward += center_bonus
            if center_bonus > 0:
                info["center_bonus"] = True

        # 5. SIDE ANGLE BONUS
        # Reward for getting ball to sides (creates better angles)
        if self._detect_side_bounce():
            shaped_reward += self.side_angle_bonus
            info["side_bounce"] = True

        # Update tracking variables
        self.prev_lives = current_lives
        self.prev_score = current_score

        # Detect if ball is in play (after FIRE action)
        if action == 1 or reward != 0 or current_lives < self.env.unwrapped.ale.lives():
            self.ball_in_play = True

        return obs, shaped_reward, terminated, truncated, info

    def _detect_paddle_hit(self):
        """
        Detect if paddle hit the ball by looking at frame differences.

        Simple heuristic: Look for changes in the lower portion of the screen
        where the paddle is located.
        """
        if len(self.frame_buffer) < 2:
            return False

        # Get the most recent frames (after FrameStack, shape is (k, 84, 84))
        prev_frame = self.frame_buffer[0]
        curr_frame = self.frame_buffer[1]

        # If using frame stacking, take the latest frame
        if len(prev_frame.shape) == 3:
            prev_frame = prev_frame[-1]
            curr_frame = curr_frame[-1]

        # Focus on paddle region (bottom 20% of screen)
        paddle_region_start = int(84 * 0.8)
        prev_paddle = prev_frame[paddle_region_start:, :]
        curr_paddle = curr_frame[paddle_region_start:, :]

        # Detect significant changes in paddle area (ball contact)
        diff = np.abs(curr_paddle.astype(float) - prev_paddle.astype(float))

        # If there's substantial change in paddle region, likely a hit
        # Using threshold to avoid noise
        change_ratio = np.sum(diff > 30) / diff.size

        return change_ratio > 0.02 and self.ball_in_play

    def _calculate_center_bonus(self, action):
        """
        Calculate bonus for staying near center.

        Encourage agent to return to center position for better ball coverage.
        """
        # Actions in Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        # Reward staying in center or moving toward center

        # Simple heuristic: Give small bonus for NOOP when ball is in play
        # (suggests paddle is well-positioned)
        if action == 0 and self.step_count % 10 == 0:
            return self.center_position_bonus * 0.5

        return 0.0

    def _detect_side_bounce(self):
        """
        Detect if ball bounced off side walls.

        Side bounces create better angles for clearing blocks.
        """
        if len(self.frame_buffer) < 2:
            return False

        prev_frame = self.frame_buffer[0]
        curr_frame = self.frame_buffer[1]

        # If using frame stacking, take the latest frame
        if len(prev_frame.shape) == 3:
            prev_frame = prev_frame[-1]
            curr_frame = curr_frame[-1]

        # Check side regions (left and right 10% of screen)
        left_region = slice(None), slice(0, 8)
        right_region = slice(None), slice(76, 84)

        # Detect changes in side regions
        left_diff = np.abs(curr_frame[left_region].astype(float) - prev_frame[left_region].astype(float))
        right_diff = np.abs(curr_frame[right_region].astype(float) - prev_frame[right_region].astype(float))

        left_change = np.sum(left_diff > 30) / left_diff.size
        right_change = np.sum(right_diff > 30) / right_diff.size

        # Threshold for detecting side bounces
        return (left_change > 0.05 or right_change > 0.05) and self.ball_in_play


class RewardShapingScheduler:
    """
    Gradually reduce reward shaping over training to avoid dependence.

    This helps the agent learn with shaping initially, then transition
    to using only the true environment rewards.
    """

    def __init__(self, initial_scale=1.0, final_scale=0.0, decay_steps=500000, decay_type="linear"):
        """
        Args:
            initial_scale: Starting scale for shaped rewards (1.0 = full shaping)
            final_scale: Final scale for shaped rewards (0.0 = no shaping)
            decay_steps: Number of steps over which to decay
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
