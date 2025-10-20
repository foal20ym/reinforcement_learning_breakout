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
        self.survival_bonus = survival_bonus

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

        # Lightweight tracking state
        self._last_track = None
        self.prev_ball = None          # (x, y) in pixel coords
        self.prev_ball_vel = None      # (vx, vy)

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
        self._last_track = None
        self.prev_ball = None
        self.prev_ball_vel = None

        return obs, info


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        if not self.enable_shaping:
            return obs, reward, terminated, truncated, info

        # Track frame for analysis (store raw obs)
        self.frame_buffer.append(obs)

        # Run lightweight tracking (ball + paddle) once per step
        self._last_track = self._track_ball_and_paddle()

        # Get current game state
        current_lives = self.env.unwrapped.ale.lives()

        # Initialize shaped reward
        shaped_reward = reward
        original_reward = reward

        if self.ball_in_play and not terminated:
            shaped_reward += self.survival_bonus
            info["survival_bonus"] = True

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

        # 3. PADDLE HIT DETECTION (using tracked ball + paddle)
        if self._detect_paddle_hit(self._last_track):
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

        # 5. SIDE ANGLE BONUS (using tracked ball)
        if self._detect_side_bounce(self._last_track):
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

    def _to_gray(self, frame):
        """Convert input frame (HWC or CHW or HW) to float32 grayscale HW."""
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        arr = np.array(frame)

        # If batch-like, squeeze it
        if arr.ndim > 3:
            arr = np.squeeze(arr)

        if arr.ndim == 2:
            gray = arr.astype(np.float32)
        elif arr.ndim == 3:
            # HWC
            if arr.shape[-1] == 3:
                r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            # CHW
            elif arr.shape[0] == 3:
                r, g, b = arr[0], arr[1], arr[2]
                gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            else:
                # Unknown 3D layout, best effort: take mean across last axis
                gray = arr.mean(axis=-1).astype(np.float32)
        else:
            # Fallback
            gray = arr.astype(np.float32)

        return gray

    def _track_ball_and_paddle(self):
        """
        Track ball position/velocity and paddle x-range from the last two frames.
        Returns dict with keys: ball (x,y), vel (vx,vy), prev_vel, paddle (x0,x1), frame_shape (H,W).
        """
        if len(self.frame_buffer) < 2:
            return {"ball": self.prev_ball, "vel": self.prev_ball_vel, "prev_vel": None, "paddle": None, "frame_shape": None}

        prev = self._to_gray(self.frame_buffer[0])
        curr = self._to_gray(self.frame_buffer[1])
        if prev.ndim != 2 or curr.ndim != 2:
            return {"ball": self.prev_ball, "vel": self.prev_ball_vel, "prev_vel": None, "paddle": None, "frame_shape": None}
        H, W = curr.shape

        paddle = self._find_paddle_range(curr)

        # Find ball using abs-diff with weighting; capped region size to avoid brick explosions
        ball = self._find_ball_position(prev, curr)
        old_vel = self.prev_ball_vel
        vel = None
        if ball is not None and self.prev_ball is not None:
            vx = float(ball[0] - self.prev_ball[0])
            vy = float(ball[1] - self.prev_ball[1])
            vel = (vx, vy)

        # Update memory after computing old_vel
        if ball is not None:
            self.prev_ball = ball
            self.prev_ball_vel = vel if vel is not None else self.prev_ball_vel

        return {"ball": self.prev_ball, "vel": self.prev_ball_vel, "prev_vel": old_vel, "paddle": paddle, "frame_shape": (H, W)}

    def _find_paddle_range(self, frame):
        """
        Detect paddle horizontal span (x0, x1) in bottom band.
        Uses adaptive threshold and longest bright run on the row with max activity.
        """
        H, W = frame.shape
        band_top = int(H * 0.88)  # bottom ~12%
        band = frame[band_top:, :].astype(np.float32)

        # Adaptive threshold from band distribution
        thr = max(50.0, float(np.percentile(band, 80)))
        band_bin = band > thr

        # Pick the row with the maximum number of "bright" pixels
        row_counts = band_bin.sum(axis=1)
        if row_counts.max() < W * 0.05:
            return None
        row_idx = int(np.argmax(row_counts))
        row = band_bin[row_idx]

        # Longest contiguous run of True
        best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
        for x in range(W):
            if row[x]:
                if cur_len == 0:
                    cur_start = x
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        if best_len < 3:
            return None
        x0 = best_start
        x1 = best_start + best_len - 1
        return (x0, x1)

    def _find_ball_position(self, prev, curr):
        """
        Estimate ball centroid from abs-diff between frames.
        Uses weighted centroid; ignores huge diff regions (brick explosions).
        Returns (x, y) or None.
        """
        H, W = curr.shape
        diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))

        # Threshold for motion; ball is small, so use a relatively high cutoff
        mask = diff > 40.0

        # If too many pixels changed (e.g., bricks), restrict to bottom 60% when we care about paddle hits
        changed = int(mask.sum())
        if changed == 0:
            return None
        if changed > (H * W) * 0.10:
            y0 = int(H * 0.40)
            mask[:y0, :] = False

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        weights = diff[ys, xs]
        wsum = float(weights.sum())
        if wsum < 1e-3:
            return None

        cx = float((xs * weights).sum() / wsum)
        cy = float((ys * weights).sum() / wsum)
        return (cx, cy)

    def _detect_paddle_hit(self, track):
        """
        A paddle hit is when the ball was moving down and now moves up
        near the paddle band, and the ball x lies within the paddle span.
        """
        if not self.ball_in_play or track is None:
            return False
        ball = track.get("ball")
        vel = track.get("vel")
        prev_vel = track.get("prev_vel")
        paddle = track.get("paddle")
        shape = track.get("frame_shape")
        if ball is None or prev_vel is None or vel is None or paddle is None or shape is None:
            return False

        H, W = shape
        x, y = ball
        vx_prev, vy_prev = prev_vel
        vx_cur, vy_cur = vel

        # Require a downward motion that flips to upward, near bottom 15% of screen
        near_bottom = y >= H * 0.85
        flipped_vertical = (vy_prev is not None) and (vy_cur is not None) and (vy_prev > 0.2 and vy_cur < -0.2)

        x0, x1 = paddle
        on_paddle_x = (x >= x0 - 2) and (x <= x1 + 2)

        return bool(near_bottom and flipped_vertical and on_paddle_x)

    def _detect_side_bounce(self, track):
        """
        A side bounce is when ball x-velocity flips sign near the left/right edges.
        """
        if not self.ball_in_play or track is None:
            return False
        ball = track.get("ball")
        vel = track.get("vel")
        prev_vel = track.get("prev_vel")
        shape = track.get("frame_shape")
        if ball is None or prev_vel is None or vel is None or shape is None:
            return False

        H, W = shape
        x, y = ball
        vx_prev, vy_prev = prev_vel
        vx_cur, vy_cur = vel

        if vx_prev is None or vx_cur is None:
            return False

        # Flip in horizontal direction
        flipped_horizontal = (vx_prev > 0.2 and vx_cur < -0.2) or (vx_prev < -0.2 and vx_cur > 0.2)
        near_edge = (x <= 4) or (x >= W - 5)
        return bool(flipped_horizontal and near_edge)
    
    def _calculate_center_bonus(self, action):
        """
        Small bonus for keeping the paddle near the horizontal screen center.
        Heuristic: find paddle in the bottom band and measure distance to center.
        """
        if not self.ball_in_play or len(self.frame_buffer) == 0:
            return 0.0

        frame = self._to_gray(self.frame_buffer[-1])
        if frame.ndim != 2:
            return 0.0

        H, W = frame.shape
        band_top = int(H * 0.88)  # bottom ~12% region
        band = frame[band_top:, :].astype(np.float32) / 255.0

        # Column intensity profile in the bottom band
        col_profile = band.mean(axis=0)
        peak = col_profile.max()
        if peak < 0.2:
            # Can't reliably find the paddle
            return 0.0

        # Estimate paddle x as weighted centroid around the max
        peak_x = int(np.argmax(col_profile))
        window = 6
        left = max(0, peak_x - window)
        right = min(W, peak_x + window + 1)
        xs = np.arange(left, right)
        weights = col_profile[left:right]
        if weights.sum() <= 1e-6:
            return 0.0
        paddle_x = (xs * weights).sum() / weights.sum()

        # Center closeness in [0, 1]
        center_x = W / 2.0
        norm_dist = abs(paddle_x - center_x) / center_x  # 0 at center, 1 at edges
        closeness = max(0.0, 1.0 - norm_dist)

        # Scale by configured bonus
        return float(self.center_position_bonus * closeness)

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
