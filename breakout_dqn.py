from visualization import plot_progress
from core.ReplayMemory import ReplayMemory
from helpers.FramePreprocess import preprocess_frame
from helpers.FrameStack import FrameStack
from core.NeuralNetwork import NeuralNetwork
from core.CNN import CNN
from environment.reward_shaping import BreakoutRewardShaping, RewardShapingScheduler
from torch import nn
import torch
import gymnasium as gym
import ale_py
import numpy as np
import random
import os
from utils import config
from helpers.FileIO import save_current_episode, save_best_avg_reward, load_best_avg_reward, load_current_episode

class BreakoutDQN:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.learning_rate = config.LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.network_sync_rate = config.NETWORK_SYNC_RATE
        self.replay_memory_size = config.REPLAY_MEMORY_SIZE
        self.mini_batch_size = config.MINI_BATCH_SIZE
        self.epsilon = config.EPSILON
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.use_neural_net = config.USE_NEURAL_NET
        self.use_cnn = config.USE_CNN
        if self.use_neural_net:
            self.num_hidden_nodes = config.HIDDEN_SIZES[0]

        # Reward shaping configuration from config
        self.enable_reward_shaping = config.REWARD_SHAPING
        self.reward_shaping_params = config.REWARD_SHAPING_PARAMS

        self.loss_fn = nn.HuberLoss()
        self.optimizer = None

    def train(self, episodes, render=False, resume_checkpoint=None, resume_replay=None):
        # Create base environment
        base_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array" if render else None)

        # Wrap with reward shaping if enabled
        if self.enable_reward_shaping:
            env = BreakoutRewardShaping(
                base_env,
                paddle_hit_bonus=self.reward_shaping_params["paddle_hit_bonus"],
                center_position_bonus=self.reward_shaping_params["center_position_bonus"],
                side_angle_bonus=self.reward_shaping_params["side_angle_bonus"],
                block_bonus_multiplier=self.reward_shaping_params["block_bonus_multiplier"],
                ball_loss_penalty=self.reward_shaping_params["ball_loss_penalty"],
                enable_shaping=True,
            )
            print("🎮 Reward shaping ENABLED")
            print(f"  Paddle hit bonus: {self.reward_shaping_params['paddle_hit_bonus']}")
            print(f"  Center bonus: {self.reward_shaping_params['center_position_bonus']}")
            print(f"  Side angle bonus: {self.reward_shaping_params['side_angle_bonus']}")
            print(f"  Block multiplier: {self.reward_shaping_params['block_bonus_multiplier']}x")
            print(f"  Ball loss penalty: {self.reward_shaping_params['ball_loss_penalty']}")
        else:
            env = base_env
            print("🎮 Reward shaping DISABLED (using original rewards)")

        num_actions = env.action_space.n
        state_dim = 4 * 84 * 84  # 4 stacked frames of size 84x84

        policy_dqn = (
            CNN(num_actions).to(self.device)
            if self.use_cnn
            else NeuralNetwork(state_dim, self.num_hidden_nodes, num_actions).to(self.device)
        )
        target_dqn = (
            CNN(num_actions).to(self.device)
            if self.use_cnn
            else NeuralNetwork(state_dim, self.num_hidden_nodes, num_actions).to(self.device)
        )
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        last_episode_done = 0
        best_avg_reward = load_best_avg_reward("environment/best_avg_reward.txt")
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            s_ep, best_avg = self.load_checkpoint(
                policy_dqn, self.optimizer, resume_checkpoint, replay_filepath=resume_replay
            )
            last_episode_done = s_ep or 0
            target_dqn.load_state_dict(policy_dqn.state_dict())
            best_avg_reward = best_avg
            print(f"Resuming from checkpoint {resume_checkpoint} at episode {last_episode_done}")
        else:
            print("Starting fresh training run")

        start_episode = last_episode_done + 1
        end_episode = last_episode_done + int(episodes)
        print(f"Training episodes {start_episode}..{end_episode} (additional: {episodes})")

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0
        last_completed_episode = last_episode_done

        for episode in range(start_episode, end_episode + 1):
            obs, info = env.reset()
            lives = info.get("lives", 5)
            obs, _, terminated, truncated, info = env.step(1)
            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            total_original_reward = 0  # Track original reward separately
            terminated = truncated = False
            episode_steps = 0

            while not (terminated or truncated):
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.to(self.device)).argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)

                # Track original reward if shaping is enabled
                if self.enable_reward_shaping and "original_reward" in info:
                    total_original_reward += info["original_reward"]
                else:
                    total_original_reward += reward

                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                # Detect life loss and reset frame stack
                current_lives = info.get("lives", lives)
                if current_lives < lives:
                    lives = current_lives

                    # Reset the frame stack after life loss
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(next_obs)
                    next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                self.memory.append((state, action, next_state, reward, terminated or truncated))
                state = next_state
                total_reward += reward
                episode_steps += 1
                step_count += 1

                if len(self.memory) > self.mini_batch_size and step_count % config.UPDATE_EVERY == 0:
                    mini_batch = self.memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(self.epsilon)

            last_completed_episode = episode
            save_current_episode(last_completed_episode)

            if episode % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])

                # Print with shaping statistics if enabled
                if self.enable_reward_shaping:
                    stats = env.get_shaping_stats()
                    print(
                        f"Episode {episode}, Avg Reward: {avg_reward:.2f} "
                        f"(Original: {total_original_reward:.2f}), "
                        f"Epsilon: {self.epsilon:.3f}"
                    )
                    print(
                        f"  Shaping stats - Paddle hits: {stats['paddle_hits']}, "
                        f"Blocks: {stats['blocks_broken']}, "
                        f"Side bounces: {stats['side_bounces']}, "
                        f"Balls lost: {stats['balls_lost']}"
                    )
                else:
                    print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

                model_filename = f"models/CNN_breakout.pt"
                torch.save(policy_dqn.state_dict(), model_filename)
                torch.save(self.optimizer.state_dict(), "models/optimizer_latest.pth")
                torch.save(policy_dqn.state_dict(), "models/model_latest.pth")
                self.save_checkpoint(policy_dqn, self.optimizer, episode, "models/checkpoint_latest.pth")

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_best_avg_reward(best_avg_reward)
                    model_filename = f"models/CNN_breakout_avg_{int(best_avg_reward)}.pt"
                    torch.save(policy_dqn.state_dict(), model_filename)
                    print(f"  New best average reward! Model saved as {model_filename}")

        env.close()
        plot_progress(rewards_per_episode, epsilon_history)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values = policy_dqn(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=config.GRADIENT_CLIP)
        self.optimizer.step()

    def test(self, episodes, model_filepath):
        """
        Runs the environment with the learned policy.
        Testing always uses original rewards (no reward shaping).
        """
        # Always use base environment for testing (no shaping)
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        num_actions = env.action_space.n
        state_dim = 4 * 84 * 84

        print("🎮 Testing with ORIGINAL rewards (no shaping)")

        # Load learned policy
        policy_dqn = (
            CNN(num_actions).to(self.device)
            if self.use_cnn
            else NeuralNetwork(state_dim, self.num_hidden_nodes, num_actions).to(self.device)
        )
        policy_dqn.load_state_dict(torch.load(model_filepath, map_location=self.device))
        policy_dqn.to(self.device)
        policy_dqn.eval()

        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            lives = info.get("lives", 5)
            obs, _, terminated, truncated, info = env.step(1)

            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0

            while not (terminated or truncated):
                with torch.no_grad():
                    q_values = policy_dqn(state.to(self.device))
                    action = q_values.argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = preprocess_frame(next_obs)

                # Detect life loss and reset frame stack
                current_lives = info.get("lives", lives)
                if current_lives < lives:
                    lives = current_lives

                    # Reset the frame stack after life loss
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(next_obs)
                    state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                    # Take a "FIRE" action to resume the game
                    next_obs, _, terminated, truncated, info = env.step(1)
                    next_obs = preprocess_frame(next_obs)
                    for _ in range(4):
                        frame_stack.append(next_obs)
                    state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                    continue

                frame_stack.append(next_obs)
                state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                total_reward += reward

            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

        env.close()

    def save_checkpoint(self, model, optimizer, episode, filepath):
        """Save model + optimizer + training state to a checkpoint file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        ckpt = {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epsilon": self.epsilon,
            "best_avg_reward": load_best_avg_reward("environment/best_avg_reward.txt"),
        }
        if getattr(self, "use_amp", False) and getattr(self, "scaler", None) is not None:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, filepath)
        # save replay memory next to checkpoint (non-blocking to disk)
        try:
            replay_path = filepath.replace(".pth", "_replay.pth")
            self.memory.save(replay_path)
            # also keep a latest symlink-style filenames
            self.memory.save("models/replay_memory_latest.pth")
        except Exception:
            # don't crash training because replay save failed
            pass

    def load_checkpoint(self, model, optimizer=None, filepath=None, replay_filepath=None):
        """Load checkpoint into model and optionally optimizer. Returns start_episode and best_avg_reward."""
        if filepath is None or not os.path.exists(filepath):
            return 0, load_best_avg_reward("environment/best_avg_reward.txt")
        ckpt = torch.load(filepath, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        if getattr(self, "use_amp", False) and getattr(self, "scaler", None) is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_episode = ckpt.get("episode", 0)
        best_avg = ckpt.get("best_avg_reward", load_best_avg_reward("environment/best_avg_reward.txt"))

        # attempt to load replay memory if provided
        if replay_filepath and os.path.exists(replay_filepath):
            try:
                self.memory = ReplayMemory.load(replay_filepath)
            except Exception:
                # fallback: keep current memory
                pass

        return start_episode, best_avg


if __name__ == "__main__":
    breakout_dqn = BreakoutDQN()
    checkpoint_path = "models/checkpoint_latest.pth"
    replay_path = "models/replay_memory_latest.pth"

    start_from_checkpoint = True

    if os.path.exists(checkpoint_path) and start_from_checkpoint:
        print(f"Auto-resume from {checkpoint_path}")
        breakout_dqn.train(episodes=1000, render=False, resume_checkpoint=checkpoint_path, resume_replay=replay_path)
    else:
        breakout_dqn.train(episodes=1000, render=False)
    
    # TESTING
    # breakout_dqn.test(10, "models/CNN_breakout.pt")
