import gymnasium as gym
import ale_py
import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from visualization import plot_progress
from model import CNN, NeuralNetwork, ReplayMemory, FrameStack, preprocess_frame
import os


class BreakoutDQN:
    learning_rate = 0.001
    discount_factor = 0.99
    network_sync_rate = 10_000
    replay_memory_size = 100_000
    mini_batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    num_hidden_nodes = 256

    loss_fn = nn.MSELoss()
    optimizer = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(self.replay_memory_size)

    def train(self, episodes, render=False):
        env = gym.make('ALE/Breakout-v5', render_mode="rgb_array" if render else None)
        num_actions = env.action_space.n
        state_dim = 4 * 84 * 84  # 4 stacked frames of size 84x84
        
        policy_dqn = CNN(num_actions).to(self.device)
        target_dqn = CNN(num_actions).to(self.device)
        # policy_dqn = NeuralNetwork(state_dim, self.num_hidden_nodes, num_actions).to(self.device)
        # target_dqn = NeuralNetwork(state_dim, self.num_hidden_nodes, num_actions).to(self.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0
        best_avg_reward = self.load_best_avg_reward()

        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            obs, _, terminated, truncated, _ = env.step(1)
            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            terminated = truncated = False
            episode_steps = 0

            while not (terminated or truncated):
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.to(self.device)).argmax().item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                self.memory.append((state, action, next_state, reward, terminated))
                state = next_state
                total_reward += reward
                episode_steps += 1
                step_count += 1

                if len(self.memory) > self.mini_batch_size and step_count % 4 == 0:
                    mini_batch = self.memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(self.epsilon)

            if episode % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

                # Check if we have a new best average reward
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_best_avg_reward(best_avg_reward)  # Save the new best average reward
                    model_filename = f"models/CNN_breakout_avg_{int(best_avg_reward)}.pt"
                    torch.save(policy_dqn.state_dict(), model_filename)
                    print(f"New best average reward! Model saved as {model_filename}")

        env.close()
        plot_progress(rewards_per_episode, epsilon_history)

    def load_best_avg_reward(self):
        """Load the best average reward from a file."""
        filepath = "environment/best_avg_reward.txt"
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                return float(file.read().strip())
        return -float("inf")  # Default to a very low value if the file doesn't exist

    def save_best_avg_reward(self, best_avg_reward):
        """Save the best average reward to a file."""
        filepath = "environment/best_avg_reward.txt"
        with open(filepath, "w") as file:
            file.write(f"{best_avg_reward}")

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
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10.0)
        self.optimizer.step()

    """
    Runs the environment with the learned policy by creating the environment and using the best actions learned during training
    """
    def test(self, episodes, model_filepath):
        env = gym.make('ALE/Breakout-v5', render_mode='human')
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = CNN(num_actions).to(self.device)
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()

        for episode in range(1, episodes + 1):
            obs, _ = env.reset()
            obs = preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            terminated = truncated = False

            # Fire the ball at the start of the episode
            env.step(1)  # Action 1 is usually "fire" in Breakout

            while not (terminated or truncated):
                with torch.no_grad():
                    q_values = policy_dqn(state.to(self.device))
                    action = q_values.argmax().item()
                    print(f"Q-values: {q_values}, Selected Action: {action}")

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                total_reward += reward

            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

        env.close()

if __name__ == "__main__":
    breakout_dqn = BreakoutDQN()
    breakout_dqn.train(episodes=300, render=False)
    #breakout_dqn.test(10, "models/CNN_breakout_avg_5.pt")

