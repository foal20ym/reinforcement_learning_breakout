import gymnasium as gym
import time
import ale_py
import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

def plot_progress(rewards_per_episode, epsilon_history):
        plt.figure(1)

        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(rewards_per_episode)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        plt.savefig('plots/dqn.png')

class NeuralNetwork1(nn.Module):

    def __init__(self, in_states, hidden_layer_nodes, out_actions):
        super().__init__()

        # Dynamically calculate the input size
        self.flattened_size = in_states  # Ensure this matches the flattened state size
        self.fc1 = nn.Linear(self.flattened_size, hidden_layer_nodes)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_layer_nodes, hidden_layer_nodes)  # Fully connected layer
        self.out = nn.Linear(hidden_layer_nodes, out_actions)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
class NeuralNetwork(nn.Module):
    def __init__(self, action_dim = 4):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1024),  # 3136 = 64*7*7
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayMemory():

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)


class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        """Call this at the beginning of an episode"""
        self.frames.clear()

    def append(self, obs):
        self.frames.append(obs)

    def get_stack(self):
        assert len(self.frames) == self.k
        return torch.cat(list(self.frames), dim=0)
    
# def preprocess_frame(obs):
#     obs = torch.tensor(obs)
#     obs = obs.permute(2, 0, 1).float()  # HWC → CHW
#     obs = T.functional.rgb_to_grayscale(obs)  # [1, 210, 160]
#     obs = obs[:, 34:34+160, :]  # crop to 160x160 to remove the score and some extra parts
#     obs = T.functional.resize(obs, (84, 84), interpolation=T.InterpolationMode.NEAREST)
#     return obs.to(torch.uint8)


class BreakoutDQN:
    learning_rate = 0.001
    discount_factor = 0.99
    network_sync_rate = 10000
    replay_memory_size = 100_000
    mini_batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    num_hidden_nodes = 512

    
    """
    best so far:
    learning_rate = 0.001
    discount_factor = 0.99
    network_sync_rate = 10000
    replay_memory_size = 100_000
    mini_batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    num_hidden_nodes = 256
    """

    loss_fn = nn.MSELoss()
    optimizer = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(self.replay_memory_size)

    def get_fire_action(self, env):
        """Return the index for the 'FIRE' action if available, else default to 1."""
        action_meanings = None
        try:
            action_meanings = env.unwrapped.get_action_meanings()
        except Exception:
            try:
                action_meanings = env.get_action_meanings()
            except Exception:
                action_meanings = None

        if action_meanings:
            for i, meaning in enumerate(action_meanings):
                if 'FIRE' in meaning.upper():
                    return i
        return 1

    def preprocess_frame(self, obs):
        obs = torch.tensor(obs)
        obs = obs.permute(2, 0, 1).float()  # HWC → CHW
        obs = T.functional.rgb_to_grayscale(obs)  # [1, 210, 160]
        obs = obs[:, 34:34+160, :]  # crop to 160x160 to remove the score and some extra parts
        obs = T.functional.resize(obs, (84, 84), interpolation=T.InterpolationMode.NEAREST)
        return obs.to(torch.uint8)

    def train(self, episodes, render=False):
        env = gym.make('ALE/Breakout-v5', render_mode="rgb_array" if render else None)
        num_actions = env.action_space.n

        policy_dqn = NeuralNetwork(num_actions).to(self.device)
        target_dqn = NeuralNetwork(num_actions).to(self.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = []
        epsilon_history = []
        step_count = 0

        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            prev_lives = info.get('lives', None)  # Track lives at the start of the episode

            fire_action = self.get_fire_action(env)
            obs, _, terminated, truncated, info = env.step(fire_action)

            obs = self.preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if random.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.to(self.device)).argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = self.preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                next_state = frame_stack.get_stack().unsqueeze(0).float() / 255.0

                # Check if lives have decreased
                current_lives = info.get('lives', prev_lives)
                if prev_lives is not None and current_lives < prev_lives:
                    # Life lost: relaunch the ball
                    obs, _, terminated, truncated, info = env.step(fire_action)
                    obs = self.preprocess_frame(obs)
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(obs)
                    state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                prev_lives = current_lives

                self.memory.append((state, action, next_state, reward, terminated))
                state = next_state
                total_reward += reward
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
                torch.save(policy_dqn.state_dict(), "models/breakout_dqn_best.pt")

        env.close()
        self.plot_progress(rewards_per_episode, epsilon_history)

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

    def plot_progress(self, rewards_per_episode, epsilon_history):
        plt.figure(1)
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.title("Rewards per Episode")
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.savefig('plots/breakout_dqn.png')
    
    """
    Runs the environment with the learned policy by creating the environment and using the best actions learned during training
    """
    def test(self, episodes, model_filepath):
        env = gym.make('ALE/Breakout-v5', render_mode='human')
        num_actions = env.action_space.n

        policy_dqn = NeuralNetwork(num_actions).to(self.device)
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()

        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            prev_lives = info.get('lives', None)  # Track lives at the start of the episode

            fire_action = self.get_fire_action(env)
            obs, _, terminated, truncated, info = env.step(fire_action)

            obs = self.preprocess_frame(obs)
            frame_stack = FrameStack(4)
            frame_stack.reset()
            for _ in range(4):
                frame_stack.append(obs)

            state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
            total_reward = 0
            terminated = truncated = False

            while not (terminated or truncated):
                with torch.no_grad():
                    action = policy_dqn(state.to(self.device)).argmax().item()

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = self.preprocess_frame(next_obs)
                frame_stack.append(next_obs)
                state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                total_reward += reward

                # Check if lives have decreased
                current_lives = info.get('lives', prev_lives)
                if prev_lives is not None and current_lives < prev_lives:
                    # Life lost: relaunch the ball
                    obs, _, terminated, truncated, info = env.step(fire_action)
                    obs = self.preprocess_frame(obs)
                    frame_stack.reset()
                    for _ in range(4):
                        frame_stack.append(obs)
                    state = frame_stack.get_stack().unsqueeze(0).float() / 255.0
                prev_lives = current_lives

            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

        env.close()

if __name__ == "__main__":
    breakout_dqn = BreakoutDQN()
    breakout_dqn.train(episodes=200, render=False)
    #breakout_dqn.test(10, "models/breakout_dqn_best.pt")

