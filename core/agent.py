import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .model import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent for Atari games.
    """

    def __init__(self, n_actions, config):
        """
        Initialize DQN Agent.

        Args:
            n_actions: Number of possible actions
            config: Configuration module with hyperparameters
        """
        self.n_actions = n_actions
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQNNetwork(n_actions).to(self.device)
        self.target_net = DQNNetwork(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)

        # Replay buffer
        self.memory = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        # Training steps counter
        self.steps = 0

        # Epsilon for exploration
        self.epsilon = config.EPS_START

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (4, 84, 84)
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        """Update epsilon value using linear decay."""
        self.epsilon = max(
            self.config.EPS_END,
            self.config.EPS_START
            - (self.config.EPS_START - self.config.EPS_END) * self.steps / self.config.EPS_DECAY_STEPS,
        )

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.config.MIN_REPLAY_SIZE:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)

        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.GAMMA * next_q_values

        # Compute loss
        loss = nn.functional.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.GRADIENT_CLIP)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps = checkpoint["steps"]
        self.epsilon = checkpoint["epsilon"]
        print(f"Checkpoint loaded from {filepath}")
