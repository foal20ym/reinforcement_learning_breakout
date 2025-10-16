import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Convolutional Neural Network for DQN on Atari games.
    Input: 4 stacked grayscale frames (4, 84, 84)
    Output: Q-values for each action
    """

    def __init__(self, n_actions):
        super(DQNNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate size after convolutions
        # Input: 84x84
        # After conv1: (84-8)/4 + 1 = 20
        # After conv2: (20-4)/2 + 1 = 9
        # After conv3: (9-3)/1 + 1 = 7
        conv_output_size = 64 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 4, 84, 84)

        Returns:
            Q-values for each action
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0

        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
