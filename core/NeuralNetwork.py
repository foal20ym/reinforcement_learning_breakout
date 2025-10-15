import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T


class NeuralNetwork(nn.Module):

    def __init__(self, in_states, hidden_layer_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, hidden_layer_nodes) # Fully connected layer
        self.fc2 = nn.Linear(hidden_layer_nodes, hidden_layer_nodes) # Fully connected layer
        self.out = nn.Linear(hidden_layer_nodes, out_actions) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
