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


class CNN(nn.Module):
    def __init__(self, num_actions = 4, **argv):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1024),  # 3136 = 64*7*7 Denna måste vara hardcoded
            nn.ReLU(),
            nn.Linear(1024, num_actions)
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
    
    
def preprocess_frame(obs):
    obs = torch.tensor(obs)
    obs = obs.permute(2, 0, 1).float()  # HWC → CHW
    obs = T.functional.rgb_to_grayscale(obs)  # [1, 210, 160]
    obs = obs[:, 34:34+160, :]  # crop to 160x160 to remove the score and some extra parts
    obs = T.functional.resize(obs, (84, 84), interpolation=T.InterpolationMode.NEAREST)
    return obs.to(torch.uint8)