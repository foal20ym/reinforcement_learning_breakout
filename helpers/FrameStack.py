from collections import deque
import torch


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
    