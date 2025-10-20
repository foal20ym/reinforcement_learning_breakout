from collections import deque
import random
import os
import torch
import numpy as np

class ReplayMemory():

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
        
    def save(self, filepath):
        """
        Save replay buffer to filepath. Tensors are moved to CPU before saving.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        buffer_cpu = []
        for item in getattr(self, "buffer", []):
            # Expect item = (state, action, next_state, reward, done)
            s, a, ns, r, d = item
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu()
            if isinstance(ns, torch.Tensor):
                ns = ns.detach().cpu()
            # normalize simple types
            if isinstance(a, torch.Tensor):
                a = a.item()
            if isinstance(r, np.ndarray):
                r = float(r)
            r = float(r)
            d = bool(d)
            buffer_cpu.append((s, a, ns, r, d))

        data = {
            "buffer": buffer_cpu,
            "capacity": getattr(self, "capacity", len(buffer_cpu)),
            "position": getattr(self, "position", 0),
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load a ReplayMemory instance from filepath. Returns a new ReplayMemory instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        data = torch.load(filepath, map_location="cpu")
        capacity = data.get("capacity", len(data.get("buffer", [])))
        mem = cls(capacity)
        mem.buffer = data.get("buffer", [])
        mem.position = data.get("position", 0)
        return mem