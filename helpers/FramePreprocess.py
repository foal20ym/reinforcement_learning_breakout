import torch
import torchvision.transforms as T

def preprocess_frame(obs):
    obs = torch.tensor(obs)
    obs = obs.permute(2, 0, 1).float()  # HWC → CHW
    obs = T.functional.rgb_to_grayscale(obs)  # [1, 210, 160]
    obs = obs[:, 34:34+160, :]  # crop to 160x160 to remove the score and some extra parts
    obs = T.functional.resize(obs, (84, 84), interpolation=T.InterpolationMode.NEAREST)
    return obs.to(torch.uint8)