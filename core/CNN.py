from torch import nn


class CNN(nn.Module):
    def __init__(self, num_actions=4):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            # Input: 4 frames x 84x84 pixels (grayscale stacked frames for motion detection)
            # Conv Layer 1: 84x84 -> 20x20
            # Formula: output_size = (input_size - kernel_size) / stride + 1
            # (84 - 8) / 4 + 1 = 76 / 4 + 1 = 19 + 1 = 20
            # Produces 32 feature maps of size 20x20
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv Layer 2: 20x20 -> 9x9
            # (20 - 4) / 2 + 1 = 16 / 2 + 1 = 8 + 1 = 9
            # Produces 64 feature maps of size 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv Layer 3: 9x9 -> 7x7
            # (9 - 3) / 1 + 1 = 6 / 1 + 1 = 7
            # Produces 64 feature maps of size 7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Flatten converts 3D tensor (64 channels x 7x7 spatial) into 1D vector
            nn.Flatten(),
            # 3136 = 64 channels * 7 height * 7 width
            # This is the total number of neurons after flattening the final conv layer
            # NOTE: This assumes input is 84x84. Will break with different input sizes.
            # The downsampling chain: 84 -> 20 -> 9 -> 7 reduces spatial dimensions
            # while increasing feature depth (4 -> 32 -> 64 -> 64), extracting
            # hierarchical features from raw pixels to abstract game state
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions),
        )

    def forward(self, x):
        return self.net(x)
