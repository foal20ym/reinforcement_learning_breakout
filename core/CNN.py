from torch import nn


class CNN(nn.Module):
    def __init__(self, num_actions=4, **argv):
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
            nn.Linear(1024, num_actions),
        )

    def forward(self, x):
        return self.net(x)
