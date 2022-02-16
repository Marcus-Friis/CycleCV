import torch.nn as nn
import torch

class Mikkel(nn.Module):
    def __init__(self):
        super(Mikkel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits