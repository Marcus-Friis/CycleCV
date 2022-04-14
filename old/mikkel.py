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


class Mads(nn.Module):
    def __init__(self):
        super(Mads, self).__init__()
        self.input_dim = 9
        self.l1 = nn.Linear(self.input_dim, 50)
        self.l2 = nn.Linear(50, 100)
        self.l3 = nn.Linear(100, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, 2)

    def forward(self, x):
        relu = nn.ReLU()
        h1 = self.l1(x)
        h1 = relu(h1)
        h2 = self.l2(h1)
        h2 = relu(h2)
        h3 = self.l3(h2)
        h3 = relu(h3)
        h4 = self.l4(h3)
        h4 = relu(h4)
        h5 = self.l5(h4)
        return h5


class Anna(nn.Module):
    def __init__(self):
        super(Anna, self).__init__()
        self.input_dim = 8
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        output = self.layers(x)
        return output
