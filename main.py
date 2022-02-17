import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pickle
import pandas as pd
import numpy as np

from pytorch import *
from mikkel import Mads

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# read and prepare training data
with open("train.pkl", "rb") as f:
    df = pickle.load(f)
x = df[['x_pos', 'y_pos', 'x_vec', 'y_vec', 'x_vec2', 'y_vec2', 'x_dest', 'y_dest']].to_numpy()
y = df[['x_tar', 'y_tar']].to_numpy()

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
train = TensorDataset(x, y)
train_loader = DataLoader(train, batch_size=64, shuffle=False, drop_last=True)

# read and prepare validation data
with open("val.pkl", "rb") as f:
    df = pickle.load(f)
x = df[['x_pos', 'y_pos', 'x_vec', 'y_vec', 'x_vec2', 'y_vec2', 'x_dest', 'y_dest']].to_numpy()
y = df[['x_tar', 'y_tar']].to_numpy()

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
val = TensorDataset(x, y)
val_loader = DataLoader(train, batch_size=64, shuffle=False, drop_last=True)

# set device, cuda does not work on my pc
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')

input_dim = 8
output_dim = 2
hidden_dim = 100
layer_dim = 3
batch_size = 64
dropout = 0.2
n_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-6

model_params = {
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'layer_dim': layer_dim,
    'output_dim': output_dim,
    'dropout_prob': dropout
}

model = Mads().to(device).float()
# model = get_model('LSTM', model_params).to(device).float()

# loss_fn = nn.MSELoss(reduction="mean")
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
# opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
# opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
# opt.plot_losses()

# predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


model.load_state_dict(torch.load('models/2022-02-17_15-16-03.pt'))
tis = torch.tensor([
        [[1050, 275, 0, 0, 0, 0, 1000, 470]],
        [[580*2, 50*2, 0, 0, 0, 0, 250*2, 280*2]],
        [[200*2, 225*2, 0, 0, 0, 0, 500*2, 250*2]],
        [[125*2, 200*2, 0, 0, 0, 0, 460*2, 50*2]],
        [[580*2, 50*2, 0, 0, 0, 0, 250*2, 300*2]],
        [[150*2, 240*2, 0, 0, 0, 0, 250*2, 300*2]]
        ]).float()
from simulator import SimulatorTorch
sim = SimulatorTorch(model, tis)
asd = sim.simulate()
sim.plot_simulation(asd)