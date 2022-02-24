import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import datetime
import pickle
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # input_size, hidden_size, num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # hidden_dim, output_dim
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    # TODO: rewrite (hn, cn) as hiddenstate
    def forward(self, x):
        lengths = x.shape[1] - torch.sum(x[:, :, -1] == torch.tensor(0), axis=1)
        x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        output_pack, hidden = self.lstm(x_pack)
        x_unpack, lens_unpack = pad_packed_sequence(output_pack, batch_first=True)
        output = self.fc(x_unpack)

        return output, hidden

    # TODO: improve predicting method
    # def mikkel(self, x, future_preds=10):
    #     out = x
    #     output = out[:, -1:]
    #     batch = output.shape[0]
    #     cn = torch.zeros(2, batch, 64)
    #     hn = torch.zeros(2, batch, 64)
    #     for i in range(future_preds):
    #         out, (hn, cn) = self.forward(out, hn=hn, cn=cn)
    #         out = out[:, -1:, :]
    #         out = torch.concat((out, output[:, -1:, :2], output[:, -1:, 4:]), axis=2)
    #         output = torch.concat((output, out), axis=1)
    #     return output


if __name__ == '__main__':
    torch.manual_seed(69)
    net = LSTM(6, 64, 2, 2)  # (input_size, hidden_size, output_size, num_layers)
    x = torch.randn(4, 3, 6)  # batch_first (batch, time_steps, input_size) else (times_steps, batch, input_size)
    out, (hn, cn) = net(x)

    # TODO: normalize data before pickling
    with open('jeff.pkl', 'rb') as f:
        df = pickle.load(f)

    inputs = []
    targets = []
    lengths = []
    for _, row in df.iterrows():
        length = len(row['xs'])
        if length:
            lengths.append(length)

            traj = torch.tensor([
                list(row['xs']),
                list(row['ys']),
                list(row['p_x']),
                list(row['p_y']),
                [row['ds_x']] * len(row['xs']),
                [row['ds_y']] * len(row['xs'])
            ]).T
            traj[:, ::2] /= 1280
            traj[:, 1::2] /= 720
            inputs.append(traj)

            target = torch.tensor([
                list(row['t_x']),
                list(row['t_y'])
            ]).T
            target[:, 0] /= 1280
            traj[:, 1] /= 720
            targets.append(target)

    # TODO: implement packed paddings to use full sequences
    inputs = pad_sequence(inputs, batch_first=True)
    # inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

    targets = pad_sequence(targets, batch_first=True)
    # targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)

    batch_size = 64
    dropout = 0.2
    n_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-6

    train = TensorDataset(inputs, targets)
    train_loader = DataLoader(train, batch_size=64, shuffle=False, drop_last=True)

    model = LSTM(6, 64, 2, 2)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()  # zero the gradient buffers
            output, (hn, cn) = model(x_batch)
            y_batch = y_batch[:,:output.shape[1]]
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

        print(
            f"[{epoch}/{n_epochs}]"
        )

    model_path = f'models/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'
    torch.save(model.state_dict(), model_path)

    # model.load_state_dict(torch.load('models/2022-02-23_16-36-55.pt'))

    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    fig, ax = plt.subplots(figsize=(16, 16))
    im = Image.open("intersection2.png")
    im = ImageOps.flip(im)
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)
    ax.imshow(im, origin='lower')

    plot_this = inputs[50:60].detach()

    for n in plot_this:
        ax.scatter(n[:,0]*1280, n[:,1]*720, c='b')

    # out = model.mikkel(plot_this, 10)
    # for n in out.detach():
    #     ax.scatter(n[:,0]*1280, n[:,1]*720, c='r')

    print(out)
    plt.show()
