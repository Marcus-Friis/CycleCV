import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
import datetime
import matplotlib.pyplot as plt

import pickle
import numpy as np

from wrangler import Wrangler


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x, x_lens):
        x_pack = pack_padded_sequence(x_pad, x_lens, batch_first=True)

        out, (hn, cn) = self.lstm(x_pack)

        out_pad, out_lens = pad_packed_sequence(out, batch_first=True)
        out_shaped = out_pad.reshape(-1, self.hidden_dim)  # torch.flatten(out_pad, start_dim=0, end_dim=1)
        zero_mask = torch.unique(torch.nonzero(out_shaped)[:, 0])
        out_fc = out_shaped[zero_mask]

        out = self.fc(out_fc)

        return out


if __name__ == '__main__':
    cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
            'l2', 'l3', 'dir_0', 'dir_1', 'dir_2'] + ['d_zone_'+str(i) for i in range(20)]

    pdf = Wrangler.load_pickle('data/pdf_sample.pkl')

    seq = []
    y_seq = []
    for i, row in pdf.iterrows():
        x = np.vstack(row[cols]).T
        seq.append(torch.from_numpy(x).float())

    sequences_sorted = sorted(seq, key=lambda x: x.shape[0], reverse=True)
    lens = torch.tensor([n.size(0) for n in sequences_sorted])
    x_pad = pad_sequence(sequences_sorted, batch_first=True)
    y = torch.from_numpy(np.concatenate(pdf['euc']))

    model = LSTM(33, 100, 1, 2, 0)
    print(model(x_pad, lens).shape)


    """
    # TODO: create y
    y = pdf['euc']

    # TODO: train test val split

    batch_size = 2
    td = TensorDataset(x_pad, y, lens)
    train_loader = DataLoader(td, batch_size=batch_size, shuffle=False)

    input_dim = len(cols)
    hidden_dim = 100
    output_dim = 1
    num_layers = 2
    dropout = 0

    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-6

    model = LSTM(input_dim, hidden_dim, output_dim, num_layers, dropout)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    for epoch in range(1, n_epochs+1):
        model.train()
        batch_loss = []
        for x_batch, y_batch, x_lens in train_loader:
            y_hat = model(x_batch, x_lens)

            loss = loss_fn(y_batch, y_hat)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_loss.append(loss.item())
        train_loss.append(np.mean(batch_loss))

        # TODO: implement val loader
        # model.eval()
        # with torch.no_grad():
        #     for x_val, y_val, x_lens_val in val_loader:
        #         pass

    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.show()

    # EXAMPLE
    # lstm = LSTM(10, 100, 1, 2, 0)
    # a = torch.rand(3, 10)
    # b = torch.rand(2, 10)
    # c = torch.rand(60, 10)
    # tensors = sorted([a, b, c], key=lambda x: x.shape[0], reverse=True)
    # lens = torch.tensor([n.size(0) for n in tensors])
    # x_pad = pad_sequence(tensors, batch_first=True)
    # x_pack = pack_padded_sequence(x_pad, lens, batch_first=True)
    # print(lstm(x_pack).shape)
    """
