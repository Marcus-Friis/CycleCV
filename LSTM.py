import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import TensorDataset, DataLoader

from wrangler import Wrangler


# LSTM regressor model with linear layer at the end. Predicts many-to-many and uses padded inputs.
# For this purpose, lengths of sequences is an input in forward needed for packing the sequences.
# This implementation also trains on the padding and is batch first.
class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float, total_length: int):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.total_length = total_length  # used for unpacking in forward

        # layers in model
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, x_lens):
        # forward takes x-shape = (batch, sequence, input_dim)

        # pack padded sequences for LSTM processing
        x_pack = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(x_pack)

        # unpack and feed to forward layer
        out_pad, out_lens = pad_packed_sequence(out, batch_first=True, total_length=self.total_length)
        out = self.fc(out_pad)

        # out-shape = (batch, sequence, output_dim)
        return out


if __name__ == '__main__':
    # cols used as features
    cols = ['x', 'y', 'd_t-1', 'd_t-2', 'd_t-3', 'd_light', 'l0', 'l1',
            'l2', 'l3', 'dir_0', 'dir_1', 'dir_2'] + ['d_zone_'+str(i) for i in range(20)]
    print('loading data...')
    file_path = 'data/pdf_train.pkl'
    pdf = Wrangler.load_pickle(file_path)
    # pdf = pdf.loc[pdf['cluster'] == 4]

    # pads sequences by shaping into tensors, calculating lengths and using pad_sequence
    print('preparing sequences...')
    seq = []
    y_seq = []
    for i, row in pdf.iterrows():
        x = np.vstack(row[cols]).T
        y = np.vstack(row['euc'])
        seq.append(torch.from_numpy(x).float())
        y_seq.append(torch.from_numpy(y).float().reshape(-1, 1))

    # sequences_sorted = sorted(seq, key=lambda x: x.shape[0], reverse=True)
    lens = torch.tensor([n.size(0) for n in seq])
    x_pad = pad_sequence(seq, batch_first=True)
    y_pad = pad_sequence(y_seq, batch_first=True)

    # prepares data splits and assigns splits to DataLoaders
    batch_size = 64
    td = TensorDataset(x_pad, y_pad, lens)
    train_size = int(0.8 * len(td))
    val_size = (len(td) - train_size) // 2
    train_dataset, val_dataset = torch.utils.data.random_split(td, [train_size, val_size],
                                                                      generator=torch.Generator().manual_seed(420))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # hyperparameters of LSTM model
    input_dim = len(cols)
    hidden_dim = 100
    output_dim = 1
    num_layers = 2
    dropout = 0.2
    total_length = x_pad.size(1)

    # training parameters
    n_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-6

    model = LSTM(input_dim, hidden_dim, output_dim, num_layers, dropout, total_length)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop: for each epoch, loops through all batches,
    # makes prediction, calculates loss and backpropagates gradients.
    # Also keeps track of training- and validation loss
    train_loss = []
    val_loss = []
    for epoch in range(1, n_epochs+1):
        print(f'epoch\t{epoch}\tout of\t{n_epochs}')

        # training loop
        model.train()
        batch_loss = []
        for x_batch, y_batch, l_batch in train_loader:
            y_hat = model(x_batch, l_batch)

            loss = loss_fn(y_batch, y_hat)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_loss.append(loss.item())
        train_loss.append(np.mean(batch_loss))

        # validation loop
        model.eval()
        batch_val_loss = []
        with torch.no_grad():
            for x_val, y_val, l_val in val_loader:
                y_hat = model(x_val, l_val)

                loss = loss_fn(y_val, y_hat)
                batch_val_loss.append(loss.item())
            val_loss.append(np.mean(batch_val_loss))

    # save trained model
    torch.save(model.state_dict(), 'models/lstm_c4.pt')

    # TODO: implement val and test score

    # plot training and validation loss
    fig, ax = plt.subplots()
    ax.set_title('train and validation loss')
    ax.plot(range(n_epochs), train_loss, label="Training loss")
    ax.plot(range(n_epochs), val_loss, label="Validation loss")
    ax.legend()
    ax.set_ylim(0)
    # plt.show()
    plt.savefig('loss_lstm.png')

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
