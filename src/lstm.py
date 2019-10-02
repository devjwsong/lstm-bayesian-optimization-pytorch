import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as f

class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            dropout=0.5,
            batch_first=True,
            num_layers=num_layers)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.Softmax(output_dim)
        self.num_layers = num_layers
        self.hidden_dim = hid_dim

    def hidden_init(self, batch_size):
        h0 = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        c0 = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, text, rnn_init=None):
        if rnn_init is None:
            rnn_init = self.hidden_init(text.size(0))
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded, rnn_init)
        output = self.fc(output)
        output = output[:, -1, :]

        return output