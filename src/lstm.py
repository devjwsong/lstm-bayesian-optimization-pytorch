import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as f

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(self.args.input_dim, self.args.emb_dim)
        self.rnn = nn.LSTM(
            input_size=self.args.emb_dim,
            hidden_size=self.args.hid_dim,
            dropout=self.args.drop_out,
            batch_first=True,
            num_layers=self.args.num_layers)
        self.fc = nn.Linear(self.args.hid_dim, self.args.output_dim)
        self.softmax = nn.Softmax(self.args.output_dim)
        self.num_layers = self.args.num_layers
        self.hidden_dim = self.args.hid_dim

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