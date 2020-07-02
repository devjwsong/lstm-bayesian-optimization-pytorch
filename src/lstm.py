from constant import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Seed fixing
        np.random.seed(777)
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        random.seed(777)

        self.embedding = nn.Embedding(vocab_size, d_w)
        self.lstm = nn.LSTM(
            input_size=d_w,
            hidden_size=d_h,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=drop_out_rate,
            num_layers=layer_num
        )
        self.dir_num = 2 if bidirectional else 1
        self.query = nn.Linear(d_h * self.dir_num, 1)
        self.output_linear = nn.Linear(d_h * self.dir_num, class_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def init_hidden(self, input_shape):
        h0 = torch.zeros((layer_num * self.dir_num, input_shape[0], d_h)).to(device)
        c0 = torch.zeros((layer_num * self.dir_num, input_shape[0], d_h)).to(device)

        return h0, c0

    def forward(self, x, lens):
        h0, c0 = self.init_hidden(x.shape)

        embedded = self.embedding(x)  # (B, L) => (B, L, d_w)
        packed_input = pack_padded_sequence(embedded, lens, batch_first=True)

        output, _ = self.lstm(packed_input, (h0, c0))
        output = pad_packed_sequence(output, batch_first=True)[0]  # (B, L, d_h)

        attn_score = self.query(output).squeeze(dim=-1)  # (B, L)
        attn_distrib = F.softmax(attn_score, dim=-1)  # (B, L)
        output = torch.bmm(attn_distrib.unsqueeze(dim=1), output).squeeze(dim=1)  # (B, d_h)

        output = self.output_linear(output)  # (B, class_num)

        return self.softmax(output)