import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)
from torch import nn

class Model(nn.Module):
    def __init__(
        self, device,
        attn_layers=5,
        lstm_layers=1,
        hidden_size=384,
        dropout=0.3,
        attn_dropout=0.3,
        decoder_dropout=0.3,
        out_layers=5,
        is_regression=False
    ):
        super(Model, self).__init__()
        self.lstm_size = 768
        self.num_layers = lstm_layers
        self._device = device
        self.hidden_size = hidden_size
        if lstm_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.attn = nn.Sequential()

        for _ in range(attn_layers):
            self.attn.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.attn.append(
                nn.Dropout(attn_dropout)
            )
            self.attn.append(
                nn.ReLU()
            )

        self.attn.append(
            nn.Linear(self.hidden_size, 1)
        )
        self.attn.append(
            nn.Softmax(1)
        )

        self.decoder = nn.Sequential()

        for _ in range(out_layers):

            self.decoder.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.decoder.append(
                nn.Dropout(decoder_dropout)
            )
            self.decoder.append(
                nn.ReLU()
            )
        self.decoder.append(
            nn.Linear(hidden_size, 1)
        )

        if is_regression:
            self.decoder.append(
                nn.Tanh()
            )
        else:
            self.decoder.append(
                nn.Sigmoid()
            )

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        output, state = self.lstm(x, (h0, c0))

        weights = self.attn(output)
        decoder_input = torch.mul(weights, output)
        decoder_input = torch.sum(decoder_input, dim=1)

        output = self.decoder(decoder_input)

        return output, state

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device))