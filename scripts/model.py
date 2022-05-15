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
        dropout=0.7,
        attn_dropout=0.7,
        decoder_dropout=0.7,
        out_layers=5
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

        self.attn = nn.Sequential(dropout=attn_dropout)

        for _ in range(attn_layers):
            self.attn.append(
                nn.Linear(self.lstm_size, self.lstm_size)
            )
            self.attn.append(
                nn.ReLU()
            )

        self.attn.append(
            nn.Linear(self.lstm_size, 1)
        )
        self.attn.append(
            nn.Softmax(1)
        )

        # self.lin_0 = nn.Linear(self.lstm_size, (self.lstm_size//4)*3)
        # self.non_lin_1 = nn.ReLU()
        # self.lin_1 = nn.Linear((self.lstm_size//4)*3, self.lstm_size//2)
        # self.non_lin_2 = nn.ReLU()
        # self.lin_2 = nn.Linear(self.lstm_size//2, self.lstm_size//4)
        # self.non_lin_3 = nn.ReLU()
        # self.lin_3 = nn.Linear(self.lstm_size//4, 1)

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.out_lin = nn.Sequential(dropout=decoder_dropout)

        for _ in range(out_layers):

            self.out_lin.append(
                nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.out_lin.append(
                nn.ReLU()
            )
        self.out_lin.append(
            nn.Linear(hidden_size, 1)
        )
        self.out_lin.append(
            nn.Tanh()
        )

        # self.fc_0 = nn.Linear(self.hidden_size, (self.hidden_size//4)*3)
        # self.non_lin_4 = nn.ReLU()
        # self.fc_1 = nn.Linear((self.hidden_size//4)*3, self.hidden_size//2)
        # self.non_lin_5 = nn.ReLU()
        # self.fc_2 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        # self.non_lin_6 = nn.ReLU()
        # self.fc_3 = nn.Linear(self.hidden_size//4, 1)

    def forward(self, x):
        # weights = self.lin_0(x)
        # weights = self.non_lin_1(weights)
        # weights = self.lin_1(weights)
        # weights = self.non_lin_2(weights)
        # weights = self.lin_2(weights)
        # weights = self.non_lin_3(weights)
        # weights = self.lin_3(weights)
        # weights_normalized = torch.softmax(weights, 1)
        weights = self.attn(x)
        lstm_input = torch.mul(weights, x)
        h0, c0 = self.init_hidden(lstm_input.size(0))
        output, state = self.lstm(lstm_input, (h0, c0))
        output = self.out_lin(output[:, -1, :])
        # output = self.fc_0(output[:, -1, :])
        # output = self.non_lin_4(output)
        # output = self.fc_1(output)
        # output = self.non_lin_5(output)
        # output = self.fc_2(output)
        # output = self.non_lin_6(output)
        # output = self.fc_3(output)
        # output = torch.tanh(output)
        return output, state

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device))