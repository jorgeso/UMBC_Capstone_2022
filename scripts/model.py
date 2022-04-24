import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, device, lstm_layers=1, hidden_size=384, dropout=0.7):
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
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        output, state = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        output = torch.sigmoid(output)
        return output, state

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device),
                torch.zeros(self.num_layers, sequence_length, self.hidden_size).to(self._device))