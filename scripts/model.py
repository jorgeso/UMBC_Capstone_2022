import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.lstm_size = 768
        self.num_layers = 1
        self._device = device

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.7,
            batch_first=True
        )
        self.fc = nn.Linear(self.lstm_size, 1)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        output, state = self.lstm(x, (h0, c0))
        print(output.size())
        print("="*50)
        print(output[:, -1, :].size())
        print("="*50)
        output = self.fc(output[:, -1, :])
        output = torch.sigmoid(output)
        return output, state

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self._device),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self._device))