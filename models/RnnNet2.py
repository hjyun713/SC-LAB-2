import torch
import torch.nn as nn
from torchinfo import summary


class Model(nn.Module):
    def __init__(self, time_slot: int = 64, depth: int = 5, num_class: int = 4, channel: int = 4):
        super(Model, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.LSTM(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.LSTM(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        # self.Dense1000 = nn.Linear(time_slot, 1000)
        self.Dense100 = nn.Linear(time_slot, 100)
        # self.DenseInput = nn.Linear(1024, 1024)
        # self.Dense64 = nn.Linear(1000, time_slot)
        self.Dense10 = nn.Linear(100, time_slot)
        self.DROP = nn.Dropout(0.7)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x) -> torch.tensor:
        x,_ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense100(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense10(x_den)
        x_cls = self.CLS(x_den)
        x_return = self.SOFT(x_cls)
        return x_return