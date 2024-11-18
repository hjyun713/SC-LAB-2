import torch.nn as nn


class TModelHJY(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 8, num_class: int = 4, channel: int = 4):
        super(TModelHJY, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.GRU(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GP1 = nn.GRU(input_size=self.channel, hidden_size=1024, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GP2 = nn.GRU(input_size=1024, hidden_size=512, batch_first=False,
                           num_layers=1, bidirectional=True)
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.DenseInput = nn.Linear(1024, 1024)
        self.Dense64 = nn.Linear(1024, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.GP1(x)
        y = self.DenseInput(x)
        x_d1 = self.DROP(x)
        x2, _ = self.GP2(x_d1)
        x_d2 = self.DROP(x)
        x_den = x_d2 + y
        x_den = self.Dense64(x_den)
        x_cls = self.CLS(x_den)
        x_return = self.SOFT(x_cls)
        return x_return


class TModelHHJ84(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 8, num_class: int = 4, channel: int = 4):
        super(TModelHHJ84, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.GRU(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=int(time_slot / 2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.Dense100 = nn.Linear(time_slot, 100)
        self.DenseInput = nn.Linear(1024, 1024)
        self.Dense64 = nn.Linear(1000, time_slot)
        self.Dense10 = nn.Linear(100, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x):
        x, _ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense1000(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense64(x_den)
        x_cls = self.CLS(x_den)
        x_return = self.SOFT(x_cls)
        return x_return


class TModel8114(nn.Module):
    def __init__(self, time_slot: int = 10, depth: int = 8, num_class: int = 4, channel: int = 4):
        super(TModel8114, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.LSTM(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.LSTM(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense100 = nn.Linear(time_slot, 100)
        self.Dense10 = nn.Linear(100, time_slot)
        self.DROP = nn.Dropout(0.7)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x):
        x,_ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense100(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense10(x_den)
        x_cls = self.CLS(x_den)
        # x_return = self.SOFT(x_cls)
        # return x_return
        return x_cls