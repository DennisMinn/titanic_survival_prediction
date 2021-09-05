import numpy as np

import torch
from torch import nn

class TitanicModel(nn.Module):
    def __init__(self, in_chans = 8):
        super().__init__()

        self.fc1 = nn.Linear(in_chans, 10)
        self.batch_norm1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 20)
        self.batch_norm2 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        out = torch.relu(self.batch_norm1(self.fc1(x)))
        out = torch.relu(self.batch_norm2(self.fc2(out)))
        out = torch.sigmoid(self.fc3(out))
        return out
