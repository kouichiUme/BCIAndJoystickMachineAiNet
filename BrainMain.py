import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np


class BrainComputerInterface(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size,hidden_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.learning_rate  = 0.01

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
        