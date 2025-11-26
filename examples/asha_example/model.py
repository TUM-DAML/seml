import torch
import torch.nn as nn


# Define your custom PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, hidden_units=[128, 64], dropout=0.0):
        super().__init__()
        layers = []
        input_size = 28 * 28
        for h in hidden_units:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_size = h
        layers.append(nn.Linear(input_size, 10))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)
