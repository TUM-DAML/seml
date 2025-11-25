import torch
import torch.nn as nn
import torch.nn.functional as F
# Define your custom PyTorch model
# class SimpleNN(nn.Module):
#     def __init__(self, hidden_units=256):
#         super().__init__()
#         self.fc1 = nn.Linear(28 * 28, hidden_units)
#         self.fc2 = nn.Linear(hidden_units, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

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
