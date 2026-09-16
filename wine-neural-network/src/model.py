import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.fc1 = nn.Linear(input_length, 8)
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
