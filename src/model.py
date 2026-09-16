from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.fc1  = nn.Linear(input_length, 16)
        self.fc2  = nn.Linear(16, 8)
        self.fc3  = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    # Forward
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
