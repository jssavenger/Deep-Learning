import torch
import torch.nn as nn

# Model Class
from model import NeuralNetwork

# Data Functions
from data import read_file, split_test_and_train

# Evaluate function
from evaluate import test

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

_EPOCHS = 30

# Model
model = NeuralNetwork(7).to(device)

# Loss Function
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

def train(train_loader):
    """Train Model
            Args:
                train_loader(): Torch DataLoader
    """
    print(f"--- Train Results ---\n")

    # Train mode
    model.train()

    for epoch in range(_EPOCHS):
        total_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Predict
            pred = model(x_batch)

            # Calculate to loss
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

            # Backward and update weights
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch} | Loss: {avg_loss}")


if __name__ == "__main__":
    print(f"\n{("-" * 29)}\n--- Train is starting now ---\n    Device: {device}\n")

    # read csv file
    X, Y = read_file()

    # call train and test splitter
    train_loader, test_loader = split_test_and_train(X, Y)

    # train to model
    train(train_loader)
    print(f"\n--- Train is completed ---\n{("-"*26)}\n\n{("-"*28)}\n--- Test is starting now ---")

    # test to model
    test(test_loader, model, loss_fn)
    print(f"--- Test is completed ---\n{("-"*25)}\n")
