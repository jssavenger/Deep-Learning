import torch
import torch.nn as nn

# Model
from model import NeuralNetwork

# Test
from evaluate import test

# Data Prepare
from data import prepare_dataset

# Device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

_EPOCHS = 30

# create model instance
model = NeuralNetwork(13).to(device)

# create optimizer instance
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# create loss functions instance
loss_fn = nn.CrossEntropyLoss()


def train(train_loader):
    """Trains Model
    """
    # Start training
    print(f"\n--- Training is starting now ---\n")
    for epoch in range(_EPOCHS):
        total_loss = 0

        for x_batch, y_batch in train_loader:

            optimizer.zero_grad()

            pred = model(x_batch)

            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch: {epoch} | Loss: {total_loss} | Avg Loss: {avg_loss}")

    print(f"\n--- Training is over ---\n")


if __name__ == "__main__":
    # Prepare to dataset
    train_loader, test_loader = prepare_dataset()
    train(train_loader)
    test(test_loader, model, loss_fn)
