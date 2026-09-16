import torch

def test(test_loader, model, loss_fn):
    """Evaluation of the trained model
            Args:
                test_loader():Torch DataLoader
                model(): Torch model
                loss_fn():
    """
    print(f"\n--- Test Results ---")

    # Data size and num of the batches
    size        = len(test_loader.dataset)
    num_batches = len(test_loader) 

    # Test Mode
    model.eval()

    # Sum of the loss and correct responses
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:

            # Predict
            pred = model(X)

            # Sum the losses
            test_loss += loss_fn(pred, y).item()

            # Call sigmoid activation function for response from model
            prob = torch.sigmoid(pred)

            # Turn to float
            pred_class = (prob >= 0.5).float()

            # Sum with correct responses
            correct += (pred_class == y).sum().item()

    test_loss /= num_batches
    correct   /= size
    print(f"\n    Accuracy: {(100*correct):>0.1f}%\n    Avg Loss: {test_loss:>8f}\n")