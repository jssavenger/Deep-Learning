import torch

def test(test_loader, model, loss_fn):
    """
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    model.eval()

    test_loss, correct = 0, 0

    print(f"\n--- Test is starting now ---\n")

    with torch.no_grad():
        for X, y in test_loader:

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            pred_class = pred.argmax(dim=1)

            correct += (pred_class == y).sum().item()

    correct /= size
    test_loss /= num_batches
    print(f"Size: {size}\nNum Batches: {num_batches}\nAccuracy: {(100*correct):>0.1f}%\nAvg Loss: {test_loss:>8f}\n\n--- Test is over ---\n")

