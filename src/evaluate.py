import torch
from torch import nn
from .utils import get_accuracy

def evaluate(model, data_loader, catcols, cat_index, cat_values):
    """Evaluates the model on the provided dataset, returning accuracy and loss."""
    criterion = nn.MSELoss()
    model.eval()
    
    # Calculate average loss
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.float()
            recon = model(data)
            loss = criterion(recon, data)
            total_loss += loss.item()
            total_batches += 1
    avg_loss = total_loss / total_batches

    # Calculate accuracy
    accuracy = get_accuracy(model, data_loader, catcols, cat_index, cat_values)

    return accuracy, avg_loss
