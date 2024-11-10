import torch
from torch import nn
import matplotlib.pyplot as plt
from .utils import zero_out_random_feature, get_accuracy

def train(model, train_data, valid_data, catcols, cat_index, cat_values, num_epochs=15, learning_rate=1e-3, batch_size=512):
    torch.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    weight_decay = 1e-5  # Set weight decay here
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # Cosine scheduler

    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.float()
            datam = zero_out_random_feature(data.clone(), catcols, cat_index, cat_values)
            recon = model(datam)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(loss.item())
        valid_loss = criterion(model(torch.tensor(valid_data, dtype=torch.float32)), torch.tensor(valid_data, dtype=torch.float32)).item()
        valid_losses.append(valid_loss)
        
        # Calculate train and validation accuracy
        train_accuracy = get_accuracy(model, train_loader, catcols, cat_index, cat_values)
        valid_accuracy = get_accuracy(model, valid_loader, catcols, cat_index, cat_values)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        # Adjust learning rate with scheduler
        scheduler.step()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, Valid Acc: {valid_accuracies[-1]:.4f}")
    
    # Plotting loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(valid_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.show()
