from src.data_loader import load_data
from src.model import AutoEncoder
from src.train import train
from src.evaluate import evaluate
import torch

def main():
    # Hyperparameters
    learning_rate = 5e-4
    batch_size = 256
    num_epochs = 20
    dropout_rate = 0.4
    weight_decay = 1e-5

    # Load data
    train_data, valid_data, test_data, catcols, cat_index, cat_values = load_data()
    model = AutoEncoder(dropout_rate=dropout_rate)  # Adjust dropout rate in model initialization

    # Train the model with specified hyperparameters
    train(model, train_data, valid_data, catcols, cat_index, cat_values, num_epochs=num_epochs, 
          learning_rate=learning_rate, batch_size=batch_size)
    
    # Save and test
    torch.save(model.state_dict(), "saved_model/autoencoder.pth")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    test_accuracy, test_loss = evaluate(model, test_loader, catcols, cat_index, cat_values)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()
