import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class MLP_torch(nn.Module):
    """
    Multi-layer perceptron model implemented using PyTorch

    Attributes:
        hidden_layer (nn.Linear): Hidden layer of the MLP
        output_layer (nn.Linear): Output layer of the MLP
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_torch, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the MLP
        
        Parameters:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.relu(self.hidden_layer(x))
        x = torch.softmax(self.output_layer(x), dim=1)
        return x


def data_preprocessing_torch(file_path):
    """
    Preprocess the dataset for training the MLP

    Parameters:
        file_path (str): Path to the dataset file

    Returns:
        torch.Tensor: Processed data tensor
        torch.Tensor: Processed label tensor
    """
    # Preprocess the dataset
    df = pd.read_csv(file_path)
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    # One-hot encode the labels for classification
    lb = LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels)

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(features.values, dtype=torch.float32) / 255  # Normalize pixel values
    label_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)

    return data_tensor, label_tensor


def train_torch(mlp, inputs, targets, epochs, learning_rate):
    """
    Trains the provided MLP model using the input data and targets.

    Parameters:
        mlp (MLP_torch): MLP model to train
        inputs (torch.Tensor): Input data tensor
        targets (torch.Tensor): Target data tensor
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for the optimizer

    Returns:
        list: Loss curve
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_curve = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0
        # Forward pass - with mlp.forward
        outs = mlp.forward(inputs)
        # Calculate Loss - with criterion (CrossEntropyLoss)
        loss = criterion(outs, targets) / 10  # Hidden size
        epoch_loss += loss.item()
        # Backward pass - Compute gradients for example
        loss.backward()
        optimizer.step()
        # Print epoch loss here if desired
        print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
        loss_curve.append(epoch_loss)

    return loss_curve


def test_torch(mlp, inputs, targets):
    """
    Tests the provided MLP model using the input data and targets.

    Parameters:
        mlp (MLP_torch): MLP model to test
        inputs (torch.Tensor): Input data tensor
        targets (torch.Tensor): Target data tensor

    Returns:
        float: Loss value
        float: Accuracy value
    """
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = mlp(inputs)
        _, targets_indices = torch.max(targets, 1)
        loss = criterion(outputs, targets_indices)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets_indices).sum().item()
        accuracy = correct / len(inputs)

    print(f"Test Loss: {loss.item()}")
    print(f"Test Accuracy: {accuracy * 100:.3f}%")

    return loss.item(), accuracy


if __name__ == "__main__":
    # Load the dataset
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    train_data, train_labels = data_preprocessing_torch(train_data_path)
    test_data, test_labels = data_preprocessing_torch(test_data_path)

    # Initialize the MLP with input, hidden, and output layers
    input_size = 28 * 28
    hidden_size = 10
    output_size = 10
    mlp_torch = MLP_torch(input_size, hidden_size, output_size)

    # Train the PyTorch model
    epochs = 50
    learning_rate = 0.1
    train_torch(mlp_torch, train_data, train_labels, epochs, learning_rate)

    # Test the PyTorch model
    test_loss, test_accuracy = test_torch(mlp_torch, test_data, test_labels)
