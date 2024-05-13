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
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


def data_preprocessing_torch(file_path):
    """
    Preprocess the dataset provided in CSV format for PyTorch.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
        torch.Tensor: Feature data
        torch.Tensor: Correct labels
    """
    df = pd.read_csv(file_path)
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:]

    lb = LabelBinarizer()
    lb.fit(labels)
    labels_indices = lb.transform(labels).argmax(axis=1)

    data_tensor = torch.tensor(features.values, dtype=torch.float32) / 255
    label_tensor = torch.tensor(labels_indices, dtype=torch.int64)
    return data_tensor, label_tensor


def train_torch(mlp, inputs, targets, epochs, learning_rate, batch_size=32):
    """
    Trains the provided MLP model using the input data and targets.

    Parameters:
        mlp (MLP_torch): The MLP model to train.
        inputs (torch.Tensor): Input data to train on.
        targets (torch.Tensor): Expected outputs.
        epochs (int): Number of epochs to run.
        learning_rate (float): Step size for gradient descent.
        batch_size (int): Number of samples to process in each batch.

    Returns:
        list: The loss values after each epoch to visualize the learning progress.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
    loss_curve = []

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            outs = mlp(batch_inputs)
            loss = criterion(outs, batch_targets)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'EPOCH: {epoch}, Loss: {epoch_loss / len(dataloader)}')
        loss_curve.append(epoch_loss / len(dataloader))
    return loss_curve


def test_torch(mlp, inputs, targets):
    """
    Tests the MLP model with the input data and computes the loss.

    Parameters:
        mlp (MLP_torch): The MLP model to test.
        inputs (torch.Tensor): Input data to test.
        targets (torch.Tensor): Expected outputs.

    Returns:
        float: The computed average loss over the test dataset.
        float: The accuracy of the model on the test dataset.
    """
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = mlp(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / len(inputs)

    print(f"Test Loss: {loss.item()}")
    print(f"Test Accuracy: {accuracy * 100:.3f}%")

    return loss.item(), accuracy


if __name__ == "__main__":
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    train_data, train_labels = data_preprocessing_torch(train_data_path)
    test_data, test_labels = data_preprocessing_torch(test_data_path)
    input_size = 28 * 28
    hidden_size = 30
    output_size = 10
    mlp_torch = MLP_torch(input_size, hidden_size, output_size)
    epochs = 10
    learning_rate = 0.001
    train_torch(mlp_torch, train_data, train_labels, epochs, learning_rate)
    test_loss, test_accuracy = test_torch(mlp_torch, test_data, test_labels)
