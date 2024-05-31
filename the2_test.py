import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

from gergen import *
from train import *


def test(mlp, inputs, targets):
    """
    Tests the MLP model with the input data and computes the loss.

    Parameters:
        mlp (MLP): The MLP model with an `ileri` method for forward propagation.
        inputs (list or generator): Input data to test (each sample should be formatted as a gergen object).
        targets (list or generator): Expected outputs (each target should be formatted as a gergen object).

    Returns:
        float: The computed average loss over the test dataset.
        float: The accuracy of the model on the test dataset.
    """
    total_loss = 0
    correct = 0

    for i in range(len(inputs)):
        # Same operations as in the training loop without the backward pass
        x_gergen = gergen(inputs[i]).devrik()
        y_gergen = gergen(targets[i]).devrik()

        # Normalize the input data
        x_gergen = x_gergen / 255

        # Forward pass: predict with the MLP
        predictions = mlp.ileri(x_gergen)

        # Compute the loss via cross-entropy
        loss = cross_entropy(predictions, y_gergen)
        total_loss += loss

        # Determine the predicted label by choosing the class with the highest probability
        predicted_index = predictions.devrik().veri[0].index(max(predictions.devrik().veri[0]))
        predicted_class = predicted_index
        actual_index = y_gergen.devrik().veri[0].index(max(y_gergen.devrik().veri[0]))
        actual_class = actual_index

        print("predicted_index: ", predicted_index)
        print("actual_index: ", actual_index)

        if predicted_class == actual_class:
            correct += 1

    # Calculate average loss and accuracy
    average_loss = total_loss / len(inputs)
    accuracy = correct / len(inputs)

    print(f"Test Loss: {average_loss}")
    print(f"Test Accuracy: {accuracy * 100:.3f}%")

    return average_loss, accuracy


def data_preprocessing(data_file):
    """
    Preprocess the dataset provided in CSV format.
    
    Args:
        data_file (str): Path to the CSV file containing the dataset.
        
    Returns:
        data (pd.DataFrame): Feature data
        labels (np.ndarray): One-hot encoded labels
    """
    # Load the data into a Pandas DataFrame
    df = pd.read_csv(data_file)

    # Extract the first column as labels
    labels = df.iloc[:, 0]

    # One-hot encode the labels using LabelBinarizer
    lb = LabelBinarizer()
    one_hot_labels = lb.fit_transform(labels)

    # Extract the remaining columns as feature data
    data = df.iloc[:, 1:]

    return data, one_hot_labels


def plot_loss_curve(loss_list):
    """
    Plots the loss curve given a list of loss values.
    
    Parameters:
        loss_list (list of floats): The list containing the loss values after each epoch.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-', color='blue')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(loss_list) + 1))
    plt.grid(True)
    plt.show()


def unit_tests():
    """
    This function contains some discrete tests inside comments, which can be used to test the 
    specific parts of the code. Uncomment the desired test and run the function to see the results.
    """
    # # Some basic tests for gergen class
    # g1 = rastgele_dogal((2,2))
    # g2 = gergen([1,2,3])
    # print(g1.us(2))
    # print(g2.us(2))

    # g1 = rastgele_dogal((2, 2))

    # # generate 2x2 numpy array and take backward pass
    # ones = [[1, 1], [1, 1], [1, 1]]
    # array = np.array([[1, 2], [3, 4], [5, 6]])
    # gArray = gergen([[1, 2], [3, 4], [5, 6]])
    # tArray = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, requires_grad=True)
    # a = np.gradient(array.mean())
    # print(a)

    # print(array.mean(axis = 1))
    # ort = Ortalama()
    # res = ort.ileri(gArray, eksen = 1)
    # print(res)
    # tMean = torch.mean(tArray, dim = 1)
    # print(tMean)

    # # Take geri pass
    # print(ort.geri(gergen(ones)))

    # # Loss calculation (using a simple target for demonstration)
    # target = torch.tensor([1, 1, 1], dtype = torch.float32)
    # loss = (tMean - target).pow(2).sum()

    # # Backward pass
    # loss.backward()

    # # Print gradients
    # print(tArray.grad)

    # test ileri of a operation
    # gArray = gArray * 2
    # print(gArray.operation.b)
    # print(gArray.operation.geri(gergen(ones))[0])
    # print(gArray.operation.ileri(gArray, gergen(ones) * 4))

    #########################################################################

    # print("-------Ic Carpim-------")

    # print("---ileri---")

    # # gergen ic carpim
    # print("gergen ic carpim")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # g2 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ic = IcCarpim()
    # res = ic.ileri(g1, g2)
    # print("result of gergen ic carpim: \n", res)

    # print("#############################################")

    # # torch ic carpim
    # print("torch ic carpim")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t_res = torch.mm(t1, t2)
    # print("result of torch ic carpim: \n", t_res)

    # print("---geri---")

    # # gergen ic carpim
    # print("gergen ic carpim")
    # ones3_3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # g2 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ic = IcCarpim()
    # res = ic.ileri(g1, g2)
    # print("result of gergen ic carpim: \n", res)
    # back = ic.geri(gergen(ones3_3))
    # print("gradient of the first tensor: \n", back[0])
    # print("gradient of the second tensor: \n", back[1])

    # print("#############################################")

    # # torch ic carpim
    # print("torch ic carpim")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    # t_res = torch.mm(t1, t2)
    # print("result of torch ic carpim: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the first tensor: \n", t1.grad)
    # print("gradient of the second tensor: \n", t2.grad)

    # test geri of a operation
    # print(gArray.operation.b)
    # print(gArray.operation.geri(gergen(ones))[1])

    #### test torch dis carpim
    # print("torch outer product")
    # t1 = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)
    # outer = torch.outer(t1, t2)
    # grad_tensor = torch.ones_like(outer)
    # print("outer product of torch.outer: \n", outer)
    # # print(t1.grad) # -> will return None because grad of outer is not calculated yet
    # outer.backward(grad_tensor)
    # print("gradient of the first tensor: \n", t1.grad)
    # print("gradient of the second tensor: \n", t2.grad)

    # print("#############################################")

    # #### test geri of Dış Çarpım
    # print("geri of Dış Çarpım")
    # ones3_3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([1, 2, 3])
    # g2 = gergen([4, 5, 6])
    # dis = DisCarpim()
    # res = dis.ileri(g1, g2)
    # print("outer product of geri: \n", res)
    # back = dis.geri(gergen(ones3_3))
    # print("gradient of the first tensor: \n", back[0])
    # print("gradient of the second tensor: \n", back[1])

    #### test relu
    # print("-------RELU-------")

    # print("---ileri---")

    # # gergen relu
    # print("gergen relu")
    # g1 = gergen([1, 2, -1])
    # relu = ReLU()
    # res = relu.ileri(g1)
    # print("result of gergen relu: \n", res)

    # print("#############################################")

    # # torch relu
    # print("torch relu")
    # t1 = torch.tensor([1, 2, -1], dtype=torch.float32, requires_grad=True)
    # t_relu = torch.nn.ReLU()
    # t_res = t_relu(t1)
    # print("result of torch relu: \n", t_res)

    # print("---geri---")

    # # gergen relu
    # print("gergen relu")
    # ones3 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    # g1 = gergen([[1, 2, -1], [-3, 0, 4], [-1, -0.3, 1.5]], operation=ReLU(), requires_grad=True)
    # relu = ReLU()
    # res = relu.ileri(g1)
    # print("result of gergen relu: \n", res)
    # back = relu.geri(gergen(ones3))
    # print("gradient of the tensor: \n", back)

    # print("#############################################")

    # # torch relu
    # print("torch relu")
    # t1 = torch.tensor([[1, 2, -1], [-3, 0, 4], [-1, -0.3, 1.5]], dtype=torch.float32, requires_grad=True)
    # t_relu = torch.nn.ReLU()
    # t_res = t_relu(t1)
    # print("result of torch relu: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the tensor: \n", t1.grad)

    #### test softmax
    # print("-------SOFTMAX-------")

    # print("---ileri---")

    # # gergen softmax
    # print("gergen softmax")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # softmax = Softmax()
    # res = softmax.ileri(g1, dim=1)
    # print("result of gergen softmax: \n", res)

    # print("#############################################")

    # # torch softmax
    # print("torch softmax")
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64, requires_grad=True)
    # t_softmax = torch.nn.Softmax(dim=1)
    # t_res = t_softmax(t1)
    # print("result of torch softmax: \n", t_res)

    # print("---geri---")

    # gergen softmax
    # print("gergen softmax")
    # g1 = gergen([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # ones3_3 = (gergen(g1.custom_zeros(g1.boyut())) + 1).veri
    # softmax = Softmax()
    # res = softmax.ileri(g1, dim=1)
    # print("result of gergen softmax: \n", res)
    # back = softmax.geri(2)
    # print("gradient of the tensor: \n", back)

    # print("#############################################")

    # torch softmax
    # print("torch softmax")
    # t1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float64, requires_grad=True)
    # t_softmax = torch.nn.Softmax()
    # t_res = t_softmax(t1)
    # print("result of torch softmax: \n", t_res)
    # grad_tensor = torch.ones_like(t_res)
    # t_res.backward(grad_tensor)
    # print("gradient of the tensor: \n", t1.grad)

    #### test Katman class
    # print("-------KATMAN-------")

    # print("---ileri---")

    # # gergen katman
    # print("gergen katman")
    # g1 = gergen([[1], [1], [1]])
    # katman = Katman(3, 2, "relu")
    # res = katman.ileri(g1)
    # print("result of gergen katman: \n", res)

    #### test MLP class
    # print("-------MLP-------")

    # print("---ileri---")

    # # gergen mlp
    # print("gergen mlp")
    # g1 = gergen([[1], [1], [1]])
    # mlp = MLP(3, 2, 2)
    # res = mlp.ileri(g1)
    # print("result of gergen mlp: \n", res)

    #### test cross entropy loss
    # print("-------CROSS ENTROPY LOSS-------")

    # # gergen cross entropy loss
    # print("gergen cross entropy loss")
    # g1 = gergen([[1, 4], [3, 6], [5, 8]])
    # g2 = gergen([[1, 0], [0, 1], [1, 0]])
    # loss = cross_entropy(g1, g2)
    # print("result of gergen cross entropy loss: \n", loss)

    # print("#############################################")

    # # torch cross entropy loss
    # print("torch cross entropy loss")
    # t1 = torch.tensor([[1, 4], [3, 6], [5, 8]], dtype=torch.float32, requires_grad=True)
    # t2 = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
    # t_loss = torch.nn.CrossEntropyLoss()
    # t_res = t_loss(t1, torch.argmax(t2, dim=1))
    # print("result of torch cross entropy loss: \n", t_res)

    pass


def optimize_hyperparameters(learning_rates, hidden_sizes, epochs, train_data_path, test_data_path):
    """
    This function is used to optimize the hyperparameters of the MLP model.

    Parameters:
        learning_rates (list): List of learning rates to test.
        hidden_sizes (list): List of hidden layer sizes to test.
        epochs (int): Number of training epochs.
        train_data_path (str): Path to the training data file.
        test_data_path (str): Path to the test data file.
    """
    # results is dictionary mapping tuples of the form
    # (learning_rate, hidden_layer_size) to tuples of the form
    # (training_loss, test_loss).
    results = {}
    best_loss = 10  # The lowest test loss that we have seen so far.
    best_model = None  # The MLP object that achieved the lowest test loss.
    best_loss_list = None  # The loss list for the best model
    best_lr = None  # The learning rate for the best model
    best_hl = None  # The hidden layer size for the best model

    train_data, train_labels = data_preprocessing(train_data_path)
    test_data, test_labels = data_preprocessing(test_data_path)

    data = train_data.values.tolist()
    labels = train_labels.tolist()

    for lr in learning_rates:

        for hl in hidden_sizes:

            # Create and train a new MLP instance
            mlp = MLP(input_size=input_size, hidden_size=hl, output_size=output_size)
            loss_list = egit(mlp, data, labels, epochs, lr)
            train_loss = loss_list[-1]

            # Predict values for test set and calculate test loss
            test_loss, test_accuracy = test(mlp, test_data, test_labels)

            print(
                f"learning rate={lr} and hidden layer size={hl} provided train_loss={train_loss:.3f} and test_loss={test_loss:.3f}"
            )

            # Save the results
            results[(lr, hl)] = (train_loss, test_loss)
            if test_loss < best_loss:
                best_lr = lr
                best_hl = hl
                best_loss = test_loss
                best_model = mlp
                best_loss_list = loss_list

    print(f'\nLowest test loss achieved: {best_loss} with params hl={best_hl} and lr={best_lr}')


if __name__ == "__main__":
    cekirdek(2)  # Set the seed for reproducibility
    # unit_tests() # To check specific functions

    # Load the data
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    data, labels = data_preprocessing(train_data_path)
    test_data, test_labels = data_preprocessing(test_data_path)

    # convert data to list

    data = data.values.tolist()
    labels = labels.tolist()
    test_data = test_data.values.tolist()
    test_labels = test_labels.tolist()

    # print("type data: ", type(data))
    # print("type labels: ", type(labels))

    # Initialize the MLP with input, hidden, and output layers
    input_size = 28 * 28
    hidden_size = 10
    output_size = 10
    mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Train the MLP using your preferred training loop
    epochs = 10
    learning_rate = 0.01

    # Egit
    loss_list = egit(mlp, data, labels, epochs, learning_rate)

    # Plot the loss curve
    plot_loss_curve(loss_list)

    # Test the MLP
    test_loss, accuracy = test(mlp, test_data, test_labels)

    # For further optimizations
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    hidden_sizes = [5, 10, 30]
    epochs = 10
    optimize_hyperparameters(learning_rates, hidden_sizes, epochs, train_data_path, test_data_path)
