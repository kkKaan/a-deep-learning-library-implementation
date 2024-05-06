import random
import math
from typing import Union
import matplotlib.pyplot as plt

from gergen import *


class ReLU(Operation):

    def ileri(self, x):
        """
        Perform the ReLU activation function on the input.

        Parameters:
            x (gergen): Input to the ReLU function.

        Returns:
            gergen: Output of the ReLU function.
        """
        self.x = x
        result_list = apply_elementwise(x, lambda val: max(0, val))
        return gergen(result_list, operation=self, requires_grad=True)

    def geri(self, grad_input):
        """
        Compute the gradient of the ReLU function.
        Gradient is passed to only those inputs where the input was greater than zero.

        Parameters:
            grad_input (gergen): Gradient of the output of the ReLU function.

        Returns:
            gergen: Gradient of the ReLU function.
        """
        grad = apply_elementwise(self.x, lambda val: 1 if val > 0 else 0)
        result_gergen = gergen(grad, operation=self)
        return grad_input * result_gergen


class Softmax(Operation):

    def ileri(self, x, dim=1):
        """
        Apply the Softmax activation function to the input.

        Parameters:
            x (gergen): Input to the Softmax function.

        Returns:
            gergen: Output of the Softmax function.
        """
        self.x = x
        self.dim = dim

        # Compute the softmax of the input x
        result = []
        data = x.veri if dim is 1 else x.devrik().veri  # Transpose if dim is 0

        for row in data:
            exps = [math.exp(val) for val in row]
            sum_exps = sum(exps)
            softmax_vals = [exp_val / sum_exps for exp_val in exps]
            result.append(softmax_vals)

        result = result if dim is 1 else gergen(result).devrik().veri  # Transpose back if dim is 0
        return gergen(result, operation=self, requires_grad=True)

    def geri(self, grad_input):
        """
        Compute the gradient of the Softmax function.

        Parameters:
            grad_input (gergen): Gradient of the output of the Softmax function. ?????

        Returns:
            gergen: Gradient of the Softmax function.
        """
        softmax_output = self.ileri(self.x, self.dim).veri
        result = []

        for i, (outputs, grads) in enumerate(zip(softmax_output, grad_input.veri)):
            # Calculate the gradient component for each output
            gradient_components = []
            for j, (s_j, g_j) in enumerate(zip(outputs, grads)):
                s_i = outputs[i]
                gradient_component = s_i * (int(i == j) - s_j) * g_j
                gradient_components.append(gradient_component)
            result.append(gradient_components)

        # Return as transposed if dim is not 1 to match the input shape
        if self.dim != 1:
            result = [list(x) for x in zip(*result)]

        return gergen(result, operation=self)


def add_vec_to_each_row(ger, vec):
    """
    Add a vector to each row of a matrix.

    Parameters:
        gergen (gergen): Matrix to which the vector will be added.
        vec (list): Vector to be added to each row.

    Returns:
        gergen: Resulting matrix.
    """
    result = []
    for row in ger.veri:
        result.append([x + y for x, y in zip(row, vec)])
    return gergen(result)


class Katman:

    def __init__(self, input_size, output_size, activation=None):
        """
        Initializes the layer with given input size, output size, and optional activation function.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Initialize weights and biases
        # Using He initialization if activation is 'relu', otherwise Xavier
        stddev = math.sqrt(2. / input_size) if activation == 'relu' else math.sqrt(1. / input_size)
        self.weights = rastgele_gercek((output_size, input_size)) * stddev
        self.biases = rastgele_gercek((output_size, 1))
        print("weights: \n", self.weights)
        print("biases: \n", self.biases)

    def ileri(self, x):
        """
        Performs the forward pass of the layer using matrix multiplication followed by adding biases.

        Parameters:
            x (gergen): Input to the layer.
        
        Returns:
            gergen: Output of the layer.
        """
        # Create an instance of IcCarpim operation
        matrix_multiplication = IcCarpim()

        # Compute the matrix multiplication of input x and weights
        z = matrix_multiplication.ileri(self.weights, x)

        # print("after matrix mult: ", z)

        z = z + self.biases

        # print("after adding biases: ", z)

        # Apply activation function if specified
        if self.activation == 'relu':
            relu_op = ReLU()
            z = relu_op.ileri(z)
        elif self.activation == 'softmax':
            softmax_op = Softmax()
            z = softmax_op.ileri(z, dim=1)

        return z

    def __str__(self):
        return f"Layer with input size {self.input_size}, output size {self.output_size}, " \
               f"activation {self.activation if self.activation else 'None'}"


class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the MLP with input, hidden, and output layers

        Parameters:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            output_size (int): Number of output units
        
        Returns:
            None
        """
        self.hidden_layer = Katman(input_size, hidden_size, activation='relu')
        self.output_layer = Katman(hidden_size, output_size, activation='softmax')

    def ileri(self, x):
        """
        Forward pass of the MLP

        Parameters:
            x (gergen): Input to the MLP
        
        Returns:
            gergen: Output of the MLP
        """
        hidden_output = self.hidden_layer.ileri(x)
        output = self.output_layer.ileri(hidden_output)
        return output


def cross_entropy(y_pred, y_true):
    """
    The cross-entropy loss function for multi-class classification.
    Remember, in a multi-class classification context, y_true is typically represented in a one-hot encoded format.
    
    Parameters:
        y_pred : Predicted probabilities for each class in each sample
        y_true : True labels.

    Returns:
        float : The cross-entropy loss
    """
    n_samples = len(y_pred)
    loss = 0.0
    epsilon = 1e-12  # Small value to avoid log(0)
    
    for i in range(n_samples):
        for j in range(len(y_pred[i])):
            loss += y_true[i][j] * math.log(y_pred[i][j] + epsilon)
    
    return -loss / n_samples


def egit(mlp, inputs, targets, epochs, learning_rate):
    """
    Trains the provided MLP model using the input data and targets.

    Parameters:
        mlp (MLP): The MLP model with an `ileri` method for forward propagation.
        inputs (list or generator): Input data to train on (each sample should be formatted as a gergen object).
        targets (list or generator): Expected outputs (each target should be formatted as a gergen object).
        epochs (int): Number of epochs to run.
        learning_rate (float): Step size for gradient descent.

    Returns:
        list: The loss values after each epoch to visualize the learning progress.
    """
    # To track loss values
    loss_curve = []

    for epoch in range(epochs):
        total_loss = 0

        for x, y in zip(inputs, targets):
            # Convert the input and target to `gergen` objects
            x_gergen = gergen(x)
            y_gergen = gergen(y)

            # Forward pass: predict with the MLP
            predictions = mlp.ileri(x_gergen)

            # Compute the loss via cross-entropy
            loss = cross_entropy(predictions.veri, y_gergen.veri)

            # Backward pass: calculate gradients using `turev_al`
            predictions.turev_al(1)

            # Update parameters of hidden and output layers
            for layer in [mlp.hidden_layer, mlp.output_layer]:
                for param in [layer.weights, layer.biases]:
                    if param.requires_grad:
                        grad_adjusted = [
                            value - learning_rate * grad
                            for value, grad in zip(param.veri, param.turev.veri)
                        ]
                        param.turev = gergen(grad_adjusted)

            # Accumulate the loss for this batch
            total_loss += loss

        # Append the average loss for this epoch
        average_loss = total_loss / len(inputs)
        loss_curve.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {average_loss}")

    return loss_curve

