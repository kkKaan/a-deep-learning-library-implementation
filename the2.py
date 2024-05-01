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
        relu_output = gergen([max(0, item) for item in x.veri], operation=self)
        return relu_output

    def geri(self, grad_input):
        """
        Compute the gradient of the ReLU function.
        Gradient is passed to only those inputs where the input was greater than zero.

        Parameters:
            grad_input (gergen): Gradient of the output of the ReLU function.

        Returns:
            gergen: Gradient of the ReLU function.
        """
        grad_output = gergen(
            [grad_input.veri[i] * (1 if self.x.veri[i] > 0 else 0) for i in range(len(self.x.veri))])
        return grad_output


class Softmax(Operation):

    def ileri(self, x):
        """
        TODO: Softmax activation function
        """
        pass

    def geri(self, grad_input):
        """
        TODO: Compute the gradient of the Softmax function
        """
        pass


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
        self.weights = gergen(
            [[random.gauss(0, stddev) for _ in range(output_size)] for _ in range(input_size)],
            requires_grad=True)
        self.biases = gergen([random.gauss(0, stddev) for _ in range(output_size)], requires_grad=True)

    def ileri(self, x):
        """
        Performs the forward pass of the layer using matrix multiplication followed by adding biases.
        """
        # Create an instance of IcCarpim operation
        matrix_multiplication = IcCarpim()

        # Compute the matrix multiplication of input x and weights
        z = matrix_multiplication.ileri(x, self.weights)

        # Add biases (ensure bias dimensions are correct for broadcasting)
        if len(z.veri) == len(self.biases.veri):
            z = z + self.biases
        else:
            raise ValueError(
                "Dimension mismatch: output of matrix multiplication and biases are not aligned.")

        # Apply activation function if specified
        if self.activation == 'relu':
            z = z.relu()
        elif self.activation == 'softmax':
            z = z.softmax()

        return z

    def __str__(self):
        return f"Layer with input size {self.input_size}, output size {self.output_size}, " \
               f"activation {self.activation if self.activation else 'None'}"


class MLP:

    def __init__(self, input_size, hidden_size, output_size):
        """
        TODO: Initialize the MLP with input, hidden, and output layers
        """
        self.hidden_layer = None
        self.output_layer = None

    def ileri(self, x):
        """
        TODO: Implement the forward pass
        """
        pass


def cross_entropy(y_pred, y_true):
    """
    TODO: Implement the cross-entropy loss function
    y_pred : Predicted probabilities for each class in each sample
    y_true : True labels.
    Remember, in a multi-class classification context, y_true is typically represented in a one-hot encoded format.
    """
    pass


def egit(mlp, inputs, targets, epochs, learning_rate):
    """
    TODO: Implement the training loop
    """
    for epoch in range(epochs):
        '''
        TODO: Implement training pipeline for each example
        '''

        # Forward pass - wtih mlp.ileri

        # Calculate Loss - with cross_entropy

        loss = None

        # Backward pass - Compute gradients for example

        # Update parameters

        # Reset gradients

        # Print epoch loss here if desired

        print("Epoch: {}, Loss: {}".format(epoch, loss))

    # return mlp, loss_history
