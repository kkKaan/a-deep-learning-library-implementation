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
        return gergen(result_list, operation=self)

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

    def ileri(self, x, dim=None):
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
        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        Compute the gradient of the Softmax function.

        Parameters:
            grad_input (gergen): Gradient of the output of the Softmax function.

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
