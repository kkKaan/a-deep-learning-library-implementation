import random
import math
from typing import Union
import matplotlib.pyplot as plt

from gergen import *

class Katman:
    def __init__(self, input_size, output_size, activation=None):
        """
        TODO: Initialize weights and biases
        """
        self.weights = None
        self.biases = None
        # Set activation function
        self.activation = activation

    def ileri(self, x):
        """
        TODO: Implement the forward pass
        """
        pass

class ReLU(Operation):
    def ileri(self, x):
        """
        TODO: ReLU activation function
        """
        pass

    def geri(self, grad_input):
        """
        TODO: Compute the gradient of the ReLU function
        """
        pass

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

