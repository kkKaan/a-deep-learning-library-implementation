# Custom MLP Implementation for MNIST Classification

This repository contains the implementation of a Multi-Layer Perceptron (MLP) classifier for the MNIST dataset using a custom tensor library. This project is designed to provide hands-on experience with the inner workings of neural networks, including forward propagation, backpropagation, and gradient descent.

## Project Structure

- `gergen.py`: Contains the custom tensor library `Gergen` class and related operations.
- `the2_test.py`: A script for testing various functionalities implemented in `gergen.py`.
- `train.py`: The training script to train the MLP model using the custom tensor library.
- `mnist_classification_torch.py`: An implementation of the MLP classifier using PyTorch for comparison.

## Objective

The main objective of this project is to extend the capabilities of the CerGen tensor library by integrating functionalities crucial for training MLPs. This includes implementing linear layers, activation functions, loss functions, and the backpropagation algorithm.

## Files Overview

### `gergen.py`

This file contains the implementation of the custom tensor class `Gergen` and the base class `Operation`. It includes various tensor operations required for forward and backward passes in the neural network.

### `the2_test.py`

This script tests the functionalities implemented in `gergen.py` to ensure correctness. It includes tests for tensor operations, linear layers, and other essential components.

### `train.py`

The main training script for the MLP model. It includes the following components:

- **Layer Class**: Represents a linear layer in the MLP.
- **MLP Class**: Represents the MLP model with one hidden layer.
- **Training Function**: Implements the training loop, including forward pass, loss computation, backward pass, and parameter updates.

### `mnist_classification_torch.py`

This file provides an implementation of the MLP model using PyTorch for comparison. It ensures that the custom implementation achieves similar performance to a well-established deep learning framework.

## Usage

You can directly use the .ipynb file, or see the outputs without running it. If you want to use .py files, you need to change main functions accordingly, assuming files are in the same directory.
