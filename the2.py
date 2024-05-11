import math

from gergen import gergen, Operation, IcCarpim, apply_elementwise, rastgele_gercek


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
        result_gergen = gergen(grad)
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
        data = x.veri if dim == 0 else x.devrik().veri  # Transpose if dim is 0

        for row in data:
            exps = [math.exp(val) for val in row]
            sum_exps = sum(exps)
            softmax_vals = [exp_val / sum_exps for exp_val in exps]
            result.append(softmax_vals)

        result = result if dim == 0 else gergen(result).devrik().veri  # Transpose back if dim is 0
        return gergen(result, operation=self, requires_grad=True)

    def geri(self, grad_input):
        """
        Compute the gradient of the Softmax function.

        Parameters:
            grad_input (gergen): Gradient of the output of the Softmax function. ?????

        Returns:
            gergen: Gradient of the Softmax function.
        """
        return grad_input  # Since we are taking this derivative with cross entropy loss inside training loop

        # softmax_output = self.ileri(self.x, self.dim).veri
        # result = []

        # # Compute the Jacobian matrix for each row in the softmax output
        # for outputs in softmax_output:
        #     jacobian_matrix = [
        #         [s_i * (int(i == j) - s_j) for j, s_j in enumerate(outputs)] for i, s_i in enumerate(outputs)
        #     ]
        #     result.append(jacobian_matrix)

        # # Transpose back if dim isn't 1 to match the input's original shape
        # if self.dim != 1:
        #     result = [list(row) for row in zip(*result)]

        # return gergen(result[0]) * grad_input


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
        stddev = math.sqrt(2. / input_size) if activation == ReLU else math.sqrt(1. / input_size)
        self.weights = rastgele_gercek((output_size, input_size)) * stddev
        self.biases = rastgele_gercek((output_size, 1))

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
        z = z + self.biases

        # Apply activation function if specified
        if self.activation == ReLU:
            relu_op = ReLU()
            z = relu_op.ileri(z)
        elif self.activation == Softmax:
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
        self.hidden_layer = Katman(input_size, hidden_size, activation=ReLU)
        self.output_layer = Katman(hidden_size, output_size, activation=Softmax)

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
        y_pred (gergen): Predicted probabilities for each class in each sample
        y_true (gergen): True labels.

    Returns:
        float : The cross-entropy loss
    """
    epsilon = 1e-12  # Small value to avoid log(0)
    n = y_pred.boyut()[0]

    loss = (y_true * (y_pred + epsilon).ln()).topla()
    loss = loss / -n  # Average loss over all samples to obtain smaller losses
    return loss.veri


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
        # print("target size: ", targets.boyut())
        # print("input size: ", inputs.boyut()) # 20.000, MNIST small dataset
        for i in range(len(inputs)):
            # Convert the input and target to `gergen` objects
            x_gergen = inputs[i]
            y_gergen = targets[i]
            # print("x_gergen type: ", type(x_gergen))
            # print("y_gergen type: ", type(y_gergen))

            x_gergen = gergen(x_gergen).devrik()
            y_gergen = gergen(y_gergen).devrik()

            # Normalize the input data to prevent overflow
            x_gergen = x_gergen / 255

            # Forward pass: predict with the MLP
            predictions = mlp.ileri(x_gergen)

            # Compute the loss via cross-entropy
            loss = cross_entropy(predictions, y_gergen)

            grad = predictions - y_gergen
            grad = grad / predictions.boyut()[0]
            # print("grad: ", grad)

            # Backward pass: calculate gradients using `turev_al`
            predictions.turev_al(grad)

            # Update parameters of hidden and output layers
            mlp.hidden_layer.weights = mlp.hidden_layer.weights - learning_rate * mlp.hidden_layer.weights.turev
            mlp.hidden_layer.biases = mlp.hidden_layer.biases - learning_rate * mlp.hidden_layer.biases.turev
            # mlp.hidden_layer.weights.turev = None
            # mlp.hidden_layer.biases.turev = None
            mlp.hidden_layer.weights.requires_grad = False
            mlp.hidden_layer.biases.requires_grad = False
            mlp.hidden_layer.weights.operation = None
            mlp.hidden_layer.biases.operation = None

            mlp.output_layer.weights = mlp.output_layer.weights - learning_rate * mlp.output_layer.weights.turev
            mlp.output_layer.biases = mlp.output_layer.biases - learning_rate * mlp.output_layer.biases.turev
            # mlp.output_layer.weights.turev = None
            # mlp.output_layer.biases.turev = None
            mlp.output_layer.weights.requires_grad = False
            mlp.output_layer.biases.requires_grad = False
            mlp.output_layer.weights.operation = None
            mlp.output_layer.biases.operation = None

            # Accumulate the loss for this batch
            total_loss += loss

        # Append the average loss for this epoch
        average_loss = total_loss / len(inputs)
        loss_curve.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {average_loss}")

    return loss_curve
