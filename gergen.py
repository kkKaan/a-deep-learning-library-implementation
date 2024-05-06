import random
import math
from typing import Union
import matplotlib.pyplot as plt


def cekirdek(sayi: int):
    # Sets the seed for random number generation
    random.seed(sayi)


def rastgele_dogal(boyut, aralik=None, dagilim='uniform'):
    """
    Generates data of specified dimensions with random integer values and returns a gergen object.

    Parameters:
        boyut (tuple): Shape of the desired data.
        aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to None, which implies a default range.
        dagilim (string, optional): Distribution of random values ('uniform' or other types). Defaults to 'uniform'.

    Returns:
        gergen: A new gergen object with random integer values.
    """

    # Set a default range if aralik is not provided
    if aralik is None:
        aralik = (0, 10)

    def generate_random_data(shape):
        if len(shape) == 1:
            return [random_value(aralik, dagilim) for _ in range(shape[0])]
        else:
            return [generate_random_data(shape[1:]) for _ in range(shape[0])]

    def random_value(aralik, dagilim):
        if dagilim == 'uniform':
            return random.randint(*aralik)
        else:
            raise ValueError(f"Unsupported distribution: {dagilim}")

    data = generate_random_data(boyut)
    return gergen(data)


def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    """
    Generates a gergen of specified dimensions with random floating-point values.

    Parameters:
        boyut (tuple): Shape of the desired gergen.
        aralik (tuple, optional): (min, max) specifying the range of random values. Defaults to (0.0, 1.0) for uniform distribution.
        dagilim (string, optional): Distribution of random value (e.g., 'uniform', 'gaussian'). Defaults to 'uniform'.

    Returns:
        gergen: A new gergen object with random floating-point values.
    """

    def generate_random_data(shape):
        if len(shape) == 1:
            return [random_value(aralik, dagilim) for _ in range(shape[0])]
        else:
            return [generate_random_data(shape[1:]) for _ in range(shape[0])]

    def random_value(aralik, dagilim):
        if dagilim == 'uniform':
            return random.uniform(*aralik)
        elif dagilim == 'gaussian':
            mean, std_dev = aralik
            return random.gauss(mean, std_dev)
        else:
            raise ValueError(f"Unsupported distribution: {dagilim}")

    data = generate_random_data(boyut)
    return gergen(data)


class Operation:

    def __call__(self, *operands, **kwargs):
        """
        Calls the operation with the provided operands and keyword arguments.

        Parameters:
            *operands: Variable length operand list.
            **kwargs: Variable length keyword argument list.

        Returns:
            gergen: The result of the operation.
        """
        self.operands = operands
        self.kwargs = kwargs  # Store keyword arguments separately
        self.outputs = None
        return self.ileri(*operands, **kwargs)

    def ileri(self, *operands, **kwargs):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.
            **kwargs: Variable length keyword argument list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError

    def geri(self, grad_input):
        """
        Defines the backward pass of the operation.
        Must be implemented by subclasses to compute the gradients.

        Parameters:
            grad_input: The gradient of the loss w.r.t. the output of this operation.
        """
        raise NotImplementedError


class Add(Operation):

    def ileri(self, a, b):
        """
        Adds two gergen objects or a gergen object and a scalar.
        
        Parameters:
            a (gergen or list): The first operand.
            b (gergen or list): The second operand.
        
        Returns:
            gergen: The result of the addition.
        """
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a = a
            self.b = b
            self.operands = [a, b]
            result = gergen(self.add_gergen(a.duzlestir().listeye(), b.duzlestir().listeye()), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (list)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.add_list(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (list)):
            self.a = b
            self.b = a
            self.operands = [b]
            result = gergen(self.add_list(b.listeye(), a), operation=self)
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.add_scalar(a.listeye(), b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            self.a = b
            self.b = a
            self.operands = [b]
            result = gergen(self.add_scalar(b.listeye(), a), operation=self)
        else:
            raise ValueError("Add operation requires at least one gergen operand.")

        result.requires_grad = True
        return result

    def add_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.add_scalar(elem, scalar) for elem in a]
        else:
            return a + scalar

    def add_gergen(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list):
            # Check if 'b' is a list
            if isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError("Dimensions of gergen objects do not match for addition.")
                return [a[i] + b[i] for i in range(len(a))]
            # If 'a' is a list and 'b' is a scalar
            elif not isinstance(b, list):
                return [item + b for item in a]

        # If 'a' is a scalar and 'b' is a list
        elif not isinstance(a, list) and isinstance(b, list):
            return [a + item for item in b]
        # Direct addition for scalars, or fallback error for unsupported types
        elif not isinstance(a, list) and not isinstance(b, list):
            return a + b

    def add_list(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list) and isinstance(b, list):
            return [self.add_list(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # If 'a' is list and b is scalar
        elif isinstance(a, list) and not isinstance(b, list):
            return [self.add_list(elem_a, b) for elem_a in a]
        elif not isinstance(a, list) and isinstance(b, list):
            return [self.add_list(a, elem_b) for elem_b in b]
        elif not isinstance(a, list) and not isinstance(b, list):
            return a + b

    def geri(self, grad_input):
        # The gradient with respect to both inputs of addition is 1 if both gergen else the second is 0
        grad_a = grad_input
        grad_b = grad_input if isinstance(self.b, gergen) else 0
        result = (grad_a, grad_b)
        return result[:len(self.operands)]


class Sub(Operation):

    def ileri(self, a, b):
        """
        Subtracts two gergen objects or a gergen object and a scalar.
        
        Parameters:
            a (gergen or list): The first operand.
            b (gergen or list): The second operand.
        
        Returns:
            gergen: The result of the subtraction.
        """
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            result = gergen(self.subtract_gergen(a.duzlestir().veri, b.duzlestir().veri), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (list)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.subtract_list(a.veri, b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (list)):
            self.a = b
            self.b = a
            self.operands = [b]
            result = gergen(self.subtract_list(a, b.veri), operation=self)
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.subtract_scalar(a.veri, b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            self.a = b
            self.b = a
            self.operands = [b]
            result = gergen(self.subtract_scalar(b.veri, a), operation=self)
        else:
            raise ValueError("Sub operation requires at least one gergen operand.")
        return result

    def subtract_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.subtract_scalar(elem, scalar) for elem in a]
        else:
            return a - scalar

    def subtract_list(self, a, b):
        # Check if 'b' is a list
        if isinstance(a, list) and isinstance(b, list):
            return [self.subtract_list(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # If 'a' is list and b is scalar
        elif isinstance(a, list) and not isinstance(b, list):
            return [self.subtract_list(elem_a, b) for elem_a in a]
        elif not isinstance(a, list) and isinstance(b, list):
            return [self.subtract_list(a, elem_b) for elem_b in b]
        elif not isinstance(a, list) and not isinstance(b, list):
            return a - b

    def subtract_gergen(self, a, b):
        # Check if 'a' is a list
        if isinstance(a, list):
            # Check if 'b' is a list
            if isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError("Dimensions of gergen objects do not match for subtraction.")
                return [a[i] - b[i] for i in range(len(a))]
            # If 'a' is a list and 'b' is a scalar
            elif not isinstance(b, list):
                return [item - b for item in a]
        # If 'a' is a scalar and 'b' is a list
        elif not isinstance(a, list) and isinstance(b, list):
            return [a - item for item in b]
        # Direct subtraction for scalars, or fallback error for unsupported types
        elif not isinstance(a, list) and not isinstance(b, list):
            return a - b

    def geri(self, grad_input):
        # The gradient with respect to the first input is 1, and with respect to the second input is -1
        grad_a = grad_input
        grad_b = -grad_input if isinstance(self.b, gergen) else 0
        result = (grad_a, grad_b)
        return result[:len(self.operands)]


class TrueDiv(Operation):

    def ileri(self, a, b):
        """
        Divides two gergen objects or a gergen object and a scalar.
        
        Parameters:
            a (gergen or list): The first operand.
            b (gergen or list): The second operand.
        
        Returns:
            gergen: The result of the division.
        """
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            result = gergen(self.divide_elements(a.duzlestir().veri, b.duzlestir().veri), operation=self)
            result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.divide_scalar(a.veri, b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            # Division of a scalar by a gergen object is not typically defined,
            # but you can implement it based on your requirements.
            raise NotImplementedError("Division of a scalar by a gergen object is not implemented.")
        else:
            raise ValueError("TrueDiv operation requires at least one gergen operand.")

        return result

    def divide_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.divide_scalar(elem, scalar) for elem in a]
        else:
            if scalar == 0:
                raise ZeroDivisionError("Division by zero.")
            return a / scalar

    def divide_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct division
        if not isinstance(a, list) and not isinstance(b, list):
            if b == 0:
                raise ZeroDivisionError("Division by zero.")
            return a / b
        # Both a and b are lists, perform element-wise division
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("Dimensions of gergen objects do not match for division.")
            return [self.divide_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, divide each element of the list by the scalar
        elif isinstance(a, list):
            return [self.divide_elements(elem, b) for elem in a]
        else:
            raise NotImplementedError("Division of scalar by a list is not typically defined.")

    def geri(self, grad_input):
        """
        Computes the backward pass of the division operation.

        Parameters:
            grad_input: The gradient of the loss w.r.t the output of this operation.

        Returns:
            Tuple: Gradients w.r.t each operand.
        """
        a, b = self.a, self.b
        if isinstance(a, gergen) and isinstance(b, gergen):
            # Gradient w.r.t a is grad_input / b
            grad_a = grad_input / b  # Assuming grad_input is not a scalar
            # Gradient w.r.t b is -grad_input * a / (b ** 2)
            grad_b = (-grad_input * a) / (b.us(2))
            return grad_a, grad_b
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            # When b is a scalar, the gradient w.r.t a is grad_input / b
            grad_a = grad_input / b  # Assuming grad_input is not a scalar
            # There is no gradient w.r.t to a scalar in the context of training neural networks
            return grad_a
        else:
            raise NotImplementedError(
                "Backpropagation for the division of a scalar by a gergen object is not supported.")


class Mul(Operation):

    def ileri(self, a, b):
        """
        Multiplies two gergen objects or a gergen object and a scalar.
        
        Parameters:
            a (gergen or list): The first operand.
            b (gergen or list): The second operand.
        
        Returns:
            gergen: The result of the multiplication.
        """
        if isinstance(a, gergen) and isinstance(b, gergen):
            self.a, self.b = a, b
            self.operands = [a, b]
            # a is a scalar gergen
            if a.uzunluk() == 1:
                result = gergen(self.multiply_scalar(b.veri, a.veri), operation=self)
            # b is a scalar gergen
            elif b.uzunluk() == 1:
                result = gergen(self.multiply_scalar(a.veri, b.veri), operation=self)
            else:
                result = gergen(self.multiply_elements(a.duzlestir().veri,
                                                       b.duzlestir().veri),
                                operation=self)
                result.boyutlandir(a.boyut())
        elif isinstance(a, gergen) and isinstance(b, (int, float)):
            self.a = a
            self.b = b
            self.operands = [a]
            result = gergen(self.multiply_scalar(a.veri, b), operation=self)
        elif isinstance(b, gergen) and isinstance(a, (int, float)):
            self.a = b
            self.b = a
            self.operands = [b]
            result = gergen(self.multiply_scalar(b.veri, a), operation=self)
        else:
            raise ValueError("Mul operation requires at least one gergen operand.")

        return result

    def multiply_scalar(self, a, scalar):
        if isinstance(a, list):
            return [self.multiply_scalar(elem, scalar) for elem in a]
        else:
            return a * scalar

    def multiply_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct multiplication
        if not isinstance(a, list) and not isinstance(b, list):
            return a * b
        # Both a and b are lists, perform element-wise multiplication
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("Dimensions of gergen objects do not match for multiplication.")
            return [self.multiply_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, multiply each element of the list by the scalar
        elif isinstance(a, list):
            return [self.multiply_elements(elem, b) for elem in a]
        else:
            return [self.multiply_elements(a, elem) for elem in b]

    def geri(self, grad_input):
        """
        Computes the backward pass of the multiplication operation.

        Parameters:
            grad_input: The gradient of the loss w.r.t the output of this operation.

        Returns:
            Tuple: Gradients w.r.t each operand.
        """
        a, b = self.a, self.b
        grad_a = grad_input * b
        grad_b = grad_input * a if isinstance(b, gergen) else 0
        result = (grad_a, grad_b)
        return result[:len(self.operands)]


class Us(Operation):

    def ileri(self, a, n):
        """
        Power operation.
        
        Parameters:
            a (gergen): The base.
            n (int): The exponent.
        
        Returns:
            gergen: The result of the power operation.
        """
        self.a = a
        self.n = n
        self.operands = [a]
        result = gergen(self.power_elements(a.veri, n), operation=self)
        return result

    def power_elements(self, a, n):
        if isinstance(a, list):
            return [self.power_elements(elem, n) for elem in a]
        else:
            return a**n

    def multiply_elements(self, a, b):
        # Both a and b are non-lists (scalars), perform direct multiplication
        if not isinstance(a, list) and not isinstance(b, list):
            return a * b
        # Both a and b are lists, perform element-wise multiplication
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("Dimensions of gergen objects do not match for multiplication.")
            return [self.multiply_elements(elem_a, elem_b) for elem_a, elem_b in zip(a, b)]
        # One of a or b is a list and the other is a scalar, multiply each element of the list by the scalar
        elif isinstance(a, list):
            return [self.multiply_elements(elem, b) for elem in a]
        else:
            return [self.multiply_elements(a, elem) for elem in b]

    def geri(self, grad_input):
        """
        Computes the backward pass of the power operation.

        Parameters:
            grad_input: The gradient of the loss w.r.t the output of this operation.

        Returns:
            Tuple: Gradients w.r.t each operand.
        """
        a, n = self.a, self.n
        grad_a = grad_input * n * (a.us(n - 1))
        return grad_a


class Log10(Operation):

    def ileri(self, a):
        """
        Log10 operation

        Parameters:
            a (gergen): The input gergen object.

        Returns:
            gergen: The result of the log10 operation.
        """
        self.a = a
        self.operands = [a]
        # Recursively check for non-positive values in the nested list structure
        if self.contains_non_positive(self.a.veri):
            raise ValueError("Logarithm undefined for non-positive values.")
        result = gergen(self.log_elements(a.veri), operation=self)
        return result

    def log_elements(self, a):
        # Recursively apply the base 10 logarithm to each element
        if isinstance(a, list):
            return [self.log_elements(elem) for elem in a]
        else:
            return math.log10(a)

    def contains_non_positive(self, a):
        # Recursively check for non-positive values and flatten the results
        def check_and_flatten(a):
            flag = False
            if isinstance(a, list):
                # Use a generator expression to recursively check each element and flatten the result
                for ele in a:
                    flag = check_and_flatten(ele)
            else:
                if a <= 0:
                    return True
            return flag

        # Use 'any' on a flattened generator of boolean values
        return check_and_flatten(a)

    def multiply_elements(self, a, scalar):
        # Recursively multiply each element by the scalar
        if isinstance(a, list):
            return [self.multiply_elements(elem, scalar) for elem in a]
        else:
            return a * scalar

    def divide_elements(self, grad_output, b):
        # Recursively divide grad_output by b, assuming they have the same structure
        if isinstance(b, list):
            return [self.divide_elements(elem_grad, elem_b) for elem_grad, elem_b in zip(grad_output, b)]
        else:
            return grad_output / b

    def geri(self, grad_input):
        """
        Computes the backward pass of the Log10 operation.

        Parameters:
            grad_input: The gradient of the loss w.r.t the output of this operation.

        Returns:
            gergen: The gradient of the loss w.r.t the input of this operation.
        """
        a = self.a
        # Calculate the gradient w.r.t a using the properties of logarithmic differentiation
        # Gradient w.r.t a is: grad_input * (1 / (a * ln(10)))
        ln10 = math.log(10)
        grad_a = grad_input * (1 / (a * ln10))
        return grad_a


class Ln(Operation):

    def ileri(self, a):
        """
        Implements the forward pass for the Ln operation.
        
        Parameters:
            a (gergen): The input gergen object.

        Returns:
            gergen: The result of the natural logarithm operation.
        """
        if not isinstance(a, gergen):
            raise ValueError("Ln operation requires a gergen operand.")
        self.a = a
        self.operands = [a]
        if self.contains_non_positive(self.a.listeye()):
            raise ValueError("Logarithm undefined for non-positive values.")

        result = gergen(self.log_elements(a.listeye()), operation=self)
        return result

    def log_elements(self, a):
        # Recursively apply the base 10 logarithm to each element
        if isinstance(a, list):
            return [self.log_elements(elem) for elem in a]
        else:
            return math.log(a) if a > 0 else math.log(a + 10**-4)

    def contains_non_positive(self, a):
        # Recursively check for non-positive values
        def check_and_flatten(a):
            if isinstance(a, list):
                return any(check_and_flatten(elem) for elem in a)
            else:
                if a <= 0:
                    a = 1
                    return True
                else:
                    return False

        # Use 'any' on a flattened generator of boolean values
        return check_and_flatten(a)

    def geri(self, grad_input):
        """
        Computes the backward pass of the Ln operation.

        Parameters:
            grad_input: The gradient of the loss w.r.t the output of this operation.

        Returns:
            gergen: The gradient of the loss w.r.t the input of this operation.
        """
        a = self.a
        # Calculate the gradient w.r.t a using the properties of logarithmic differentiation
        # Gradient w.r.t a is: grad_input * (1 / a)
        grad_a = grad_input / a
        return grad_a


def apply_elementwise(g, func):
    """
    Applies a given function element-wise to the data in a gergen object.
    This version is capable of handling nested lists of any depth.

    Parameters:
        g (gergen): The input gergen object.
        func (function): The function to apply to the data.
    
    Returns:
        list: A new veri for a gergen object with the function applied element-wise.
    """

    def recursive_apply(data):
        if isinstance(data, list):
            # Recursively apply func to each element if data is a list
            return [recursive_apply(sublist) for sublist in data]
        else:
            # Apply func directly if data is a scalar (non-list)
            return func(data)

    # Use the recursive function to apply the operation to the gergen object's data
    return recursive_apply(g.listeye())


class Sin(Operation):

    def ileri(self, a):
        """
        Implements the forward pass for the Sin operation.
        
        Parameters:
            a (gergen): The input gergen object.

        Returns:
            gergen: The result of the sine operation.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.sin), operation=self)
        return result

    def geri(self, grad_output):
        """
        Computes the backward pass of the Sin operation.
        """
        a = self.operands[0]
        # The gradient with respect to a is grad_output multiplied by the derivative of sin(a), which is cos(a).
        cos_a = apply_elementwise(a, math.cos)
        grad_a = grad_output * cos_a
        return grad_a


class Cos(Operation):

    def ileri(self, a):
        """
        Implements the forward pass for the Cos operation.
        
        Parameters:
            a (gergen): The input gergen object.

        Returns:
            gergen: The result of the cosine operation.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.cos), operation=self)
        return result

    def geri(self, grad_output):
        """
        Computes the backward pass of the Cos operation.
        """
        a = self.operands[0]
        # The gradient with respect to a is grad_output multiplied by the derivative of cos(a), which is -sin(a).
        sin_a = apply_elementwise(a, math.sin)
        grad_a = grad_output * -sin_a
        return grad_a


class Tan(Operation):

    def ileri(self, a):
        """
        Implements the forward pass for the Tan operation.
        
        Parameters:
            a (gergen): The input gergen object.

        Returns:
            gergen: The result of the tangent operation.
        """
        self.operands = [a]
        result = gergen(apply_elementwise(a, math.tan), operation=self)
        return result

    def geri(self, grad_output):
        """
        Computes the backward pass of the Tan operation.
        """
        a = self.operands[0]
        # The gradient with respect to a is grad_output multiplied by the derivative of tan(a), which is sec^2(a).
        sec2_a = apply_elementwise(a, lambda x: 1 / math.cos(x)**2)
        grad_a = grad_output * sec2_a
        return grad_a


class Topla(Operation):

    def ileri(self, a, eksen=None):
        """
        Forward pass for the Topla operation.
        
        Parameters:
            a (gergen): The input gergen object.
            eksen (int, optional): The axis along which to sum the elements. Defaults to None.
        
        Returns:
            gergen: The result of the sum operation.
        """

        def sum_elements(lst):
            if isinstance(lst[0], list):
                return [sum_elements(sublst) for sublst in zip(*lst)]
            else:
                return sum(lst)

        def sum_along_axis(data, axis):
            if axis == 0:
                return sum_elements(data)
            else:
                return [sum_along_axis(subdata, axis - 1) for subdata in data]

        self.operands = [a]
        self.eksen = eksen
        if eksen is None:
            result = sum(a.duzlestir().listeye())
        elif isinstance(eksen, int):
            if eksen < 0 or eksen >= len(a.boyut()):
                raise ValueError("Axis out of bounds for gergen's dimensionality")
            result = sum_along_axis(a.listeye(), eksen)
        else:
            raise TypeError("Axis must be an integer or None")

        return gergen(result, operation=self)

    def geri(self, grad_output):
        """
        (Optional, not tested) Computes the backward pass of the Topla operation.
        """
        a = self.operands[0]
        if self.eksen is None:
            # If the sum was across the entire tensor, every element contributes equally.
            grad_input_shape = [1] * len(a.boyut())  # Create a shape of ones
            expanded_grad = grad_output.boyutlandir(grad_input_shape)  # Expand grad to match input shape
            grad_input = expanded_grad * gergen.custom_zeros(a.boyut())  # Multiply by a tensor of ones
        else:
            # If sum was along a particular axis, replicate the gradient along that axis
            repeats = [1] * len(a.boyut())
            repeats[self.eksen] = a.boyut()[self.eksen]
            grad_input = grad_output.repeat(repeats)

        return grad_input


class Ortalama(Operation):

    def ileri(self, a, eksen=None):
        """
        Forward pass for the Ortalama operation.

        Parameters:
            a (gergen): The input gergen object.
            eksen (int, optional): The axis along which to compute the average. Defaults to None.

        Returns:
            gergen: The result of the average operation.
        """

        def average_elements(total_sum, total_elements):
            # Compute the average
            if isinstance(total_sum, list):
                # If total_sum is a list (multi-dimensional case), calculate the average for each sublist
                return [average_elements(ts, total_elements) for ts in total_sum]
            else:
                # For a single number, just divide
                return total_sum / total_elements

        self.operands = [a]
        self.eksen = eksen
        sum_op = Topla()  # Instantiate the Sum operation

        total_sum = sum_op.ileri(a, eksen=eksen).listeye()

        if eksen is None:
            total_elements = a.uzunluk()
        else:
            if eksen < 0 or eksen >= len(a.boyut()):
                raise ValueError("Axis out of bounds for gergen's dimensionality")
            total_elements = a.boyut()[eksen]

        # Compute the average
        average_result = average_elements(total_sum, total_elements)
        return gergen(average_result, operation=self)

    def geri(self, grad_output):
        """
        Computes the backward pass of the Ortalama operation. ?????
        """
        a = self.operands[0]

        # Calculate the number of elements involved in the averaging operation.
        if self.eksen is None:
            # Average over all elements
            total_elements = a.uzunluk()
        else:
            # Average over a specific axis
            total_elements = a.boyut()[len(a.boyut()) - self.eksen - 1]

        # The gradient of the average is the incoming gradient divided by the number of elements.
        # This division gives us the gradient per element that was averaged.
        # We then need to create a tensor with this gradient for each element that was part of the average.

        # First, create a gergen object with the same shape as 'a', filled with 1/total_elements
        grad_per_element = gergen(1 / total_elements)

        # Then, we'll broadcast this gradient to the shape of 'a' to distribute it across all elements.
        # Multiplication with the broadcasted gradient tensor gives us the gradient with respect to the input 'a'.
        grad_a = grad_output * grad_per_element

        return grad_a


class IcCarpim(Operation):

    def ileri(self, a, b):
        """
        Computes the dot product of two gergen objects.

        Parameters:
            a (gergen): The first operand.
            b (gergen): The second operand.

        Returns:
            gergen: The result of the dot product operation.
        """
        self.a = a
        self.b = b
        self.operands = [a, b]
        if not isinstance(a, type(b)):
            raise ValueError("Both operands must be gergen objects.")

        def is_vector(v):
            return len(v.boyut()) == 1

        def is_matrix(m):
            return len(m.boyut()) == 2

        def vector_dot_product(v1, v2):
            if len(v1) != len(v2):
                raise ValueError("Vectors must have the same length for dot product.")
            return sum(x * y for x, y in zip(v1, v2))

        def matrix_multiply(m1, m2):
            if len(m1[0]) != len(m2):
                raise ValueError(
                    "The number of columns in the first matrix must match the number of rows in the second matrix."
                )
            return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*m2)] for row_a in m1]

        if len(a.boyut()) > 2 or len(b.boyut()) > 2:
            raise ValueError("Operands must both be either 1-D vectors or 2-D matrices.")
        elif is_vector(a) and is_vector(b):
            # Perform vector dot product
            result = vector_dot_product(a.listeye(), b.listeye())
        elif is_matrix(a) and is_matrix(b):
            # Perform matrix multiplication
            result = matrix_multiply(a.listeye(), b.listeye())
        else:
            raise ValueError("Operands must both be either 1-D vectors or 2-D matrices.")

        # Return result
        return gergen(result, operation=self, requires_grad=True)

    def geri(self, grad_output):
        """
        Computes the backward pass of the IcCarpim operation.

        Parameters:
            grad_output: The gradient of the loss w.r.t the output of this operation.
        
        Returns:
            Tuple: Gradients w.r.t each operand.
        """
        a = self.operands[0]
        b = self.operands[1]

        # Ensure the gradient output is a gergen object
        if not isinstance(grad_output, gergen):
            grad_output = gergen(grad_output)

        # Compute gradients with respect to inputs
        if len(a.boyut()) == 2 and len(b.boyut()) == 2:
            # Matrix multiplication case
            grad_a = grad_output.ic_carpim(b.devrik())  # grad_output (m x p) * b^T (p x n) => (m x n)
            grad_b = a.devrik().ic_carpim(grad_output)  # a^T (n x m) * grad_output (m x p) => (n x p)
        else:
            raise ValueError(
                "Operands and gradient outputs must both be 2-D matrices for matrix multiplication.")

        return grad_a, grad_b


class DisCarpim(Operation):

    def ileri(self, a, b):
        """
        Computes the outer product of two gergen objects.

        Parameters:
            a (gergen): The first operand.
            b (gergen): The second operand.

        Returns:
            gergen: The result of the outer product operation.
        """
        if not isinstance(a, gergen) or not isinstance(b, gergen):
            raise ValueError("Both operands must be gergen objects.")

        # Ensure the veri attributes are lists representing vectors
        if not all(isinstance(x, (int, float))
                   for x in a.listeye()) or not all(isinstance(y, (int, float)) for y in b.listeye()):
            raise ValueError("Both gergen objects must contain 1-D numerical data.")

        self.operands = [a, b]
        # Compute the outer product
        result = [[x * y for y in b.listeye()] for x in a.listeye()]

        # Return a new gergen object with the outer product as its veri
        return gergen(result, operation=self)

    def geri(self, grad_input):
        """
        Computes the backward pass of the DisCarpim operation.
        """
        a = self.operands[0]
        b = self.operands[1]

        # Gradients for a are the dot product of grad_input with each column of b
        # grad_input.shape = (len(a.veri), len(b.veri))
        # We sum over the columns of grad_input for each element in b
        grad_a = [
            sum(grad_input.veri[i][j] * b.veri[j] for j in range(len(b.veri))) for i in range(len(a.veri))
        ]

        # Gradients for b are the dot product of grad_input with each row of a
        # We sum over the rows of grad_input for each element in a
        grad_b = [
            sum(grad_input.veri[i][j] * a.veri[i] for i in range(len(a.veri))) for j in range(len(b.veri))
        ]

        # Return gradients wrapped in gergen objects
        return gergen(grad_a, operation=self), gergen(grad_b, operation=self)


class gergen:

    #TODO: You should modify this class implementation

    __veri = None  # A nested list of numbers representing the data
    D = None  # Transpose of data
    turev = None  # Stores the derivate
    operation = None  # Stores the operation that produced the gergen
    __boyut = None  # Dimensions of the gergen (Shape)
    requires_grad = True  # Flag to determine if the gradient should be computed

    def __init__(self, veri=None, operation=None, requires_grad=None):
        # The constructor for the 'gergen' class.
        if veri is None:
            self.__veri = []
            self.__boyut = (0,)
            self.D = None
            self.turev = None
            self.operation = None
            self.requires_grad = None
        else:
            self.__veri = veri
            self.__boyut = self.get_shape(veri, ())  # Assuming rectangular data
            self.D = None
            self.turev = None
            self.operation = operation
            self.requires_grad = requires_grad

    def __iter__(self):
        # The __iter__ method returns the iterator object itself.
        # You can reset the iterator here if you want to allow multiple passes over the data.
        pass

    def __next__(self):
        # The __next__ method should return the next value from the iterator.
        pass

    def __getitem__(self, key):
        """
        Allows for indexing or slicing the gergen object's data.

        Parameters:
            key (int, slice, tuple): An integer or slice for one-dimensional indexing,
                                     or a tuple for multi-dimensional indexing/slicing.

        Returns:
            The element or a new gergen object corresponding to the provided key.
        """

        # Helper function to handle recursive indexing/slicing
        def index_or_slice(data, key):
            if isinstance(key, int) or isinstance(key, slice):
                return data[key]
            elif isinstance(key, tuple):
                result = data
                for k in key:
                    result = index_or_slice(result, k)
                return result
            else:
                raise TypeError(f"Invalid index type: {type(key)}")

        # Perform the indexing or slicing operation
        result = index_or_slice(self.__veri, key)
        # If the result is a list, return it wrapped in a new gergen object
        return gergen(result)

    def __str__(self):
        # Generates a string representation
        if self.uzunluk() == 0:
            return "Empty Gergen"
        else:
            shape_str = ""
            for b in self.boyut():
                shape_str += str(b) + "x"
            if shape_str == "":
                shape_str += "0x"
            return shape_str[:-1] + " boyutlu gergen:" + "\n" + self.str_helper(
                self.listeye(), len(self.boyut()))

    def str_helper(self, data, shape, depth=0):
        if not shape:
            return str(data)
        elif not isinstance(data[0], list):
            return str(data)
        else:
            inner_results = []
            for subdata in data:
                inner_results.append(self.str_helper(subdata, shape, depth + 1))

            result = "[" + ("\n" * (shape - depth - 1)).join(r for r in inner_results) + "]"
            return result

    @property
    def veri(self):
        return self.__veri

    @staticmethod
    def get_shape(lst, shape=()):
        if not isinstance(lst, list):
            # base case
            return shape
        # peek ahead and assure all lists in the next depth
        # have the same length
        if isinstance(lst[0], list):
            l = len(lst[0])
            if not all(len(item) == l for item in lst):
                msg = 'not all lists have the same length'
                raise ValueError(msg)

        shape += (len(lst),)
        # recurse
        shape = gergen.get_shape(lst[0], shape)

        return shape

    @staticmethod
    def custom_zeros(shape):
        """
        Creates a multi-dimensional array of zeros with the specified shape.

        Parameters:
            shape (tuple): A tuple representing the dimensions of the array.

        Returns:
            A nested list (multi-dimensional array) filled with zeros.
        """
        if not shape:  # If shape is empty or reaches the end of recursion
            return 0
        # Recursively build nested lists
        return [gergen.custom_zeros(shape[1:]) for _ in range(shape[0])]

    # HELPER
    @staticmethod
    def prod(iterable):
        """
        Utility function to calculate the product of elements in an iterable.
        """
        result = 1
        for i in iterable:
            result *= i
        return result

    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        mul_operation = Mul()
        result_gergen = mul_operation(self, other)
        return result_gergen

    def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
        mul_operation = Mul()
        result_gergen = mul_operation(self, other)
        return result_gergen

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        div_operation = TrueDiv()
        result_gergen = div_operation(self, other)
        return result_gergen

    def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        div_operation = TrueDiv()
        result_gergen = div_operation(self, other)
        return result_gergen

    def __add__(self, other):
        add_operation = Add()
        result_gergen = add_operation(self, other)
        return result_gergen

    def __radd__(self, other):
        add_operation = Add()
        result_gergen = add_operation(self, other)
        return result_gergen

    def __sub__(self, other):
        sub_operation = Sub()
        result_gergen = sub_operation(self, other)
        return result_gergen

    def __rsub__(self, other):
        sub_operation = Sub()
        result_gergen = sub_operation(other, self)
        return result_gergen

    def uzunluk(self):
        # Returns the total number of elements in the gergen
        total = 1
        for ele in self.__boyut:
            total *= ele
        return total

    def boyut(self):
        # Returns the shape of the gergen
        return self.__boyut

    def devrik(self):
        """
        Returns the transpose of the gergen object.

        Returns:
            gergen: The transpose of the gergen object.
        """
        if self.uzunluk() == 1:
            return gergen(self.__veri)
        # Check if the gergen object represents a 1D list (vector)
        if isinstance(self.__veri, list) and all(not isinstance(item, list) for item in self.__veri):
            # Convert each element into a list (column vector)
            return gergen([[item] for item in self.__veri])
        else:
            # Handle higher-dimensional cases (e.g., 2D matrices, 3D tensors, etc.)
            new_boyut = tuple(reversed(self.__boyut))
            order = list(reversed(range(len(self.__boyut))))
            arr = self.custom_zeros(
                new_boyut)  # Assuming custom_zeros initializes an array with the given shape
            paths = [0] * len(self.__boyut)
            while paths[0] < self.__boyut[0]:
                ref = self.listeye()
                place = arr
                for i in range(len(paths) - 1):
                    ref = ref[paths[i]]
                    place = place[paths[order[i]]]

                place[paths[order[-1]]] = ref[paths[-1]]
                paths[-1] += 1
                for i in range(len(paths) - 1, 0, -1):
                    if paths[i] >= self.__boyut[i]:
                        paths[i] = 0
                        paths[i - 1] += 1
                    else:
                        break
            self.D = gergen(arr)
            return gergen(arr)

    def L1(self):
        # Calculates and returns the L1 norm
        flattened_data = self.duzlestir().__veri  # Assuming flatten returns a gergen object
        # Calculate the L1 norm by summing the absolute values of elements in the flattened list
        l1_norm = sum(abs(item) for item in flattened_data)
        return l1_norm

    def L2(self):
        # Assuming flatten returns a gergen object and __veri holds the flattened data
        flattened_data = self.duzlestir().__veri
        # Calculate the L2 norm by summing the squares of elements in the flattened list and then taking the square root
        l2_norm = sum(item**2 for item in flattened_data)**0.5
        return l2_norm

    def Lp(self, p):
        # Calculates and returns the Lp norm, where p should be positive integer
        if p <= 0:
            raise ValueError("p must be a positive integer for Lp norm.")
        # Assuming flatten returns a gergen object and __veri holds the flattened data
        flattened_data = self.duzlestir().__veri

        # Calculate the Lp norm by raising elements to the power of p, summing, and then taking the p-th root
        lp_norm = sum(abs(item)**p for item in flattened_data)**(1 / p)

        return lp_norm

    def listeye(self):
        # Converts the gergen object into a list or a nested list, depending on its dimensions.
        if isinstance(self.__veri, list):
            if not self.__veri:
                return []
            return self.__veri.copy()
        else:
            return self.__veri

    def duzlestir(self):
        """
        Flattens a multidimensional list (self.__veri) into a 1D list.

        Returns:
            gergen: A new gergen object with the flattened list.
        """
        if not isinstance(self.__veri, list):
            return gergen(self.__veri)
        flattened_list = []
        # Create a stack with the initial list
        stack = [self.__veri]

        # Process the stack
        while stack:
            current_item = stack.pop()
            if isinstance(current_item, list):
                # Extend the stack by reversing the current item list
                # to maintain the original order in the flattened list
                stack.extend(current_item[::-1])
            else:
                # If it's not a list, add it to the flattened list
                flattened_list.append(current_item)

        # Since we're appending elements to the end, but processing the stack in LIFO order,
        # we need to reverse the flattened list to restore the original element order
        flattened_list.reverse()

        # Create a new gergen instance with the flattened list
        return gergen(flattened_list)

    def boyutlandir(self, yeni_boyut):
        """
        Reshapes the gergen object to a new shape 'yeni_boyut', specified as a tuple.
        """
        # Flatten the data first
        flat_data = list(self.duzlestir().__veri)

        def reshape_helper(data, dims):
            if not dims:
                return data.pop(0)
            return [reshape_helper(data, dims[1:]) for _ in range(dims[0])]

        # Check if the new shape is compatible with the number of elements
        if self.prod(yeni_boyut) != len(flat_data):
            raise ValueError("New shape must have the same number of elements as the original.")

        # Use the helper to create the reshaped data and update the object's internal state
        self.__veri = reshape_helper(flat_data, yeni_boyut)
        self.__boyut = yeni_boyut

    def ic_carpim(self, other):
        ic_carpim_operation = IcCarpim()
        result_gergen = ic_carpim_operation(self, other)
        return result_gergen

    def dis_carpim(self, other):
        dis_carpim_operation = DisCarpim()
        result_gergen = dis_carpim_operation(self, other)
        return result_gergen

    def us(self, n):
        # Applies the power function to each element of the gergen object.
        power_operation = Us()
        result_gergen = power_operation(self, n)
        return result_gergen

    def log(self):
        # Applies the log function to each element of the gergen object.
        log_operation = Log10()
        result_gergen = log_operation(self)
        return result_gergen

    def ln(self):
        # Applies the ln function to each element of the gergen object.
        log_operation = Ln()
        result_gergen = log_operation(self)
        return result_gergen

    def sin(self):
        # Applies the sin function to each element of the gergen object.
        sin_operation = Sin()
        result_gergen = sin_operation(self)
        return result_gergen

    def cos(self):
        # Applies the cos function to each element of the gergen object.
        cos_operation = Cos()
        result_gergen = cos_operation(self)
        return result_gergen

    def tan(self):
        # Applies the tan function to each element of the gergen object.
        tan_operation = Tan()
        result_gergen = tan_operation(self)
        return result_gergen

    def topla(self, eksen=None):
        # Calculates the sum of the elements of the gergen object, optionally along a specified axis 'eksen'.
        topla_operation = Topla()
        result_gergen = topla_operation(self, eksen=eksen)
        return result_gergen

    def ortalama(self, eksen=None):
        # Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        ortalama_operation = Ortalama()
        result = ortalama_operation(self, eksen=eksen)
        return result

    def turev_al(self, grad_output=1):
        """
        Computes the backward pass of the operation that produced this gergen object.

        Parameters:
            grad_output: The gradient of the loss w.r.t the output of the operation.

        Returns:
            None: Gradients are propagated recursively.
        """
        # If the current gergen doesn't require gradient calculation
        if not self.requires_grad:
            return

        # If no operation produced this gergen, it's a leaf node
        if self.operation is None:
            self.turev = grad_output
        else:
            # Get gradients of the input(s) by calling geri()
            gradients = self.operation.geri(grad_output)

            # If there are multiple inputs (a tuple of gradients)
            if isinstance(gradients, tuple):
                for inp, grad in zip(self.operation.operands, gradients):
                    if inp.requires_grad:
                        inp.turev_al(grad)
            else:
                # Single input operation
                operand = self.operation.operands[0]
                if operand.requires_grad:
                    operand.turev_al(gradients)

