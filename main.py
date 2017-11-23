import numpy
import math
import random

# hyper parameters
input_size = 2
hidden_size = 5
output_size = 1

learning_rate = 0.2
max_iterations = 15000

# training data
training_input = ((0, 0), (0, 1), (1, 0), (1, 1))
training_output = (0, 1, 1, 0)


# activation functions - forward and backward
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid_forward(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


class xornn:
    ''' Neural network for the XOR operation

    The input layer has two neurons representing the boolean inputs.
    The output layer has one neuron representing the output.
    '''

    input_size = 2
    output_size = 1

    def __init__(self, hidden_size=5, learning_rate=.1):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        random.seed(1)


if __name__ == "__main__":
    pass
