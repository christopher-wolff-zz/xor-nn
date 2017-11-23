import numpy as np

# training data
training_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_output = np.array([[0], [1], [1], [0]])

# hyper parameters
input_size = 2
hidden_size = 3
output_size = 1

num_epochs = 60000


# activation function - forward pass
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of activation function - backward pass
def sigmoid_prime(x):
    return x * (1.0 - x)


def main():
    # randomly initialize weights
    w0 = np.random.uniform(size=(input_size, hidden_size))   # hidden weights
    w1 = np.random.uniform(size=(hidden_size, output_size))  # output weights

    # training
    for i in range(num_epochs):
        # forward pass
        hidden = sigmoid(np.dot(training_input, w0))  # hidden activations
        output = sigmoid(np.dot(hidden, w1))          # output activations
        # error
        error = training_output - output
        # update
        delta_output = error * sigmoid_prime(output)
        delta_hidden = delta_output.dot(w1.T) * sigmoid_prime(hidden)
        w1 += hidden.T.dot(delta_output)
        w0 += training_input.T.dot(delta_hidden)
        # print status
        if i % 1000 == 0:
            print(error)

    print("Training complete.")
    print(f"Final output: {output}")
    print(f"Final error: {error}")


if __name__ == "__main__":
    main()
