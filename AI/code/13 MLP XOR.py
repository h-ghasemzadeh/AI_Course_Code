import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Logical datasets
def get_logical_function_data(logic_type="XOR"):
    if logic_type == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
        y = np.array([[0], [1], [1], [0]])  # XOR outputs
    elif logic_type == "XNOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
        y = np.array([[1], [0], [0], [1]])  # XNOR outputs
    else:
        raise ValueError("Unsupported logic type. Use 'XOR' or 'XNOR'.")
    return X, y

# Neural network training function
def train_neural_network(X, y, hidden_size=2, learning_rate=0.1, epochs=10000):
    # Initialize weights and biases
    input_size = X.shape[1]
    output_size = y.shape[1]

    np.random.seed(42)  # For reproducibility
    weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
    bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
    bias_output = np.random.uniform(-1, 1, (1, output_size))

    errors = []  # To track error for visualization

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)

        # Calculate error
        error = y - output_layer_output
        errors.append(np.mean(np.abs(error)))  # Store error for visualization

        # Backpropagation
        d_output = error * sigmoid_derivative(output_layer_output)

        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Print error every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {np.mean(np.abs(error))}")

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, errors

# Plot training error
def plot_training_error(errors, logic_type):
    plt.figure()
    plt.plot(errors)
    plt.title(f'Training Error Over Epochs ({logic_type} Logic)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.show()

# Plot decision boundary
def plot_decision_boundary(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, X, y, logic_type):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict on grid
    Z = []
    for i in np.c_[xx.ravel(), yy.ravel()]:
        hidden_layer_input = np.dot(i, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)
        Z.append(np.round(output_layer_output[0]))
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolor='k', cmap=plt.cm.Paired)
    plt.title(f'Decision Boundary ({logic_type} Logic)')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.show()

# Train and visualize for a given logic type
def train_and_visualize_logic(logic_type="XOR"):
    X, y = get_logical_function_data(logic_type)
    print(f"\nTraining Neural Network for {logic_type} Logic:")
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, errors = train_neural_network(X, y)
    plot_training_error(errors, logic_type)
    plot_decision_boundary(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, X, y, logic_type)

# Train and visualize for XOR logic
train_and_visualize_logic("XOR")

# Train and visualize for XNOR logic
train_and_visualize_logic("XNOR")
