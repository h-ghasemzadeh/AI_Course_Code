import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Train the neural network and return first and final iteration states with errors
def train_neural_network(input_values, target_output, hidden_size=2, learning_rate=0.1, epochs=1000):
    input_size = len(input_values[0])
    output_size = len(target_output[0])

    # Initialize weights and biases
    np.random.seed(42)
    weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
    bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
    bias_output = np.random.uniform(-1, 1, (1, output_size))

    # Save the initial state for the first iteration
    initial_state = {
        "weights_input_hidden": weights_input_hidden.copy(),
        "bias_hidden": bias_hidden.copy(),
        "weights_hidden_output": weights_hidden_output.copy(),
        "bias_output": bias_output.copy(),
    }

    # To track training error
    errors = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(input_values, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)

        # Compute error
        error = target_output - output_layer_output
        errors.append(np.mean(np.abs(error)))

        # Backpropagation
        d_output = error * sigmoid_derivative(output_layer_output)
        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += input_values.T.dot(d_hidden) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Final state after training
    final_state = {
        "weights_input_hidden": weights_input_hidden,
        "bias_hidden": bias_hidden,
        "weights_hidden_output": weights_hidden_output,
        "bias_output": bias_output,
    }

    return initial_state, final_state, errors

# Plot the neural network diagram
def plot_neural_network(state, input_values, title):
    # Define layer positions
    layer_positions = {
        "input": [(-4, 2), (-4, -2)],
        "hidden": [(-1, 3), (-1, -3)],
        "output": [(2, 0)],
    }

    # Forward pass to compute neuron values
    weights_input_hidden = state["weights_input_hidden"]
    bias_hidden = state["bias_hidden"]
    weights_hidden_output = state["weights_hidden_output"]
    bias_output = state["bias_output"]

    hidden_inputs = np.dot(input_values, weights_input_hidden) + bias_hidden
    hidden_outputs = sigmoid(hidden_inputs)
    output_input = np.dot(hidden_outputs, weights_hidden_output) + bias_output
    output_value = sigmoid(output_input)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    ax.set_xlim(-5, 3)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

    # Draw neurons and annotate values
    for layer, positions in layer_positions.items():
        for idx, pos in enumerate(positions):
            if layer == "input":
                color, value = "blue", f"X{idx + 1}={input_values[0][idx]:.2f}"
            elif layer == "hidden":
                color, value = "green", f"z={hidden_inputs[0, idx]:.2f}\no={hidden_outputs[0, idx]:.2f}"
            else:
                color, value = "orange", f"z={output_input[0, 0]:.2f}\no={output_value[0, 0]:.2f}"

            # Main neuron
            circle = plt.Circle(pos, radius=0.7, color=color, fill=True, alpha=0.8)
            ax.add_artist(circle)
            ax.text(pos[0], pos[1], value, fontsize=14, color="white", ha="center", va="center")

            # Bias neuron
            if layer != "input":
                bias_pos = (pos[0] - 1.5, pos[1])
                bias_value = bias_hidden[0, idx] if layer == "hidden" else bias_output[0, 0]
                bias_circle = plt.Circle(bias_pos, radius=0.4, color="red", fill=True, alpha=0.8)
                ax.add_artist(bias_circle)
                ax.plot([bias_pos[0], pos[0]], [bias_pos[1], pos[1]], "r--", lw=1)
                ax.text(bias_pos[0], bias_pos[1], f"{bias_value:.2f}", fontsize=14, color="yellow", ha="center", va="center")

    # Draw connections and annotate weights
    for i, input_pos in enumerate(layer_positions["input"]):
        for j, hidden_pos in enumerate(layer_positions["hidden"]):
            ax.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]], "k-", lw=1)
            weight = weights_input_hidden[i, j]
            ax.text((input_pos[0] + hidden_pos[0]) / 2, (input_pos[1] + hidden_pos[1]) / 2, f"{weight:.2f}", fontsize=14, ha="center")

    for i, hidden_pos in enumerate(layer_positions["hidden"]):
        for j, output_pos in enumerate(layer_positions["output"]):
            ax.plot([hidden_pos[0], output_pos[0]], [hidden_pos[1], output_pos[1]], "k-", lw=1)
            weight = weights_hidden_output[i, j]
            ax.text((hidden_pos[0] + output_pos[0]) / 2, (hidden_pos[1] + output_pos[1]) / 2, f"{weight:.2f}", fontsize=14, ha="center")

    plt.show()

# Plot training error
def plot_training_error(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, color="blue", label="Training Error")
    plt.title("Training Error Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Absolute Error", fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()

# Main function
def main():
    input_values = np.array([[0.5, 0.8]])  # Example input
    target_output = np.array([[1]])  # Target output

    # Train the network
    initial_state, final_state, errors = train_neural_network(input_values, target_output, hidden_size=2, epochs=1000)

    # Plot first iteration
    plot_neural_network(initial_state, input_values, "First Iteration")

    # Plot final iteration
    plot_neural_network(final_state, input_values, "Final Iteration")

    # Plot training error
    plot_training_error(errors)

# Run the main function
main()
