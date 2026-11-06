import matplotlib.pyplot as plt

# Function to plot the neural network
def plot_nn_with_7_layers():
    # Define the number of neurons in each layer
    input_neurons = 5
    hidden_layers = [9, 14, 15, 19, 19, 13, 8]  # 7 hidden layers with different sizes
    output_neurons = 4

    # Define layer colors
    layer_colors = ["blue", "green", "orange", "purple", "red", "cyan", "yellow"]

    # Initialize figure
    plt.figure(figsize=(12, 8))
    plt.title("Neural Network with 7 Hidden Layers", fontsize=16)
    plt.axis("off")

    # Helper function to calculate neuron positions
    def get_layer_positions(layer_size, x_offset, y_center=0):
        return [(x_offset, y_center + i - layer_size // 2) for i in range(layer_size)]

    # Plot layers
    previous_layer_positions = get_layer_positions(input_neurons, -4)  # Input layer at x=-4
    plt.scatter(
        [pos[0] for pos in previous_layer_positions],
        [pos[1] for pos in previous_layer_positions],
        s=200,
        color="r",
        label="Input Layer",
    )
    for i, layer_size in enumerate(hidden_layers):
        current_layer_positions = get_layer_positions(layer_size, -3 + i)
        plt.scatter(
            [pos[0] for pos in current_layer_positions],
            [pos[1] for pos in current_layer_positions],
            s=200,
            color=layer_colors[i % len(layer_colors)],
            label=f"Hidden Layer {i + 1}" if i == 0 else None,
        )
        # Draw connections between layers
        for prev_pos in previous_layer_positions:
            for curr_pos in current_layer_positions:
                plt.plot(
                    [prev_pos[0], curr_pos[0]],
                    [prev_pos[1], curr_pos[1]],
                    "g-",
                    alpha=0.1,
                )
        previous_layer_positions = current_layer_positions

    # Plot output layer
    output_layer_positions = get_layer_positions(output_neurons, 4)
    plt.scatter(
        [pos[0] for pos in output_layer_positions],
        [pos[1] for pos in output_layer_positions],
        s=200,
        color="orange",
        label="Output Layer",
    )
    for prev_pos in previous_layer_positions:
        for curr_pos in output_layer_positions:
            plt.plot(
                [prev_pos[0], curr_pos[0]],
                [prev_pos[1], curr_pos[1]],
                "g-",
                alpha=0.1,
            )

    # Add legend
    plt.legend(loc="upper right", fontsize=10)
    plt.show()

# Call the function to plot the neural network
plot_nn_with_7_layers()
