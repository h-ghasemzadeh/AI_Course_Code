import numpy as np
import matplotlib.pyplot as plt

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for the bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []  # To store weights for visualization
        self.errors = []   # To store errors for each epoch

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Add bias term to input
        x_with_bias = np.insert(x, 0, 1)  # Add bias input (x0 = 1)
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, x_with_bias)
        # Apply activation function
        return self.activation_function(weighted_sum)

    def train(self, X, y):
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                # Add bias term to input
                x_with_bias = np.insert(X[i], 0, 1)
                # Calculate prediction
                prediction = self.predict(X[i])
                # Calculate error
                error = y[i] - prediction
                total_error += abs(error)  # Track total error for this epoch
                # Update weights
                self.weights += self.learning_rate * error * x_with_bias
            # Save weights after each epoch for visualization
            self.history.append(self.weights.copy())
            self.errors.append(total_error)
            print(f"Epoch {epoch+1}/{self.epochs}, Weights: {self.weights}, Total Error: {total_error}")

# Logical function dataset
def get_logical_function_data(logic_type="AND"):
    if logic_type == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
        y = np.array([0, 0, 0, 1])  # Target output (AND logic)
    elif logic_type == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
        y = np.array([0, 1, 1, 1])  # Target output (OR logic)
    else:
        raise ValueError("Unsupported logic type. Use 'AND' or 'OR'.")
    return X, y

# Plot decision boundary
def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict on grid
    Z = np.array([perceptron.predict([x, y]) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.show()

# Plot training error
def plot_training_error(perceptron, logic_type):
    plt.figure()
    plt.plot(range(1, perceptron.epochs + 1), perceptron.errors)
    plt.title(f'Training Error Over Epochs ({logic_type} Logic)')
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.show()

# Train and visualize for a given logic type
def train_and_visualize_logic(logic_type="AND"):
    X, y = get_logical_function_data(logic_type)
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron.train(X, y)
    plot_decision_boundary(perceptron, X, y, title=f"{logic_type} Logic Decision Boundary")
    plot_training_error(perceptron, logic_type)
    return perceptron

# Train and visualize for AND logic
print("Training Perceptron for AND Logic:")
train_and_visualize_logic("AND")

# Train and visualize for OR logic
print("\nTraining Perceptron for OR Logic:")
train_and_visualize_logic("OR")
