import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate y values for sigmoid and its derivative
y = sigmoid(x)
y_derivative = sigmoid_derivative(x)

# Plot the sigmoid function and its derivative
plt.plot(x, y, label='Sigmoid Function')
plt.plot(x, y_derivative, label='Sigmoid Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Sigmoid Function and Its Derivative')
plt.grid(True)
plt.show()
