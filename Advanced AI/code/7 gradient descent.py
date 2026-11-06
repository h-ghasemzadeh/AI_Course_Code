import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic cost function: f(x) = 1/2 * x^T A x - b^T x
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])

def cost_function(x):
    return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

# Gradient of the cost function
def cost_function_grad(x):
    return A @ x - b

# Gradient Descent parameters
learning_rate = 0.1
initial_guess = np.array([0.0, 0.0])
num_iterations = 50

# Initialize the path list to store points for plotting the optimization path
path = [initial_guess]

# Gradient Descent loop
x = initial_guess
for _ in range(num_iterations):
    grad = cost_function_grad(x)
    x = x - learning_rate * grad
    path.append(x)

# Convert path to an array for easier plotting
path = np.array(path)

# Generate a grid of points for contour plot
x_vals = np.linspace(-1, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2) - b[0] * X - b[1] * Y

# Plot the contours of the cost function
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'o-', color='red', label="Gradient Descent Path")
plt.scatter(*path[-1], color='blue', s=100, label="Minimum")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Gradient Descent Minimization Path")
plt.legend()
plt.show()
