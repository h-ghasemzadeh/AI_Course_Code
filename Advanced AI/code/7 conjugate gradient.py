import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a simple quadratic cost function: f(x) = 1/2 * x^T A x - b^T x
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
def cost_function(x):
    return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

# Gradient of the cost function
def cost_function_grad(x):
    return A @ x - b

# Initialize the path list to store points for plotting the optimization path
path = []

# Minimize the cost function using Conjugate Gradient method
initial_guess = np.array([-0.5, 2.5])
result = minimize(cost_function, initial_guess, method='CG', jac=cost_function_grad, callback=lambda xk: path.append(xk))

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
plt.plot(path[:, 0], path[:, 1], 'o-', color='red', label="Conjugate Gradient Path")
plt.scatter(*result.x, color='blue', s=100, label="Minimum")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Conjugate Gradient Minimization Path")
plt.legend()
plt.show()
