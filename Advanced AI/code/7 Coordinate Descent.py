import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic cost function: f(x) = 1/2 * x^T A x - b^T x
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])

def cost_function(x):
    return 0.5 * np.dot(x, A @ x) - np.dot(b, x)

# Initialize path list to store points for plotting the optimization path
path = []

# Coordinate Descent parameters
initial_guess = np.array([0.0, 0.0])
x = initial_guess.copy()
tolerance = 1e-6
max_iterations = 100

# Perform Coordinate Descent optimization
for iteration in range(max_iterations):
    # Store the current position for plotting
    path.append(x.copy())
    
    # Update each coordinate (variable) in x sequentially
    for i in range(len(x)):
        # Calculate the partial derivative with respect to x[i]
        gradient = A[i, 0] * x[0] + A[i, 1] * x[1] - b[i]
        
        # Update x[i] based on the partial derivative, keeping other coordinates fixed
        x[i] = x[i] - 0.1 * gradient  # Step size is 0.1 for simplicity

    # Check for convergence
    if np.linalg.norm(path[-1] - x) < tolerance:
        break

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
plt.plot(path[:, 0], path[:, 1], 'o-', color='red', label="Coordinate Descent Path")
plt.scatter(*x, color='blue', s=100, label="Minimum")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Coordinate Descent Minimization Path")
plt.legend()
plt.show()
