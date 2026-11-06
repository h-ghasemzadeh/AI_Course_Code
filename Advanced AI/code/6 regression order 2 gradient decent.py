import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = '3D_data.xlsx'
data = pd.read_excel(file_path)

# Extract X, Y, Z columns
X_data = data[['X', 'Y']].values
Z_data = data['Z'].values

# Generate polynomial features of degree 2 manually
X_poly = np.c_[np.ones(X_data.shape[0]), X_data[:, 0], X_data[:, 1], X_data[:, 0]**2, X_data[:, 0]*X_data[:, 1], X_data[:, 1]**2]

# Initialize parameters for gradient descent
theta = np.random.randn(X_poly.shape[1])
learning_rate = 0.01
num_iterations = 10000

# Define the cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Define the gradient descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    
    for i in range(num_iterations):
        # Calculate predictions
        predictions = X @ theta
        
        # Calculate the gradients
        gradients = (1 / m) * X.T @ (predictions - y)
        
        # Update the parameters
        theta -= learning_rate * gradients
        
        # Store the cost for this iteration
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Optional: print cost every 1000 iterations for monitoring
        if i % 1000 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    
    return theta, cost_history

# Run gradient descent
theta, cost_history = gradient_descent(X_poly, Z_data, theta, learning_rate, num_iterations)

# Display final parameters
intercept = theta[0]
equation = (f"Z = {intercept:.2f} "
            f"+ ({theta[1]:.2f})*X "
            f"+ ({theta[2]:.2f})*Y "
            f"+ ({theta[3]:.2f})*X^2 "
            f"+ ({theta[4]:.2f})*X*Y "
            f"+ ({theta[5]:.2f})*Y^2")
print("Equation:", equation)

# Plotting the 3D scatter and regression surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Original data points
ax.scatter(X_data[:, 0], X_data[:, 1], Z_data, color='blue', label="Original Data", alpha=0.5)

# Create mesh grid for X and Y for the regression surface
x_range = np.linspace(X_data[:, 0].min(), X_data[:, 0].max(), 100)
y_range = np.linspace(X_data[:, 1].min(), X_data[:, 1].max(), 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
X_mesh_poly = np.c_[np.ones(x_mesh.ravel().shape[0]), x_mesh.ravel(), y_mesh.ravel(), x_mesh.ravel()**2, x_mesh.ravel()*y_mesh.ravel(), y_mesh.ravel()**2]

# Predict Z values for the grid
z_mesh_pred = X_mesh_poly @ theta
z_mesh_pred = z_mesh_pred.reshape(x_mesh.shape)

# Plot the regression surface
ax.plot_surface(x_mesh, y_mesh, z_mesh_pred, color='orange', alpha=0.6, rstride=100, cstride=100)

# Add the polynomial equation to the plot
ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Labels and title
ax.set_title("3D Polynomial Regression Surface using Gradient Descent")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
