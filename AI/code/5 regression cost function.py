import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data from the table
x = np.array([21, 35, 7, 33, 5, 14, 25])  # Income
y = np.array([15, 21, 2, 13, 3, 8, 9])    # Food Expenditure

# Number of data points
M = len(x)  # Changed from N to M

# Defining the cost function
def cost_function(B0, B1):
    return (1 / (2 * M)) * np.sum((y - (B0 + B1 * x)) ** 2)

# Generating values for B0 and B1
B0_values = np.linspace(-5, 10, 1000)
B1_values = np.linspace(-1, 2, 2000)
B0_mesh, B1_mesh = np.meshgrid(B0_values, B1_values)

# Calculating cost values for each combination of B0 and B1
cost_values = np.zeros(B0_mesh.shape)
for i in range(B0_mesh.shape[0]):
    for j in range(B0_mesh.shape[1]):
        cost_values[i, j] = cost_function(B0_mesh[i, j], B1_mesh[i, j])

# Logarithmic scale for cost values (add a small value to avoid log(0))
log_cost_values = np.log(cost_values + 1e-10)  # Adding a small constant to avoid log(0)

# Finding the minimum point
min_index = np.unravel_index(np.argmin(cost_values), cost_values.shape)
B0_min = B0_mesh[min_index]
B1_min = B1_mesh[min_index]
min_cost = cost_values[min_index]

# Plotting the 2D contour plot of the cost function
plt.figure(figsize=(12, 5))

# 2D Contour Plot with contour lines
plt.subplot(1, 2, 1)
cp = plt.contourf(B0_mesh, B1_mesh, cost_values, levels=50, cmap='viridis')
plt.colorbar(cp)
contour_lines = plt.contour(B0_mesh, B1_mesh, cost_values, levels=10, colors='black', linewidths=0.5)
plt.clabel(contour_lines, inline=True, fontsize=8)
plt.scatter(B0_min, B1_min, color="red")
plt.annotate(f"Minimum Point\n(B0={B0_min:.1f}, B1={B1_min:.1f}, Cost={min_cost:.1f})",
             (B0_min, B1_min), 
             textcoords="offset points", 
             xytext=(10, 10),  # Adjust these values to move the text
             ha='center', 
             color="red")
plt.xlabel("Intercept (B0)")
plt.ylabel("Slope (B1)")
plt.title("Cost Function Contour Plot with Contour Lines")
plt.legend()

# 3D Surface Plot with logarithmic cost values
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B0_mesh, B1_mesh, log_cost_values, cmap='viridis', edgecolor='k', alpha=0.7)
ax.scatter(B0_min, B1_min, np.log(min_cost + 1e-10), color="red", s=50, label=f"Minimum Point (B0={B0_min:.3f}, B1={B1_min:.3f}, Cost={min_cost:.3f})")

# Set the best view for the 3D plot
ax.view_init(elev=20, azim=45)  # Adjust these values to find the best angle
# Labels and title
ax.set_xlabel("Intercept (B0)")
ax.set_ylabel("Slope (B1)")
ax.set_zlabel("Log(Cost)")
ax.set_title("3D Plot of Logarithmic Cost Function")
ax.legend()

# Print minimum values
print("Optimal Intercept (B0):", B0_min)
print("Optimal Slope (B1):", B1_min)
print("Minimum Cost:", min_cost)

plt.show()   # Show plots
