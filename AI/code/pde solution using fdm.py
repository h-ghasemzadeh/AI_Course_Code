import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 1  # Given constant r in the boundary condition u(x, 1) = e^(-rx)
Lx = 1  # Length in x direction
Ly = 1   # Length in y direction
Nx = 100  # Number of grid points in x direction
Ny = 100  # Number of grid points in y direction
dx = Lx / (Nx - 1)  # Step size in x direction
dy = Ly / (Ny - 1)  # Step size in y direction

# Initialize the solution grid
u = np.zeros((Nx, Ny))

# Boundary conditions
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# u(x, 1) = e^(-rx)
u[:, -1] = np.exp(-r * x)

# u(x, 0) = 0 (this is already initialized to zero)

# u(0, y) = 0 (this is already initialized to zero)

# Finite Difference Method to solve u_xx + u_yy = 0
for iteration in range(2000):  # Number of iterations for convergence
    u_old = u.copy()
    
    # Update the internal grid points
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i, j] = 0.5 * (((u_old[i+1, j] + u_old[i-1, j]) / dx**2) + 
                             ((u_old[i, j+1] + u_old[i, j-1]) / dy**2)) / \
                      (1/dx**2 + 1/dy**2)

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Plot the results
plt.figure(figsize=(6, 4))
cp = plt.contourf(X, Y, u.T, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.title("Solution to the PDE u_xx + u_yy = 0")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
