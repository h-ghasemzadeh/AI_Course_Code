import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1  # Length of the domain (0 <= x <= 1)
Nx = 100  # Number of spatial grid points
dx = L / (Nx - 1)  # Spatial step size
x = np.linspace(0, L, Nx)  # Spatial grid
T_max = 0.5  # Maximum time for simulation

# Calculate stable dt based on dx
dt = 0.5 * dx**2  # Time step to ensure stability
time_steps = int(T_max / dt)  # Number of time steps

# Check stability condition
alpha = dt / dx**2
print(f"Alpha (dt / dx^2): {alpha}")  # This should now be <= 0.5

# Initial condition U(x, 0)
U_initial = 1 + np.cos(np.pi * x) + 0.5 * np.cos(3 * np.pi * x)

# Pre-allocate solution matrix (U at each time step)
U = np.zeros((time_steps, Nx))
U[0, :] = U_initial

# Finite Difference Method (Explicit scheme for diffusion equation)
for n in range(0, time_steps - 1):
    U[n+1, 1:-1] = U[n, 1:-1] + alpha * (U[n, :-2] - 2 * U[n, 1:-1] + U[n, 2:])
    
    # Neumann boundary conditions (Ux(0,t) = Ux(1,t) = 0)
    U[n+1, 0] = U[n+1, 1]  # Zero flux at x=0
    U[n+1, -1] = U[n+1, -2]  # Zero flux at x=1

# Plot results at different times
plt.figure(figsize=(8, 6))
for i in range(0, time_steps, int(time_steps / 5)):
    plt.plot(x, U[i, :], label=f't={i * dt:.4f}')

plt.title('Solution to the Heat Equation with Neumann BCs')
plt.xlabel('x')
plt.ylabel('U(x, t)')
plt.legend()
plt.grid(True)
plt.show()
