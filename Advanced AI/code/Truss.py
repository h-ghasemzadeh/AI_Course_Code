import numpy as np
import matplotlib.pyplot as plt

# Define the problem
nodes = np.array([
    [0, 0],    # Node 1
    [1, 0],    # Node 2
    [2, 0],    # Node 3
    [1, 1]     # Node 4
])

elements = np.array([
    [0, 1],    # Element 1: Node 1 to Node 2
    [1, 2],    # Element 2: Node 2 to Node 3
    [0, 3],    # Element 3: Node 1 to Node 4
    [1, 3],    # Element 4: Node 2 to Node 4
    [2, 3]     # Element 5: Node 3 to Node 4
])

# Material properties
E = 210e9  # Young's modulus (Pa)
A = 0.01   # Cross-sectional area (m^2)

# External forces
F = np.zeros(2 * len(nodes))  # Global force vector
F[4] = 200  # Force in the x-direction at Node 3 (N)

# Boundary conditions
fixed_dofs = [0, 1, 5]  # Fixed DOFs (Node 1: x, y; Node 3: y)
print(fixed_dofs)
# Function to compute the stiffness matrix of a truss element
def truss_stiffness_matrix(node1, node2, E, A):
    L = np.linalg.norm(node2 - node1)
    c = (node2[0] - node1[0]) / L  # Cosine
    s = (node2[1] - node1[1]) / L  # Sine
    k = (E * A / L) * np.array([
        [c * c, c * s, -c * c, -c * s],
        [c * s, s * s, -c * s, -s * s],
        [-c * c, -c * s, c * c, c * s],
        [-c * s, -s * s, c * s, s * s]
    ])
    return k

# Assemble the global stiffness matrix
K_global = np.zeros((2 * len(nodes), 2 * len(nodes)))

for element in elements:
    node1 = nodes[element[0]]
    node2 = nodes[element[1]]
    k_local = truss_stiffness_matrix(node1, node2, E, A)
    
    # Local to global mapping
    dofs = [2 * element[0], 2 * element[0] + 1, 2 * element[1], 2 * element[1] + 1]
    for i in range(4):
        for j in range(4):
            K_global[dofs[i], dofs[j]] += k_local[i, j]

# Apply boundary conditions
free_dofs = np.setdiff1d(np.arange(2 * len(nodes)), fixed_dofs)
K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
F_reduced = F[free_dofs]

# Solve for displacements
U_reduced = np.linalg.solve(K_reduced, F_reduced)
U = np.zeros(2 * len(nodes))
U[free_dofs] = U_reduced

# Calculate stresses in each element
stresses = []
for element in elements:
    node1 = nodes[element[0]]
    node2 = nodes[element[1]]
    L = np.linalg.norm(node2 - node1)
    c = (node2[0] - node1[0]) / L
    s = (node2[1] - node1[1]) / L
    dofs = [2 * element[0], 2 * element[0] + 1, 2 * element[1], 2 * element[1] + 1]
    u_local = U[dofs]
    stress = (E / L) * np.array([-c, -s, c, s]) @ u_local
    stresses.append(stress)

# Output results
print("Displacements (U):")
for i, displacement in enumerate(U):
    print(f"Node {i // 2 + 1}, {'x' if i % 2 == 0 else 'y'}: {displacement:.6e} m")

print("\nStresses in elements:")
for i, stress in enumerate(stresses):
    print(f"Element {i + 1}: {stress:.6f} Pa")

# Function to plot the truss with supports and forces
def plot_truss(nodes, elements, U=None, scale=1.0, fixed_dofs=None, forces=None):
    plt.figure(figsize=(10, 6))
    plt.title("Truss Structure with Supports and Forces")
    
    # Plot original truss
    for element in elements:
        node1 = nodes[element[0]]
        node2 = nodes[element[1]]
        plt.plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-', label="Original" if element[0] == 0 else "")
    
    # Plot deformed truss if displacements are provided
    if U is not None:
        deformed_nodes = nodes + U.reshape(-1, 2) * scale
        for element in elements:
            node1 = deformed_nodes[element[0]]
            node2 = deformed_nodes[element[1]]
            plt.plot([node1[0], node2[0]], [node1[1], node2[1]], 'r--', label="Deformed" if element[0] == 0 else "")
    
    # Plot nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], c='k', label="Nodes")
    if U is not None:
        plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], c='r', label="Deformed Nodes")
    
    # Plot supports
    if fixed_dofs is not None:
        for dof in fixed_dofs:
            node_idx = dof // 2
            direction = dof % 2  # 0: x, 1: y
            if direction == 0:
                plt.plot(nodes[node_idx, 0], nodes[node_idx, 1], 'ks', markersize=10, label="Support (x)" if node_idx == 0 else "")
            elif direction == 1:
                plt.plot(nodes[node_idx, 0], nodes[node_idx, 1], 'k^', markersize=10, label="Support (y)" if node_idx == 0 else "")
    
    # Plot forces
    if forces is not None:
        for node_idx in range(len(nodes)):
            fx = forces[2 * node_idx]
            fy = forces[2 * node_idx + 1]
            if fx != 0 or fy != 0:
                plt.arrow(nodes[node_idx, 0], nodes[node_idx, 1], fx * 1e-3, fy * 1e-3, color='g', width=0.01, label="Force" if node_idx == 2 else "")
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot the original and deformed truss with supports and forces
plot_truss(nodes, elements, fixed_dofs=fixed_dofs, forces=F)  # Original truss
plot_truss(nodes, elements, U, scale=1e6, fixed_dofs=fixed_dofs, forces=F)  # Deformed truss (scaled for visibility)