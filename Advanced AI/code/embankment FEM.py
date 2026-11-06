import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, linalg

class FEMDamSettlement:
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.material_props = None  # Fixed variable name
        self.boundary_conditions = None
        self.loads = None
        
    def create_mesh(self, length, height, nx, ny):
        """
        Create a rectangular mesh for the dam
        :param length: Length of the dam (m)
        :param height: Height of the dam (m)
        :param nx: Number of elements in x-direction
        :param ny: Number of elements in y-direction
        """
        self.nodes = []
        self.elements = []
        
        # Create nodes
        node_id = 0
        for j in range(ny + 1):
            y = height * (j / ny)
            for i in range(nx + 1):
                x = length * (i / nx)
                self.nodes.append([x, y])
                node_id += 1
        
        # Create elements (4-node quadrilateral elements)
        for j in range(ny):
            for i in range(nx):
                n1 = i + j * (nx + 1)
                n2 = n1 + 1
                n3 = n2 + (nx + 1)
                n4 = n1 + (nx + 1)
                self.elements.append([n1, n2, n3, n4])
    
    def set_material_properties(self, E, nu, rho):
        """
        Set material properties
        :param E: Young's modulus (Pa)
        :param nu: Poisson's ratio
        :param rho: Density (kg/m3)
        """
        self.material_props = {  # Fixed variable name
            'E': E,
            'nu': nu,
            'rho': rho
        }
    
    def set_boundary_conditions(self, fixed_nodes):
        """
        Set boundary conditions (fixed nodes)
        :param fixed_nodes: List of node indices to be fixed
        """
        self.boundary_conditions = fixed_nodes
    
    def set_loads(self, loads):
        """
        Set nodal loads
        :param loads: Dictionary of {node_id: [Fx, Fy]}
        """
        self.loads = loads
    
    def calculate_stiffness_matrix(self):
        """
        Calculate global stiffness matrix
        """
        nnodes = len(self.nodes)
        ndof = 2 * nnodes  # 2 DOF per node (u, v)
        
        # Initialize sparse matrix in COO format
        rows = []
        cols = []
        data = []
        
        E = self.material_props['E']
        nu = self.material_props['nu']
        
        # Plane strain constitutive matrix
        D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu)/2]
        ])
        
        # Numerical integration points (2x2 Gauss quadrature)
        gauss_points = [
            (-1/np.sqrt(3), -1/np.sqrt(3)),
            (1/np.sqrt(3), -1/np.sqrt(3)),
            (1/np.sqrt(3), 1/np.sqrt(3)),
            (-1/np.sqrt(3), 1/np.sqrt(3))
        ]
        weights = [1, 1, 1, 1]
        
        for elem in self.elements:
            elem_nodes = [self.nodes[n] for n in elem]
            ke = np.zeros((8, 8))  # Element stiffness matrix
            
            for gp, w in zip(gauss_points, weights):
                xi, eta = gp
                
                # Shape functions and derivatives
                N = 0.25 * np.array([
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta)
                ])
                
                dN_dxi = 0.25 * np.array([
                    [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                ])
                
                # Jacobian matrix
                J = dN_dxi @ np.array(elem_nodes)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                
                # B matrix (strain-displacement)
                dN_dx = invJ @ dN_dxi
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i] = dN_dx[0, i]
                    B[1, 2*i+1] = dN_dx[1, i]
                    B[2, 2*i] = dN_dx[1, i]
                    B[2, 2*i+1] = dN_dx[0, i]
                
                # Element stiffness contribution
                ke += B.T @ D @ B * detJ * w
            
            # Assemble into global matrix
            for i in range(4):
                for j in range(4):
                    for di in range(2):
                        for dj in range(2):
                            rows.append(2*elem[i] + di)
                            cols.append(2*elem[j] + dj)
                            data.append(ke[2*i+di, 2*j+dj])
        
        # Create sparse matrix
        self.K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()
    
    def calculate_force_vector(self):
        """
        Calculate global force vector
        """
        nnodes = len(self.nodes)
        self.F = np.zeros(2 * nnodes)
        
        # Apply nodal loads
        if self.loads:
            for node, load in self.loads.items():
                self.F[2*node] = load[0]  # Fx
                self.F[2*node+1] = load[1]  # Fy
        
        # Apply gravity loads (self-weight)
        rho = self.material_props['rho']
        g = 9.81  # gravity
        
        for elem in self.elements:
            elem_nodes = [self.nodes[n] for n in elem]
            
            # Element area (simplified)
            x = [n[0] for n in elem_nodes]
            y = [n[1] for n in elem_nodes]
            area = 0.5 * abs((x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]) - 
                            (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0]))
            
            # Distribute weight equally to nodes
            elem_weight = area * rho * g
            nodal_weight = elem_weight / 4
            
            for n in elem:
                self.F[2*n+1] -= nodal_weight  # Only vertical component
    
    def apply_boundary_conditions(self):
        """
        Apply boundary conditions by modifying stiffness matrix and force vector
        """
        ndof = len(self.F)
        
        # Create a list of constrained DOFs
        constrained_dofs = []
        for node in self.boundary_conditions:
            constrained_dofs.append(2*node)    # x-direction
            constrained_dofs.append(2*node+1)  # y-direction
        
        # Modify stiffness matrix and force vector
        for dof in constrained_dofs:
            self.K[dof, :] = 0
            self.K[:, dof] = 0
            self.K[dof, dof] = 1
            self.F[dof] = 0
    
    def solve(self):
        """
        Solve the system of equations
        """
        self.calculate_stiffness_matrix()
        self.calculate_force_vector()
        self.apply_boundary_conditions()
        
        # Solve KU = F
        self.U = linalg.spsolve(self.K, self.F)
    
    def get_displacements(self):
        """
        Get displacements at nodes
        :return: List of [u, v] displacements for each node
        """
        nnodes = len(self.nodes)
        displacements = []
        for i in range(nnodes):
            displacements.append([self.U[2*i], self.U[2*i+1]])
        return displacements
    
    def plot_displacements(self, scale=1.0):
        """
        Plot deformed shape
        :param scale: Displacement scaling factor for visualization
        """
        nnodes = len(self.nodes)
        original = np.array(self.nodes)
        displaced = original + scale * np.array(self.get_displacements())
        
        plt.figure(figsize=(12, 6))
        
        # Plot original mesh
        for elem in self.elements:
            x = [self.nodes[n][0] for n in elem] + [self.nodes[elem[0]][0]]
            y = [self.nodes[n][1] for n in elem] + [self.nodes[elem[0]][1]]
            plt.plot(x, y, 'b-', alpha=0.5)
        
        # Plot deformed mesh
        for elem in self.elements:
            x = [displaced[n][0] for n in elem] + [displaced[elem[0]][0]]
            y = [displaced[n][1] for n in elem] + [displaced[elem[0]][1]]
            plt.plot(x, y, 'r-')
        
        plt.title(f'Dam Deformation (Scale factor: {scale})')
        plt.xlabel('Length (m)')
        plt.ylabel('Height (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize FEM model
    dam = FEMDamSettlement()
    
    # Create mesh (100m long, 50m high, 20 elements in x, 10 in y)
    dam.create_mesh(length=100, height=50, nx=20, ny=10)
    
    # Set material properties (typical for compacted earthfill)
    dam.set_material_properties(E=50e6, nu=0.3, rho=2000)
    
    # Set boundary conditions (fix base nodes)
    base_nodes = [i for i, node in enumerate(dam.nodes) if node[1] == 0]
    dam.set_boundary_conditions(base_nodes)
    
    # Solve the system
    dam.solve()
    
    # Get and print maximum settlement
    displacements = dam.get_displacements()
    max_settlement = min(d[1] for d in displacements)  # Negative y is downward
    print(f"Maximum settlement: {max_settlement:.4f} m")
    
    # Plot deformed shape (exaggerated for visualization)
    dam.plot_displacements(scale=100)