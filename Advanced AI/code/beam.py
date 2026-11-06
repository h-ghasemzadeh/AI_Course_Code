import sympy as sp

# Define symbols
x = sp.symbols('x')
L1, L2, L3 = sp.symbols('L1 L2 L3')  # Lengths of spans
P1, P2, P3 = sp.symbols('P1 P2 P3')  # Point loads
w1, w2, w3 = sp.symbols('w1 w2 w3')  # Uniformly distributed loads
R_A, R_B, R_C, R_D = sp.symbols('R_A R_B R_C R_D')  # Reactions at supports

# Define beam spans and loads
# Example: 3-span beam with lengths L1, L2, L3
# Point loads P1, P2, P3 at specific locations
# UDLs w1, w2, w3 over each span

# Equations for reactions (sum of forces and moments)
# Sum of vertical forces: R_A + R_B + R_C + R_D = P1 + P2 + P3 + w1*L1 + w2*L2 + w3*L3
force_eq = sp.Eq(R_A + R_B + R_C + R_D, P1 + P2 + P3 + w1*L1 + w2*L2 + w3*L3)

# Sum of moments about point A (assuming A is at x=0)
moment_eq_A = sp.Eq(R_B*L1 + R_C*(L1 + L2) + R_D*(L1 + L2 + L3), 
                    P1*L1/2 + P2*(L1 + L2/2) + P3*(L1 + L2 + L3/2) + 
                    w1*L1*(L1/2) + w2*L2*(L1 + L2/2) + w3*L3*(L1 + L2 + L3/2))

# Sum of moments about point B
moment_eq_B = sp.Eq(R_A*L1 + R_C*L2 + R_D*(L2 + L3), 
                    P1*(L1/2) + P2*(L2/2) + P3*(L2 + L3/2) + 
                    w1*L1*(L1/2) + w2*L2*(L2/2) + w3*L3*(L2 + L3/2))

# Solve the system of equations for reactions
reactions = sp.solve([force_eq, moment_eq_A, moment_eq_B], (R_A, R_B, R_C, R_D))

# Display reactions
print("Reactions at supports:")
for reaction, value in reactions.items():
    print(f"{reaction} = {value}")

# Shear force and bending moment equations
# Shear force V(x) = R_A - w1*x (for 0 < x < L1)
V1 = reactions[R_A] - w1*x
V2 = reactions[R_B] - w2*(x - L1)
V3 = reactions[R_C] - w3*(x - L1 - L2)

# Bending moment M(x) = R_A*x - w1*x^2/2 (for 0 < x < L1)
M1 = reactions[R_A]*x - w1*x**2/2
M2 = reactions[R_B]*(x - L1) - w2*(x - L1)**2/2
M3 = reactions[R_C]*(x - L1 - L2) - w3*(x - L1 - L2)**2/2

# Display shear force and bending moment equations
print("\nShear Force Equations:")
print(f"V1(x) = {V1} for 0 < x < {L1}")
print(f"V2(x) = {V2} for {L1} < x < {L1 + L2}")
print(f"V3(x) = {V3} for {L1 + L2} < x < {L1 + L2 + L3}")

print("\nBending Moment Equations:")
print(f"M1(x) = {M1} for 0 < x < {L1}")
print(f"M2(x) = {M2} for {L1} < x < {L1 + L2}")
print(f"M3(x) = {M3} for {L1 + L2} < x < {L1 + L2 + L3}")

# Deflection equations (using double integration method)
# EI * d^2y/dx^2 = M(x)
# Integrate twice to get y(x)

# Example: Deflection for span 1
EI = sp.symbols('EI')  # Flexural rigidity
y1 = sp.integrate(sp.integrate(M1/EI, x), x) + sp.symbols('C1')*x + sp.symbols('C2')

# Apply boundary conditions (e.g., y(0) = 0, y(L1) = 0) to solve for constants C1 and C2
boundary_conditions = [
    sp.Eq(y1.subs(x, 0), 0),
    sp.Eq(y1.subs(x, L1), 0)
]
constants = sp.solve(boundary_conditions, (sp.symbols('C1'), sp.symbols('C2')))

# Substitute constants back into deflection equation
y1_deflection = y1.subs(constants)

print("\nDeflection Equation for Span 1:")
print(f"y1(x) = {y1_deflection}")

# Repeat for other spans as needed