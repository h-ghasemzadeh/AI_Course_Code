import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = '3D_data.xlsx'
data = pd.read_excel(file_path)

# Extract X, Y, Z columns
X_data = data[['X', 'Y']]
Z_data = data['Z']

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_data)

# Fit a linear regression model on the polynomial features
model = LinearRegression()
model.fit(X_poly, Z_data)

# Predict Z values using the model
Z_pred = model.predict(X_poly)

# Get coefficients for the polynomial equation
intercept = model.intercept_
coefficients = model.coef_

# Format the polynomial equation as a string
equation = (f"Z = {intercept:.2f} "
            f"+ ({coefficients[1]:.2f})*X "
            f"+ ({coefficients[2]:.2f})*Y "
            f"+ ({coefficients[3]:.2f})*X^2 "
            f"+ ({coefficients[4]:.2f})*X*Y "
            f"+ ({coefficients[5]:.2f})*Y^2")

# Create a mesh grid for X and Y for a smooth surface plot
x_range = np.linspace(X_data['X'].min(), X_data['X'].max(), 100)
y_range = np.linspace(X_data['Y'].min(), X_data['Y'].max(), 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# Transform mesh grid data for prediction
X_mesh_poly = poly.transform(np.c_[x_mesh.ravel(), y_mesh.ravel()])
z_mesh_pred = model.predict(X_mesh_poly).reshape(x_mesh.shape)

# Plotting the 3D scatter and regression surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Plot the original data points
ax.scatter(X_data['X'], X_data['Y'], Z_data, color='blue', label="Original Data", alpha=0.5)
# Plot the regression surface
ax.plot_surface(x_mesh, y_mesh, z_mesh_pred, color='orange', alpha=0.6, rstride=100, cstride=100)
# Add the polynomial equation to the plot
ax.text2D(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
# Labels and title
ax.set_title("3D Polynomial Regression Surface")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
