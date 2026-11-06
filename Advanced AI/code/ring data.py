import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_ring_data(num_points, inner_radius, outer_radius):
    """
    Generate a ring-shaped dataset with a given number of points, inner radius, and outer radius.

    Parameters:
    - num_points: Number of points in the ring.
    - inner_radius: Inner radius of the ring.
    - outer_radius: Outer radius of the ring.

    Returns:
    - x, y: Coordinates of the points in the ring.
    """
    # Generate evenly spaced angles and random radii within the range
    angles = np.linspace(0, 2 * np.pi, num_points)
    radii = np.random.uniform(inner_radius, outer_radius, num_points)
    # Compute x and y coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

# Generate two rings with different radii
num_points = 500
x1, y1 = generate_ring_data(num_points, 1, 2)  # Ring 1: Radius range [1, 2]
x2, y2 = generate_ring_data(num_points, 3, 4)  # Ring 2: Radius range [3, 4]

# Combine data into a DataFrame and assign labels
ring1_data = pd.DataFrame({'x': x1, 'y': y1, 'label': 'Ring 1'})
ring2_data = pd.DataFrame({'x': x2, 'y': y2, 'label': 'Ring 2'})
all_data = pd.concat([ring1_data, ring2_data], ignore_index=True)

# Write data to a CSV file
csv_file_path = "two_rings_data.csv"
all_data.to_csv(csv_file_path, index=False)
print(f"Data saved to {csv_file_path}")

# Plot the rings
plt.figure(figsize=(8, 8))
plt.scatter(x1, y1, c='gray', label='Ring 1')  # Inner ring in blue
plt.scatter(x2, y2, c='gray', label='Ring 2')   # Outer ring in red
plt.title("Two Rings Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling on both axes
plt.show()
