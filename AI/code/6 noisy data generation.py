import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for the normal distribution
mean = [0, 0, 10]   # Mean for each axis
std_dev = [1, 2, 3]        # Standard deviation
num_points = 1000  # Number of data points

# Generating 3D normally distributed data
data_x = np.random.normal(mean[0], std_dev[0], num_points)
data_y = np.random.normal(mean[1], std_dev[1], num_points)
data_z = data_x**2 + data_y**2

# Adding noise to each axis
noise_x = np.random.normal(0, 0.2, num_points)  # Adjust noise level as needed
noise_y = np.random.normal(0, 0.5, num_points)
noise_z = np.random.normal(0, 0.1, num_points)

# Creating noisy data
noisy_data_x = data_x + noise_x
noisy_data_y = data_y + noise_y
noisy_data_z = data_z + noise_z

# Creating a DataFrame and saving it to Excel
data = pd.DataFrame({'X': noisy_data_x, 'Y': noisy_data_y, 'Z': noisy_data_z})
excel_path = "3D_data.xlsx"
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# Save DataFrame to an Excel sheet
data.to_excel(writer, sheet_name='3D Normal Data', index=False)

# Saving the 3D plot as an image
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(noisy_data_x, noisy_data_y, noisy_data_z, alpha=0.6, edgecolor='w')
ax.set_title("Noisy 3D Normally Distributed Data")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
image_path = "3D_data.png"
plt.savefig(image_path)  # Save the plot as a PNG file
plt.show()
print(f"Noisy 3D data has been saved to {excel_path}")
print(f"3D plot image has been saved to {image_path}")

# Save and close the Excel writer
writer.close()


