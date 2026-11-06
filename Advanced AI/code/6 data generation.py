import numpy as np
import pandas as pd

# Parameters for the normal distribution
mean = [0, 0, 0]   # Mean for each axis
std_dev = 1        # Standard deviation
num_points = 1000  # Number of data points

# Generating 3D normally distributed data
data_x = np.random.normal(mean[0], std_dev, num_points)
data_y = np.random.normal(mean[1], std_dev, num_points)
data_z = np.random.normal(mean[2], std_dev, num_points)

# Adding noise to each axis
noise_x = np.random.normal(0, 0.2, num_points)
noise_y = np.random.normal(0, 0.2, num_points)
noise_z = np.random.normal(0, 0.2, num_points)

# Creating noisy data
noisy_data_x = data_x + noise_x
noisy_data_y = data_y + noise_y
noisy_data_z = data_z + noise_z

# Creating a DataFrame
data = pd.DataFrame({'X': noisy_data_x, 'Y': noisy_data_y, 'Z': noisy_data_z})
excel_path = "noisy_3D_data.xlsx"

# Saving to Excel with a context manager
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    data.to_excel(writer, sheet_name='3D Normal Data', index=False)

print(f"Noisy 3D data has been saved to {excel_path}")
